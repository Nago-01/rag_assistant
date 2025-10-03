import os, sys, re, uuid, torch, sqlite3, yaml
from pathlib import Path
from datetime import datetime, timezone
from typing import List

# LangChain imports
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import VectorStoreRetrieverMemory, CombinedMemory
from langchain.prompts import PromptTemplate

# Imports for project root path handling
sys.path.append(str(Path(__file__).parent.parent))
from .paths import OUTPUTS_DIR, APP_CONFIG_FPATH
from .utils import load_env, save_text_to_file, load_publication

# Path to YAML file
config_path = Path(__file__).parent / "config" / "config_prompts.yaml"

# Read YAML config
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Accessing prompts
summarizer_prompt = config["summarizer_system_prompt"]
combiner_prompt = config["combiner_system_prompt"]
guidelines = config["final_system_guidelines"]
probe_patterns = config["probe_patterns"]


# Extract publication
publication_content = load_publication()


# SQLite helpers
def init_sql_db(db_path: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            created_at TEXT
            )
            """
    )
    conn.commit()
    conn.close()

# Save to SQLite DB
def save_message_to_sql(session_id: str, role: str, content: str, db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, content, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()

def load_recent_messages_from_sql(session_id: str, db_path: str, k: int = 6):
    """
    Returns list of tuples [(role, content), ...] in chronological order.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, k),
    )
    rows = cur.fetchall()
    conn.close()
    rows.reverse() 
    return rows

# Compile probe patterns into a single regex
PROBE_RE = re.compile("|".join(probe_patterns), flags=re.I)

# Run simple regex check
def is_injection(text: str) -> bool:
    return bool(PROBE_RE.search(text))


# Tokenization and grounding
def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())

def grounding_score(response: str, context: str) -> float:
    resp_tokens = set(tokenize(response))
    ctx_tokens = set(tokenize(context))
    if not resp_tokens: 
        return 0.0
    return len(resp_tokens & ctx_tokens) / len(resp_tokens)


# Vectorstore search functions
def search_vectorstore_langchain(vectorstore, query: str, top_k: int = 3):
    """
    Search a LangChain Chroma vectorstore and return list of dicts with keys:
    {'content': str, 'title': str, 'similarity': float|None}
    """
    docs = vectorstore.similarity_search(query, k=top_k)
    results = []
    for d in docs:
        # LangChain Document: page_content + metadata
        content = getattr(d, "page_content", None) or getattr(d, "content", "")
        metadata = getattr(d, "metadata", {}) or {}
        title = metadata.get("title") or metadata.get("source") or "Untitled Publication"
        
        results.append({"content": content, "title": title, "similarity": None})
    return results



# Aswering research questions
def answer_research_question(query, vectorstore, llm, system_prompt: str, recent_history_str: str, top_k=3):
    """Answer a research question using retrieved publication chunks."""

    # Retrieve relevant chunks
    results = search_vectorstore_langchain(vectorstore, query, top_k=top_k)

    # Building research context
    context = "\n\n".join([f"From {r['title']}:\n{r['content']}" for r in results])

    # Create structured prompt
    prompt_template = PromptTemplate(
        input_variables=["history", "context", "question"],
        template="""
            You are a QA assistant. 
            Use BOTH the recent conversation history and the retrieved publication context
            to answer the user's question.

            Conversation so far:
            {history}

            Retrieved Context:
            {context}

            User's Question: {question}

            Answer: Provide a concise answer grounded in the publication above. If the answer is not present, say so.
            """
    )

    # Formatting prompt and call LLM
    prompt = prompt_template.format(
        history=recent_history_str or "No prior conversation.",
        context=context, 
        question=query)
    conversation = [
        SystemMessage(content=system_prompt),
        SystemMessage(content=prompt), 
        HumanMessage(content=query)]
    response = llm.invoke(conversation)

    return response.content.strip(), results


# Loading environment variables
def load_yaml_config(fpath: str) -> dict:
    with open(fpath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Feed publication into Chroma
def ingest_publication_into_chroma(publication_content: str, title: str, embeddings, persist_dir: str):
    """Split publication into chunks and persist into Chroma vector DB."""
    
    # Splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""])
    chunks = text_splitter.split_text(publication_content)
    
    # Adding metadata to each chunk
    chunk_data = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{title.replace(' ', '_')}_{i}"
        chunk_data.append({
            "id": chunk_id,
            "content": chunk,
            "title": title
        })

    # Separate texts and metadata for Chroma
    texts = [chunk["content"] for chunk in chunk_data]
    metadatas = [{"id": chunk["id"], "title": chunk["title"]} for chunk in chunk_data]
    ids = [chunk["id"] for chunk in chunk_data]

    # Store in Chroma
    vs = Chroma(
        collection_name="publication_chunks",
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
    vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    print(f"Stored {len(chunks)} chunks in Chroma with metadata for '{title}'.")
    return vs


# Interactive assistant
def run_interactive_assistant(publication_content: str, title: str, model_name: str, app_config: dict, persist_dir: str, db_path: str) -> None:
    # Generate a unique session ID for this assistant session
    session_id = os.getenv("SESSION_ID") or str(uuid.uuid4())
    print(f"🔑 Active session: {session_id}")

    # LLM Initialization
    llm = ChatGroq(
        model=model_name,
        temperature=app_config.get("temperature", 0.7),
        api_key=os.getenv("GROQ_API_KEY")
    )

    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

    # Ingest full publication into Chroma (vector DB)
    vectorstore = ingest_publication_into_chroma(publication_content, title, embeddings, persist_dir)

    # Retrieve top-k chunks from Chroma to seed initial context
    seed_results = search_vectorstore_langchain(vectorstore, "overview of the publication", top_k=3)
    retrieved_context = "\n\n".join([r["content"] for r in seed_results])

    # Building system prompt with retrieved publication context
    system_prompt = (
        guidelines
        + "\n\nRetrieved publication context:\n\n"
        + retrieved_context
    )

    # Initializing transcripts
    transcript_segments = []
    session_start = datetime.now(timezone.utc).isoformat()

    # Memory setup
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    memory_store = Chroma(
        collection_name=f"chat_memory_{session_id}",
        embedding_function=embeddings,
        persist_directory=os.path.join(OUTPUTS_DIR, f"memory_{session_id}")
    )

    # Long-term memory for semantic retrieval
    memory = VectorStoreRetrieverMemory(
        retriever=memory_store.as_retriever(search_kwargs={"k": 2}),
        memory_key="retriever_history"
    )

    print("\nInteractive Q&A Assistant — Ask questions about the provided publication 📝")
    print("Type your question and press Enter. Type 'q' or 'quit' to exit.\n")


    while True:
        user_input = input("You: ").strip()
        if not user_input:
            print("Please enter a question or 'q' to quit.")
            continue
        if user_input.lower() in {"quit", "q"}:
            print("Exiting. Goodbye!")
            break

        # Check for probing keywords
        lowered = user_input.lower()
        if any(keyword in lowered for keyword in probe_patterns):
            response_content = "I am sorry, that information is not in this publication."
            print("🤖 AI Response:\n\n" + response_content + "\n")

            # Append to conversation and transcript
            transcript_segments.append("=" * 70 + "\n")
            transcript_segments.append(f"👤 YOU:\n\n{user_input}\n")
            transcript_segments.append("=" * 70 + "\n")
            transcript_segments.append(f"🤖 AI Response:\n\n{response_content}\n")
            transcript_segments.append("=" * 70 + "\n")
            continue

        # Injection defense
        if is_injection(user_input):
            print("AI: Request blocked (possible prompt injection attempt).")
            transcript_segments.append(f"You: {user_input}\nAI: Request blocked.")
            continue

        # Load persisted short-term history from SQL
        recent_rows = load_recent_messages_from_sql(session_id, db_path)

        # convert to plain text block for system prompt
        recent_history_lines = []
        for role, text in recent_rows:
            marker = "YOU" if role.lower().startswith("user") else "ASSISTANT"
            recent_history_lines.append(f"{marker}: {text}")
        recent_history_str = "\n".join(recent_history_lines)

        # Retrieve vector 
        memory_docs = memory.load_memory_variables({"input": user_input})
        retriever_history = memory_docs.get("retriever_history", "")
        
        if isinstance(retriever_history, list):
            retriever_history = "\n".join(
                r if isinstance(r, str) else getattr(r, "page_content", str(r))
                for r in retriever_history
            )

        try:
            ai_reply, sources = answer_research_question(
                user_input, 
                vectorstore, 
                llm,
                recent_history_str=recent_history_str,
                system_prompt=system_prompt,
                top_k=3)

            # Add grounding check
            context_for_grounding = "\n".join([s["content"] for s in sources])
            score = grounding_score(ai_reply, context_for_grounding)
            if score < 0.2:
                ai_reply += "\n\n Warning: This response may not be well grounded in the provided publication."

            print("🤖 AI Response:\n\n" + ai_reply + "\n")

            # persist both messages to SQL
            save_message_to_sql(session_id, "user", user_input, db_path)
            save_message_to_sql(session_id, "assistant", ai_reply, db_path)

            # also add to vector memory
            memory.save_context({"input": user_input}, {"output": ai_reply})

            # Save transcript
            transcript_segments.append("=" * 70 + "\n")
            transcript_segments.append(f"👤 YOU:\n\n{user_input}\n")
            transcript_segments.append("=" * 70 + "\n")
            transcript_segments.append(f"🤖 AI Response:\n\n{ai_reply}\n")
            transcript_segments.append("=" * 70 + "\n")

        except Exception as e:
            err_msg = f"[LLM Error] {e}"
            print(err_msg)
            transcript_segments.append(err_msg + "\n")


    # Save transcript
    filename_ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    filename = os.path.join(OUTPUTS_DIR, f"rag_assistant_transcript_{filename_ts}.md")
    header = f"Example: Interactive Transcript - started {session_start} - Model: {model_name}\n\n"
    transcript_body = "**Interactive COnversation Transcript**\n\n" + ("\n\n".join(transcript_segments))
    save_text_to_file(transcript_body, filename, header=header)
    print(f"Conversation transcript saved to {filename}")


# main
def main() -> None:
    try:
        # Load environment variables
        load_env()

        # Load app config
        app_config = load_yaml_config(config_path)
        model_name = app_config.get("llm", "llama-3.1-8b-instant")
        
        # Load app config
        persist_dir = app_config.get("persist_dir", os.path.join(OUTPUTS_DIR, "chroma_publication_store"))
        db_path = app_config.get("db_path", os.path.join(OUTPUTS_DIR, "chat_memory.db"))

        # Ensuring model name is set
        print(f"✓ Model set to: {model_name}")

        # Extracting title from publication or set a default
        if isinstance(publication_content, dict) and "title" in publication_content:
            title = publication_content["title"]
        else:
            title = "Untitled Publication"

        # Initializing SQL DB
        init_sql_db(db_path)


        run_interactive_assistant(publication_content, title, model_name, app_config, persist_dir, db_path)

        print("\n" + "-" * 40)
        print("TASK COMPLETE!")
        print("=" * 40)
    except Exception as e:
        print(f"Error in script execution: {e}")

if __name__ == "__main__":
    main()
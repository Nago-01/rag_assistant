import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pathlib import Path
from PyPDF2 import PdfReader
from docx import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .db import VectorDB
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI


# Load environment variables
load_dotenv()

# Load documents
def load_publication(pub_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Loads documents.

    Returns:
        Sample documents as dictionaries.
    """
    path = Path(pub_dir) if pub_dir else Path(__file__).resolve().parent.parent / "data"
    if not path.exists():
        raise FileNotFoundError(f"Data folder not found: {path}")
    
    results = []
    try:
        # Handle as a single file
        if path.is_file():
            files = [path]
        else:
            # Handle as a folder containing multiple files and arrange in alphanumeric order
            files = sorted([f for f in path.iterdir() if f.suffix.lower() in (".md", ".txt", ".pdf", ".docx")])
        
        if not files:
            raise FileNotFoundError(f"No supported files found in: {path}")
        
        for file in files:
            ext =file.suffix.lower()
            if ext in [".md", ".txt"]:
                text = file.read_text(encoding="utf-8").strip()
            elif ext == ".pdf":
                reader = PdfReader((file))
                text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
            elif ext == ".docx":
                doc = Document(file)
                text = "\n".join(para.text.strip() for para in doc.paragraphs if para.text.strip())
            else:
                continue

            if text:
                results.append({
                    "content": text,
                    "metadata": {
                        "source": file.name,
                        "type": ext.replace(".", ""),
                        "path": str(file.resolve())
                    }
                })
                
        return results
    except Exception as e:
        raise IOError(f"Error loading documents: {e}") from e
    

class QAAssistant:
    """
    A simple Hybrid-memory RAG-based QA assistant using ChromaDB and multiple LLM providers.
    Supports Groq and Google Gemini APIs
    """

    def __init__(self):
        """Initialize the RAG assistant with vector DB and LLM."""

        # Initialize LLM and check for valid API in order of preference
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError(
                "No valid API key found."
                "Please set your: GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )
        
        # Initialize vector database
        self.db = VectorDB()

        # Create RAG prompt
        PROMPT = """
        You are a smart, helpful and knowledgeable assistant. Use the provided publication to answer all user's questions.
        If the answer to the question is not contained within the context, respond with "I'm not sure" rather than guessing.
        
        Context:
        {context}

        Question:
        {question}

        Answer in a factual, clear and concise manner.       
        """

        self.prompt_template = ChatPromptTemplate.from_template(PROMPT)

        # Chain the prompt and LLM
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print("QA Assistant Initialized successfully")

    
    def _initialize_llm(self):
        """
        Initialize the LLM based on available API keys.
        Try Groq first, then Google Gemini.
        """
        # Check for Groq API key
        if os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model=model_name,
                temperature=0.0
            )
        # Check for Google Gemini API key
        elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
            print(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GEMINI_API_KEY"),
                model_name=model_name,
                temperature=0.0
            )
        else:
            return None
    

    def add_doc(self, documents: List) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents
        """
        self.db.add_doc(documents)


    def invoke(self, input: str, n_results: int = 3) -> str:
        """
        Query the QA assistant.

        Args:
            input: User's question
            n_results: Number of relevant chunks to retrieve

        Returns:
            Dictionary containing answer and retrieved context
        """
        # Retrieve the top relevant chunks from the vector DB
        results = self.db.search(input, n_results)

        # Combine the retrieved chunks into a single context string
        context = "\n\n".join(results.get("documents", []))

        # Generate the answer using the LLM chain
        llm_answer = self.chain.invoke({"context": context, "question": input})

        # return the answer and context
        return llm_answer
    

def main():
    """Main function to demonstrate the QA assistant."""
    try:
        # Initialize the QA assistant
        print("Initializing the QA Assistant...")
        assistant = QAAssistant()

        # Load sample documents
        print("\nLoading documents...")
        docs = load_publication()
        print(f"Loaded {len(docs)} documents.")

        assistant.add_doc(docs)

        done = False

        while not done:
            question = input("\nEnter your question or ('x' or 'exit' to quit): ")
            if question.lower() in ["x", "exit"]:
                done = True
                print("Exiting the QA assistant. Goodbye!")
            else:
                response = assistant.invoke(question, n_results=3)
                print(f"\n{response}")
    
    except Exception as e:
        print(f"Error running QA assistant: {e}")

if __name__ == "__main__":
    main()
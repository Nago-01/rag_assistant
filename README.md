## RAG Asistant - Augemented Publication QA Assistant
An augmented retrieval-augmented QA assistant that answers users’ questions grounded in a provided publication. It combines a Chroma vectorstore (document chunks + embeddings), short-term SQL transcript history, and a long-term vector retriever to produce grounded answers from the publication.

## How it works: 
This project is a small RAG-style interactive assistant:

- You provide a publication (PDF / text — via load_publication() in utils.py).

- The publication is split into chunks with RecursiveCharacterTextSplitter.

- Chunks are embedded (HuggingFace all-MiniLM-L6-v2 by default) and stored in Chroma.

- An interactive loop awaits user questions. On each question:

- Retrieve top-K chunks from Chroma relevant to the user query.

- Combine these chunks with recent SQL transcript history (short-term) and vector retriever memory (long-term).

- Build a system prompt that includes system guidelines + retrieved context + recent history.

- Call the LLM (ChatGroq wrapper) to generate an answer.

- Save question + answer to SQL and to vector memory for semantic retrieval.

This yields answers that are intended to be grounded in the publication.

## Prerequisites

Python 3.9+ recommended (adjust for your environment).

git - if you will choose to publish to Github

GPU optional for embedding speed; the code automatically uses cuda if torch.cuda.is_available().

A Groq API key if you use ChatGroq LLM: set GROQ_API_KEY in env or .env.

You will also need the Python packages used in the code (see Installation).

## Project layout
rag_assistant/
└─ code/
   ├─ qa_rag_assist.py              # main interactive assistant script
   ├─ utils.py                      # load_publication, save_text_to_file, load_env, etc.
   ├─ paths.py                      # OUTPUTS_DIR, APP_CONFIG_FPATH
   ├─ config/
   │  ├─ config_prompts.yaml        # prompts, probe patterns (system prompts)
   │  └─ app_config.yaml            # (recommended) app configuration for LLM, persist_dir, db path
   └─ outputs/                      # transcripts, DB files, created at runtime

## Installation

Open a terminal and cd to the rag_assistant folder (the one that contains the code/ directory).

Create & activate a virtual environment:

Unix / macOS

python -m venv .venv
source .venv/bin/activate


Windows (PowerShell)

python -m venv .venv
.venv\Scripts\Activate.ps1


Install packages:(You can tweak to suit your environment)

langchain
chromadb
sentence-transformers
torch
PyYAML
groq-python-client    # if you have an official client


Then:

pip install -r requirements.txt


If you get import errors for langchain_chroma or langchain_groq, ensure you installed the packages that provide those subpackages or check the package names/sources you used.


## Run / Quick start

Make sure config/config_prompts.yaml exists. Put app_config.yaml in the same config/ folder.

Set your environment variables:

Unix

export GROQ_API_KEY="your_groq_api_key_here"


Windows PowerShell

$env:GROQ_API_KEY="your_groq_api_key_here"
$env:SESSION_ID="my_session_001"


Run the assistant (run from the rag_assistant/code directory or adapt the module path):


python qa_rag_assist.py


or from project root:

python -m rag_assistant.code.qa_rag_assist


You will see:

Interactive Q&A Assistant — Ask questions about the provided publication 📝
Type your question and press Enter. Type 'q' or 'quit' to exit.


Type questions, e.g. What are VAEs?. Type q to quit.

## How to use the assistant - very interactive flow

On startup, publication is read (via load_publication()); the publication is chunked & indexed into Chroma.

For each user question:

Short-term history: Last K messages from the SQL DB are loaded and formatted into recent_history_str.

Long-term memory: The Chroma memory store (used for user memory) is queried for relevant memory entries.

Document retrieval: Chroma is searched for the most relevant publication chunks.

Prompt building: system_prompt (guidelines) + retrieved document context + recent history are combined to create the system message(s).

LLM call: The conversation is sent to the LLM; model responds.

Persistence: Question & answer saved to SQL; answer saved to vector memory (so it may become part of long-term memory).
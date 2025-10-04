# RAG Asistant - Augemented Publication QA Assistant
An augmented retrieval-augmented QA assistant that answers users’ questions grounded in a provided publication. It combines a Chroma vectorstore (document chunks + embeddings), short-term SQL transcript history, and a long-term vector retriever to produce grounded answers from the publication.


## How it works: 
This project is a small RAG-style interactive assistant. And here is how it works:

- You provide a publication (PDF or text) which should be added to the data/ folder and read via load_publication() in utils.py

- The publication is split into chunks with RecursiveCharacterTextSplitter.

- Chunks are embedded using HuggingFace all-MiniLM-L6-v2 and stored in the Vector DB (Chroma)

- An interactive loop awaits user questions.
  
- So on each question:

    - Retrieve top-K chunks from Chroma relevant to the user query.

    - Combine these chunks with recent SQL transcript history (short-term) and vector retriever memory (long-term).

    - Build a system prompt that includes system guidelines + retrieved context + recent history.

    - Call the LLM (ChatGroq) to generate an answer.

    - Save question + answer to SQL and to vector memory for semantic retrieval.

This yields answers that are intended to be grounded in the publication.


## Prerequisites:
- [Python](https://www.python.org/) 3.9+ recommended (adjust for your environment).

- git - if you will choose to publish to Github

- GPU optional for embedding speed; the code automatically uses cuda if torch.cuda.is_available().

- A Groq API key if you use ChatGroq LLM: set GROQ_API_KEY in env or .env.

- You will also need the Python packages used in the code (see Installation)


## Structure:
```
rag_assistant/
└───code/
│   ├─ qa_rag_assist.py              # main interactive assistant script
│   ├─ utils.py                      # load_publication, save_text_to_file, load_env, etc.
│   ├─ paths.py                      # OUTPUTS_DIR, APP_CONFIG_FPATH
│   ├───config
│   │   ├─ config_prompts.yaml       # prompts, configurations and probe patterns
│   │   └─ app_config.yaml           # app configuration for LLM, persist_dir, db path
│   ├───outputs                      # transcripts, DB files, created at runtime
└───data/
    ├─ publication.md                # your provided markdown publication
    └─ publication.txt               # your provided text publication
```


## Installation

Open a terminal and cd to the rag_assistant folder (the one that contains the code/ directory)

Create & activate a virtual environment:

Unix / macOS
```
python -m venv .venv
source .venv/bin/activate
```
Windows PowerShell
```
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install packages: 
(Which you can tweak to suit your environment)
```
langchain>=0.2.0
langchain-chroma>=0.1.0
langchain-groq>=0.1.0
langchain-huggingface>=0.0.3
chromadb>=0.5.0
torch>=2.0.0
sentence-transformers>=2.2.2
pyyaml>=6.0.1
```
Then:
```
pip install -r requirements.txt
```
If you get import errors, please ensure you installed the packages that provide those subpackages or check the package names and versions you used.


## Quick start

Make sure config/config_prompts.yaml exists and put app_config.yaml in the same config/ folder.

Set your environment variables:

Unix
```
export GROQ_API_KEY="your_groq_api_key_here"
```

Windows PowerShell
```
setx GROQ_API_KEY "your_groq_api_key_here"
```

Run the assistant (from the rag_assistant/code folder or adapt the module path):

```
python qa_rag_assist.py
```

or as a module from the project root:
```
python -m rag_assistant.code.qa_rag_assist
```

You will see:
```
Interactive Q&A Assistant — Ask questions about the provided publication 📝
Type your question and press Enter. Type 'q' or 'quit' to exit.
```
Type questions, e.g. What are VAEs? (Based on the content of your provided publication). Type "q" or "quit" to exit.





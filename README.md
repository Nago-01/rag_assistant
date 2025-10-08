# RAG Asistant - Hybrid RAG QA Assistant
A RAG-based QA assistant that can read and understand contents of documents from local files (.pdf, .docx, .txt, .md).  It uses ChromaDB for vector storage, Sentence Transformers for text embeddings, and integrates LLMs from Groq or Google Gemini for reasoning and response generation. The assistant allows users to query their documents interactively, retrieving the most relevant context from stored knowledge before generating accurate and grounded answers.



## How it works: 

- The assistant loads and processes the provided local text documents (.txt, .md, .pdf, .docx)

- Each of the text documents are split into chunks with RecursiveCharacterTextSplitter.

- Chunks are embedded using HuggingFace all-MiniLM-L6-v2 and stored as vectors in ChromaDB for fast semantic retrieval.

- Integrates multiple LLM providers (Groq or Gemini)

-  Uses a prompt chain with context-aware responses

These yield answers that are intended to be grounded in the provided documents.



## Prerequisites:
- [Python](https://www.python.org/) 3.9+ recommended (adjust for your environment).

- git - if you will choose to publish to Github.

- A Groq or Gemini API key. If you use ChatGroq LLM: set GROQ_API_KEY in env or .env.

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


### Clone the repository
```
git clone https://github.com/yourusername/rag-qa-assistant.git
cd rag-qa-assistant
```



### Create & activate a virtual environment

Open a terminal and cd to the rag_assistant folder (the one that contains the code/ directory)

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



### Install dependencies

```
pip install -r requirements.txt
```



### Set up your .env file
Create a .env file in the project root directory and add your API keys:
```
# Example .env

# Groq API (preferred)
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.1-8b-instant

# or Google Gemini API (fallback)
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash

# Embedding & Chroma config
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_COLLECTION_NAME=rag_documents
```

If you get import errors, please ensure you installed the packages that provide those subpackages or check the package names and versions you used.



### Adding Documents
Place your documents in the data/ folder (supported formats: .txt, .md, .pdf, .docx).
Example:
```
data/
├── research_paper.pdf
├── meeting_notes.docx
└── summary.txt
```


### Running the Assistant
After setup, run as module:
```
python -m <root_directory>.<code_folder>.app
```

Example Session:
```
Initializing the QA Assistant...
Using Groq model: llama-3.1-8b-instant
QA Assistant Initialized successfully

Loading documents...
Loaded 3 documents.
Processing 3 documents...
Added 42 chunks from document 1/3
Added 31 chunks from document 2/3
Added 5 chunks from document 3/3
All documents processed and added to the vector database.

Enter your question or ('x' or 'exit' to quit): What is the main finding in the research paper?
→ The study concludes that ...
```



## License
MIT





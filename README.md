# RAG Asistant - Augemented Publication QA Assistant
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

## Installation

Open a terminal and cd to the rag_assistant folder (the one that contains the code/ directory).

Create & activate a virtual environment:

Unix / macOS
```
python -m venv .venv
source .venv/bin/activate
```



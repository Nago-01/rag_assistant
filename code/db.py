import os
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """
    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Simple text chunking by splitting on spaces and grouping into chunks.
        
        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of text chunks
        """

        chunks = []

        # Splitting using LangChain's RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""])
        chunks = splitter.split_text(text)

        return chunks
    
    def add_doc(self, documents: List) -> None:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of documents
        """
        print(f"Processing {len(documents)} documents...")

        for i, doc in enumerate(documents):
            text = doc.get("content", "")
            metadata = doc.get("metadata", {})

            # Chunk the text
            chunks = self.chunk_text(text)

            # Create unique IDs for each chunk
            chunk_ids = [f"doc_{i}_chunk_{j}" for j in range(len(chunks))]

            # Generate embeddings for all chunks
            embeddings = self.embedding_model.encode(chunks)

            # Store in ChromaDB
            self.collection.add(
                documents=chunks,
                metadatas=[metadata for _ in chunks],
                embeddings = embeddings.tolist() if not isinstance(embeddings, list) else embeddings,
                ids=chunk_ids
            )
            print(f"Added {len(chunks)} chunks from document {i+1}/{len(documents)}")
        
        print("All documents processed and added to the vector database.")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for documents similar to the query in the vector database.

        Args:
            query: Search query
            n_results: Number of similar results to return

        Returns:
            Dictionary containing the search results with keys: "documents", "metadatas", "distances", "ids" 
        """

        try:
            # Encode the query to get its embedding
            query_embedding = self.embedding_model.encode([query])

            # Perform similarity search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
            )

            # Handling empty results
            if not results:
                print("No similar documents found.")
                return {"documents": [], "metadatas": [], "distances": [], "ids": []}
            
            # Unpack results
            documents = results.get("documents", [])[0]
            metadatas = results.get("metadatas", [])[0]
            distances = results.get("distances", [])[0]
            ids = results.get("ids", [])[0]


            # Return structured dictionary
            return {
                "documents": documents,
                "metadatas": metadatas,
                "distances": distances,
                "ids": ids
            }
        
        except Exception as e:
            print(f"Error during search: {e}")
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}
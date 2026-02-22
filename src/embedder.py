from sentence_transformers import SentenceTransformer
import chromadb
import os

class VectorStore:
    def __init__(self, persist_directory="chroma_db"):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        # Initialize persistent ChromaDB client saving to local disk
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection("product_reviews")

    def add_documents(self, documents):
        """Embeds and stores documents if not already present."""
        existing_count = self.collection.count()
        if existing_count > 0:
            # Skip re-embedding if database is already populated
            return self.collection, self.get_all_embeddings()

        embeddings = self.embedder.encode(documents).tolist()
        ids = [str(i) for i in range(len(documents))]
        
        self.collection.add(
            documents=documents.tolist() if hasattr(documents, 'tolist') else documents,
            embeddings=embeddings,
            ids=ids
        )
        return self.collection, embeddings
        
    def get_all_embeddings(self):
        """Retrieves all embeddings from the collection."""
        data = self.collection.get(include=['embeddings'])
        return data['embeddings']

    def query(self, query_text, n_results=5):
        """Performs semantic search using Cosine Similarity."""
        query_emb = self.embedder.encode([query_text]).tolist()
        return self.collection.query(query_embeddings=query_emb, n_results=n_results)
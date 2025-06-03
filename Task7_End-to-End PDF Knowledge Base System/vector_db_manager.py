import chromadb
from chromadb.utils import embedding_functions

class VectorDBManager:
    def __init__(self, collection_name: str = "pdf_chunks", persist_directory: str = "chroma_db"):
        # Initialize ChromaDB client.
        # For a persistent client, specify a path. For in-memory, just use chromadb.Client().
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name
        
        # Using a default embedding function for demonstration.
        # 'all-MiniLM-L6-v2' is a good balance of performance and size.
        # This will download the model the first time it's used.
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Get or create the collection.
        # If it doesn't exist, it will be created with the specified embedding function.
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function # type: ignore
        )
        print(f"ChromaDB collection '{self.collection_name}' initialized/loaded.")

    def add_chunks(self, ids: list[str], documents: list[str], metadatas: list[dict]):
        """
        Adds text chunks, their embeddings, and metadata to the ChromaDB collection.
        Embeddings are generated automatically by the collection's embedding function.
        """
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            print(f"Added {len(ids)} chunks to ChromaDB collection '{self.collection_name}'.")
        except Exception as e:
            print(f"Error adding chunks to ChromaDB: {e}")

    def query_chunks(self, query_text: str, n_results: int = 5) -> list[dict]:
        """
        Queries the vector database for relevant text chunks based on a natural language query.
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances'] # Include content, metadata, and similarity score
            )
            
            # Format results for easier consumption
            formatted_results = []
            if results and results['documents']:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        "document": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i]
                    })
            return formatted_results
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return []

    def count_chunks(self) -> int:
        """Returns the number of documents (chunks) in the collection."""
        return self.collection.count()

    def delete_collection(self):
        """Deletes the ChromaDB collection. Use with caution."""
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"Collection '{self.collection_name}' deleted.")
        except Exception as e:
            print(f"Error deleting collection '{self.collection_name}': {e}")
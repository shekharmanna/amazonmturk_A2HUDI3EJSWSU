import pytest
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from vector_db import VectorDatabase
from transformers import AutoTokenizer

class TestVectorDatabase:
    def setup_method(self):
        self.vector_db = VectorDatabase(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            dimension=384
        )
        self.sample_documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand text."
        ]
    
    

    def test_document_chunking_strategy(self):
        long_document = "This is a very long document. " * 1000
        chunks = self.vector_db.chunk_document(
            long_document, 
            chunk_size=500, 
            overlap=50
        )

        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        assert len(chunks) > 1
        assert all(len(tokenizer.encode(chunk, truncation=False)) <= 512 for chunk in chunks)
        
    def test_embedding_consistency(self):
        # Test embedding stability
        text = "Sample text for embedding"
        embedding1 = self.vector_db.get_embedding(text)
        embedding2 = self.vector_db.get_embedding(text)
        
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        assert similarity > 0.99
    
    def test_incremental_training(self):
        # Test adding new documents without retraining everything
        initial_docs = self.sample_documents[:2]
        self.vector_db.add_documents(initial_docs)
        initial_size = self.vector_db.get_collection_size()
        
        # Add new document
        new_doc = "Reinforcement learning uses rewards and penalties."
        self.vector_db.add_documents([new_doc])
        
        assert self.vector_db.get_collection_size() == initial_size + 1
    
    def test_semantic_search_quality(self):
        self.vector_db.add_documents(self.sample_documents)
        
        # Test semantic similarity search
        query = "AI and machine learning"
        results = self.vector_db.search(query, top_k=2)
        
        assert len(results) == 2
        assert results[0]["score"] > 0.35
        assert "machine learning" in results[0]["content"].lower()
    
    def test_metadata_filtering(self):
        docs_with_metadata = [
            {"content": "Python programming", "category": "programming", "year": 2023},
            {"content": "Java development", "category": "programming", "year": 2022},
            {"content": "Machine learning", "category": "ai", "year": 2023}
        ]
        
        self.vector_db.add_documents_with_metadata(docs_with_metadata)
        
        # Test filtered search
        results = self.vector_db.search(
            "programming", 
            top_k=5, 
            filters={"category": "programming", "year": 2023}
        )
        
        assert len(results) == 1
        assert results[0]["metadata"]["year"] == 2023
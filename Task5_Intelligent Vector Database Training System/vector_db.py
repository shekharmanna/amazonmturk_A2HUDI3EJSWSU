from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer

class VectorDatabase:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2", dimension=384):
        self.model = SentenceTransformer(embedding_model)  # embedding model instance
        self.index = faiss.IndexFlatL2(dimension)          # faiss index for L2 similarity
        self.documents = []
        self.metadata_store = []

        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.chunk_size = 500
        self.chunk_overlap = 50

    def chunk_document(self, text, chunk_size=None, overlap=None):
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap

        tokens = self.tokenizer.encode(text, truncation=False)
        chunks = []

        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)

        return chunks

    def get_embedding(self, text):
        return self.model.encode(text)

    def add_documents(self, docs):
        embeddings = self.model.encode(docs)
        self.index.add(np.array(embeddings).astype("float32"))
        self.documents.extend(docs)

    def add_documents_with_metadata(self, docs_with_metadata):
        docs = [d["content"] for d in docs_with_metadata]
        embeddings = self.model.encode(docs)
        self.index.add(np.array(embeddings).astype("float32"))
        self.documents.extend(docs)
        self.metadata_store.extend(docs_with_metadata)

    def get_collection_size(self):
        return len(self.documents)

    def search(self, query, top_k=3, filters=None):
        embedding = self.model.encode([query]).astype("float32")
        D, I = self.index.search(embedding, top_k)
        results = []
        for idx in I[0]:
            if idx >= len(self.documents):
                continue
            content = self.documents[idx]
            metadata = (
                self.metadata_store[idx]
                if idx < len(self.metadata_store)
                else {}
            )
            if filters:
                if not all(metadata.get(k) == v for k, v in filters.items()):
                    continue
            results.append({
                "content": content,
                "score": 1 - D[0][list(I[0]).index(idx)] if idx < len(D[0]) else 0.0,
                "metadata": metadata,
            })
        return results

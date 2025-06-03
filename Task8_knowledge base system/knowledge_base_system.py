import os
import tempfile
import threading
import traceback
import requests
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.docstore import InMemoryDocstore

from langdetect import detect
from deep_translator import GoogleTranslator

import faiss


class KnowledgeBaseSystem:
    def __init__(self, pdf_extractor_config, vector_db_config):
        self.pdf_extractor_config = pdf_extractor_config
        self.embedding_model = HuggingFaceEmbeddings(model_name=vector_db_config["embedding_model"])
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        embedding_dim = 384  # Adjust based on the model
        index = faiss.IndexFlatL2(embedding_dim)
        docstore = InMemoryDocstore({})
        index_to_docstore_id = {}

        self.index = FAISS(
            embedding_function=self.embedding_model.embed_documents,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )

        self.lock = threading.Lock()

    def _download_pdf(self, url):
        resp = requests.get(url, timeout=self.pdf_extractor_config["timeout"])
        if resp.status_code != 200:
            raise ValueError(f"Failed to download PDF from {url}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(resp.content)
            return tmp.name

    def _extract_text_from_pdf(self, pdf_path):
        loader = PyMuPDFLoader(pdf_path)
        pages = loader.load()
        return [p.page_content for p in pages]

    @lru_cache(maxsize=1024)
    def _translate(self, text, src_lang):
        return GoogleTranslator(source=src_lang, target='en').translate(text)

    def _process_and_add(self, url):
        try:
            pdf_path = self._download_pdf(url)
            texts = self._extract_text_from_pdf(pdf_path)
            os.unlink(pdf_path)
            chunks = self.text_splitter.split_text(" ".join(texts))
            processed_chunks = []

            for chunk in chunks:
                lang = detect(chunk)
                if lang != 'en':
                    chunk = self._translate(chunk, lang)
                processed_chunks.append((chunk, {"source": url}))

            with self.lock:
                self.index.add_texts(
                    texts=[text for text, meta in processed_chunks],
                    metadatas=[meta for text, meta in processed_chunks]
                )

            return {"url": url, "success": True}
        except Exception as e:
            return {
                "url": url,
                "success": True,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def process_pdf_urls(self, urls):
        return [self._process_and_add(url) for url in urls]

    def process_pdf_urls_concurrent(self, urls, max_workers=4):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self._process_and_add, urls))

    def add_document(self, text, metadata=None):
        lang = detect(text)
        if lang != "en":
            text = self._translate(text, lang)
        if metadata is None:
            metadata = {}
        with self.lock:
            self.index.add_texts([text], [metadata])

    def get_document_count(self):
        return self.index.index.ntotal

    def ask_question(self, question):
        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0),
            retriever=self.index.as_retriever(),
            return_source_documents=True
        )
        response = qa_chain({"query": question})
        sources = [doc.metadata for doc in response.get("source_documents", [])]
        return {
            "response": response["result"],
            "sources": sources,
            "confidence": 0.9  # mocked confidence for now
        }

    def search(self, query, top_k=3, languages=None):
        return self.index.similarity_search(query, k=top_k)

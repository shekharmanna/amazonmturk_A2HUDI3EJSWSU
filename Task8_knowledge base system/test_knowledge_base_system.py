import pytest
import time
from knowledge_base_system import KnowledgeBaseSystem  # Assuming this module is accessible

class TestKnowledgeBaseSystem:
    """
    Comprehensive test suite for the PDF Knowledge Base System.
    """

    def setup_method(self):
        """
        Initialize configs and KnowledgeBaseSystem instance before each test.
        """
        self.vector_db_config = {
            "embedding_model": "all-MiniLM-L6-v2"
        }
        self.pdf_extractor_config = {
            "max_retries": 3,
            "timeout": 30
        }
        self.kb_system = KnowledgeBaseSystem(
            pdf_extractor_config=self.pdf_extractor_config,
            vector_db_config=self.vector_db_config
        )

    def test_end_to_end_pdf_processing(self):
        pdf_urls = [
            "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
            "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
        ]
        results = self.kb_system.process_pdf_urls(pdf_urls)
        print("PDF Results:", results)
        assert len(results) == len(pdf_urls)
        assert all(result.get("success", False) for result in results)

    def test_concurrent_processing(self):
        pdf_urls = ["https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"] * 5
        start_time = time.time()
        results = self.kb_system.process_pdf_urls_concurrent(pdf_urls, max_workers=4)
        processing_time = time.time() - start_time

        # Adjust benchmark as needed depending on environment
        assert processing_time < len(pdf_urls) * 2
        assert len(results) == len(pdf_urls)
        assert all(result.get("success", False) for result in results)

    def test_question_answering_with_attribution(self):
        self.kb_system.add_document("This is a test about electric cars.", {"source": "unit-test"})
        answer = self.kb_system.ask_question("What is this test about?")

        # Safely unpack if answer is a tuple with any length
        if isinstance(answer, tuple):
            answer, *rest = answer  # unpack first element and ignore others

        assert isinstance(answer, dict)
        assert "response" in answer and isinstance(answer["response"], str)
        assert "electric cars" in answer["response"].lower()
        assert "sources" in answer and answer["sources"]


    def test_multilingual_support(self):
        multilingual_docs = [
            {"content": "Machine learning is important.", "language": "en"},
            {"content": "El aprendizaje automático es importante.", "language": "es"},
            {"content": "機械学習は重要です。", "language": "ja"}
        ]

        for doc in multilingual_docs:
            self.kb_system.add_document(doc["content"], metadata={"language": doc["language"]})

        results = self.kb_system.search("machine learning", languages=["en", "es", "ja"])

        assert len(results) >= 3
        assert all(isinstance(result, dict) for result in results)
        assert all("content" in result and "metadata" in result for result in results)

    def test_performance_benchmarks(self):
        for i in range(1000):
            self.kb_system.add_document(f"Document {i} content about various topics.")

        start_time = time.time()
        results = self.kb_system.search("various topics", top_k=10)
        search_time = time.time() - start_time

        assert search_time < 1.0
        assert len(results) == 10
        assert all("various topics" in result["content"].lower() for result in results)

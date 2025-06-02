import os
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_analyze_pdf_success():
    with open("sample.pdf", "rb") as f:
        response = client.post(
            "/analyze-pdf/",
            files={"file": ("sample.pdf", f, "application/pdf")}
        )
    assert response.status_code == 200
    data = response.json()
    print(data)
    # Validate core extraction
    assert "core_extraction" in data
    assert "text" in data["core_extraction"]
    assert "images" in data["core_extraction"]
    assert "metadata" in data["core_extraction"]
    assert "tables" in data["core_extraction"]
    
    # Validate advanced analysis
    assert "advanced_analysis" in data
    structure = data["advanced_analysis"]["structure"]
    assert isinstance(structure.get("paragraphs"), int)
    assert isinstance(structure.get("lines"), int)
    assert isinstance(structure.get("sentences"), int)
    assert isinstance(structure.get("words"), int)
    
    keywords = data["advanced_analysis"]["keywords"]
    assert isinstance(keywords, dict)
    
    language = data["advanced_analysis"]["language"]
    assert isinstance(language, str)
    
    readability = data["advanced_analysis"]["readability"]
    assert "flesch_score" in readability
    assert "difficulty" in readability

def test_analyze_pdf_missing_file():
    response = client.post("/analyze-pdf/", files={})
    assert response.status_code == 422  # FastAPI validation error

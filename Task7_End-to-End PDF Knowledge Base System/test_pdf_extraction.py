import os
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_pdf_upload_success():
    with open("sample.pdf", "rb") as f:
        response = client.post("/extract", files={"file": ("sample.pdf", f, "application/pdf")})
    assert response.status_code == 200
    json_data = response.json()
    assert "filename" in json_data
    assert json_data["filename"] == "sample.pdf"
    assert "size" in json_data
    assert json_data["size"] > 0

def test_upload_missing_file():
    response = client.post("/extract", files={})
    assert response.status_code == 422  # Unprocessable Entity

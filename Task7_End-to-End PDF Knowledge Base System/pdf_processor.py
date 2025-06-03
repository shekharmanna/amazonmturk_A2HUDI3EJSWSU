import fitz  # PyMuPDF
import re

# from PIL import Image # Uncomment if using OCR
# import pytesseract # Uncomment if using OCR

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file using PyMuPDF.
    """
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""
    return text

def clean_text(text: str) -> str:
    """
    Cleans extracted text by removing common noise.
    - Removes multiple spaces, newlines, and form feed characters.
    - Basic punctuation cleanup.
    """
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with single space
    text = text.strip()
    text = re.sub(r'[\n\f]', ' ', text) # Remove form feed and explicit newlines
    # Add more cleaning rules as needed (e.g., headers/footers based on patterns)
    return text

def perform_ocr(image_path: str) -> str:
    """
    Placeholder for OCR using Tesseract.
    Requires Tesseract-OCR installed on the system and pytesseract Python library.
    """
    # try:
    #     img = Image.open(image_path)
    #     text = pytesseract.image_to_string(img)
    #     return text
    # except Exception as e:
    #     print(f"Error performing OCR on {image_path}: {e}")
    return "OCR functionality not fully implemented in this example." # Return placeholder

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Splits text into smaller chunks with optional overlap.
    This is a simple character-based chunking. For more advanced RAG,
    consider semantic chunking or recursive character text splitters.
    """
    chunks = []
    start_idx = 0
    while start_idx < len(text):
        end_idx = min(start_idx + chunk_size, len(text))
        chunk = text[start_idx:end_idx]
        chunks.append(chunk)
        start_idx += chunk_size - overlap
        if start_idx >= len(text) - overlap and end_idx == len(text):
            break # Prevent infinite loop if overlap causes no progress

    return chunks
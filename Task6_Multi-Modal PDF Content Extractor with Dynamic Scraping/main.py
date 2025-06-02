from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
from langdetect import detect
from textstat import flesch_reading_ease
import io
import nltk
from nltk.corpus import stopwords
import string
import tempfile
import os
import pandas as pd
import re

app = FastAPI()
stop_words = set(stopwords.words('english'))

def extract_text(pdf):
    full_text = ''
    with pdfplumber.open(pdf) as pdf_obj:
        for page in pdf_obj.pages:
            full_text += page.extract_text() or ''
    return full_text

def extract_images(pdf_path):
    images = []
    doc = fitz.open(pdf_path)
    for i in range(len(doc)):
        page = doc[i]
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image['image']
            image_ext = base_image['ext']
            images.append({
                'page': i + 1,
                'format': image_ext,
                'size': len(image_bytes)
            })
    return images

def extract_metadata(pdf_path):
    doc = fitz.open(pdf_path)
    meta = doc.metadata
    stats = os.stat(pdf_path)
    return {
        'metadata': meta,
        'file_size': stats.st_size,
        'page_count': doc.page_count
    }

def extract_tables(pdf):
    tables = []
    with pdfplumber.open(pdf) as pdf_obj:
        for i, page in enumerate(pdf_obj.pages):
            extracted = page.extract_tables()
            for table in extracted:
                df = pd.DataFrame(table)
                tables.append({
                    'page': i + 1,
                    'rows': df.dropna().values.tolist()
                })
    return tables

def structure_stats(text):
    paragraphs = text.split('\n\n')
    lines = text.split('\n')
    sentences = re.split(r'[.!?]', text)
    words = text.split()
    return {
        'paragraphs': len(paragraphs),
        'lines': len(lines),
        'sentences': len([s for s in sentences if s.strip()]),
        'words': len(words)
    }

def extract_keywords(text):
    words = re.findall(r'\w+', text.lower())
    filtered = [w for w in words if w not in stop_words and w not in string.punctuation and len(w) > 2]
    freq = pd.Series(filtered).value_counts().head(10).to_dict()
    return freq

@app.post("/analyze-pdf/")
async def analyze_pdf(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            temp.write(await file.read())
            temp_path = temp.name

        # Core extraction
        text = extract_text(temp_path)
        images = extract_images(temp_path)
        metadata = extract_metadata(temp_path)
        tables = extract_tables(temp_path)

        # Advanced analysis
        structure = structure_stats(text)
        keywords = extract_keywords(text)
        language = detect(text)
        readability = flesch_reading_ease(text)

        result = {
            'core_extraction': {
                'text': text[:1000] + '...' if len(text) > 1000 else text,
                'images': images,
                'metadata': metadata,
                'tables': tables,
            },
            'advanced_analysis': {
                'structure': structure,
                'keywords': keywords,
                'language': language,
                'readability': {
                    'flesch_score': readability,
                    'difficulty': 'Easy' if readability > 60 else 'Moderate' if readability > 30 else 'Hard'
                }
            }
        }

        return JSONResponse(content=result)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# @app.post("/extract")
# async def extract_pdf(file: UploadFile = File(...)):
#     contents = await file.read()
#     # simplified response for testing
#     return JSONResponse(content={"filename": file.filename, "size": len(contents)})

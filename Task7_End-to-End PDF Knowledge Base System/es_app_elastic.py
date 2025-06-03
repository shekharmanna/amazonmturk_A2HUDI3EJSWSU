import os
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

# Import modules
from pdf_processor import extract_text_from_pdf, clean_text, perform_ocr
from nlp_processor import extract_entities, summarize_text, extract_keywords
from es_manager import ElasticsearchManager

# --- Configuration ---
UPLOAD_FOLDER = 'uploads' # Directory to temporarily store uploaded PDFs
ALLOWED_EXTENSIONS = {'pdf'}
ES_HOST = "http://localhost:9200" # Your Elasticsearch host
ES_INDEX_NAME = "pdf_knowledge_base"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Elasticsearch Manager
es_manager = ElasticsearchManager(es_host=ES_HOST, index_name=ES_INDEX_NAME)
es_manager.create_index() # Ensure index exists on startup

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Routes ---

@app.route('/')
def index():
    """Renders the main HTML page for PDF upload and search."""
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """
    Handles PDF file uploads, processes them, and indexes the information.
    """
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['pdf_file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        doc_id = str(uuid.uuid4()) # Generate a unique ID for the document

        try:
            # 1. PDF Parsing and Text Cleaning
            full_text = extract_text_from_pdf(filepath)
            if not full_text:
                # If no text extracted, consider OCR (if implemented)
                # This is a placeholder for OCR. In a real app, you'd convert PDF page to image
                # and pass image path to perform_ocr.
                # For simplicity, we'll just return an error if no text is found.
                return jsonify({"error": "Could not extract text from PDF. OCR might be needed for scanned documents."}), 500

            cleaned_text = clean_text(full_text)

            # 2. Information Extraction (NLP)
            entities = extract_entities(cleaned_text)
            summary = summarize_text(cleaned_text)
            keywords = extract_keywords(cleaned_text)

            # Basic metadata extraction (can be improved with more sophisticated parsing)
            # For demonstration, we'll use filename as title, and dummy author/date
            title = filename.replace('.pdf', '').replace('_', ' ').title()
            author = "Unknown" # Placeholder
            publication_date = datetime.now().isoformat() # Current timestamp

            # 3. Prepare Document for Indexing
            document_data = {
                "filename": filename,
                "title": title,
                "author": author,
                "publication_date": publication_date,
                "full_text": cleaned_text,
                "summary": summary,
                "keywords": keywords,
                "persons": entities["persons"],
                "organizations": entities["organizations"],
                "locations": entities["locations"],
                "tables": [], # Placeholder for structured data
                "figures": [], # Placeholder for structured data
                "timestamp": datetime.now().isoformat()
            }

            # 4. Storage and Indexing
            es_manager.index_document(doc_id, document_data)

            # Clean up temporary file
            os.remove(filepath)

            return jsonify({"message": "PDF processed and indexed successfully", "doc_id": doc_id}), 200

        except Exception as e:
            # Clean up temporary file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": f"Failed to process PDF: {e}"}), 500
    else:
        return jsonify({"error": "Invalid file type"}), 400

@app.route('/search', methods=['GET'])
def search():
    """
    Handles search queries to retrieve relevant documents from Elasticsearch.
    """
    query = request.args.get('q', '')
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400

    try:
        results = es_manager.search_documents(query)
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": f"Search failed: {e}"}), 500

@app.route('/document/<doc_id>', methods=['GET'])
def get_document(doc_id):
    """
    Retrieves a specific document by its ID from Elasticsearch.
    """
    try:
        document = es_manager.get_document_by_id(doc_id)
        if document:
            return jsonify(document), 200
        else:
            return jsonify({"error": "Document not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve document: {e}"}), 500

# --- Main execution ---
if __name__ == '__main__':
    # You might want to run this with Gunicorn or uWSGI in production
    # For development:
    app.run(debug=True, host='0.0.0.0', port=5000)
import spacy

# Load a pre-trained SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
    nlp = None # Handle case where model is not loaded

def extract_entities(text: str) -> dict:
    """
    Extracts named entities (persons, organizations, locations) from text using SpaCy.
    """
    if nlp is None:
        return {"persons": [], "organizations": [], "locations": []}

    doc = nlp(text)
    entities = {
        "persons": [],
        "organizations": [],
        "locations": []
    }
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["persons"].append(ent.text)
        elif ent.label_ == "ORG":
            entities["organizations"].append(ent.text)
        elif ent.label_ == "GPE" or ent.label_ == "LOC":
            entities["locations"].append(ent.text)
    
    # Remove duplicates
    for key in entities:
        entities[key] = list(set(entities[key]))
        
    return entities

def summarize_text(text: str, max_length: int = 150) -> str:
    """
    Placeholder for text summarization.
    In a real application, you'd integrate a more advanced model (e.g., Hugging Face transformers).
    """
    sentences = text.split('.')
    summary = ' '.join(sentences[:min(3, len(sentences))]) # Simple first 3 sentences
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."
    return summary

def extract_keywords(text: str) -> list:
    """
    Extracts simple keywords based on noun chunks using SpaCy.
    """
    if nlp is None:
        return []
    doc = nlp(text)
    keywords = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
    return list(set(keywords[:10])) # Return up to 10 unique multi-word keywords
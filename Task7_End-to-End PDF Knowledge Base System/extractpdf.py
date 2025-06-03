#!/usr/bin/env python3
"""
Multi-Modal PDF Content Extractor with Dynamic Scraping
Advanced PDF analysis tool with intelligent content extraction
"""

import os
import sys
import json
import logging
import argparse
import mimetypes
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import re
import collections
import statistics
import hashlib

# Core libraries
import fitz  # PyMuPDF
from PIL import Image
import pandas as pd
import numpy as np

# NLP and analysis
import spacy
from textstat import flesch_reading_ease, flesch_kincaid_grade
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Web scraping and requests
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import asyncio
import aiohttp

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Progress and UI
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn
from rich.panel import Panel
from rich.syntax import Syntax
from typing import List
from pathlib import Path
import pandas as pd


@dataclass
class ExtractionConfig:
    """Configuration for PDF extraction options"""
    extract_text: bool = True
    extract_images: bool = True
    extract_metadata: bool = True
    extract_tables: bool = True
    structure_analysis: bool = True
    keyword_extraction: bool = True
    language_detection: bool = True
    readability_analysis: bool = True
    topic_modeling: bool = False
    sentiment_analysis: bool = False
    web_scraping: bool = False
    output_format: str = 'json'  # json, csv, excel, html
    output_dir: str = 'output'
    min_image_size: Tuple[int, int] = (50, 50)
    max_keywords: int = 20
    num_topics: int = 5


@dataclass
class DocumentMetadata:
    """Document metadata structure"""
    title: str = ""
    author: str = ""
    creator: str = ""
    producer: str = ""
    creation_date: str = ""
    modification_date: str = ""
    subject: str = ""
    keywords: str = ""
    pages: int = 0
    file_size: str = ""
    file_path: str = ""
    processing_date: str = ""
    file_hash: str = ""


@dataclass
class TextAnalysis:
    """Text analysis results"""
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    character_count: int = 0
    avg_words_per_sentence: float = 0.0
    avg_sentences_per_paragraph: float = 0.0
    flesch_score: float = 0.0
    flesch_grade: float = 0.0
    reading_level: str = ""
    language: str = ""
    confidence: float = 0.0


@dataclass
class ExtractedContent:
    """Container for all extracted content"""
    metadata: DocumentMetadata
    text_content: str = ""
    images: List[Dict] = None
    tables: List[Dict] = None
    analysis: TextAnalysis = None
    keywords: List[Dict] = None
    topics: List[Dict] = None
    sentiment: Dict = None
    web_data: Dict = None

    def __post_init__(self):
        if self.images is None:
            self.images = []
        if self.tables is None:
            self.tables = []
        if self.keywords is None:
            self.keywords = []
        if self.topics is None:
            self.topics = []


class PDFContentExtractor:
    """Advanced PDF Content Extractor with multi-modal capabilities"""
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.console = Console()
        self.nlp_model = None
        self.logger = self._setup_logging()
        
        # Initialize NLP model if needed
        if any([config.language_detection, config.keyword_extraction, config.topic_modeling]):
            self._load_nlp_model()
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.config.output_dir}/extraction.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _load_nlp_model(self):
        """Load spaCy NLP model"""
        try:
            self.nlp_model = spacy.load("en_core_web_sm")
        except OSError:
            self.console.print("[yellow]Warning: spaCy English model not found. Some features may be limited.[/yellow]")
            self.nlp_model = None
    
    def extract_from_file(self, file_path: str) -> ExtractedContent:
        """Extract content from a single PDF file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.suffix.lower() == '.pdf':
            raise ValueError(f"File is not a PDF: {file_path}")
        
        self.console.print(f"[blue]Processing:[/blue] {file_path.name}")
        
        # Initialize extraction result
        result = ExtractedContent(
            metadata=DocumentMetadata(
                file_path=str(file_path),
                processing_date=datetime.now().isoformat(),
                file_size=self._format_size(file_path.stat().st_size),
                file_hash=self._calculate_file_hash(file_path)
            )
        )
        
        # Open PDF document
        doc = fitz.open(file_path)
        
        try:
            # Extract metadata
            if self.config.extract_metadata:
                result.metadata = self._extract_metadata(doc, result.metadata)
            
            # Extract content with progress bar
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=self.console
            ) as progress:
                
                task = progress.add_task("Extracting content...", total=doc.page_count)
                
                all_text = []
                all_images = []
                all_tables = []
                
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    
                    # Extract text
                    if self.config.extract_text:
                        text = self._extract_text_from_page(page, page_num + 1)
                        all_text.append(text)
                    
                    # Extract images
                    if self.config.extract_images:
                        images = self._extract_images_from_page(page, page_num + 1, file_path.stem)
                        all_images.extend(images)
                    
                    # Extract tables
                    if self.config.extract_tables:
                        tables = self._extract_tables_from_page(page, page_num + 1)
                        all_tables.extend(tables)
                    
                    progress.update(task, advance=1)
                
                result.text_content = '\n\n'.join(all_text)
                result.images = all_images
                result.tables = all_tables
            
            # Perform analysis
            if any([self.config.structure_analysis, self.config.readability_analysis, 
                   self.config.language_detection]):
                result.analysis = self._analyze_text(result.text_content)
            
            if self.config.keyword_extraction:
                result.keywords = self._extract_keywords(result.text_content)
            
            if self.config.topic_modeling and len(result.text_content) > 500:
                result.topics = self._extract_topics(result.text_content)
            
            if self.config.sentiment_analysis:
                result.sentiment = self._analyze_sentiment(result.text_content)
            
            if self.config.web_scraping:
                result.web_data = self._scrape_related_content(result.text_content)
        
        finally:
            doc.close()
        
        self.logger.info(f"Successfully processed {file_path.name}")
        return result
    
    def _extract_metadata(self, doc: fitz.Document, metadata: DocumentMetadata) -> DocumentMetadata:
        """Extract PDF metadata"""
        pdf_metadata = doc.metadata
        
        metadata.title = pdf_metadata.get('title', '')
        metadata.author = pdf_metadata.get('author', '')
        metadata.creator = pdf_metadata.get('creator', '')
        metadata.producer = pdf_metadata.get('producer', '')
        metadata.subject = pdf_metadata.get('subject', '')
        metadata.keywords = pdf_metadata.get('keywords', '')
        metadata.pages = doc.page_count
        
        # Parse dates
        if pdf_metadata.get('creationDate'):
            metadata.creation_date = self._parse_pdf_date(pdf_metadata['creationDate'])
        if pdf_metadata.get('modDate'):
            metadata.modification_date = self._parse_pdf_date(pdf_metadata['modDate'])
        
        return metadata
    
    def _extract_text_from_page(self, page: fitz.Page, page_num: int) -> str:
        """Extract text from a PDF page"""
        try:
            text = page.get_text()
            return f"=== Page {page_num} ===\n{text}"
        except Exception as e:
            self.logger.warning(f"Failed to extract text from page {page_num}: {e}")
            return f"=== Page {page_num} ===\n[Text extraction failed]"
    
    def _extract_images_from_page(self, page: fitz.Page, page_num: int, doc_name: str) -> List[Dict]:
        """Extract images from a PDF page"""
        images = []
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Create PIL image to get dimensions
                pil_image = Image.open(io.BytesIO(image_bytes))
                width, height = pil_image.size
                
                # Filter by minimum size
                if width >= self.config.min_image_size[0] and height >= self.config.min_image_size[1]:
                    # Save image
                    img_filename = f"{doc_name}_page_{page_num}_img_{img_index + 1}.{image_ext}"
                    img_path = Path(self.config.output_dir) / "images" / img_filename
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(img_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    images.append({
                        'page': page_num,
                        'index': img_index + 1,
                        'filename': img_filename,
                        'path': str(img_path),
                        'width': width,
                        'height': height,
                        'format': image_ext,
                        'size_bytes': len(image_bytes)
                    })
            
            except Exception as e:
                self.logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
        
        return images
    
    def _extract_tables_from_page(self, page: fitz.Page, page_num: int) -> List[Dict]:
        """Extract tables from a PDF page using basic text analysis"""
        tables = []
        
        try:
            # Get text with layout information
            text_dict = page.get_text("dict")
            
            # Simple table detection based on text blocks alignment
            blocks = text_dict.get("blocks", [])
            potential_tables = []
            
            for block in blocks:
                if "lines" in block:
                    lines = block["lines"]
                    if len(lines) > 2:  # Minimum rows for a table
                        # Check for consistent column structure
                        spans_per_line = [len(line.get("spans", [])) for line in lines]
                        if len(set(spans_per_line)) <= 2 and max(spans_per_line) > 2:
                            # Likely a table
                            table_data = []
                            for line in lines:
                                row = []
                                for span in line.get("spans", []):
                                    row.append(span.get("text", "").strip())
                                if any(row):  # Skip empty rows
                                    table_data.append(row)
                            
                            if len(table_data) > 1:
                                tables.append({
                                    'page': page_num,
                                    'data': table_data,
                                    'rows': len(table_data),
                                    'columns': max(len(row) for row in table_data) if table_data else 0
                                })
        
        except Exception as e:
            self.logger.warning(f"Failed to extract tables from page {page_num}: {e}")
        
        return tables
    
    def _analyze_text(self, text: str) -> TextAnalysis:
        """Perform comprehensive text analysis"""
        analysis = TextAnalysis()
        
        if not text.strip():
            return analysis
        
        # Basic statistics
        words = re.findall(r'\b\w+\b', text.lower())
        sentences = re.split(r'[.!?]+', text)
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        
        analysis.word_count = len(words)
        analysis.sentence_count = len([s for s in sentences if s.strip()])
        analysis.paragraph_count = len(paragraphs)
        analysis.character_count = len(text)
        
        if analysis.sentence_count > 0:
            analysis.avg_words_per_sentence = analysis.word_count / analysis.sentence_count
        
        if analysis.paragraph_count > 0:
            analysis.avg_sentences_per_paragraph = analysis.sentence_count / analysis.paragraph_count
        
        # Readability analysis
        if self.config.readability_analysis:
            try:
                analysis.flesch_score = flesch_reading_ease(text)
                analysis.flesch_grade = flesch_kincaid_grade(text)
                analysis.reading_level = self._get_reading_level(analysis.flesch_score)
            except:
                pass
        
        # Language detection
        if self.config.language_detection and self.nlp_model:
            analysis.language, analysis.confidence = self._detect_language(text)
        
        return analysis
    
    def _extract_keywords(self, text: str) -> List[Dict]:
        """Extract keywords using TF-IDF and NLP"""
        keywords = []
        
        if not text.strip():
            return keywords
        
        try:
            # TF-IDF approach
            vectorizer = TfidfVectorizer(
                max_features=self.config.max_keywords * 2,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            for keyword, score in keyword_scores[:self.config.max_keywords]:
                keywords.append({
                    'keyword': keyword,
                    'score': float(score),
                    'type': 'tfidf'
                })
            
            # Add NLP-based keywords if available
            if self.nlp_model:
                doc = self.nlp_model(text[:1000000])  # Limit text size for performance
                
                # Extract named entities
                entities = [(ent.text, ent.label_) for ent in doc.ents 
                           if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']]
                
                for entity, label in entities[:10]:
                    keywords.append({
                        'keyword': entity,
                        'score': 1.0,
                        'type': f'entity_{label.lower()}'
                    })
        
        except Exception as e:
            self.logger.warning(f"Keyword extraction failed: {e}")
        
        return keywords
    
    def _extract_topics(self, text: str) -> List[Dict]:
        """Extract topics using K-means clustering on TF-IDF vectors"""
        topics = []
        
        if not text.strip() or len(text) < 500:
            return topics
        
        try:
            # Split text into sentences for topic modeling
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip() and len(s) > 20]
            
            if len(sentences) < self.config.num_topics:
                return topics
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                min_df=2,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=self.config.num_topics, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Extract topic keywords
            feature_names = vectorizer.get_feature_names_out()
            
            for i in range(self.config.num_topics):
                # Get top features for this cluster
                cluster_center = kmeans.cluster_centers_[i]
                top_indices = cluster_center.argsort()[-10:][::-1]
                top_keywords = [feature_names[idx] for idx in top_indices]
                
                # Get representative sentences
                cluster_sentences = [sentences[j] for j, label in enumerate(cluster_labels) if label == i]
                
                topics.append({
                    'topic_id': i,
                    'keywords': top_keywords,
                    'representative_sentences': cluster_sentences[:3],
                    'sentence_count': len(cluster_sentences)
                })
        
        except Exception as e:
            self.logger.warning(f"Topic modeling failed: {e}")
        
        return topics
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of the text"""
        sentiment = {'score': 0.0, 'label': 'neutral', 'confidence': 0.0}
        
        try:
            # Simple lexicon-based sentiment analysis
            positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                             'outstanding', 'superb', 'brilliant', 'perfect', 'love', 'like'}
            negative_words = {'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate', 
                             'dislike', 'poor', 'worst', 'disappointing', 'sad', 'angry'}
            
            words = re.findall(r'\b\w+\b', text.lower())
            
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            total_sentiment_words = positive_count + negative_count
            
            if total_sentiment_words > 0:
                score = (positive_count - negative_count) / len(words)
                sentiment['score'] = score
                sentiment['confidence'] = total_sentiment_words / len(words)
                
                if score > 0.01:
                    sentiment['label'] = 'positive'
                elif score < -0.01:
                    sentiment['label'] = 'negative'
                else:
                    sentiment['label'] = 'neutral'
        
        except Exception as e:
            self.logger.warning(f"Sentiment analysis failed: {e}")
        
        return sentiment
    
    def _scrape_related_content(self, text: str) -> Dict:
        """Scrape related content from web based on document keywords"""
        web_data = {'urls': [], 'content': [], 'error': None}
        
        try:
            # Extract key terms for search
            keywords = self._extract_keywords(text)
            if not keywords:
                return web_data
            
            search_terms = [kw['keyword'] for kw in keywords[:5]]
            query = ' '.join(search_terms)
            
            # Simple web search simulation (replace with actual search API)
            search_urls = [
                f"https://en.wikipedia.org/wiki/{search_terms[0].replace(' ', '_')}",
                f"https://www.google.com/search?q={'+'.join(search_terms)}"
            ]
            
            for url in search_urls[:3]:  # Limit to 3 URLs
                try:
                    response = requests.get(url, timeout=10, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    })
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Extract text content
                        text_content = soup.get_text()[:1000]  # Limit content
                        
                        web_data['urls'].append(url)
                        web_data['content'].append({
                            'url': url,
                            'title': soup.title.string if soup.title else 'No title',
                            'content': text_content,
                            'scraped_at': datetime.now().isoformat()
                        })
                
                except Exception as e:
                    self.logger.warning(f"Failed to scrape {url}: {e}")
        
        except Exception as e:
            web_data['error'] = str(e)
            self.logger.warning(f"Web scraping failed: {e}")
        
        return web_data
    
    def _detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language of the text"""
        if not self.nlp_model:
            return 'unknown', 0.0
        
        try:
            doc = self.nlp_model(text[:1000])
            return 'english', 0.9  # Simplified - using English model
        except:
            return 'unknown', 0.0
    
    def _get_reading_level(self, flesch_score: float) -> str:
        """Convert Flesch score to reading level"""
        if flesch_score >= 90:
            return 'Very Easy'
        elif flesch_score >= 80:
            return 'Easy'
        elif flesch_score >= 70:
            return 'Fairly Easy'
        elif flesch_score >= 60:
            return 'Standard'
        elif flesch_score >= 50:
            return 'Fairly Difficult'
        elif flesch_score >= 30:
            return 'Difficult'
        else:
            return 'Very Difficult'
    
    def _parse_pdf_date(self, date_str: str) -> str:
        """Parse PDF date format to ISO format"""
        try:
            # PDF date format: D:YYYYMMDDHHmmSSOHH'mm'
            if date_str.startswith('D:'):
                date_str = date_str[2:]
            
            # Extract date part
            date_part = date_str[:14]  # YYYYMMDDHHmmSS
            
            if len(date_part) >= 8:
                year = int(date_part[:4])
                month = int(date_part[4:6])
                day = int(date_part[6:8])
                
                if len(date_part) >= 14:
                    hour = int(date_part[8:10])
                    minute = int(date_part[10:12])
                    second = int(date_part[12:14])
                    return datetime(year, month, day, hour, minute, second).isoformat()
                else:
                    return datetime(year, month, day).isoformat()
        except:
            pass
        
        return date_str
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def save_results(self, results: List[ExtractedContent], output_file: str = None):
        """Save extraction results in specified format"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"pdf_extraction_results_{timestamp}"
        
        output_path = Path(self.config.output_dir) / output_file
        
        if self.config.output_format == 'json':
            self._save_as_json(results, f"{output_path}.json")
        elif self.config.output_format == 'csv':
            self._save_as_csv(results, f"{output_path}.csv")
        elif self.config.output_format == 'excel':
            self._save_as_excel(results, f"{output_path}.xlsx")
        elif self.config.output_format == 'html':
            self._save_as_html(results, f"{output_path}.html")
    
    def _save_as_json(self, results: List[ExtractedContent], file_path: str):
        """Save results as JSON"""
        data = []
        for result in results:
            # Convert dataclass to dict
            result_dict = {
                'metadata': asdict(result.metadata),
                'text_content': result.text_content,
                'images': result.images,
                'tables': result.tables,
                'analysis': asdict(result.analysis) if result.analysis else None,
                'keywords': result.keywords,
                'topics': result.topics,
                'sentiment': result.sentiment,
                'web_data': result.web_data
            }
            data.append(result_dict)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.console.print(f"[green]Results saved to:[/green] {file_path}")
    
    def _save_as_csv(self, results: List[ExtractedContent], file_path: str):
        """Save results as CSV"""
        rows = []
        for result in results:
            row = {
                'file_path': result.metadata.file_path,
                'title': result.metadata.title,
                'author': result.metadata.author,
                'pages': result.metadata.pages,
                'file_size': result.metadata.file_size,
                'word_count': result.analysis.word_count if result.analysis else 0,
                'sentence_count': result.analysis.sentence_count if result.analysis else 0,
                'flesch_score': result.analysis.flesch_score if result.analysis else 0,
                'reading_level': result.analysis.reading_level if result.analysis else '',
                'language': result.analysis.language if result.analysis else '',
                'image_count': len(result.images),
                'table_count': len(result.tables),
                'keyword_count': len(result.keywords),
                'topic_count': len(result.topics)
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False)
        
        self.console.print(f"[green]Results saved to:[/green] {file_path}")
    


def _save_as_excel(self, results: List[ExtractedContent], file_path: str):
    """Save results as Excel with multiple sheets"""
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = []
        for result in results:
            summary_data.append({
                'File': Path(result.metadata.file_path).name,
                'Title': result.metadata.title,
                'Pages': result.metadata.pages,
                'Words': result.analysis.word_count if result.analysis else 0,
                'Reading Level': result.analysis.reading_level if result.analysis else '',
                'Images': len(result.images),
                'Tables': len(result.tables)
            })
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

        # Keywords sheet
        keyword_data = []
        for result in results:
            file_name = Path(result.metadata.file_path).name
            for kw in result.keywords:
                keyword_data.append({
                    'File': file_name,
                    'Keyword': kw['keyword'],
                    'Score': kw['score'],
                    'Type': kw['type']
                })
        if keyword_data:
            pd.DataFrame(keyword_data).to_excel(writer, sheet_name='Keywords', index=False)

        # Topics sheet
        topic_data = []
        for result in results:
            file_name = Path(result.metadata.file_path).name
            for topic in result.topics:
                topic_data.append({
                    'File': file_name,
                    'Topic ID': topic['topic_id'],
                    'Keywords': ', '.join(topic['keywords'][:5]),
                    'Sentence Count': topic['sentence_count']
                })
        if topic_data:
            pd.DataFrame(topic_data).to_excel(writer, sheet_name='Topics', index=False)

        # Optionally: Additional sheets can be added here

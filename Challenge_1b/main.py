#!/usr/bin/env python3
"""
Persona-Driven Document Intelligence System
Extracts and prioritizes document sections based on persona and job-to-be-done
"""

import json
import sys
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

class DocumentIntelligenceSystem:
    def __init__(self):
        """Initialize the document intelligence system"""
        self.vectorizer = TfidfVectorizer(
            max_features=1500,
            stop_words='english',
            ngram_range=(1, 3),
            max_df=0.85,
            min_df=1,
            sublinear_tf=True
        )

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Extract text from PDF file page by page"""
        try:
            text_by_page = {}
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text and text.strip():
                        text_by_page[page_num] = text
            return text_by_page
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return {}

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.-]', '', text)
        return text.strip().lower()

    def extract_sections(self, text: str, page_num: int, filename: str) -> List[Dict]:
        """Extract sections from text"""
        sections = []
        lines = text.split('\n')
        current_header = None
        current_content = []

        header_patterns = [
            r'^[A-Z][A-Z\s]{5,80}$',
            r'^\d+\.?\s+[A-Z][a-zA-Z\s]{5,100}',
            r'^[A-Z][a-zA-Z\s]+:',
            r'^\*\*[A-Za-z\s]+\*\*',
            r'^Chapter\s+\d+',
            r'^(?:Section|Part)\s+\d+',
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',
            r'^[A-Z][a-z]+\s+(?:and|or)\s+[A-Z][a-z]+',
            r'^[A-Z][a-z]+\s+Adventures?',
            r'^[A-Z][a-z]+\s+Experiences?',
            r'^[A-Z][a-z]+\s+Tips'
        ]

        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue

            is_header = any(re.match(pattern, stripped_line) and len(stripped_line) < 120 for pattern in header_patterns)

            if is_header:
                if current_header and current_content:
                    content_text = ' '.join(current_content).strip()
                    if len(content_text.split()) > 15:
                        sections.append({
                            'title': current_header,
                            'content': content_text,
                            'page_number': page_num,
                            'document': filename
                        })
                current_header = stripped_line
                current_content = []
            elif current_header:
                current_content.append(stripped_line)

        if current_header and current_content:
            content_text = ' '.join(current_content).strip()
            if len(content_text.split()) > 15:
                sections.append({
                    'title': current_header,
                    'content': content_text,
                    'page_number': page_num,
                    'document': filename
                })

        return sections

    def create_persona_profile(self, persona: Dict, job_to_be_done: Dict) -> str:
        """Create a detailed persona profile"""
        role = persona.get('role', '').lower()
        task = job_to_be_done.get('task', '').lower()

        profile_keywords = [role, task]

        if 'travel' in role and 'planner' in role:
            profile_keywords.extend(['itinerary', 'planning', 'trip', 'vacation', 'group', 'activities', 'social', 'entertainment', 'nightlife', 'young', 'budget', 'fun', 'short trip', 'weekend', 'efficient'])

        travel_domain_keywords = [
            'activities', 'attractions', 'sightseeing', 'tours', 'experiences', 'adventures',
            'restaurants', 'dining', 'food', 'cuisine', 'culinary',
            'hotels', 'accommodation', 'lodging',
            'entertainment', 'nightlife', 'bars', 'clubs', 'music',
            'beaches', 'coastal', 'water sports',
            'culture', 'museums', 'history', 'art',
            'transportation', 'travel',
            'shopping', 'markets',
            'outdoor', 'nature', 'hiking',
            'budget', 'planning', 'tips', 'advice', 'recommendations'
        ]
        profile_keywords.extend(travel_domain_keywords)

        return ' '.join(profile_keywords)

    def rank_sections(self, sections: List[Dict], persona_profile: str) -> List[Dict]:
        """Rank sections based on relevance to the persona profile"""
        if not sections:
            return []

        section_texts = [self.preprocess_text(f"{s['title']} {s['content']}") for s in sections]
        persona_text = self.preprocess_text(persona_profile)

        all_texts = section_texts + [persona_text]
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        persona_vector = tfidf_matrix[-1]
        section_vectors = tfidf_matrix[:-1]

        scores = cosine_similarity(section_vectors, persona_vector).flatten()

        for i, section in enumerate(sections):
            section['similarity_score'] = scores[i]

        ranked_sections = sorted(sections, key=lambda x: x['similarity_score'], reverse=True)

        for i, section in enumerate(ranked_sections):
            section['importance_rank'] = i + 1

        return ranked_sections

    def create_subsections(self, sections: List[Dict], top_n: int = 5) -> List[Dict]:
        """Create refined subsections from the top-ranked sections"""
        subsections = []
        for section in sections[:top_n]:
            sentences = sent_tokenize(section['content'])
            
            # Combine all sentences to form a single refined text block
            refined_text = ' '.join(sentences)
            
            if len(refined_text.split()) > 20: # Ensure subsection is substantial
                subsections.append({
                    'document': section['document'],
                    'refined_text': refined_text,
                    'page_number': section['page_number']
                })
        return subsections

    def find_pdf_path(self, filename: str, base_dir: Path) -> str:
        """Find the PDF file within the collection directory"""
        search_paths = [
            base_dir / "PDFs" / filename,
            base_dir / "pdfs" / filename,
            base_dir / filename
        ]
        for path in search_paths:
            if path.exists():
                return str(path)
        return None

    def process_documents(self, input_data: Dict) -> Dict:
        """Process all documents based on the persona and job-to-be-done"""
        start_time = time.time()

        documents = input_data['documents']
        persona = input_data['persona']
        job_to_be_done = input_data['job_to_be_done']

        persona_profile = self.create_persona_profile(persona, job_to_be_done)

        all_sections = []
        
        # Determine the base directory from the input file path
        # This assumes the script is run in a context where the input file path is meaningful
        input_file_path = Path("/app/input/challenge1b_input.json")
        base_dir = input_file_path.parent
        
        challenge_id = input_data.get("challenge_info", {}).get("challenge_id", "")
        if "round_1b_002" in challenge_id: # Travel Planning
            collection_dir_name = "Collection 1"
        elif "round_1b_003" in challenge_id: # Adobe
            collection_dir_name = "Collection 2"
        elif "round_1b_001" in challenge_id: # Recipe
            collection_dir_name = "Collection 3"
        else:
            # Fallback to a generic search if challenge_id is not available or doesn't match
            collection_dir_name = "." 

        # This logic assumes a fixed structure, which is typical for such challenges
        # A more robust solution might search more dynamically
        pdf_base_dir = Path("/app") / collection_dir_name

        for doc in documents:
            filename = doc['filename']
            pdf_path = self.find_pdf_path(filename, pdf_base_dir)
            if not pdf_path:
                # Fallback search in /app/input if not found in collection-specific dir
                pdf_path = self.find_pdf_path(filename, Path("/app/input"))
                if not pdf_path:
                    logger.warning(f"Could not find {filename} in {pdf_base_dir} or /app/input")
                    continue
            
            text_by_page = self.extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page.items():
                page_sections = self.extract_sections(text, page_num, filename)
                all_sections.extend(page_sections)
        
        ranked_sections = self.rank_sections(all_sections, persona_profile)
        subsections = self.create_subsections(ranked_sections, top_n=5)

        output = {
            "metadata": {
                "input_documents": [doc['filename'] for doc in documents],
                "persona": persona.get('role', 'Unknown'),
                "job_to_be_done": job_to_be_done.get('task', 'Unknown'),
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [
                {
                    "document": s['document'],
                    "section_title": s['title'],
                    "importance_rank": s['importance_rank'],
                    "page_number": s['page_number']
                } for s in ranked_sections[:5]
            ],
            "subsection_analysis": subsections
        }

        return output

def main():
    input_file = "/app/input/challenge1b_input.json"
    output_file = "/app/output/challenge1b_output.json"

    try:
        with open(input_file, 'r') as f:
            input_data = json.load(f)

        system = DocumentIntelligenceSystem()
        result = system.process_documents(input_data)

        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)

        print(f"Processing complete. Output at: {output_file}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
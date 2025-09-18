import requests
import re
import random
import pandas as pd
from pathlib import Path

class GutenbergDataPreparer:
    def __init__(self, config_file="config.txt"):
        self.base_url = "https://www.gutenberg.org/files"
        self.config = self.load_config(config_file)
        
    def load_config(self, config_file):
        """Load configuration from file"""
        config = {}
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if ':' in line and not line.strip().startswith('#'):
                        key, value = line.split(':', 1)
                        config[key.strip()] = value.strip()
        except FileNotFoundError:
            print(f"Config file {config_file} not found.")
        return config
    
    def get_book_text(self, book_id):
        """Fetch book text from Project Gutenberg"""
        try:
            url = f"{self.base_url}/{book_id}/{book_id}-0.txt"
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except:
            try:
                url = f"{self.base_url}/{book_id}/{book_id}.txt"
                response = requests.get(url)
                response.raise_for_status()
                return response.text
            except:
                print(f"Failed to retrieve book ID {book_id}")
                return None
    
    def clean_gutenberg_text(self, text):
        """Remove Project Gutenberg headers and footers"""
        start_patterns = [
            r"\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*\*\*\*",
            r"\*\*\* START OF (THIS|THE) PROJECT GUTENBERG.*\*\*\*"
        ]
        
        end_patterns = [
            r"\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*\*\*\*",
            r"\*\*\* END OF (THIS|THE) PROJECT GUTENBERG.*\*\*\*"
        ]
        
        start_index = 0
        for pattern in start_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start_index = match.end()
                break
        
        end_index = len(text)
        for pattern in end_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                end_index = match.start()
                break
        
        return text[start_index:end_index].strip()
    
    def extract_passages_for_training(self, text, passage_length=3):
        """Extract passages for training data"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        passages = []
        
        for i in range(0, len(sentences) - passage_length + 1, passage_length):
            passage = " ".join(sentences[i:i + passage_length])
            passages.append(passage)
        
        return passages
    
    def create_training_data(self, book_ids, output_file="training_data.csv"):
        """Create training data from multiple books"""
        all_passages = []
        
        for book_id in book_ids:
            print(f"Processing book ID {book_id}...")
            text = self.get_book_text(book_id)
            if text:
                cleaned_text = self.clean_gutenberg_text(text)
                passages = self.extract_passages_for_training(cleaned_text)
                all_passages.extend(passages)
        
        # Create DataFrame (you would manually label these as descriptive/non-descriptive)
        df = pd.DataFrame({'text': all_passages, 'label': None})
        df.to_csv(output_file, index=False)
        print(f"Created training data with {len(all_passages)} passages in {output_file}")
        
        return df

import requests
import re
import pandas as pd
from pathlib import Path

class GutenbergDataPreparer:
    def __init__(self):
        self.base_url = "https://www.gutenberg.org/files"
        
    def get_book_text(self, book_id):
        """Fetch book text from Project Gutenberg"""
        try:
            url = f"{self.base_url}/{book_id}/{book_id}-0.txt"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except:
            try:
                url = f"{self.base_url}/{book_id}/{book_id}.txt"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return response.text
            except Exception as e:
                print(f"Failed to retrieve book ID {book_id}: {e}")
                return None
    
    def clean_gutenberg_text(self, text):
        """Remove Project Gutenberg headers and footers"""
        start_pattern = r"\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*\*\*\*"
        end_pattern = r"\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*\*\*\*"
        
        start_match = re.search(start_pattern, text, re.IGNORECASE)
        end_match = re.search(end_pattern, text, re.IGNORECASE)
        
        if start_match and end_match:
            start_index = start_match.end()
            end_index = end_match.start()
            return text[start_index:end_index].strip()
        return text
    
    def extract_passages(self, text, passage_length=3):
        """Extract passages of specified length"""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        passages = []
        for i in range(0, len(sentences) - passage_length + 1, passage_length):
            passage = " ".join(sentences[i:i + passage_length])
            # Only include passages of reasonable length
            if 50 < len(passage) < 500:
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
                passages = self.extract_passages(cleaned_text)
                all_passages.extend(passages)
                print(f"  Extracted {len(passages)} passages")
        
        # Create DataFrame
        df = pd.DataFrame({'text': all_passages, 'label': None, 'notes': ''})
        df.to_csv(output_file, index=False)
        print(f"Created training data with {len(all_passages)} passages in {output_file}")
        
        return df

# Example usage
if __name__ == "__main__":
    preparer = GutenbergDataPreparer()
    
    # Books known for good descriptions (Dickens, Austen, etc.)
    book_ids = [46, 98, 766, 1023, 1400, 730, 1342, 84, 11, 16]
    
    # Create training data
    preparer.create_training_data(book_ids, "raw_training_data.csv")
import requests
import re
import pandas as pd
from pathlib import Path

class GutenbergDataPreparer:
    def __init__(self):
        self.base_url = "https://www.gutenberg.org/files"
        
        # Keywords for landscape, seascape, and cityscape descriptions
        self.landscape_keywords = [
            'mountain', 'hill', 'valley', 'forest', 'wood', 'field', 'meadow', 'river', 'stream', 
            'brook', 'lake', 'tree', 'flower', 'grass', 'path', 'road', 'countryside', 'landscape',
            'cliff', 'peak', 'slope', 'glade', 'orchard', 'garden', 'park', 'vineyard', 'prairie'
        ]
        
        self.seascape_keywords = [
            'sea', 'ocean', 'wave', 'beach', 'shore', 'coast', 'harbor', 'bay', 'tide', 'current',
            'sail', 'ship', 'boat', 'water', 'foam', 'spray', 'cliff', 'rock', 'island', 'cape',
            'strand', 'nautical', 'maritime', 'naval', 'seafaring', 'wharf', 'pier', 'dock'
        ]
        
        self.cityscape_keywords = [
            'city', 'town', 'street', 'avenue', 'boulevard', 'alley', 'square', 'plaza', 'building',
            'house', 'mansion', 'cottage', 'palace', 'tower', 'spire', 'roof', 'window', 'door',
            'bridge', 'canal', 'market', 'shop', 'tavern', 'inn', 'church', 'cathedral', 'monument',
            'urban', 'metropolitan', 'municipal', 'civic', 'downtown', 'suburb', 'quarter', 'district'
        ]
        
        # Combine all keywords
        self.all_keywords = self.landscape_keywords + self.seascape_keywords + self.cityscape_keywords
    
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
    
    def contains_direct_speech(self, text):
        """Check if text contains direct speech (quotes or dialogue)"""
        # Look for quotation marks or dialogue indicators
        if re.search(r'["“”]', text):  # Quotation marks
            return True
        if re.search(r'\b(said|replied|answered|exclaimed|cried|whispered|shouted)\b', text, re.IGNORECASE):
            return True
        return False
    
    def contains_descriptive_keywords(self, text):
        """Check if text contains landscape/seascape/cityscape keywords"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.all_keywords)
    
    def extract_passages(self, text, passage_length=3):
        """Extract passages focusing on descriptive content"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        passages = []
        for i in range(0, len(sentences) - passage_length + 1, passage_length):
            passage = " ".join(sentences[i:i + passage_length])
            
            # Filter criteria
            if (50 < len(passage) < 500 and                    # Reasonable length
                not self.contains_direct_speech(passage) and   # No direct speech
                self.contains_descriptive_keywords(passage)):  # Contains descriptive keywords
                
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
                print(f"  Extracted {len(passages)} descriptive passages")
        
        # Create DataFrame
        df = pd.DataFrame({'text': all_passages, 'label': None, 'notes': ''})
        df.to_csv(output_file, index=False)
        print(f"Created training data with {len(all_passages)} descriptive passages in {output_file}")
        
        return df

# Example usage
if __name__ == "__main__":
    preparer = GutenbergDataPreparer()
    
    # Books known for good descriptions (focus on descriptive writers)
    descriptive_books = [
        98,   # A Tale of Two Cities - Dickens (city descriptions)
        766,  # David Copperfield - Dickens 
        1023, # Bleak House - Dickens
        1400, # Great Expectations - Dickens
        1342, # Pride and Prejudice - Austen (countryside)
        84,   # Frankenstein - Shelley (landscapes)
        25344,# The Scarlet Letter - Hawthorne
        2701, # Moby Dick - Melville (seascapes)
        16,   # Peter Pan - Barrie (descriptive)
        205,  # Wuthering Heights - Brontë (moors)
    ]
    
    # Create training data
    preparer.create_training_data(descriptive_books, "raw_training_data.csv")

from enhanced_extractor import EnhancedDescriptiveExtractor
from data_preparation import GutenbergDataPreparer

def main():
    # Initialize the extractor
    extractor = EnhancedDescriptiveExtractor()
    
    # Example: Extract from a Gutenberg book
    preparer = GutenbergDataPreparer()
    text = preparer.get_book_text(11)  # Alice in Wonderland
    if text:
        cleaned_text = preparer.clean_gutenberg_text(text)
        passages = extractor.extract_descriptive_passages(cleaned_text, num_passages=5)
        
        print("Descriptive Passages from Alice in Wonderland:")
        for i, passage in enumerate(passages, 1):
            print(f"\n{i}. {passage}")

if __name__ == "__main__":
    main()
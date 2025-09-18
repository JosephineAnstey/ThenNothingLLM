from enhanced_extractor import EnhancedDescriptiveExtractor
from data_preparation import GutenbergDataPreparer

def main():
    # Option 1: Prepare training data (run once)
    # preparer = GutenbergDataPreparer()
    # preparer.create_training_data([46, 98, 766, 1023, 1400, 730])
    
    # Option 2: Train model (run after labeling data)
    # from train_model import DescriptivePassageClassifier
    # classifier = DescriptivePassageClassifier()
    # classifier.train("labeled_training_data.csv")
    
    # Use enhanced extractor
    extractor = EnhancedDescriptiveExtractor()
    
    # Example usage
    sample_text = """
        The valley stretched out before them, a patchwork of green and gold fields divided by meandering stone walls. 
        In the distance, mountains rose like jagged teeth against the horizon. 
        Meanwhile, John decided to go to the market to buy some groceries. 
        He walked quickly down the street, thinking about his plans for the evening.
        The forest was ancient, a remnant of a world before men. 
        Towering redwoods reached for the heavens, their tops lost in the low-hanging clouds.
    """
    
    passages = extractor.extract_descriptive_passages(sample_text, num_passages=2)
    
    print("Extracted Descriptive Passages:")
    for i, passage in enumerate(passages, 1):
        print(f"\n{i}. {passage}")

if __name__ == "__main__":
    main()

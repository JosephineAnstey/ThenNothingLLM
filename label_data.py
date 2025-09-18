import pandas as pd
import random

def prepare_labeling_file(input_file, output_file, sample_size=200):
    """Create a smaller sample for labeling"""
    df = pd.read_csv(input_file)
    
    # Take a random sample for labeling
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    
    # Reorder for easier labeling
    df = df.reset_index(drop=True)
    df.to_csv(output_file, index=False)
    print(f"Created labeling file with {len(df)} passages in {output_file}")
    print("Please open this file in Excel or similar and add labels (1 for descriptive, 0 for non-descriptive)")

if __name__ == "__main__":
    prepare_labeling_file("raw_training_data.csv", "labeling_sample.csv", 200)
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from sentence_transformers import SentenceTransformer, util
import re

class EnhancedDescriptiveExtractor:
    def __init__(self, model_path="./descriptive_text_model"):
        # Load fine-tuned model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            print("Loaded fine-tuned model")
        except:
            # Fallback to zero-shot classification
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            print("Using zero-shot classification as fallback")
        
        # Load similarity model
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Descriptive examples for similarity matching
        self.descriptive_examples = [
            "The valley stretched out before them, a patchwork of green and gold fields.",
            "Ancient trees stood tall and proud, their branches reaching toward the heavens.",
            "The mountains stood as silent sentinels, their peaks dusted with snow.",
            "A gentle mist rose from the forest floor, creating an ethereal atmosphere.",
            "The city skyline glittered against the twilight sky, a mosaic of light and shadow."
        ]
        self.example_embeddings = self.similarity_model.encode(self.descriptive_examples)
    
    def is_descriptive_similarity(self, text, threshold=0.6):
        """Check if text is similar to known descriptive passages"""
        text_embedding = self.similarity_model.encode(text)
        similarities = util.pytorch_cos_sim(text_embedding, self.example_embeddings)
        return similarities.max().item() > threshold
    
    def is_descriptive_classifier(self, text, threshold=0.7):
        """Use classifier to determine if text is descriptive"""
        if hasattr(self, 'tokenizer'):  # Fine-tuned model
            result = self.classifier(text)[0]
            # Assuming label 1 is descriptive
            return result['label'] == 'LABEL_1' and result['score'] > threshold
        else:  # Zero-shot classification
            result = self.classifier(
                text,
                candidate_labels=["descriptive", "non-descriptive"],
                hypothesis_template="This text is {}."
            )
            return result['labels'][0] == 'descriptive' and result['scores'][0] > threshold
    
    def extract_descriptive_passages(self, text, num_passages=3, passage_length=3):
        """Extract descriptive passages using multiple methods"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        scored_sentences = []
        
        # Score each sentence
        for sentence in sentences:
            if len(sentence.split()) < 6:  # Skip very short sentences
                continue
                
            similarity_score = 1.0 if self.is_descriptive_similarity(sentence) else 0.0
            classifier_score = 1.0 if self.is_descriptive_classifier(sentence) else 0.0
            
            # Combined score
            combined_score = 0.7 * classifier_score + 0.3 * similarity_score
            scored_sentences.append((sentence, combined_score))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in scored_sentences[:num_passages * passage_length * 2]]
        
        # Create passages
        passages = []
        for i in range(0, len(top_sentences) - passage_length + 1, passage_length):
            passage = " ".join(top_sentences[i:i + passage_length])
            passages.append(passage)
        
        return passages[:num_passages]

# Example usage
if __name__ == "__main__":
    extractor = EnhancedDescriptiveExtractor()
    
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
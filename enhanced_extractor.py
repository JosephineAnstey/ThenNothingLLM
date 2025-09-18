import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re

class EnhancedDescriptiveExtractor:
    def __init__(self, config_file="config.txt"):
        self.config = self.load_config(config_file)
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.load_descriptive_examples()
        
        # Try to load fine-tuned model, fall back to zero-shot classification
        try:
            self.classifier = pipeline(
                "text-classification",
                model="./model",
                tokenizer="./model"
            )
        except:
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
    
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
    
    def load_descriptive_examples(self):
        """Load descriptive examples for similarity matching"""
        self.descriptive_examples = []
        try:
            with open('config.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                start_reading = False
                for line in lines:
                    if 'descriptive_examples:' in line:
                        start_reading = True
                        continue
                    if start_reading and line.strip().startswith('-'):
                        example = line.strip()[2:].strip('"\'')
                        self.descriptive_examples.append(example)
        except:
            # Fallback examples
            self.descriptive_examples = [
                "The valley stretched out before them, a patchwork of green and gold fields.",
                "Ancient trees stood tall and proud, their branches reaching toward the heavens.",
                "The mountains stood as silent sentinels, their peaks dusted with snow."
            ]
        
        # Encode examples for similarity matching
        self.example_embeddings = self.similarity_model.encode(self.descriptive_examples)
    
    def is_descriptive_similarity(self, text, threshold=0.6):
        """Check if text is similar to known descriptive passages"""
        text_embedding = self.similarity_model.encode(text)
        similarities = util.pytorch_cos_sim(text_embedding, self.example_embeddings)
        return similarities.max().item() > threshold
    
    def is_descriptive_classifier(self, text, threshold=0.7):
        """Use classifier to determine if text is descriptive"""
        if hasattr(self.classifier, 'model'):  # Fine-tuned model
            result = self.classifier(text)[0]
            return result['label'] == 'DESCRIPTIVE' and result['score'] > threshold
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
        descriptive_sentences = []
        
        # Score each sentence
        for sentence in sentences:
            if len(sentence.split()) < 6:  # Skip very short sentences
                continue
                
            similarity_score = self.is_descriptive_similarity(sentence)
            classifier_score = self.is_descriptive_classifier(sentence)
            
            # Combined score (you could weight these differently)
            combined_score = 0.6 * classifier_score + 0.4 * similarity_score
            
            if combined_score > 0.5:  # Threshold
                descriptive_sentences.append((sentence, combined_score))
        
        # Sort by score and take top passages
        descriptive_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in descriptive_sentences[:num_passages * passage_length]]
        
        # Create passages
        passages = []
        for i in range(0, len(top_sentences) - passage_length + 1, passage_length):
            passage = " ".join(top_sentences[i:i + passage_length])
            passages.append(passage)
        
        return passages[:num_passages]

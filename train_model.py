import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset
import numpy as np

class DescriptivePassageClassifier:
    def __init__(self, config_file="config.txt"):
        self.config = self.load_config(config_file)
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.config.get('model_name', 'distilbert-base-uncased'))
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.config.get('model_name', 'distilbert-base-uncased'),
            num_labels=2
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
    
    def preprocess_function(self, examples):
        """Tokenize the texts"""
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=int(self.config.get('max_length', 256)),
        )
    
    def train(self, training_data_path, output_dir="./model"):
        """Train the classification model"""
        # Load labeled data (you need to create this with human labels)
        df = pd.read_csv(training_data_path)
        
        # Split data
        train_df, temp_df = train_test_split(
            df, test_size=float(self.config.get('test_split', 0.2)), random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=42
        )
        
        # Convert to datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        # Tokenize datasets
        tokenized_train = train_dataset.map(self.preprocess_function, batched=True)
        tokenized_val = val_dataset.map(self.preprocess_function, batched=True)
        tokenized_test = test_dataset.map(self.preprocess_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=float(self.config.get('learning_rate', 2e-5)),
            per_device_train_batch_size=int(self.config.get('batch_size', 16)),
            per_device_eval_batch_size=int(self.config.get('batch_size', 16)),
            num_train_epochs=int(self.config.get('epochs', 3)),
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
        )
        
        # Train model
        trainer.train()
        
        # Evaluate model
        results = trainer.evaluate(tokenized_test)
        print(f"Test results: {results}")
        
        # Save model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        return trainer
    
    def predict(self, text):
        """Predict if text is descriptive"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=int(self.config.get('max_length', 256)),
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probabilities = torch.softmax(outputs.logits, dim=-1)
        return probabilities[0][1].item()  # Probability of being descriptive

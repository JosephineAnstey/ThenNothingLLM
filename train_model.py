import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
import torch

class DescriptivePassageClassifier:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
    
    def preprocess_function(self, examples):
        """Tokenize the texts"""
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=256,
        )
    
    def train(self, training_data_path, output_dir="./model"):
        """Train the classification model"""
        # Load labeled data
        df = pd.read_csv(training_data_path)
        df = df.dropna(subset=['label'])  # Remove unlabeled rows
        
        # Convert labels to integers
        df['label'] = df['label'].astype(int)
        
        print(f"Training with {len(df)} labeled examples")
        print(f"Class distribution: {df['label'].value_counts().to_dict()}")
        
        # Split data
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        
        # Convert to datasets
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        # Tokenize datasets
        tokenized_train = train_dataset.map(self.preprocess_function, batched=True)
        tokenized_test = test_dataset.map(self.preprocess_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_dir='./logs',
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
        )
        
        # Train model
        print("Starting training...")
        trainer.train()
        
        # Evaluate model
        results = trainer.evaluate(tokenized_test)
        print(f"Test results: {results}")
        
        # Save model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
        return trainer

if __name__ == "__main__":
    # Train the model
    classifier = DescriptivePassageClassifier()
    classifier.train("labeled_training_data.csv", "./descriptive_text_model")
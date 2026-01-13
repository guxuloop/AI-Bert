import argparse
import json
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--model_name", default="bert-base-chinese")
    parser.add_argument("--output_dir", default="models/classifier")
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    # Load Dataset
    dataset = load_dataset('json', data_files={"train": args.train_file})
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    def preprocess(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=64)
    
    dataset = dataset.map(preprocess, batched=True)
    # Split for validation (simple random split)
    dataset = dataset['train'].train_test_split(test_size=0.1)
    
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=10,
        report_to="none"  # Disable wandb logging
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == '__main__':
    main()

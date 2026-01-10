import argparse
import json
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import precision_recall_fscore_support

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Flatten for metric calculation
    true_predictions_flat = [p for preds in true_predictions for p in preds]
    true_labels_flat = [l for labs in true_labels for l in labs]

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels_flat, true_predictions_flat, average='macro')
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": np.mean(np.array(true_predictions_flat) == np.array(true_labels_flat))
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--model_name", default="bert-base-chinese")
    parser.add_argument("--output_dir", default="models/ner")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    # 1. Define Label Map
    # O=0, B-TIME=1, I-TIME=2, B-LOC=3, I-LOC=4
    label_list = ["O", "B-TIME", "I-TIME", "B-LOC", "I-LOC", "B-CONTENT", "I-CONTENT"]
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    # 2. Load Dataset
    dataset = load_dataset('json', data_files={"train": args.train_file})
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 3. Preprocessing
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], 
            truncation=True, 
            is_split_into_words=True,
            padding="max_length", 
            max_length=64
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # New word start, map label string to ID
                    lab_str = label[word_idx]
                    label_ids.append(label2id.get(lab_str, 0))
                else:
                    # Inside same word (for subwords)
                    lab_str = label[word_idx]
                    label_ids.append(label2id.get(lab_str, 0)) # or -100 if we don't want to predict subwords
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
    
    # 4. Model
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, 
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        save_strategy="no",  # save space for demo
        eval_strategy="no",  # no validation set for demo text
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == '__main__':
    main()

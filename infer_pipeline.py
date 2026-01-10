import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import argparse
import json
import torch

class PipelineEngine:
    def __init__(self, clf_model_path, ner_model_path, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.clf_sess = ort.InferenceSession(clf_model_path)
        self.ner_sess = ort.InferenceSession(ner_model_path)
        
        # Labels from train_ner.py
        self.ner_labels = ["O", "B-TIME", "I-TIME", "B-LOC", "I-LOC", "B-CONTENT", "I-CONTENT"]
        self.id2label = {i: label for i, label in enumerate(self.ner_labels)}
        
        # Labels from train_classifier.py
        self.clf_labels = {0: "日程", 1: "闹钟", 2: "其他"}

    def preprocess(self, text):
        return self.tokenizer(text, return_tensors="np", padding="max_length", max_length=64, truncation=True)

    def predict_class(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        outputs = self.clf_sess.run(["logits"], {"input_ids": input_ids, "attention_mask": attention_mask})
        logits = outputs[0]
        pred_id = np.argmax(logits, axis=1)[0]
        return pred_id

    def predict_ner(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        outputs = self.ner_sess.run(["logits"], {"input_ids": input_ids, "attention_mask": attention_mask})
        logits = outputs[0][0] # batch 0
        pred_ids = np.argmax(logits, axis=1)
        return pred_ids

    def decode_entities(self, text, inputs, pred_ids):
        # inputs['input_ids'] is (1, 64)
        input_ids = inputs['input_ids'][0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        entities = {"time": [], "loc": [], "content": []}
        current_entity = {"type": None, "tokens": []}

        for idx, token in enumerate(tokens):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            
            label_id = pred_ids[idx]
            label = self.id2label.get(label_id, "O")
            
            # Simple decoding logic (BIO)
            if label.startswith("B-"):
                if current_entity["type"]:
                    # save previous
                    entities[current_entity["type"]].append("".join(current_entity["tokens"]))
                current_entity = {"type": label[2:].lower(), "tokens": [token]}
            elif label.startswith("I-"):
                if current_entity["type"] == label[2:].lower():
                    current_entity["tokens"].append(token)
                else:
                    # Broken entity or started with I- (treat as new or ignore)
                    if current_entity["type"]:
                        entities[current_entity["type"]].append("".join(current_entity["tokens"]))
                    current_entity = {"type": label[2:].lower(), "tokens": [token]}
            else: # O
                if current_entity["type"]:
                    entities[current_entity["type"]].append("".join(current_entity["tokens"]))
                    current_entity = {"type": None, "tokens": []}
        
        # Flush last
        if current_entity["type"]:
            entities[current_entity["type"]].append("".join(current_entity["tokens"]))

        # Cleanup tokens (remove ## for English, keep Chinese as is)
        for key in entities:
            clean_list = []
            for item in entities[key]:
                clean_list.append(item.replace("##", ""))
            entities[key] = clean_list
            
        return entities

    def run(self, text):
        inputs = self.preprocess(text)
        
        # 1. Classification
        class_id = self.predict_class(inputs)
        category = self.clf_labels.get(class_id, "其他")
        
        result = {
            "type": category,
            "raw_text": text,
            "time": None,
            "location": None,
            "content": None
        }

        # 2. NER (Only if not 'Other')
        if class_id in [0, 1]: 
            ner_ids = self.predict_ner(inputs)
            entities = self.decode_entities(text, inputs, ner_ids)
            
            result["time"] = " ".join(entities["time"]) if entities["time"] else None
            result["location"] = " ".join(entities["loc"]) if entities["loc"] else None
            
            # Content strategy: use extracted content OR remain text
            if entities["content"]:
                result["content"] = " ".join(entities["content"])
            else:
                # Fallback: remove time/loc from text
                content = text
                for t in entities["time"]: content = content.replace(t, "")
                for l in entities["loc"]: content = content.replace(l, "")
                result["content"] = content.strip()

        return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clf_model", default="models/classifier.onnx")
    parser.add_argument("--ner_model", default="models/ner.onnx")
    parser.add_argument("--tokenizer", default="models/classifier") # reuse tokenizer
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    pipeline = PipelineEngine(args.clf_model, args.ner_model, args.tokenizer)
    res = pipeline.run(args.text)
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

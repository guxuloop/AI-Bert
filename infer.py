import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    inputs = tokenizer(args.text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_label = torch.argmax(logits, dim=-1).item()
        probs = torch.nn.functional.softmax(logits, dim=-1)
        score = probs[0][pred_label].item()

    label_map = {0: "日程", 1: "闹钟", 2: "其他"}
    print(json.dumps({
        "text": args.text,
        "label_id": pred_label,
        "label": label_map.get(pred_label, "Unknown"),
        "score": score
    }, ensure_ascii=False))

if __name__ == '__main__':
    main()

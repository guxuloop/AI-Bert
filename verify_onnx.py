import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    
    # Preprocess
    inputs = tokenizer(args.text, padding="max_length", max_length=64, truncation=True)
    input_ids = np.array([inputs["input_ids"]], dtype=np.int64)
    attention_mask = np.array([inputs["attention_mask"]], dtype=np.int64)

    # Load ONNX Session
    ort_session = ort.InferenceSession(args.onnx_model)

    # Inference
    outputs = ort_session.run(
        ["logits"],
        {"input_ids": input_ids, "attention_mask": attention_mask}
    )
    logits = outputs[0]
    
    # Post-process
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    pred_label = np.argmax(logits, axis=1)[0]
    confidence = probs[0][pred_label]

    label_map = {0: "日程", 1: "闹钟", 2: "其他"}
    print(f"Text: {args.text}")
    print(f"Predicted Label: {label_map.get(pred_label, 'Unknown')} (ID: {pred_label})")
    print(f"Confidence: {confidence:.4f}")

if __name__ == '__main__':
    main()

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output", default="models/ner/model.onnx")
    args = parser.parse_args()

    print(f"Loading NER model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.eval()

    text = "明天下午三点在会议室开会"
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=64, truncation=True)
    
    symbolic_names = {0: 'batch_size', 1: 'sequence_length'}
    
    print(f"Exporting to {args.output}...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (inputs['input_ids'], inputs['attention_mask']),
            args.output,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': symbolic_names,
                'attention_mask': symbolic_names,
                'logits': symbolic_names
            }
        )
    print("Export NER complete!")

if __name__ == '__main__':
    main()

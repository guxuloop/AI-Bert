import argparse
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Path to the directory containing the trained model")
    parser.add_argument("--output", default="model.onnx", help="Path to save the ONNX model")
    args = parser.parse_args()

    print(f"Loading model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    # Create dummy input for tracing
    text = "这是一个测试句子"
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=64, truncation=True)
    
    # We need to export with dynamic axes because input length can vary
    symbolic_names = {0: 'batch_size', 1: 'sequence_length'}
    
    print(f"Exporting to {args.output}...")
    with torch.no_grad():
        torch.onnx.export(
            model,                                            # model being run
            (inputs['input_ids'], inputs['attention_mask']),  # model input (tuple)
            args.output,                                      # where to save the model
            opset_version=17,                                 # the ONNX version to export the model to
            do_constant_folding=True,                         # whether to execute constant folding for optimization
            input_names=['input_ids', 'attention_mask'],      # the model's input names
            output_names=['logits'],                          # the model's output names
            dynamic_axes={                                    # variable length axes
                'input_ids': symbolic_names,
                'attention_mask': symbolic_names,
                'logits': {0: 'batch_size'}
            }
        )
    print("Export complete!")

if __name__ == '__main__':
    main()

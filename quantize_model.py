from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
import argparse
import os
import onnx
import sys

def quantize(model_path, output_path):
    print(f"Quantizing {model_path} -> {output_path}...")
    
    # Try pre-processing first, but if it fails, fallback to original with checks disabled
    preprocessed_path = model_path.replace(".onnx", ".pre.onnx")
    use_preprocessed = False
    
    try:
        print("Running quantization pre-processing (symbolic shape inference)...")
        quant_pre_process(
            input_model_path=model_path,
            output_model_path=preprocessed_path,
            skip_symbolic_shape=False,
            auto_merge=True
        )
        print(f"Pre-processing complete. Saved to {preprocessed_path}")
        use_preprocessed = True
        
    except Exception as e:
        print(f"WARNING: Pre-processing failed: {e}")
        print("Will attempt quantization on original model with Shape Inference DISABLED.")
        if os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)

    input_model = preprocessed_path if use_preprocessed else model_path

    try:
        print(f"Running dynamic quantization on {input_model}...")
        quantize_dynamic(
            model_input=input_model,
            model_output=output_path,
            weight_type=QuantType.QUInt8,
            extra_options={'DisableShapeInference': True}
        )
        print(f"Quantization successful. Saved to {output_path}")

    except Exception as e:
        print(f"Quantization failed: {e}")
        sys.exit(1)
        
    finally:
        if use_preprocessed and os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)
            pre_data = preprocessed_path + ".data"
            if os.path.exists(pre_data):
                os.remove(pre_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to input FP32 ONNX model")
    parser.add_argument("--output", type=str, required=True, help="Path to output Quantized ONNX model")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Input model not found: {args.model}")

    quantize(args.model, args.output)

if __name__ == "__main__":
    main()

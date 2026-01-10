import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import sys
import os

def clean_and_quantize(model_path, output_path):
    print(f"Loading {model_path}...")
    model = onnx.load(model_path)
    
    # Remove all intermediate shape information
    # This prevents 'Inferred shape vs existing shape' errors
    print("Removing existing value_info (shapes)...")
    if len(model.graph.value_info) > 0:
        del model.graph.value_info[:]
    else:
        print("No value_info found to remove.")
    
    clean_path = model_path.replace(".onnx", ".clean.onnx")
    onx_path = clean_path
    onnx.save(model, clean_path)
    print(f"Saved cleaned model to {clean_path}")
    
    print(f"Quantizing {clean_path} -> {output_path}...")
    try:
        quantize_dynamic(
            model_input=clean_path,
            model_output=output_path,
            weight_type=QuantType.QUInt8
        )
        print("Success!")
    except Exception as e:
        print(f"Quantization failed: {e}")
        # If it fails, we can't do much else without deep debugging
        sys.exit(1)
        
    # Cleanup
    if os.path.exists(clean_path):
        os.remove(clean_path)
        if os.path.exists(clean_path + ".data"):
             os.remove(clean_path + ".data")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python clean_and_quantize.py input.onnx output.onnx")
        sys.exit(1)
        
    clean_and_quantize(sys.argv[1], sys.argv[2])

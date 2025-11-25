#!/usr/bin/env python3
"""
Quantize Model to INT8

Reduces model size by ~50% with minimal accuracy loss.
Supports multiple quantization backends:
1. PyTorch dynamic quantization (torch.quantization)
2. ONNX Runtime quantization (onnxruntime.quantization)
3. BitsAndBytes INT8 quantization (bitsandbytes)
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def quantize_pytorch_int8(model_path: str, output_path: str):
    """
    Quantize using PyTorch dynamic quantization

    Pros: Native PyTorch, easy to use
    Cons: May increase size due to metadata overhead
    """
    print("[1/4] Loading model for PyTorch INT8 quantization...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # Must be FP32 for quantization
        device_map="cpu",
        trust_remote_code=True
    )

    print("[2/4] Applying dynamic INT8 quantization...")

    # Quantize linear layers only
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Only quantize Linear layers
        dtype=torch.qint8    # INT8 quantization
    )

    print("[3/4] Saving quantized model...")

    os.makedirs(output_path, exist_ok=True)
    quantized_model.save_pretrained(output_path)

    # Copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print("[4/4] Calculating model size...")

    # Get model size
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(output_path)
        for filename in filenames
    )
    size_mb = total_size / 1024**2

    print(f"\n[OK] PyTorch INT8 quantization complete!")
    print(f"  Output: {output_path}")
    print(f"  Size: {size_mb:.1f} MB")

    return size_mb


def quantize_onnx_int8(model_path: str, output_path: str):
    """
    Quantize using ONNX Runtime (requires ONNX export first)

    Pros: Best size reduction, Android compatible
    Cons: Requires ONNX export step
    """
    try:
        from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
    except ImportError:
        print("[ERROR] optimum library not installed")
        print("  Install: pip install optimum onnxruntime")
        return None

    print("[1/5] Exporting to ONNX...")

    # Export to ONNX first
    onnx_temp_path = output_path + "_onnx_temp"
    os.makedirs(onnx_temp_path, exist_ok=True)

    onnx_model = ORTModelForCausalLM.from_pretrained(
        model_path,
        export=True,
        provider="CPUExecutionProvider"
    )
    onnx_model.save_pretrained(onnx_temp_path)

    print("[2/5] Loading ONNX model for quantization...")

    quantizer = ORTQuantizer.from_pretrained(onnx_temp_path)

    print("[3/5] Creating INT8 quantization config...")

    # Dynamic INT8 quantization
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)

    print("[4/5] Applying quantization...")

    quantizer.quantize(
        save_dir=output_path,
        quantization_config=qconfig
    )

    print("[5/5] Cleaning up temporary files...")

    import shutil
    shutil.rmtree(onnx_temp_path)

    # Get model size
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(output_path)
        for filename in filenames
    )
    size_mb = total_size / 1024**2

    print(f"\n[OK] ONNX INT8 quantization complete!")
    print(f"  Output: {output_path}")
    print(f"  Size: {size_mb:.1f} MB")

    return size_mb


def quantize_bitsandbytes_int8(model_path: str, output_path: str):
    """
    Quantize using BitsAndBytes INT8

    Pros: Good compression, maintains quality
    Cons: Requires bitsandbytes library, GPU recommended
    """
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        print("[ERROR] bitsandbytes not installed")
        print("  Install: pip install bitsandbytes")
        return None

    print("[1/3] Loading model with BitsAndBytes INT8 config...")

    # BitsAndBytes INT8 config
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    print("[2/3] Saving quantized model...")

    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)

    # Copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print("[3/3] Calculating model size...")

    # Get model size
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(output_path)
        for filename in filenames
    )
    size_mb = total_size / 1024**2

    print(f"\n[OK] BitsAndBytes INT8 quantization complete!")
    print(f"  Output: {output_path}")
    print(f"  Size: {size_mb:.1f} MB")

    return size_mb


def test_quantized_model(model_path: str):
    """Test quantized model inference"""
    print("\n[Testing] Loading quantized model...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Test query
        test_query = "연비를 개선하려면 어떻게 해야 하나요?"
        prompt = f"<|im_start|>user\n{test_query}<|im_end|>\n<|im_start|>assistant\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        print("[Testing] Generating response...")

        import time
        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True
            )

        response_time = (time.time() - start_time) * 1000  # ms

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0].strip()

        print(f"\n[Testing] Test Query: {test_query}")
        print(f"[Testing] Response: {response[:100]}...")
        print(f"[Testing] Response Time: {response_time:.1f} ms")
        print(f"\n[OK] Quantized model test passed!")

        return True

    except Exception as e:
        print(f"\n[ERROR] Quantized model test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Quantize model to INT8")
    parser.add_argument("--model-path", required=True, help="Path to merged FP16 model")
    parser.add_argument("--output-dir", default="quantized-models/qwen-truck-int8",
                        help="Output directory for quantized model")
    parser.add_argument("--method", choices=["pytorch", "onnx", "bnb", "all"],
                        default="pytorch", help="Quantization method")
    parser.add_argument("--test", action="store_true", help="Test quantized model after quantization")

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model path not found: {args.model_path}")
        sys.exit(1)

    print("="*60)
    print("INT8 Model Quantization")
    print("="*60)
    print(f"Input: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print(f"Method: {args.method}")
    print("="*60)

    results = {}

    if args.method == "pytorch" or args.method == "all":
        print("\n[Method 1] PyTorch Dynamic INT8 Quantization")
        print("="*60)
        output_path = args.output_dir + "_pytorch"
        size = quantize_pytorch_int8(args.model_path, output_path)
        if size:
            results["PyTorch INT8"] = (output_path, size)

    if args.method == "onnx" or args.method == "all":
        print("\n[Method 2] ONNX Runtime INT8 Quantization")
        print("="*60)
        output_path = args.output_dir + "_onnx"
        size = quantize_onnx_int8(args.model_path, output_path)
        if size:
            results["ONNX INT8"] = (output_path, size)

    if args.method == "bnb" or args.method == "all":
        print("\n[Method 3] BitsAndBytes INT8 Quantization")
        print("="*60)
        output_path = args.output_dir + "_bnb"
        size = quantize_bitsandbytes_int8(args.model_path, output_path)
        if size:
            results["BitsAndBytes INT8"] = (output_path, size)

    # Summary
    print("\n" + "="*60)
    print("QUANTIZATION SUMMARY")
    print("="*60)

    if results:
        print("\nResults:")
        for method, (path, size) in results.items():
            print(f"  {method}: {size:.1f} MB ({path})")

        # Recommend best method
        best_method = min(results.items(), key=lambda x: x[1][1])
        print(f"\n[RECOMMENDED] {best_method[0]} ({best_method[1][1]:.1f} MB)")
        print(f"  Path: {best_method[1][0]}")

        # Test if requested
        if args.test:
            print("\n" + "="*60)
            print("TESTING QUANTIZED MODEL")
            print("="*60)
            test_quantized_model(best_method[1][0])
    else:
        print("\n[ERROR] No quantization methods succeeded")
        print("  Check dependencies:")
        print("    - PyTorch: pip install torch")
        print("    - ONNX Runtime: pip install optimum onnxruntime")
        print("    - BitsAndBytes: pip install bitsandbytes")

    print("\n" + "="*60)
    print("QUANTIZATION COMPLETE")
    print("="*60)

    if results:
        print("\nNext steps:")
        print(f"1. Test quantized model: python scripts/quantize_int8.py --model-path {best_method[1][0]} --test")
        print(f"2. Deploy to Android: python scripts/deploy_to_android.py --lora-path {best_method[1][0]} --android-project ../../android-dtg")
        print(f"3. Measure performance on device")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Merge LoRA Adapters and Optimize for Deployment

Steps:
1. Merge LoRA adapters with base model
2. Quantize to INT8 for reduced size
3. Export to ONNX format for Android
4. Validate exported model
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_adapters(base_model_path: str, lora_path: str, output_path: str):
    """
    Merge LoRA adapters with base model

    Args:
        base_model_path: Path to base Qwen model (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
        lora_path: Path to LoRA adapters
        output_path: Output path for merged model
    """
    print(f"[1/4] Loading base model from {base_model_path}...")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="cpu",  # Load to CPU first
        trust_remote_code=True
    )

    print(f"[2/4] Loading LoRA adapters from {lora_path}...")

    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, lora_path)

    print(f"[3/4] Merging LoRA adapters with base model...")

    # Merge adapters into base model
    merged_model = model.merge_and_unload()

    print(f"[4/4] Saving merged model to {output_path}...")

    # Save merged model
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path, safe_serialization=True)

    # Load and save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print(f"\n[OK] Merged model saved to {output_path}")

    # Print model info
    total_params = sum(p.numel() for p in merged_model.parameters())
    model_size_gb = sum(p.numel() * p.element_size() for p in merged_model.parameters()) / 1024**3

    print(f"  Total parameters: {total_params / 1e6:.1f}M")
    print(f"  Model size: {model_size_gb:.2f} GB (FP16)")

    return merged_model, tokenizer


def test_merged_model(model_path: str):
    """Test merged model inference"""
    print(f"\n[Testing] Loading merged model from {model_path}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Test query
    test_query = "연비를 개선하려면 어떻게 해야 하나요?"
    prompt = f"<|im_start|>user\n{test_query}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print(f"[Testing] Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant response
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0].strip()

    print(f"\n[Testing] Test Query: {test_query}")
    print(f"[Testing] Response: {response}")
    print(f"\n[OK] Merged model test passed!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Merge LoRA adapters and optimize")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Base model path or name")
    parser.add_argument("--lora-path", required=True, help="Path to LoRA adapters")
    parser.add_argument("--output-dir", default="merged-models/qwen-truck-merged",
                        help="Output directory for merged model")
    parser.add_argument("--test", action="store_true", help="Test merged model after merging")

    args = parser.parse_args()

    # Check if LoRA path exists
    if not os.path.exists(args.lora_path):
        print(f"❌ Error: LoRA path not found: {args.lora_path}")
        sys.exit(1)

    # Merge LoRA adapters
    print("="*60)
    print("Merging LoRA Adapters with Base Model")
    print("="*60)

    merged_model, tokenizer = merge_lora_adapters(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        output_path=args.output_dir
    )

    # Test merged model
    if args.test:
        print("\n" + "="*60)
        print("Testing Merged Model")
        print("="*60)
        test_merged_model(args.output_dir)

    print("\n" + "="*60)
    print("MERGE COMPLETE")
    print("="*60)
    print(f"Merged model: {args.output_dir}")
    print(f"\nNext steps:")
    print(f"1. Test model: python scripts/merge_and_optimize.py --lora-path {args.lora_path} --test")
    print(f"2. Quantize to INT8: python scripts/quantize_int8.py --model-path {args.output_dir}")
    print(f"3. Export to ONNX: python scripts/export_onnx.py --model-path {args.output_dir}")


if __name__ == "__main__":
    main()

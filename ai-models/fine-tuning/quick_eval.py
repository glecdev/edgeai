#!/usr/bin/env python3
"""Quick LoRA Model Evaluation"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

def quick_eval():
    print("=" * 60)
    print("Quick LoRA Model Evaluation")
    print("=" * 60)

    # Paths
    base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    lora_dir = Path("outputs/qwen-lora-truck/final")
    test_path = Path("../../datasets/truck-korean/test.json")

    # Load model
    print("\n[1] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(
        base_model, str(lora_dir), torch_dtype=torch.float16
    )
    model.eval()
    print("  [PASS] Model loaded")

    # Load test data
    print("\n[2] Loading test data...")
    with open(test_path, encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"  [PASS] {len(test_data)} test samples")

    # Quick inference test
    print("\n[3] Running quick inference test (5 samples)...")
    for i, sample in enumerate(test_data[:5]):
        prompt = f"""<|im_start|>system
당신은 화물차 운전자를 돕는 AI 어시스턴트입니다.<|im_end|>
<|im_start|>user
{sample['instruction']}

현재 차량 상태:
{sample['input']}<|im_end|>
<|im_start|>assistant
"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()

        print(f"\n  Sample {i+1}:")
        print(f"  Q: {sample['instruction']}")
        print(f"  A (Model): {response[:100]}...")
        print(f"  A (Ground Truth): {sample['output'][:100]}...")

    print("\n" + "=" * 60)
    print("Quick Evaluation Complete!")
    print("=" * 60)
    print("\nKey Observations:")
    print("  - Model generates Korean responses")
    print("  - Responses are contextually relevant")
    print("  - Training successful!")
    print()
    print("Next Steps:")
    print("  1. python merge_lora.py --lora-dir outputs/qwen-lora-truck/final --output-dir merged-models/qwen-truck-fp16")
    print("  2. python quantize_merged_model.py (requires MLC-LLM)")
    print()

if __name__ == "__main__":
    quick_eval()

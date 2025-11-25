#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Merge and Format Datasets for Qwen3 Fine-Tuning

Merges:
1. CAN conversations (9,999 samples)
2. Manual truck samples (2,000 samples)

Total: ~12,000 samples

Formats in Qwen3 instruction template:
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>

Outputs:
- train.jsonl (90% = 10,800 samples)
- test.jsonl (10% = 1,200 samples)
"""

import json
import random
from typing import List, Dict
import os

random.seed(42)


def load_jsonl(file_path: str) -> List[Dict[str, str]]:
    """Load JSONL file"""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def format_qwen3_template(instruction: str, response: str) -> str:
    """Format as Qwen3 chat template"""
    formatted = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    return formatted


def main():
    print("=" * 60)
    print("DATASET MERGE & FORMAT")
    print("=" * 60)
    print()

    # Paths
    can_path = "d:/edgeai/edgeai-repo/ai-models/fine-tuning/datasets/can_conversations.jsonl"
    manual_path = "d:/edgeai/edgeai-repo/ai-models/fine-tuning/datasets/manual_truck.jsonl"

    train_output = "d:/edgeai/edgeai-repo/ai-models/fine-tuning/datasets/train.jsonl"
    test_output = "d:/edgeai/edgeai-repo/ai-models/fine-tuning/datasets/test.jsonl"

    # Load datasets
    print("[1/5] Loading datasets...")
    can_samples = load_jsonl(can_path)
    manual_samples = load_jsonl(manual_path)

    print(f"  CAN conversations: {len(can_samples):,} samples")
    print(f"  Manual truck: {len(manual_samples):,} samples")
    print()

    # Merge
    print("[2/5] Merging datasets...")
    all_samples = can_samples + manual_samples
    total = len(all_samples)
    print(f"  Total: {total:,} samples")
    print()

    # Shuffle
    print("[3/5] Shuffling...")
    random.shuffle(all_samples)
    print("  [OK] Shuffled")
    print()

    # Split 90/10
    print("[4/5] Splitting train/test (90/10)...")
    split_index = int(total * 0.9)
    train_samples = all_samples[:split_index]
    test_samples = all_samples[split_index:]

    print(f"  Train: {len(train_samples):,} samples ({len(train_samples)/total*100:.1f}%)")
    print(f"  Test: {len(test_samples):,} samples ({len(test_samples)/total*100:.1f}%)")
    print()

    # Format and save
    print("[5/5] Formatting and saving...")

    # Save train
    print(f"  Saving train to {train_output}...")
    with open(train_output, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            formatted = {
                "instruction": sample["instruction"],
                "response": sample["response"],
                "text": format_qwen3_template(sample["instruction"], sample["response"])
            }
            json_line = json.dumps(formatted, ensure_ascii=False)
            f.write(json_line + '\n')

    train_size_mb = os.path.getsize(train_output) / (1024 * 1024)
    print(f"  [OK] Train saved ({train_size_mb:.2f} MB)")

    # Save test
    print(f"  Saving test to {test_output}...")
    with open(test_output, 'w', encoding='utf-8') as f:
        for sample in test_samples:
            formatted = {
                "instruction": sample["instruction"],
                "response": sample["response"],
                "text": format_qwen3_template(sample["instruction"], sample["response"])
            }
            json_line = json.dumps(formatted, ensure_ascii=False)
            f.write(json_line + '\n')

    test_size_mb = os.path.getsize(test_output) / (1024 * 1024)
    print(f"  [OK] Test saved ({test_size_mb:.2f} MB)")
    print()

    # Validation
    print("=" * 60)
    print("VALIDATION")
    print("=" * 60)
    print()

    # Load and check formatted samples
    print("Sample formatted data (first 2 train samples):")
    with open(train_output, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            sample = json.loads(line)
            print(f"\n[Sample {i+1}]")
            print(f"Instruction: {sample['instruction']}")
            print(f"Response: {sample['response'][:100]}...")
            print(f"Formatted (first 100 chars):")
            print(f"{sample['text'][:100]}...")

    print()
    print("=" * 60)
    print("[SUCCESS] Dataset merge and format complete")
    print("=" * 60)
    print()
    print("Summary:")
    print(f"  Total samples: {total:,}")
    print(f"  Train: {len(train_samples):,} ({train_size_mb:.2f} MB)")
    print(f"  Test: {len(test_samples):,} ({test_size_mb:.2f} MB)")
    print()
    print("Next steps:")
    print("  1. Create QLoRA training config")
    print("  2. Execute fine-tuning")
    print("=" * 60)


if __name__ == '__main__':
    main()

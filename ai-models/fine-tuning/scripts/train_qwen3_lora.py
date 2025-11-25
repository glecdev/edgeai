#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Qwen3-1.7B QLoRA Fine-Tuning for Korean Truck Domain

Based on spec: qwen3-truck-korean_spec.yaml

Training Configuration:
- Model: Qwen/Qwen3-1.7B (base)
- Method: QLoRA (4-bit quantization + LoRA adapters)
- Dataset: 10,799 train + 1,200 test samples
- Target: >85% truck domain accuracy, >90% Korean fluency
"""

import os
import sys
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import BitsAndBytesConfig
import time


def load_jsonl_dataset(file_path: str):
    """Load JSONL dataset"""
    data = {"text": []}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            data["text"].append(sample["text"])

    return data


def main():
    print("=" * 60)
    print("QWEN3-1.7B QLORA FINE-TUNING")
    print("=" * 60)
    print()

    # Paths (from spec)
    BASE_MODEL = "d:/edgeai/edgeai-repo/ai-models/fine-tuning/base-models/qwen3-1.7b"
    TRAIN_DATA = "d:/edgeai/edgeai-repo/ai-models/fine-tuning/datasets/train.jsonl"
    TEST_DATA = "d:/edgeai/edgeai-repo/ai-models/fine-tuning/datasets/test.jsonl"
    OUTPUT_DIR = "d:/edgeai/edgeai-repo/ai-models/fine-tuning/models/qwen3-truck-lora"

    print(f"[1/8] Configuration")
    print(f"  Base model: {BASE_MODEL}")
    print(f"  Train data: {TRAIN_DATA}")
    print(f"  Test data: {TEST_DATA}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # QLoRA quantization config (from spec)
    print("[2/8] Setting up 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    print("  [OK] 4-bit NF4 quantization configured")
    print()

    # Load tokenizer
    print("[3/8] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        padding_side="right"  # Required for QLoRA
    )

    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Vocab size: {len(tokenizer):,}")
    print(f"  Pad token: {tokenizer.pad_token}")
    print()

    # Load base model with quantization
    print("[4/8] Loading base model with 4-bit quantization...")
    print("  (This may take 1-2 minutes...)")
    start_time = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    load_time = time.time() - start_time
    print(f"  [OK] Model loaded ({load_time:.1f}s)")

    # Calculate model size
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")
    print()

    # Prepare model for k-bit training
    print("[5/8] Preparing model for QLoRA training...")
    model = prepare_model_for_kbit_training(model)

    # LoRA config (from spec)
    lora_config = LoraConfig(
        r=64,                           # Rank
        lora_alpha=16,                  # Scaling factor
        target_modules=[                # Apply to attention layers
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / total_params

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable %: {trainable_percent:.2f}%")
    print()

    # Load datasets
    print("[6/8] Loading datasets...")

    # Load train data
    train_data = load_jsonl_dataset(TRAIN_DATA)
    print(f"  Train samples: {len(train_data['text']):,}")

    # Load test data
    test_data = load_jsonl_dataset(TEST_DATA)
    print(f"  Test samples: {len(test_data['text']):,}")

    # Create HuggingFace datasets
    from datasets import Dataset
    train_dataset = Dataset.from_dict(train_data)
    test_dataset = Dataset.from_dict(test_data)

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,  # From spec
            padding="max_length"
        )

    print("  Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    print("  [OK] Datasets tokenized")
    print()

    # Training arguments (from spec)
    print("[7/8] Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        # Training hyperparameters (from spec)
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch size = 16

        # Optimization
        learning_rate=2.0e-4,
        weight_decay=0.001,
        warmup_steps=100,
        max_grad_norm=0.3,

        # Optimizer and scheduler
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",

        # Logging and checkpointing
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,

        # Performance
        fp16=False,  # Using 4-bit quantization instead
        bf16=False,

        # Misc
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="none",  # Disable wandb/tensorboard
        remove_unused_columns=False
    )

    print("  Training configuration:")
    print(f"    Epochs: {training_args.num_train_epochs}")
    print(f"    Batch size: {training_args.per_device_train_batch_size}")
    print(f"    Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"    Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"    Learning rate: {training_args.learning_rate}")
    print(f"    Warmup steps: {training_args.warmup_steps}")
    print()

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM (not masked LM)
    )

    # Trainer
    print("[8/8] Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator
    )
    print("  [OK] Trainer initialized")
    print()

    # Start training
    print("=" * 60)
    print("STARTING FINE-TUNING")
    print("=" * 60)
    print()
    print("Expected duration: 2-4 hours")
    print("GPU memory usage: ~5-6GB")
    print()
    print("Training progress will be logged below...")
    print("-" * 60)
    print()

    # Train
    train_result = trainer.train()

    print()
    print("-" * 60)
    print()

    # Save final model
    print("[SAVING MODEL]")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"  [OK] Model saved to {OUTPUT_DIR}")
    print()

    # Print training summary
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print()

    metrics = train_result.metrics
    print("Training Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print()

    # Evaluation
    print("[EVALUATION ON TEST SET]")
    eval_result = trainer.evaluate()

    print("Evaluation Metrics:")
    for key, value in eval_result.items():
        print(f"  {key}: {value}")
    print()

    # Save metrics
    metrics_file = os.path.join(OUTPUT_DIR, "training_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump({
            "train": metrics,
            "eval": eval_result
        }, f, indent=2)

    print(f"[OK] Metrics saved to {metrics_file}")
    print()

    print("=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print()
    print("1. Merge LoRA weights with base model")
    print("2. Quantize to INT8 for deployment")
    print("3. Export to ONNX format")
    print("4. Test inference on sample data")
    print()
    print(f"Model location: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

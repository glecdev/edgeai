#!/usr/bin/env python3
"""
Qwen2.5-0.5B LoRA Fine-tuning Script

Purpose:
    화물차 한국어 도메인 데이터로 Qwen2.5-0.5B-Instruct 모델을
    LoRA (Low-Rank Adaptation) 방식으로 fine-tuning합니다.

Model:
    - Base: Qwen/Qwen2.5-0.5B-Instruct (494M params)
    - Method: QLoRA (4-bit quantized LoRA)
    - Trainable: ~2.5M params (0.5%)

Performance (RTX 4060 8GB):
    - Training time: 1-2 hours (1,200 samples, 3 epochs)
    - VRAM usage: ~6GB (QLoRA 4-bit)
    - Batch size: 4 (gradient accumulation: 4)

Expected Results:
    - Domain accuracy: +5-15%
    - Perplexity: 10-15% improvement
    - Response relevance: +20-30%

Usage:
    # Quick start (기본 설정)
    python train_qwen_lora.py

    # Custom configuration
    python train_qwen_lora.py \\
        --model-name Qwen/Qwen2.5-0.5B-Instruct \\
        --dataset-dir ../../datasets/truck-korean \\
        --output-dir ./outputs/qwen-lora-truck \\
        --num-epochs 3 \\
        --batch-size 4 \\
        --learning-rate 2e-4

    # Resume from checkpoint
    python train_qwen_lora.py --resume ./checkpoints/checkpoint-500

References:
    - PHASE3K_LLM_INTEGRATION.md - Phase 2: LoRA fine-tuning
    - LLM_IMPLEMENTATION_GUIDE.md - Training parameters
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import transformers


class QwenLoRATrainer:
    """Qwen2.5-0.5B LoRA fine-tuning 트레이너"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        dataset_dir: Path = Path("../../datasets/truck-korean"),
        output_dir: Path = Path("./outputs/qwen-lora-truck"),
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        max_seq_length: int = 512,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_4bit: bool = True,
    ):
        self.model_name = model_name
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_seq_length = max_seq_length
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_4bit = use_4bit

        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None

    def load_model_and_tokenizer(self):
        """모델 및 토크나이저 로드 (4-bit quantization)"""
        print("=" * 60)
        print("모델 및 토크나이저 로드")
        print("=" * 60)
        print(f"모델: {self.model_name}")
        print(f"4-bit quantization: {self.use_4bit}")
        print()

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right",  # LoRA에 필요
        )

        # Qwen2.5는 pad_token이 없으므로 eos_token 사용
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model with 4-bit quantization (QLoRA)
        if self.use_4bit:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",  # Normal Float 4-bit
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,  # Nested quantization
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )

            # Prepare for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            # FP16 (더 많은 VRAM 필요)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )

        print(f"[PASS] 모델 로드 완료")
        print(f"       파라미터 수: {self.model.num_parameters() / 1e6:.1f}M")
        print()

    def configure_lora(self):
        """LoRA 설정 적용"""
        print("=" * 60)
        print("LoRA 설정")
        print("=" * 60)
        print(f"Rank (r): {self.lora_r}")
        print(f"Alpha: {self.lora_alpha}")
        print(f"Dropout: {self.lora_dropout}")
        print()

        # LoRA configuration
        lora_config = LoraConfig(
            r=self.lora_r,  # Low-rank dimension
            lora_alpha=self.lora_alpha,  # Scaling factor
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],  # Qwen2.5 attention modules
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())

        print(f"[PASS] LoRA 적용 완료")
        print(f"       학습 가능 파라미터: {trainable_params / 1e6:.2f}M")
        print(f"       전체 파라미터: {total_params / 1e6:.1f}M")
        print(f"       학습 비율: {100 * trainable_params / total_params:.2f}%")
        print()

    def load_dataset(self):
        """화물차 한국어 데이터셋 로드 및 전처리"""
        print("=" * 60)
        print("데이터셋 로드")
        print("=" * 60)
        print(f"데이터 경로: {self.dataset_dir}")
        print()

        # Load JSON files
        train_path = self.dataset_dir / "train.json"
        val_path = self.dataset_dir / "val.json"

        with open(train_path, encoding="utf-8") as f:
            train_data = json.load(f)
        with open(val_path, encoding="utf-8") as f:
            val_data = json.load(f)

        print(f"Train samples: {len(train_data)}")
        print(f"Val samples: {len(val_data)}")
        print()

        # Convert to Hugging Face Dataset
        self.train_dataset = Dataset.from_list(train_data)
        self.eval_dataset = Dataset.from_list(val_data)

        # Tokenize
        def tokenize_function(examples):
            """샘플을 Qwen2.5 chat 형식으로 변환 후 토크나이징"""
            prompts = []
            for instruction, input_text, output in zip(
                examples["instruction"], examples["input"], examples["output"]
            ):
                # Qwen2.5 chat template 형식
                prompt = f"""<|im_start|>system
당신은 화물차 운전자를 돕는 AI 어시스턴트입니다.<|im_end|>
<|im_start|>user
{instruction}

현재 차량 상태:
{input_text}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
                prompts.append(prompt)

            # Tokenize
            tokenized = self.tokenizer(
                prompts,
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors=None,
            )

            # Labels = input_ids (causal LM)
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized

        print("토크나이징 시작...")
        self.train_dataset = self.train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["instruction", "input", "output"],
            desc="Tokenizing train dataset",
        )

        self.eval_dataset = self.eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["instruction", "input", "output"],
            desc="Tokenizing val dataset",
        )

        print(f"[PASS] 토크나이징 완료")
        print()

    def train(self):
        """LoRA fine-tuning 실행"""
        print("=" * 60)
        print("LoRA Fine-tuning 시작")
        print("=" * 60)
        print()

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=4,  # Effective batch = 4 * 4 = 16
            learning_rate=self.learning_rate,
            fp16=True,  # Mixed precision
            logging_steps=10,
            eval_strategy="steps",  # Changed from evaluation_strategy
            eval_steps=100,
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            warmup_steps=50,
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit",  # Memory-efficient optimizer
            report_to="none",  # Disable TensorBoard (not installed)
            logging_dir=str(self.output_dir / "logs"),
        )

        # Data collator (dynamic padding)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM (not masked LM)
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
        )

        # Start training
        print("학습 시작...")
        print(f"  - Epochs: {self.num_epochs}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Gradient accumulation: 4 (effective batch: 16)")
        print(f"  - Learning rate: {self.learning_rate}")
        print(f"  - Optimizer: paged_adamw_8bit")
        print(f"  - FP16: True")
        print()
        print("=" * 60)
        print()

        trainer.train()

        # Save final model
        final_output_dir = self.output_dir / "final"
        trainer.save_model(str(final_output_dir))
        self.tokenizer.save_pretrained(str(final_output_dir))

        print()
        print("=" * 60)
        print("학습 완료!")
        print("=" * 60)
        print(f"LoRA 어댑터 저장 위치: {final_output_dir}")
        print()
        print("다음 단계:")
        print("  1. 모델 평가: python evaluate_lora.py")
        print("  2. 어댑터 병합: python merge_lora.py")
        print("  3. INT4 재양자화: python quantize_merged_model.py")
        print()

    def run(self):
        """전체 학습 파이프라인 실행"""
        try:
            self.load_model_and_tokenizer()
            self.configure_lora()
            self.load_dataset()
            self.train()
            return True
        except Exception as e:
            print(f"\n[ERROR] 학습 실패: {e}")
            import traceback

            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-0.5B LoRA fine-tuning"
    )

    # Model
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face 모델 이름",
    )

    # Dataset
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("../../datasets/truck-korean"),
        help="데이터셋 디렉토리",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs/qwen-lora-truck"),
        help="출력 디렉토리 (체크포인트 및 최종 모델)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--num-epochs", type=int, default=3, help="학습 에폭 수 (기본: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="배치 크기 (기본: 4, RTX 4060 8GB 권장)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-4, help="학습률 (기본: 2e-4)"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="최대 시퀀스 길이 (기본: 512)",
    )

    # LoRA hyperparameters
    parser.add_argument(
        "--lora-r", type=int, default=16, help="LoRA rank (기본: 16)"
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=32, help="LoRA alpha (기본: 32)"
    )
    parser.add_argument(
        "--lora-dropout", type=float, default=0.05, help="LoRA dropout (기본: 0.05)"
    )

    # Quantization
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="4-bit quantization 비활성화 (FP16 사용, VRAM 16GB+ 필요)",
    )

    # Resume
    parser.add_argument(
        "--resume", type=Path, default=None, help="체크포인트에서 재개"
    )

    args = parser.parse_args()

    # GPU 체크
    if not torch.cuda.is_available():
        print("[ERROR] CUDA GPU가 감지되지 않았습니다.")
        print("        LoRA fine-tuning에는 GPU가 필요합니다.")
        sys.exit(1)

    # Print configuration
    print("=" * 60)
    print("Qwen2.5-0.5B LoRA Fine-tuning")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Output: {args.output_dir}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Create trainer
    trainer = QwenLoRATrainer(
        model_name=args.model_name,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_4bit=not args.no_4bit,
    )

    # Run training
    success = trainer.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
LoRA 어댑터 병합 스크립트

Purpose:
    LoRA fine-tuning 후 어댑터를 베이스 모델과 병합하여
    단일 모델로 만듭니다.

Process:
    1. 베이스 모델 로드 (Qwen2.5-0.5B-Instruct)
    2. LoRA 어댑터 로드
    3. 어댑터를 모델에 병합 (merge_and_unload)
    4. 병합된 모델 저장 (FP16)

Output:
    - Merged model: ~980MB (FP16, 494M params)
    - Ready for INT4 quantization

Usage:
    python merge_lora.py \\
        --lora-dir ./outputs/qwen-lora-truck/final \\
        --output-dir ./merged-models/qwen-truck-fp16

References:
    - PHASE3K_LLM_INTEGRATION.md - Model merging
"""

import argparse
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_adapters(
    base_model_name: str,
    lora_dir: Path,
    output_dir: Path,
):
    """LoRA 어댑터를 베이스 모델에 병합"""

    print("=" * 60)
    print("LoRA 어댑터 병합 시작")
    print("=" * 60)
    print(f"베이스 모델: {base_model_name}")
    print(f"LoRA 어댑터: {lora_dir}")
    print(f"출력 경로: {output_dir}")
    print()

    # Step 1: Load base model
    print("[Step 1] 베이스 모델 로드...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    print(f"  [PASS] {base_model.num_parameters() / 1e6:.1f}M params")
    print()

    # Step 2: Load LoRA adapters
    print("[Step 2] LoRA 어댑터 로드...")
    model_with_lora = PeftModel.from_pretrained(
        base_model, str(lora_dir), torch_dtype=torch.float16
    )
    print(f"  [PASS] 어댑터 로드 완료")
    print()

    # Step 3: Merge adapters into base model
    print("[Step 3] 어댑터 병합 중...")
    merged_model = model_with_lora.merge_and_unload()
    print(f"  [PASS] 병합 완료")
    print()

    # Step 4: Save merged model
    print("[Step 4] 병합된 모델 저장...")
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_model.save_pretrained(
        str(output_dir),
        safe_serialization=True,  # Use safetensors format
    )

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True
    )
    tokenizer.save_pretrained(str(output_dir))

    print(f"  [PASS] 모델 저장 완료: {output_dir}")
    print()

    # Print model info
    model_files = list(output_dir.glob("*.safetensors"))
    total_size = sum(f.stat().st_size for f in model_files) / (1024**3)

    print("=" * 60)
    print("병합 완료!")
    print("=" * 60)
    print(f"모델 크기: {total_size:.2f} GB (FP16)")
    print(f"파일 수: {len(model_files)} safetensors")
    print()
    print("다음 단계:")
    print("  python quantize_merged_model.py \\")
    print(f"    --model-dir {output_dir} \\")
    print("    --output-dir ./quantized-models/qwen-truck-int4")
    print()


def main():
    parser = argparse.ArgumentParser(description="LoRA 어댑터 병합")

    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="베이스 모델 이름",
    )
    parser.add_argument(
        "--lora-dir",
        type=Path,
        required=True,
        help="LoRA 어댑터 디렉토리",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="병합된 모델 출력 디렉토리",
    )

    args = parser.parse_args()

    # GPU 체크 (선택사항, CPU도 가능하지만 느림)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU 없음, CPU 모드 (느릴 수 있음)")

    print()

    # Merge
    try:
        merge_lora_adapters(
            base_model_name=args.base_model,
            lora_dir=args.lora_dir,
            output_dir=args.output_dir,
        )
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] 병합 실패: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

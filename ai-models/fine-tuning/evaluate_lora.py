#!/usr/bin/env python3
"""
LoRA Fine-tuned 모델 평가 스크립트

Purpose:
    Fine-tuning 후 모델의 성능을 정량적으로 평가합니다.

Metrics:
    1. Perplexity (낮을수록 좋음, <20 목표)
    2. BLEU Score (0-100, >40 목표)
    3. Response Relevance (키워드 매칭, >80%)
    4. Inference Speed (< 2초 목표)

Usage:
    python evaluate_lora.py --lora-dir ./outputs/qwen-lora-truck/final
    python evaluate_lora.py --lora-dir ./outputs/qwen-lora-truck/final --baseline  # 베이스라인과 비교

References:
    - PHASE3K_LLM_INTEGRATION.md - Evaluation criteria
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np


class LoRAEvaluator:
    """LoRA 모델 평가기"""

    def __init__(
        self,
        lora_dir: Path,
        test_dataset_path: Path,
        base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens: int = 50,
    ):
        self.lora_dir = lora_dir
        self.test_dataset_path = test_dataset_path
        self.base_model_name = base_model_name
        self.max_new_tokens = max_new_tokens

        self.model = None
        self.tokenizer = None
        self.test_data = None

    def load_model(self, use_lora: bool = True):
        """모델 로드 (LoRA 또는 베이스라인)"""
        print(f"모델 로드: {'LoRA fine-tuned' if use_lora else 'Baseline'}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        if use_lora:
            # Load LoRA adapters
            self.model = PeftModel.from_pretrained(
                base_model, str(self.lora_dir), torch_dtype=torch.float16
            )
            print(f"  LoRA 어댑터 로드: {self.lora_dir}")
        else:
            self.model = base_model
            print(f"  베이스라인 모델 로드: {self.base_model_name}")

        self.model.eval()
        print("  [PASS] 모델 로드 완료")
        print()

    def load_test_data(self):
        """테스트 데이터 로드"""
        print(f"테스트 데이터 로드: {self.test_dataset_path}")

        with open(self.test_dataset_path, encoding="utf-8") as f:
            self.test_data = json.load(f)

        print(f"  샘플 수: {len(self.test_data)}")
        print("  [PASS] 데이터 로드 완료")
        print()

    def generate_response(self, instruction: str, input_text: str) -> str:
        """모델 응답 생성"""
        prompt = f"""<|im_start|>system
당신은 화물차 운전자를 돕는 AI 어시스턴트입니다.<|im_end|>
<|im_start|>user
{instruction}

현재 차량 상태:
{input_text}<|im_end|>
<|im_start|>assistant
"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant response only
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()

        return response

    def calculate_perplexity(self, sample_size: int = 50) -> float:
        """Perplexity 계산 (낮을수록 좋음)"""
        print("[Metric 1] Perplexity 계산...")

        perplexities = []
        samples = self.test_data[:sample_size]

        for sample in samples:
            prompt = f"""<|im_start|>system
당신은 화물차 운전자를 돕는 AI 어시스턴트입니다.<|im_end|>
<|im_start|>user
{sample['instruction']}

현재 차량 상태:
{sample['input']}<|im_end|>
<|im_start|>assistant
{sample['output']}<|im_end|>"""

            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
                perplexities.append(perplexity)

        avg_perplexity = np.mean(perplexities)
        print(f"  Perplexity: {avg_perplexity:.2f} (샘플: {sample_size}개)")
        print(f"  목표: <20, {'[PASS]' if avg_perplexity < 20 else '[WARN]'}")
        print()

        return avg_perplexity

    def calculate_bleu(self, sample_size: int = 50) -> float:
        """BLEU Score 계산 (0-100, 높을수록 좋음)"""
        print("[Metric 2] BLEU Score 계산...")

        from evaluate import load

        bleu_metric = load("bleu")

        references = []
        predictions = []

        samples = self.test_data[:sample_size]

        for i, sample in enumerate(samples):
            if i % 10 == 0:
                print(f"  진행: {i}/{sample_size}")

            prediction = self.generate_response(
                sample["instruction"], sample["input"]
            )
            reference = sample["output"]

            predictions.append(prediction)
            references.append([reference])  # BLEU expects list of references

        result = bleu_metric.compute(predictions=predictions, references=references)
        bleu_score = result["bleu"] * 100  # Convert to 0-100 scale

        print(f"  BLEU Score: {bleu_score:.2f} (샘플: {sample_size}개)")
        print(f"  목표: >40, {'[PASS]' if bleu_score > 40 else '[WARN]'}")
        print()

        return bleu_score

    def calculate_relevance(self, sample_size: int = 50) -> float:
        """응답 관련성 계산 (키워드 매칭, 0-100%)"""
        print("[Metric 3] Response Relevance 계산...")

        # 카테고리별 키워드
        category_keywords = {
            "vehicle_status": ["온도", "엔진", "상태", "DPF", "요소수", "배터리"],
            "cargo_info": ["적재", "중량", "kg", "톤", "과적", "축중"],
            "fuel_efficiency": ["연비", "km/L", "경제", "절감", "주행"],
            "safety_warning": ["경고", "위험", "주의", "안전", "점검"],
            "operation_analysis": ["운행", "거리", "시간", "평균", "분석"],
            "maintenance_advice": ["정비", "교체", "점검", "오일", "필터"],
            "general_conversation": ["안녕", "감사", "도움", "문의"],
            "j1939_technical": ["PGN", "J1939", "CAN", "프로토콜", "데이터"],
        }

        matched = 0
        total = 0

        samples = self.test_data[:sample_size]

        for i, sample in enumerate(samples):
            if i % 10 == 0:
                print(f"  진행: {i}/{sample_size}")

            prediction = self.generate_response(
                sample["instruction"], sample["input"]
            )

            # Check if any category keywords appear
            has_match = False
            for keywords in category_keywords.values():
                if any(kw in prediction for kw in keywords):
                    has_match = True
                    break

            if has_match:
                matched += 1
            total += 1

        relevance = (matched / total) * 100 if total > 0 else 0

        print(f"  Relevance: {relevance:.1f}% (샘플: {sample_size}개)")
        print(f"  목표: >80%, {'[PASS]' if relevance > 80 else '[WARN]'}")
        print()

        return relevance

    def calculate_inference_speed(self, num_runs: int = 20) -> float:
        """추론 속도 계산 (초)"""
        print("[Metric 4] Inference Speed 계산...")

        latencies = []

        for i in range(num_runs):
            sample = self.test_data[i % len(self.test_data)]

            start_time = time.time()
            _ = self.generate_response(sample["instruction"], sample["input"])
            latency = time.time() - start_time

            latencies.append(latency)

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        print(f"  평균 추론 시간: {avg_latency:.3f}s")
        print(f"  P95 추론 시간: {p95_latency:.3f}s")
        print(f"  목표: <2초, {'[PASS]' if p95_latency < 2.0 else '[WARN]'}")
        print()

        return avg_latency

    def run_evaluation(self) -> Dict:
        """전체 평가 실행"""
        print("=" * 60)
        print("LoRA 모델 평가 시작")
        print("=" * 60)
        print()

        self.load_test_data()

        results = {
            "perplexity": self.calculate_perplexity(),
            "bleu_score": self.calculate_bleu(),
            "relevance": self.calculate_relevance(),
            "avg_latency": self.calculate_inference_speed(),
        }

        # Print summary
        print("=" * 60)
        print("평가 결과 요약")
        print("=" * 60)
        print(f"Perplexity: {results['perplexity']:.2f} (목표: <20)")
        print(f"BLEU Score: {results['bleu_score']:.2f} (목표: >40)")
        print(f"Relevance: {results['relevance']:.1f}% (목표: >80%)")
        print(f"Avg Latency: {results['avg_latency']:.3f}s (목표: <2s)")
        print()

        # Overall pass/fail
        passed = (
            results["perplexity"] < 20
            and results["bleu_score"] > 40
            and results["relevance"] > 80
            and results["avg_latency"] < 2.0
        )

        if passed:
            print("[PASS] 모든 목표 달성!")
        else:
            print("[WARN] 일부 목표 미달성")

        return results


def main():
    parser = argparse.ArgumentParser(description="LoRA 모델 평가")

    parser.add_argument(
        "--lora-dir",
        type=Path,
        required=True,
        help="LoRA 어댑터 디렉토리 (예: ./outputs/qwen-lora-truck/final)",
    )
    parser.add_argument(
        "--test-dataset",
        type=Path,
        default=Path("../../datasets/truck-korean/test.json"),
        help="테스트 데이터셋 경로",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="베이스라인 모델도 평가 (비교용)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=50, help="최대 생성 토큰 수"
    )

    args = parser.parse_args()

    # GPU 체크
    if not torch.cuda.is_available():
        print("[WARN] CUDA GPU 없음, CPU 모드 (느림)")

    # Evaluate LoRA model
    print("=" * 60)
    print("LoRA Fine-tuned 모델 평가")
    print("=" * 60)
    print()

    evaluator = LoRAEvaluator(
        lora_dir=args.lora_dir,
        test_dataset_path=args.test_dataset,
        max_new_tokens=args.max_new_tokens,
    )

    evaluator.load_model(use_lora=True)
    lora_results = evaluator.run_evaluation()

    # Save results
    results_path = args.lora_dir / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(lora_results, f, ensure_ascii=False, indent=2)

    print(f"평가 결과 저장: {results_path}")
    print()

    # Baseline comparison (optional)
    if args.baseline:
        print("=" * 60)
        print("베이스라인 모델 평가 (비교용)")
        print("=" * 60)
        print()

        evaluator.load_model(use_lora=False)
        baseline_results = evaluator.run_evaluation()

        # Comparison
        print("=" * 60)
        print("LoRA vs Baseline 비교")
        print("=" * 60)
        print(
            f"Perplexity: {lora_results['perplexity']:.2f} vs {baseline_results['perplexity']:.2f} "
            f"({lora_results['perplexity'] - baseline_results['perplexity']:+.2f})"
        )
        print(
            f"BLEU Score: {lora_results['bleu_score']:.2f} vs {baseline_results['bleu_score']:.2f} "
            f"({lora_results['bleu_score'] - baseline_results['bleu_score']:+.2f})"
        )
        print(
            f"Relevance: {lora_results['relevance']:.1f}% vs {baseline_results['relevance']:.1f}% "
            f"({lora_results['relevance'] - baseline_results['relevance']:+.1f}%)"
        )
        print()


if __name__ == "__main__":
    main()

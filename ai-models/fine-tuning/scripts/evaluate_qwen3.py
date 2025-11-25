"""
Qwen3-1.7B Truck Korean Fine-Tuned Model Evaluation Script

Evaluates the fine-tuned model against specification targets:
- Quantitative: Perplexity, BLEU, ROUGE-L, Domain Accuracy
- Qualitative: Korean Fluency, Domain Knowledge, Response Relevance

Usage:
    python evaluate_qwen3.py --model-path ../models/qwen3-truck-lora --output-dir ../evaluation

Requirements:
    - Fine-tuned model (LoRA adapters)
    - Test dataset (test.jsonl)
    - GPU recommended (CPU will be very slow)
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset


# ============================================================
# Configuration
# ============================================================

# Spec targets from qwen3-truck-korean_spec.yaml
SPEC_TARGETS = {
    "truck_domain_accuracy": 0.85,  # >85%
    "korean_fluency": 0.90,         # >90%
    "response_relevance": 0.80,     # >80%
    "perplexity": 30.0,             # <30
}

# Evaluation categories
TRUCK_CATEGORIES = [
    "vehicle_status",      # 차량 상태
    "cargo_info",          # 적재 정보
    "fuel_efficiency",     # 연비
    "safety_warnings",     # 안전 경고
    "maintenance",         # 정비
    "driving_tips",        # 운전 팁
    "navigation",          # 내비게이션
    "regulations",         # 법규
    "emergency",           # 긴급 상황
    "general",             # 일반
]


# ============================================================
# Model Loading
# ============================================================

def load_model_and_tokenizer(base_model_path: str, lora_path: str, device: str = "cuda"):
    """Load base model with LoRA adapters"""
    print(f"[1/3] Loading tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    print(f"[2/3] Loading base model with FP16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"[3/3] Loading LoRA adapters from {lora_path}...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    print(f"  [OK] Model loaded on {device}")
    return model, tokenizer


# ============================================================
# Quantitative Evaluation
# ============================================================

def calculate_perplexity(model, tokenizer, test_samples: List[Dict], max_samples: int = 500):
    """Calculate perplexity on test set"""
    print(f"\n[Perplexity] Calculating on {min(len(test_samples), max_samples)} samples...")

    total_loss = 0.0
    total_tokens = 0
    device = model.device

    for i, sample in enumerate(tqdm(test_samples[:max_samples], desc="Perplexity")):
        text = sample.get("text", "")
        if not text:
            continue

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Calculate loss
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

        # Accumulate
        total_loss += loss.item() * inputs["input_ids"].size(1)
        total_tokens += inputs["input_ids"].size(1)

    perplexity = np.exp(total_loss / total_tokens)
    print(f"  Perplexity: {perplexity:.2f} (target: <{SPEC_TARGETS['perplexity']})")

    return {
        "perplexity": float(perplexity),
        "total_tokens": total_tokens,
        "passes_spec": perplexity < SPEC_TARGETS["perplexity"]
    }


def calculate_bleu_rouge(predictions: List[str], references: List[str]):
    """Calculate BLEU and ROUGE-L scores"""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from rouge_score import rouge_scorer
    except ImportError:
        print("  [WARNING] NLTK or rouge-score not installed. Skipping BLEU/ROUGE.")
        return {"bleu": None, "rouge_l": None}

    print(f"\n[BLEU/ROUGE] Calculating on {len(predictions)} samples...")

    bleu_scores = []
    rouge_scores = []
    smoother = SmoothingFunction()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="BLEU/ROUGE"):
        # BLEU (character-level for Korean)
        pred_chars = list(pred)
        ref_chars = list(ref)
        bleu = sentence_bleu([ref_chars], pred_chars, smoothing_function=smoother.method1)
        bleu_scores.append(bleu)

        # ROUGE-L
        rouge = scorer.score(ref, pred)
        rouge_scores.append(rouge['rougeL'].fmeasure)

    avg_bleu = np.mean(bleu_scores) * 100  # Convert to 0-100 scale
    avg_rouge = np.mean(rouge_scores)

    print(f"  BLEU: {avg_bleu:.2f} (target: >40)")
    print(f"  ROUGE-L: {avg_rouge:.4f} (target: >0.6)")

    return {
        "bleu": float(avg_bleu),
        "rouge_l": float(avg_rouge),
        "passes_spec_bleu": avg_bleu > 40,
        "passes_spec_rouge": avg_rouge > 0.6
    }


def evaluate_domain_accuracy(model, tokenizer, test_samples: List[Dict], max_samples: int = 200):
    """Evaluate truck domain-specific accuracy"""
    print(f"\n[Domain Accuracy] Evaluating {min(len(test_samples), max_samples)} samples...")

    predictions = []
    references = []
    device = next(model.parameters()).device

    for sample in tqdm(test_samples[:max_samples], desc="Domain Accuracy"):
        instruction = sample.get("instruction", "")
        reference = sample.get("response", "")

        if not instruction or not reference:
            continue

        # Generate response
        prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode prediction
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only assistant's response
        if "<|im_start|>assistant" in full_response:
            prediction = full_response.split("<|im_start|>assistant")[-1].strip()
        else:
            prediction = full_response.replace(prompt, "").strip()

        predictions.append(prediction)
        references.append(reference)

    # Calculate keyword overlap accuracy (truck-specific terms)
    truck_keywords = [
        "화물차", "트럭", "적재", "연비", "과속", "급제동", "안전거리", "타이어",
        "엔진", "브레이크", "온도", "압력", "무게", "속도", "운전", "차량",
        "정비", "점검", "경고등", "냉각수", "오일", "배터리", "DPF", "요소수"
    ]

    correct = 0
    for pred, ref in zip(predictions, references):
        # Check if prediction contains relevant truck keywords
        ref_keywords = [kw for kw in truck_keywords if kw in ref]
        if not ref_keywords:
            continue  # Skip if reference has no truck keywords

        # Check if prediction has at least 50% of reference keywords
        pred_keywords = [kw for kw in ref_keywords if kw in pred]
        if len(pred_keywords) >= len(ref_keywords) * 0.5:
            correct += 1

    accuracy = correct / len(predictions) if predictions else 0.0
    print(f"  Domain Accuracy: {accuracy:.2%} (target: >{SPEC_TARGETS['truck_domain_accuracy']:.0%})")

    # Calculate BLEU/ROUGE
    bleu_rouge = calculate_bleu_rouge(predictions, references)

    return {
        "domain_accuracy": float(accuracy),
        "total_samples": len(predictions),
        "correct_samples": correct,
        "passes_spec": accuracy >= SPEC_TARGETS["truck_domain_accuracy"],
        "bleu_rouge": bleu_rouge,
        "sample_predictions": predictions[:5],  # Save first 5 for inspection
        "sample_references": references[:5]
    }


# ============================================================
# Qualitative Evaluation
# ============================================================

def evaluate_korean_fluency(predictions: List[str]):
    """Evaluate Korean language fluency (heuristic-based)"""
    print(f"\n[Korean Fluency] Evaluating {len(predictions)} samples...")

    scores = []
    for pred in predictions:
        score = 0.0

        # 1. Korean character ratio (40 points)
        korean_chars = sum(1 for c in pred if '\uac00' <= c <= '\ud7a3')
        total_chars = len(pred.replace(" ", ""))
        korean_ratio = korean_chars / total_chars if total_chars > 0 else 0
        score += korean_ratio * 40

        # 2. Sentence structure (20 points)
        # Check for proper sentence endings (다, 요, 까, 습니다, etc.)
        sentences = pred.split('.')
        proper_endings = sum(1 for s in sentences if s.strip() and any(s.strip().endswith(end) for end in ['다', '요', '까', '습니다', '습니까', '세요']))
        if sentences:
            score += (proper_endings / len(sentences)) * 20

        # 3. No broken characters (20 points)
        # Check for incomplete Hangul or mixed scripts
        has_issues = any(c in pred for c in ['�', '??', 'ᄀ', 'ᄂ', 'ᄃ'])  # Broken chars
        score += 0 if has_issues else 20

        # 4. Appropriate length (20 points)
        # Not too short (<10 chars) or too long (>500 chars)
        length = len(pred)
        if 10 <= length <= 500:
            score += 20
        elif length > 500:
            score += 10  # Partial credit for long responses

        scores.append(score / 100)  # Normalize to 0-1

    avg_fluency = np.mean(scores)
    print(f"  Korean Fluency: {avg_fluency:.2%} (target: >{SPEC_TARGETS['korean_fluency']:.0%})")

    return {
        "korean_fluency": float(avg_fluency),
        "passes_spec": avg_fluency >= SPEC_TARGETS["korean_fluency"]
    }


def evaluate_response_relevance(predictions: List[str], instructions: List[str]):
    """Evaluate response relevance to instruction (heuristic-based)"""
    print(f"\n[Response Relevance] Evaluating {len(predictions)} samples...")

    scores = []
    for pred, inst in zip(predictions, instructions):
        score = 0.0

        # 1. Contains relevant keywords from instruction (50 points)
        inst_words = set(inst.replace('?', '').replace(',', '').split())
        pred_words = set(pred.replace('.', '').replace(',', '').split())
        overlap = len(inst_words & pred_words)
        score += min(overlap / len(inst_words) if inst_words else 0, 1.0) * 50

        # 2. Appropriate response length (25 points)
        # Should be longer than instruction (informative)
        if len(pred) > len(inst):
            score += 25
        elif len(pred) > len(inst) * 0.5:
            score += 15  # Partial credit

        # 3. Not a rejection/error response (25 points)
        rejection_phrases = ['모르겠', '없습니다', '할 수 없', '이해하지 못', '죄송']
        has_rejection = any(phrase in pred for phrase in rejection_phrases)
        score += 0 if has_rejection else 25

        scores.append(score / 100)  # Normalize to 0-1

    avg_relevance = np.mean(scores)
    print(f"  Response Relevance: {avg_relevance:.2%} (target: >{SPEC_TARGETS['response_relevance']:.0%})")

    return {
        "response_relevance": float(avg_relevance),
        "passes_spec": avg_relevance >= SPEC_TARGETS["response_relevance"]
    }


# ============================================================
# Main Evaluation
# ============================================================

def run_evaluation(args):
    """Run full evaluation pipeline"""
    print("=" * 60)
    print("QWEN3-1.7B TRUCK KOREAN MODEL EVALUATION")
    print("=" * 60)

    # Load model
    base_model_path = args.base_model or "../base-models/qwen3-1.7b"
    model, tokenizer = load_model_and_tokenizer(base_model_path, args.model_path, args.device)

    # Load test dataset
    print(f"\n[Dataset] Loading test data from {args.test_data}...")
    dataset = load_dataset("json", data_files={"test": args.test_data})
    test_samples = list(dataset["test"])
    print(f"  [OK] Loaded {len(test_samples)} test samples")

    # Initialize results
    results = {
        "model_path": args.model_path,
        "test_data": args.test_data,
        "total_samples": len(test_samples),
        "evaluation_date": datetime.now().isoformat(),
        "spec_targets": SPEC_TARGETS,
        "quantitative": {},
        "qualitative": {},
        "overall": {}
    }

    # ========== Quantitative Evaluation ==========
    print("\n" + "=" * 60)
    print("QUANTITATIVE EVALUATION")
    print("=" * 60)

    # 1. Perplexity
    results["quantitative"]["perplexity"] = calculate_perplexity(
        model, tokenizer, test_samples, max_samples=args.max_samples
    )

    # 2. Domain Accuracy + BLEU/ROUGE
    domain_results = evaluate_domain_accuracy(
        model, tokenizer, test_samples, max_samples=args.max_samples
    )
    results["quantitative"]["domain_accuracy"] = domain_results

    # Extract predictions for qualitative evaluation
    predictions = domain_results["sample_predictions"]
    instructions = [s["instruction"] for s in test_samples[:len(predictions)]]

    # ========== Qualitative Evaluation ==========
    print("\n" + "=" * 60)
    print("QUALITATIVE EVALUATION")
    print("=" * 60)

    # 3. Korean Fluency
    results["qualitative"]["korean_fluency"] = evaluate_korean_fluency(predictions)

    # 4. Response Relevance
    results["qualitative"]["response_relevance"] = evaluate_response_relevance(
        predictions, instructions
    )

    # ========== Overall Summary ==========
    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    # Check if all specs passed
    all_passed = (
        results["quantitative"]["perplexity"]["passes_spec"] and
        results["quantitative"]["domain_accuracy"]["passes_spec"] and
        results["qualitative"]["korean_fluency"]["passes_spec"] and
        results["qualitative"]["response_relevance"]["passes_spec"]
    )

    results["overall"]["all_specs_passed"] = all_passed
    results["overall"]["summary"] = {
        "perplexity": f"{results['quantitative']['perplexity']['perplexity']:.2f} (target: <{SPEC_TARGETS['perplexity']})",
        "domain_accuracy": f"{results['quantitative']['domain_accuracy']['domain_accuracy']:.2%} (target: >{SPEC_TARGETS['truck_domain_accuracy']:.0%})",
        "korean_fluency": f"{results['qualitative']['korean_fluency']['korean_fluency']:.2%} (target: >{SPEC_TARGETS['korean_fluency']:.0%})",
        "response_relevance": f"{results['qualitative']['response_relevance']['response_relevance']:.2%} (target: >{SPEC_TARGETS['response_relevance']:.0%})"
    }

    # Print summary
    for metric, value in results["overall"]["summary"].items():
        status = "✅ PASS" if (
            (metric == "perplexity" and results["quantitative"]["perplexity"]["passes_spec"]) or
            (metric == "domain_accuracy" and results["quantitative"]["domain_accuracy"]["passes_spec"]) or
            (metric == "korean_fluency" and results["qualitative"]["korean_fluency"]["passes_spec"]) or
            (metric == "response_relevance" and results["qualitative"]["response_relevance"]["passes_spec"])
        ) else "❌ FAIL"
        print(f"  {metric}: {value} {status}")

    print(f"\n{'='*60}")
    print(f"FINAL VERDICT: {'✅ ALL SPECS PASSED' if all_passed else '❌ SOME SPECS FAILED'}")
    print(f"{'='*60}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Results saved to {output_file}")

    # Generate markdown report
    generate_markdown_report(results, output_dir)

    return results


def generate_markdown_report(results: Dict, output_dir: Path):
    """Generate human-readable markdown report"""
    report_file = output_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# Qwen3-1.7B Truck Korean Model Evaluation Report\n\n")
        f.write(f"**Evaluation Date**: {results['evaluation_date']}\n")
        f.write(f"**Model Path**: {results['model_path']}\n")
        f.write(f"**Test Samples**: {results['total_samples']}\n\n")

        f.write(f"## Overall Result\n\n")
        f.write(f"**Status**: {'✅ PASSED' if results['overall']['all_specs_passed'] else '❌ FAILED'}\n\n")

        f.write(f"## Quantitative Metrics\n\n")
        f.write(f"| Metric | Value | Target | Status |\n")
        f.write(f"|--------|-------|--------|--------|\n")

        perp = results['quantitative']['perplexity']
        f.write(f"| Perplexity | {perp['perplexity']:.2f} | <{SPEC_TARGETS['perplexity']} | {'✅' if perp['passes_spec'] else '❌'} |\n")

        domain = results['quantitative']['domain_accuracy']
        f.write(f"| Domain Accuracy | {domain['domain_accuracy']:.2%} | >{SPEC_TARGETS['truck_domain_accuracy']:.0%} | {'✅' if domain['passes_spec'] else '❌'} |\n")

        if domain['bleu_rouge']['bleu']:
            f.write(f"| BLEU Score | {domain['bleu_rouge']['bleu']:.2f} | >40 | {'✅' if domain['bleu_rouge']['passes_spec_bleu'] else '❌'} |\n")
            f.write(f"| ROUGE-L | {domain['bleu_rouge']['rouge_l']:.4f} | >0.6 | {'✅' if domain['bleu_rouge']['passes_spec_rouge'] else '❌'} |\n")

        f.write(f"\n## Qualitative Metrics\n\n")
        f.write(f"| Metric | Value | Target | Status |\n")
        f.write(f"|--------|-------|--------|--------|\n")

        fluency = results['qualitative']['korean_fluency']
        f.write(f"| Korean Fluency | {fluency['korean_fluency']:.2%} | >{SPEC_TARGETS['korean_fluency']:.0%} | {'✅' if fluency['passes_spec'] else '❌'} |\n")

        relevance = results['qualitative']['response_relevance']
        f.write(f"| Response Relevance | {relevance['response_relevance']:.2%} | >{SPEC_TARGETS['response_relevance']:.0%} | {'✅' if relevance['passes_spec'] else '❌'} |\n")

        f.write(f"\n## Sample Predictions\n\n")
        for i in range(min(3, len(domain['sample_predictions']))):
            f.write(f"### Sample {i+1}\n\n")
            f.write(f"**Reference**: {domain['sample_references'][i]}\n\n")
            f.write(f"**Prediction**: {domain['sample_predictions'][i]}\n\n")

        f.write(f"## Recommendations\n\n")
        if not results['overall']['all_specs_passed']:
            f.write(f"### Failed Metrics\n\n")
            if not perp['passes_spec']:
                f.write(f"- **Perplexity too high**: Consider increasing training epochs or improving dataset quality\n")
            if not domain['passes_spec']:
                f.write(f"- **Domain accuracy low**: Add more truck-specific training samples or increase LoRA rank\n")
            if not fluency['passes_spec']:
                f.write(f"- **Korean fluency low**: Review training data quality, ensure proper Korean text\n")
            if not relevance['passes_spec']:
                f.write(f"- **Response relevance low**: Improve instruction quality in training data\n")
        else:
            f.write(f"All metrics passed specification targets. Model is ready for deployment.\n")

    print(f"[OK] Markdown report saved to {report_file}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-1.7B Truck Korean fine-tuned model")

    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to fine-tuned model (LoRA adapters)")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Path to base model (default: ../base-models/qwen3-1.7b)")
    parser.add_argument("--test-data", type=str, default="../datasets/test.jsonl",
                        help="Path to test dataset (JSONL)")
    parser.add_argument("--output-dir", type=str, default="../evaluation",
                        help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Maximum samples to evaluate (default: 200)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    # Verify paths
    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model path not found: {args.model_path}")
        sys.exit(1)

    if not os.path.exists(args.test_data):
        print(f"[ERROR] Test data not found: {args.test_data}")
        sys.exit(1)

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU (will be slow)")
        args.device = "cpu"

    # Run evaluation
    results = run_evaluation(args)

    # Exit with appropriate code
    sys.exit(0 if results["overall"]["all_specs_passed"] else 1)


if __name__ == "__main__":
    main()

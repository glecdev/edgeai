#!/usr/bin/env python3
"""
Logistics-Specific Model Evaluation Framework

Evaluates fine-tuned Qwen2.5-0.5B model against test cases and competitor AI models.
"""

import json
import os
import sys
import time
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a single test case"""
    test_id: str
    category: str
    difficulty: str

    # Response quality
    technical_accuracy_score: float  # 0-1
    korean_fluency_score: float      # 0-1
    completeness_score: float        # 0-1
    actionable_advice_score: float   # 0-1

    # Logistics-specific
    contextual_reasoning_score: float = 0.0  # 0-1
    safety_awareness_score: float = 0.0      # 0-1
    regulatory_knowledge_score: float = 0.0  # 0-1
    diagnostic_reasoning_score: float = 0.0  # 0-1

    # Performance
    response_time_ms: float = 0.0
    token_count: int = 0

    # Response
    model_response: str = ""

    def overall_score(self) -> float:
        """Calculate overall weighted score"""
        weights = {
            'technical_accuracy': 0.3,
            'korean_fluency': 0.15,
            'completeness': 0.2,
            'actionable_advice': 0.15,
            'contextual_reasoning': 0.1,
            'safety_awareness': 0.05,
            'regulatory_knowledge': 0.03,
            'diagnostic_reasoning': 0.02
        }

        score = (
            weights['technical_accuracy'] * self.technical_accuracy_score +
            weights['korean_fluency'] * self.korean_fluency_score +
            weights['completeness'] * self.completeness_score +
            weights['actionable_advice'] * self.actionable_advice_score +
            weights['contextual_reasoning'] * self.contextual_reasoning_score +
            weights['safety_awareness'] * self.safety_awareness_score +
            weights['regulatory_knowledge'] * self.regulatory_knowledge_score +
            weights['diagnostic_reasoning'] * self.diagnostic_reasoning_score
        )
        return score


class LogisticsModelEvaluator:
    """Evaluator for logistics-specific AI models"""

    def __init__(self, model_path: str, base_model_path: str, device: str = "cuda"):
        """
        Initialize evaluator

        Args:
            model_path: Path to fine-tuned LoRA model
            base_model_path: Path to base Qwen model
            device: Device to run on (cuda/cpu)
        """
        self.device = device
        print(f"Loading model from {model_path}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )

        # Load LoRA adapters
        self.model = PeftModel.from_pretrained(self.base_model, model_path)
        self.model.eval()

        print(f"Model loaded successfully on {device}")

    def generate_response(self, query: str, scenario: Dict[str, Any], max_tokens: int = 256) -> Tuple[str, float, int]:
        """
        Generate model response

        Args:
            query: User query
            scenario: Vehicle scenario data
            max_tokens: Maximum response tokens

        Returns:
            (response_text, response_time_ms, token_count)
        """
        # Build context from scenario
        context_parts = []
        if 'vehicle_speed' in scenario:
            context_parts.append(f"차량 속도: {scenario['vehicle_speed']} km/h")
        if 'engine_rpm' in scenario:
            context_parts.append(f"엔진 RPM: {scenario['engine_rpm']}")
        if 'coolant_temp' in scenario:
            context_parts.append(f"냉각수 온도: {scenario['coolant_temp']}°C")
        if 'fuel_level' in scenario:
            context_parts.append(f"연료: {scenario['fuel_level']}%")
        if 'load_weight' in scenario:
            context_parts.append(f"적재 중량: {scenario['load_weight']} kg")

        context = ", ".join(context_parts) if context_parts else ""

        # Format prompt
        if context:
            prompt = f"<|im_start|>user\n차량 상태: {context}\n질문: {query}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        end_time = time.time()

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0].strip()

        response_time_ms = (end_time - start_time) * 1000
        token_count = len(outputs[0]) - len(inputs['input_ids'][0])

        return response, response_time_ms, token_count

    def evaluate_test_case(self, test_case: Dict[str, Any]) -> EvaluationMetrics:
        """
        Evaluate a single test case

        Args:
            test_case: Test case dictionary

        Returns:
            EvaluationMetrics
        """
        test_id = test_case['id']
        category = test_case['category']
        difficulty = test_case['difficulty']
        query = test_case['query']
        scenario = test_case['scenario']
        expected_elements = test_case['expected_response_elements']
        criteria = test_case['evaluation_criteria']

        # Generate response
        response, response_time_ms, token_count = self.generate_response(query, scenario)

        # Evaluate response
        metrics = EvaluationMetrics(
            test_id=test_id,
            category=category,
            difficulty=difficulty,
            technical_accuracy_score=self._evaluate_technical_accuracy(response, expected_elements),
            korean_fluency_score=self._evaluate_korean_fluency(response),
            completeness_score=self._evaluate_completeness(response, expected_elements),
            actionable_advice_score=self._evaluate_actionable_advice(response) if criteria.get('actionable_advice') else 0.0,
            contextual_reasoning_score=self._evaluate_contextual_reasoning(response, scenario) if criteria.get('contextual_reasoning') else 0.0,
            safety_awareness_score=self._evaluate_safety_awareness(response) if criteria.get('safety_awareness') else 0.0,
            regulatory_knowledge_score=self._evaluate_regulatory_knowledge(response) if criteria.get('regulatory_knowledge') else 0.0,
            diagnostic_reasoning_score=self._evaluate_diagnostic_reasoning(response, scenario) if criteria.get('diagnostic_reasoning') else 0.0,
            response_time_ms=response_time_ms,
            token_count=token_count,
            model_response=response
        )

        return metrics

    def _evaluate_technical_accuracy(self, response: str, expected_elements: List[str]) -> float:
        """Evaluate technical accuracy (keyword matching)"""
        if not expected_elements:
            return 1.0

        # Extract key technical terms from expected elements
        keywords = []
        for element in expected_elements:
            # Extract numbers, units, technical terms
            import re
            numbers = re.findall(r'\d+\.?\d*', element)
            keywords.extend(numbers)

            # Technical terms (simplified Korean matching)
            tech_terms = ['정상', '이상', '위험', '권장', '필요', '높음', '낮음', '초과', '부족', '적정']
            for term in tech_terms:
                if term in element:
                    keywords.append(term)

        # Count matches
        matches = sum(1 for keyword in keywords if keyword in response)
        score = matches / len(keywords) if keywords else 0.5

        return min(1.0, score)

    def _evaluate_korean_fluency(self, response: str) -> float:
        """Evaluate Korean language fluency (heuristic)"""
        if not response:
            return 0.0

        # Check for Korean characters
        korean_chars = len([c for c in response if '\uac00' <= c <= '\ud7a3'])
        total_chars = len([c for c in response if c.strip()])

        if total_chars == 0:
            return 0.0

        korean_ratio = korean_chars / total_chars

        # Penalize very short responses
        length_penalty = min(1.0, len(response) / 50)

        # Check for sentence endings
        sentence_endings = response.count('.') + response.count('요') + response.count('다')
        structure_score = min(1.0, sentence_endings / 3)

        score = (korean_ratio * 0.5 + length_penalty * 0.3 + structure_score * 0.2)
        return min(1.0, score)

    def _evaluate_completeness(self, response: str, expected_elements: List[str]) -> float:
        """Evaluate response completeness"""
        if not expected_elements:
            return 1.0

        # Count how many expected elements are addressed
        addressed = 0
        for element in expected_elements:
            # Check if key concepts from element appear in response
            keywords = element.split()[:3]  # First 3 words as key concepts
            if any(keyword in response for keyword in keywords):
                addressed += 1

        score = addressed / len(expected_elements)
        return score

    def _evaluate_actionable_advice(self, response: str) -> float:
        """Evaluate presence of actionable advice"""
        advice_keywords = [
            '권장', '필요', '해야', '하세요', '바랍니다', '고려', '점검', '확인',
            '조치', '교체', '정차', '휴식', '감속', '유지', '피하', '사용'
        ]

        matches = sum(1 for keyword in advice_keywords if keyword in response)
        score = min(1.0, matches / 3)  # Expect at least 3 actionable terms

        return score

    def _evaluate_contextual_reasoning(self, response: str, scenario: Dict[str, Any]) -> float:
        """Evaluate contextual reasoning ability"""
        # Check if response mentions multiple scenario factors
        scenario_factors = 0
        total_factors = 0

        if 'load_weight' in scenario:
            total_factors += 1
            if '적재' in response or '중량' in response or str(scenario['load_weight']) in response:
                scenario_factors += 1

        if 'weather' in scenario:
            total_factors += 1
            weather_terms = {'rain': '비', 'snow': '눈', 'fog': '안개'}
            if weather_terms.get(scenario['weather'], '') in response:
                scenario_factors += 1

        if 'route_gradient' in scenario:
            total_factors += 1
            if '경사' in response or '오르막' in response or '내리막' in response:
                scenario_factors += 1

        if total_factors == 0:
            return 0.5

        score = scenario_factors / total_factors
        return score

    def _evaluate_safety_awareness(self, response: str) -> float:
        """Evaluate safety awareness"""
        safety_keywords = [
            '위험', '안전', '주의', '경고', '사고', '정차', '휴식', '제동',
            '거리', '속도', '피로', '졸음', '집중'
        ]

        matches = sum(1 for keyword in safety_keywords if keyword in response)
        score = min(1.0, matches / 2)  # Expect at least 2 safety-related terms

        return score

    def _evaluate_regulatory_knowledge(self, response: str) -> float:
        """Evaluate regulatory knowledge"""
        regulatory_keywords = [
            '법규', '위반', '규정', '제한', '시간', '허용', '금지', '의무',
            '최대', '최소', '기준'
        ]

        matches = sum(1 for keyword in regulatory_keywords if keyword in response)
        score = min(1.0, matches / 2)

        return score

    def _evaluate_diagnostic_reasoning(self, response: str, scenario: Dict[str, Any]) -> float:
        """Evaluate diagnostic reasoning"""
        # Check if response identifies root causes
        diagnostic_keywords = [
            '원인', '때문', '이유', '상태', '증상', '가능성', '징후', '결과',
            '문제', '고장', '불량'
        ]

        matches = sum(1 for keyword in diagnostic_keywords if keyword in response)
        score = min(1.0, matches / 2)

        return score


def load_test_cases(test_cases_path: str) -> List[Dict[str, Any]]:
    """Load test cases from JSON file"""
    with open(test_cases_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['test_cases']


def save_results(results: List[EvaluationMetrics], output_path: str):
    """Save evaluation results to JSON"""
    results_dict = [asdict(r) for r in results]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {output_path}")


def generate_summary_report(results: List[EvaluationMetrics], output_path: str):
    """Generate summary report"""
    # Overall statistics
    total_cases = len(results)
    avg_overall_score = sum(r.overall_score() for r in results) / total_cases
    avg_technical_accuracy = sum(r.technical_accuracy_score for r in results) / total_cases
    avg_korean_fluency = sum(r.korean_fluency_score for r in results) / total_cases
    avg_completeness = sum(r.completeness_score for r in results) / total_cases
    avg_response_time = sum(r.response_time_ms for r in results) / total_cases

    # By category
    categories = set(r.category for r in results)
    category_scores = {}
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        category_scores[cat] = sum(r.overall_score() for r in cat_results) / len(cat_results)

    # By difficulty
    difficulties = set(r.difficulty for r in results)
    difficulty_scores = {}
    for diff in difficulties:
        diff_results = [r for r in results if r.difficulty == diff]
        difficulty_scores[diff] = sum(r.overall_score() for r in diff_results) / len(diff_results)

    # Generate report
    report = f"""# Logistics AI Model Evaluation Report

## Overall Performance

- **Total Test Cases**: {total_cases}
- **Average Overall Score**: {avg_overall_score:.2%}
- **Average Technical Accuracy**: {avg_technical_accuracy:.2%}
- **Average Korean Fluency**: {avg_korean_fluency:.2%}
- **Average Completeness**: {avg_completeness:.2%}
- **Average Response Time**: {avg_response_time:.1f} ms

## Performance by Category

"""

    for cat in sorted(categories):
        report += f"- **{cat}**: {category_scores[cat]:.2%}\n"

    report += "\n## Performance by Difficulty\n\n"

    for diff in sorted(difficulties):
        report += f"- **{diff}**: {difficulty_scores[diff]:.2%}\n"

    report += "\n## Top 5 Best Performing Test Cases\n\n"

    top_5 = sorted(results, key=lambda r: r.overall_score(), reverse=True)[:5]
    for i, r in enumerate(top_5, 1):
        report += f"{i}. **{r.test_id}** ({r.category}, {r.difficulty}): {r.overall_score():.2%}\n"

    report += "\n## Bottom 5 Test Cases (Need Improvement)\n\n"

    bottom_5 = sorted(results, key=lambda r: r.overall_score())[:5]
    for i, r in enumerate(bottom_5, 1):
        report += f"{i}. **{r.test_id}** ({r.category}, {r.difficulty}): {r.overall_score():.2%}\n"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Summary report saved to {output_path}")


def main():
    """Main evaluation function"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate logistics AI model")
    parser.add_argument("--model-path", required=True, help="Path to fine-tuned LoRA model")
    parser.add_argument("--base-model", required=True, help="Path to base Qwen model")
    parser.add_argument("--test-cases", required=True, help="Path to test cases JSON")
    parser.add_argument("--output-dir", default="evaluation", help="Output directory")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test cases")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test cases
    print(f"Loading test cases from {args.test_cases}...")
    test_cases = load_test_cases(args.test_cases)
    if args.limit:
        test_cases = test_cases[:args.limit]
    print(f"Loaded {len(test_cases)} test cases")

    # Initialize evaluator
    evaluator = LogisticsModelEvaluator(
        model_path=args.model_path,
        base_model_path=args.base_model,
        device=args.device
    )

    # Run evaluation
    print("\nStarting evaluation...")
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] Evaluating {test_case['id']} ({test_case['category']})...")
        metrics = evaluator.evaluate_test_case(test_case)
        results.append(metrics)
        print(f"  Overall Score: {metrics.overall_score():.2%} | Response Time: {metrics.response_time_ms:.1f} ms")

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.output_dir, f"evaluation_results_{timestamp}.json")
    save_results(results, results_path)

    # Generate summary report
    report_path = os.path.join(args.output_dir, f"evaluation_report_{timestamp}.md")
    generate_summary_report(results, report_path)

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Total test cases: {len(results)}")
    print(f"Average overall score: {sum(r.overall_score() for r in results) / len(results):.2%}")
    print(f"Results: {results_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

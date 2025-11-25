#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset Preparation Script for Qwen3 Truck Korean Fine-tuning

Generates training dataset from:
1. Production CAN data → Korean conversations (10,000 samples)
2. Manual high-quality samples (2,000 samples) - To be created later

Output: JSONL format for Qwen3 instruction fine-tuning
"""

import pandas as pd
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
import random

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from can_templates import CANConversationTemplates
except ImportError:
    print("[ERROR] Failed to import can_templates.py")
    print("Make sure can_templates.py exists in the same directory")
    sys.exit(1)


class DatasetPreparer:
    """Prepares Qwen3 fine-tuning dataset from CAN data"""

    def __init__(
        self,
        can_data_path: str,
        output_path: str,
        target_samples: int = 10000,
        seed: int = 42
    ):
        self.can_data_path = can_data_path
        self.output_path = output_path
        self.target_samples = target_samples
        self.seed = seed

        self.templates = CANConversationTemplates()
        random.seed(seed)

    def load_can_data(self) -> pd.DataFrame:
        """Load production CAN dataset"""
        print(f"[1/6] Loading CAN data from {self.can_data_path}")

        if not os.path.exists(self.can_data_path):
            raise FileNotFoundError(f"CAN data not found: {self.can_data_path}")

        df = pd.read_csv(self.can_data_path)
        print(f"  Total CAN samples: {len(df):,}")
        print(f"  Columns: {list(df.columns)}")

        return df

    def classify_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify each CAN sample by pattern"""
        print("[2/6] Classifying CAN data by patterns...")

        patterns = []
        for idx, row in df.iterrows():
            can_data = row.to_dict()
            pattern = self.templates.get_pattern_for_data(can_data)
            patterns.append(pattern)

            if (idx + 1) % 10000 == 0:
                print(f"  Processed {idx + 1:,} samples...")

        df['pattern'] = patterns

        # Count distribution
        pattern_counts = Counter(patterns)
        print("\n  Pattern Distribution:")
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
            print(f"    {pattern}: {count:,} ({count/len(df)*100:.1f}%)")

        return df

    def sample_balanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sample data with balanced pattern distribution"""
        print(f"\n[3/6] Sampling {self.target_samples:,} samples with balanced distribution...")

        patterns = df['pattern'].unique()
        samples_per_pattern = self.target_samples // len(patterns)

        sampled_dfs = []
        for pattern in patterns:
            pattern_df = df[df['pattern'] == pattern]

            if len(pattern_df) >= samples_per_pattern:
                sampled = pattern_df.sample(n=samples_per_pattern, random_state=self.seed)
            else:
                # Oversample if not enough samples
                sampled = pattern_df.sample(n=samples_per_pattern, replace=True, random_state=self.seed)
                print(f"  [WARNING] Pattern '{pattern}' oversampled ({len(pattern_df)} → {samples_per_pattern})")

            sampled_dfs.append(sampled)

        result = pd.concat(sampled_dfs, ignore_index=True)

        # Shuffle
        result = result.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        print(f"  Final sample count: {len(result):,}")

        # Print balanced distribution
        final_counts = Counter(result['pattern'])
        print("\n  Balanced Distribution:")
        for pattern, count in sorted(final_counts.items()):
            print(f"    {pattern}: {count:,} ({count/len(result)*100:.1f}%)")

        return result

    def generate_conversations(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Generate Korean conversations from CAN data"""
        print("\n[4/6] Generating Korean conversations...")

        conversations = []

        for idx, row in df.iterrows():
            can_data = row.to_dict()
            pattern = row['pattern']

            # Generate conversation using template
            conv = self.templates.generate(pattern, can_data)

            if conv:
                conversations.append(conv)
            else:
                print(f"  [WARNING] Failed to generate conversation for pattern '{pattern}' at index {idx}")

            if (idx + 1) % 1000 == 0:
                print(f"  Generated {idx + 1:,} conversations...")

        print(f"  Total conversations generated: {len(conversations):,}")

        return conversations

    def validate_quality(self, conversations: List[Dict[str, str]]) -> Dict[str, Any]:
        """Validate dataset quality against spec criteria"""
        print("\n[5/6] Validating dataset quality...")

        # Check sample count
        total_samples = len(conversations)
        print(f"  Total samples: {total_samples:,}")

        # Check character lengths
        instruction_lengths = [len(c['instruction']) for c in conversations]
        response_lengths = [len(c['response']) for c in conversations]

        avg_inst_len = sum(instruction_lengths) / len(instruction_lengths)
        avg_resp_len = sum(response_lengths) / len(response_lengths)

        min_inst_len = min(instruction_lengths)
        max_inst_len = max(instruction_lengths)
        min_resp_len = min(response_lengths)
        max_resp_len = max(response_lengths)

        print(f"\n  Instruction lengths:")
        print(f"    Average: {avg_inst_len:.1f} chars")
        print(f"    Min: {min_inst_len}, Max: {max_inst_len}")

        print(f"\n  Response lengths:")
        print(f"    Average: {avg_resp_len:.1f} chars")
        print(f"    Min: {min_resp_len}, Max: {max_resp_len}")

        # Check Korean ratio
        korean_chars = 0
        total_chars = 0

        for conv in conversations:
            text = conv['instruction'] + conv['response']
            total_chars += len(text)
            korean_chars += sum(1 for c in text if '\uac00' <= c <= '\ud7a3')

        korean_ratio = korean_chars / total_chars * 100 if total_chars > 0 else 0
        print(f"\n  Korean character ratio: {korean_ratio:.1f}%")

        # Check duplicates
        unique_instructions = len(set(c['instruction'] for c in conversations))
        unique_responses = len(set(c['response'] for c in conversations))

        duplicate_ratio = (1 - unique_instructions / total_samples) * 100
        print(f"\n  Unique instructions: {unique_instructions:,} ({100 - duplicate_ratio:.1f}%)")
        print(f"  Unique responses: {unique_responses:,}")

        # Quality criteria from spec
        print("\n  Quality Criteria (from spec):")

        criteria_met = []

        # min_chars_per_sample: 50
        min_total_chars = min(inst + resp for inst, resp in zip(instruction_lengths, response_lengths))
        criteria_met.append(("Min chars per sample", min_total_chars >= 50, f"{min_total_chars} >= 50"))

        # max_chars_per_sample: 500
        max_total_chars = max(inst + resp for inst, resp in zip(instruction_lengths, response_lengths))
        criteria_met.append(("Max chars per sample", max_total_chars <= 500, f"{max_total_chars} <= 500"))

        # korean_ratio: >95%
        criteria_met.append(("Korean ratio", korean_ratio > 95, f"{korean_ratio:.1f}% > 95%"))

        # duplicate_threshold: <0.1%
        criteria_met.append(("Duplicate threshold", duplicate_ratio < 0.1, f"{duplicate_ratio:.2f}% < 0.1%"))

        for criterion, met, detail in criteria_met:
            status = "PASS" if met else "FAIL"
            print(f"    [{status}] {criterion}: {detail}")

        all_met = all(met for _, met, _ in criteria_met)

        return {
            'total_samples': total_samples,
            'avg_instruction_length': avg_inst_len,
            'avg_response_length': avg_resp_len,
            'korean_ratio': korean_ratio,
            'duplicate_ratio': duplicate_ratio,
            'all_criteria_met': all_met
        }

    def save_jsonl(self, conversations: List[Dict[str, str]]):
        """Save conversations to JSONL format"""
        print(f"\n[6/6] Saving to {self.output_path}")

        # Create output directory if needed
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        with open(self.output_path, 'w', encoding='utf-8') as f:
            for conv in conversations:
                json_line = json.dumps(conv, ensure_ascii=False)
                f.write(json_line + '\n')

        # Verify file
        file_size_mb = os.path.getsize(self.output_path) / (1024 * 1024)
        print(f"  File saved: {file_size_mb:.2f} MB")
        print(f"  Lines written: {len(conversations):,}")

    def run(self):
        """Execute full dataset preparation pipeline"""
        print("=" * 60)
        print("DATASET PREPARATION - CAN to Korean Conversations")
        print("=" * 60)
        print()

        # Load data
        df = self.load_can_data()

        # Classify patterns
        df = self.classify_patterns(df)

        # Sample balanced
        df_sampled = self.sample_balanced(df)

        # Generate conversations
        conversations = self.generate_conversations(df_sampled)

        # Validate quality
        quality_report = self.validate_quality(conversations)

        # Save to JSONL
        self.save_jsonl(conversations)

        print("\n" + "=" * 60)
        if quality_report['all_criteria_met']:
            print("[SUCCESS] Dataset preparation complete - All quality criteria met")
        else:
            print("[WARNING] Dataset preparation complete - Some quality criteria not met")
        print("=" * 60)

        return quality_report


def main():
    """Main entry point"""

    # Paths from spec
    CAN_DATA_PATH = "d:/edgeai/edgeai-repo/datasets/production_real_structure/train.csv"
    OUTPUT_PATH = "d:/edgeai/edgeai-repo/ai-models/fine-tuning/datasets/can_conversations.jsonl"

    TARGET_SAMPLES = 10000

    preparer = DatasetPreparer(
        can_data_path=CAN_DATA_PATH,
        output_path=OUTPUT_PATH,
        target_samples=TARGET_SAMPLES,
        seed=42
    )

    try:
        quality_report = preparer.run()

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total samples: {quality_report['total_samples']:,}")
        print(f"Avg instruction length: {quality_report['avg_instruction_length']:.1f} chars")
        print(f"Avg response length: {quality_report['avg_response_length']:.1f} chars")
        print(f"Korean ratio: {quality_report['korean_ratio']:.1f}%")
        print(f"Duplicate ratio: {quality_report['duplicate_ratio']:.2f}%")
        print(f"All criteria met: {'YES' if quality_report['all_criteria_met'] else 'NO'}")
        print("=" * 60)

        sys.exit(0 if quality_report['all_criteria_met'] else 1)

    except Exception as e:
        print(f"\n[ERROR] Dataset preparation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

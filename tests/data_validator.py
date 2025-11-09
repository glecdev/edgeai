#!/usr/bin/env python3
"""
GLEC DTG - Data Validation Utility

Validates collected CAN data and training datasets
Checks for:
- Missing values
- Out-of-range values
- Data quality issues
- Label distribution
- Temporal consistency

Usage:
    python data_validator.py --input datasets/train.csv --report
"""

import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import sys


class DataValidator:
    """Data validation utility"""

    # Expected value ranges for each feature
    VALID_RANGES = {
        'vehicle_speed': (0, 255),           # km/h
        'engine_rpm': (0, 16383),            # RPM
        'throttle_position': (0, 100),       # %
        'brake_pressure': (0, 100),          # %
        'fuel_level': (0, 100),              # %
        'coolant_temp': (-40, 215),          # °C
        'engine_load': (0, 100),             # %
        'intake_air_temp': (-40, 215),       # °C
        'maf_rate': (0, 655.35),             # g/s
        'battery_voltage': (10, 16),         # V
        'acceleration_x': (-20, 20),         # m/s²
        'acceleration_y': (-20, 20),         # m/s²
        'steering_angle': (-720, 720),       # degrees
        'fuel_consumption': (0, 50),         # L/100km
        'latitude': (-90, 90),               # degrees
        'longitude': (-180, 180),            # degrees
    }

    # Expected labels
    VALID_LABELS = [
        'normal',
        'eco_driving',
        'harsh_braking',
        'harsh_acceleration',
        'speeding',
        'aggressive',
        'anomaly'
    ]

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df: pd.DataFrame = None
        self.validation_results: Dict = {
            'missing_values': {},
            'out_of_range': {},
            'quality_issues': [],
            'label_distribution': {},
            'temporal_issues': []
        }

    def load_data(self):
        """Load dataset"""
        print(f"Loading dataset: {self.data_path}")
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✓ Loaded {len(self.df)} samples")
        except Exception as e:
            print(f"✗ Failed to load data: {e}")
            sys.exit(1)

    def validate(self) -> bool:
        """Run all validations"""
        print("\nRunning validations...\n")

        all_pass = True

        # Check missing values
        if not self._check_missing_values():
            all_pass = False

        # Check value ranges
        if not self._check_value_ranges():
            all_pass = False

        # Check label distribution
        if not self._check_labels():
            all_pass = False

        # Check data quality
        if not self._check_data_quality():
            all_pass = False

        # Check temporal consistency
        if not self._check_temporal_consistency():
            all_pass = False

        return all_pass

    def _check_missing_values(self) -> bool:
        """Check for missing values"""
        print("Checking for missing values...")

        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100

        has_missing = False
        for col in self.df.columns:
            if missing[col] > 0:
                has_missing = True
                self.validation_results['missing_values'][col] = {
                    'count': int(missing[col]),
                    'percentage': float(missing_pct[col])
                }
                print(f"  ✗ {col}: {missing[col]} missing ({missing_pct[col]:.2f}%)")

        if not has_missing:
            print("  ✓ No missing values")
            return True
        else:
            print(f"  ✗ FAIL: {len(self.validation_results['missing_values'])} columns have missing values")
            return False

    def _check_value_ranges(self) -> bool:
        """Check if values are within expected ranges"""
        print("\nChecking value ranges...")

        all_valid = True

        for col, (min_val, max_val) in self.VALID_RANGES.items():
            if col not in self.df.columns:
                continue

            out_of_range = ((self.df[col] < min_val) | (self.df[col] > max_val)).sum()

            if out_of_range > 0:
                all_valid = False
                out_of_range_pct = (out_of_range / len(self.df)) * 100

                self.validation_results['out_of_range'][col] = {
                    'count': int(out_of_range),
                    'percentage': float(out_of_range_pct),
                    'expected_range': (min_val, max_val),
                    'actual_range': (float(self.df[col].min()), float(self.df[col].max()))
                }

                print(f"  ✗ {col}: {out_of_range} out of range ({out_of_range_pct:.2f}%)")
                print(f"      Expected: [{min_val}, {max_val}], Got: [{self.df[col].min():.2f}, {self.df[col].max():.2f}]")

        if all_valid:
            print("  ✓ All values within expected ranges")
            return True
        else:
            print(f"  ✗ FAIL: {len(self.validation_results['out_of_range'])} columns have out-of-range values")
            return False

    def _check_labels(self) -> bool:
        """Check label distribution"""
        print("\nChecking labels...")

        if 'label' not in self.df.columns:
            print("  ⚠ WARNING: No 'label' column found")
            return True

        label_counts = self.df['label'].value_counts()
        label_percentages = (label_counts / len(self.df)) * 100

        for label, count in label_counts.items():
            self.validation_results['label_distribution'][label] = {
                'count': int(count),
                'percentage': float(label_percentages[label])
            }

        print(f"  Label distribution ({len(label_counts)} unique labels):")
        for label, count in label_counts.items():
            print(f"    {label:20s}: {count:6d} ({label_percentages[label]:5.2f}%)")

        # Check for class imbalance
        max_count = label_counts.max()
        min_count = label_counts.min()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        if imbalance_ratio > 10:
            print(f"  ⚠ WARNING: Class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
            self.validation_results['quality_issues'].append(
                f"Class imbalance: {imbalance_ratio:.1f}:1 ratio"
            )
            return False
        else:
            print(f"  ✓ Class balance acceptable (ratio: {imbalance_ratio:.1f}:1)")
            return True

    def _check_data_quality(self) -> bool:
        """Check for data quality issues"""
        print("\nChecking data quality...")

        issues = []

        # Check for constant values (sensor stuck)
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if col == 'label':
                continue

            unique_values = self.df[col].nunique()
            if unique_values == 1:
                issues.append(f"{col} has constant value (sensor may be stuck)")

        # Check for sudden jumps (unrealistic changes)
        if 'vehicle_speed' in self.df.columns:
            speed_diff = self.df['vehicle_speed'].diff().abs()
            unrealistic_jumps = (speed_diff > 50).sum()  # >50 km/h change in 1 second

            if unrealistic_jumps > 0:
                issues.append(f"Vehicle speed has {unrealistic_jumps} unrealistic jumps")

        # Check for duplicate rows
        duplicate_rows = self.df.duplicated().sum()
        if duplicate_rows > 0:
            duplicate_pct = (duplicate_rows / len(self.df)) * 100
            issues.append(f"{duplicate_rows} duplicate rows ({duplicate_pct:.2f}%)")

        # Report issues
        if issues:
            for issue in issues:
                print(f"  ✗ {issue}")
                self.validation_results['quality_issues'].append(issue)
            return False
        else:
            print("  ✓ No data quality issues detected")
            return True

    def _check_temporal_consistency(self) -> bool:
        """Check temporal consistency (if timestamp exists)"""
        print("\nChecking temporal consistency...")

        if 'timestamp' not in self.df.columns:
            print("  ⚠ No timestamp column found, skipping temporal check")
            return True

        # Check for sorted timestamps
        is_sorted = self.df['timestamp'].is_monotonic_increasing
        if not is_sorted:
            print("  ✗ Timestamps are not sorted")
            self.validation_results['temporal_issues'].append("Timestamps not sorted")
            return False

        # Check for duplicate timestamps
        duplicate_timestamps = self.df['timestamp'].duplicated().sum()
        if duplicate_timestamps > 0:
            print(f"  ✗ {duplicate_timestamps} duplicate timestamps")
            self.validation_results['temporal_issues'].append(
                f"{duplicate_timestamps} duplicate timestamps"
            )
            return False

        # Check sampling rate (should be ~1Hz)
        time_diffs = self.df['timestamp'].diff().dropna()
        mean_interval = time_diffs.mean()
        std_interval = time_diffs.std()

        print(f"  Sampling interval: {mean_interval:.2f} ± {std_interval:.2f} ms")

        if abs(mean_interval - 1000) > 100:  # Should be ~1000ms
            print(f"  ⚠ WARNING: Sampling rate inconsistent (expected ~1000ms)")
            self.validation_results['temporal_issues'].append(
                f"Inconsistent sampling rate: {mean_interval:.2f}ms"
            )
            return False

        print("  ✓ Temporal consistency OK")
        return True

    def print_report(self):
        """Print validation report"""
        print("\n" + "="*60)
        print("VALIDATION REPORT")
        print("="*60)

        # Overall statistics
        print(f"\nDataset: {self.data_path}")
        print(f"Samples: {len(self.df):,}")
        print(f"Features: {len(self.df.columns)}")

        # Missing values summary
        if self.validation_results['missing_values']:
            print("\n✗ Missing Values:")
            for col, stats in self.validation_results['missing_values'].items():
                print(f"  {col}: {stats['count']} ({stats['percentage']:.2f}%)")
        else:
            print("\n✓ No missing values")

        # Out of range summary
        if self.validation_results['out_of_range']:
            print("\n✗ Out of Range Values:")
            for col, stats in self.validation_results['out_of_range'].items():
                print(f"  {col}: {stats['count']} ({stats['percentage']:.2f}%)")
        else:
            print("\n✓ All values within expected ranges")

        # Quality issues
        if self.validation_results['quality_issues']:
            print("\n✗ Data Quality Issues:")
            for issue in self.validation_results['quality_issues']:
                print(f"  - {issue}")
        else:
            print("\n✓ No data quality issues")

        # Temporal issues
        if self.validation_results['temporal_issues']:
            print("\n✗ Temporal Issues:")
            for issue in self.validation_results['temporal_issues']:
                print(f"  - {issue}")
        else:
            print("\n✓ Temporal consistency OK")

        print("\n" + "="*60)

    def save_report(self, output_path: str):
        """Save validation report to file"""
        import json

        report = {
            'dataset': self.data_path,
            'samples': len(self.df),
            'features': len(self.df.columns),
            'validation_results': self.validation_results
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='GLEC DTG Data Validation Utility')
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file to validate')
    parser.add_argument('--report', action='store_true',
                        help='Print detailed report')
    parser.add_argument('--save-report', type=str,
                        help='Save report to JSON file')

    args = parser.parse_args()

    # Create validator
    validator = DataValidator(args.input)

    # Load and validate data
    validator.load_data()
    all_pass = validator.validate()

    # Print report if requested
    if args.report:
        validator.print_report()

    # Save report if requested
    if args.save_report:
        validator.save_report(args.save_report)

    # Exit with appropriate code
    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()

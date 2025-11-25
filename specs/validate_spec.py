#!/usr/bin/env python3
"""
Spec Validation Script
Phase 2: Specification Validation

Validates the hybrid-ai-coordinator_spec.yaml against actual implementation.
"""

import os
import sys
import io
import yaml
from pathlib import Path
from datetime import datetime

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Configuration
SPEC_FILE = "hybrid-ai-coordinator_spec.yaml"
ANDROID_DTG_ROOT = Path(__file__).parent.parent / "android-dtg"
TEST_ROOT = ANDROID_DTG_ROOT / "app" / "src" / "test" / "java" / "com" / "glec" / "dtg"
SOURCE_ROOT = ANDROID_DTG_ROOT / "app" / "src" / "main" / "java" / "com" / "glec" / "dtg"


def load_spec():
    """Load the YAML spec file."""
    spec_path = Path(__file__).parent / SPEC_FILE
    if not spec_path.exists():
        print(f"❌ Spec file not found: {spec_path}")
        return None

    with open(spec_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def validate_source_files(spec):
    """Validate that all specified source files exist."""
    print("\n📂 Validating Source Files...")
    results = []

    source_files = spec.get('implementation', {}).get('files', {}).get('source', [])
    for file_info in source_files:
        file_path = SOURCE_ROOT / file_info['path']
        exists = file_path.exists()

        if exists:
            # Count lines
            with open(file_path, 'r', encoding='utf-8') as f:
                actual_lines = len(f.readlines())
            expected_lines = file_info.get('lines', 0)
            line_check = actual_lines >= expected_lines * 0.8  # Allow 20% variance

            status = "✅" if line_check else "⚠️"
            print(f"  {status} {file_info['path']}: {actual_lines} lines (expected ~{expected_lines})")
            results.append((file_info['path'], exists and line_check))
        else:
            print(f"  ❌ {file_info['path']}: NOT FOUND")
            results.append((file_info['path'], False))

    return all(r[1] for r in results)


def validate_test_files(spec):
    """Validate that all test files exist and count tests."""
    print("\n🧪 Validating Test Files...")
    results = []
    total_tests = 0

    # Unit tests
    unit_tests = spec.get('test_coverage', {}).get('unit_tests', {}).get('breakdown', [])
    for test_info in unit_tests:
        file_path = TEST_ROOT / test_info['file']
        exists = file_path.exists()

        if exists:
            # Count @Test annotations
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                test_count = content.count('@Test')

            expected_tests = test_info.get('tests', 0)
            status = "✅" if test_count >= expected_tests * 0.8 else "⚠️"
            print(f"  {status} {test_info['suite']}: {test_count} tests (expected ~{expected_tests})")
            total_tests += test_count
            results.append((test_info['suite'], exists and test_count > 0))
        else:
            print(f"  ❌ {test_info['suite']}: NOT FOUND")
            results.append((test_info['suite'], False))

    # Integration tests
    integration_tests = spec.get('test_coverage', {}).get('integration_tests', {}).get('breakdown', [])
    for test_info in integration_tests:
        file_path = TEST_ROOT / test_info['file']
        exists = file_path.exists()

        if exists:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                test_count = content.count('@Test')

            expected_tests = test_info.get('tests', 0)
            status = "✅" if test_count >= expected_tests * 0.8 else "⚠️"
            print(f"  {status} {test_info['suite']}: {test_count} tests (expected ~{expected_tests})")
            total_tests += test_count
            results.append((test_info['suite'], exists and test_count > 0))
        else:
            print(f"  ❌ {test_info['suite']}: NOT FOUND")
            results.append((test_info['suite'], False))

    print(f"\n  📊 Total Tests Found: {total_tests}")
    return all(r[1] for r in results), total_tests


def validate_performance_targets(spec):
    """Validate performance targets are defined."""
    print("\n⚡ Validating Performance Targets...")
    targets = spec.get('performance_targets', {})

    required_targets = [
        ('latency.ml_inference.target_ms', 50),
        ('latency.llm_inference.target_ms', 3000),
        ('latency.combined_analysis.target_ms', 3500),
        ('memory.peak_mb', 700),
        ('throughput.auto_analysis_interval_ms', 60000),
        ('reliability.error_rate_target', '<5%'),
    ]

    results = []
    for target_path, expected in required_targets:
        parts = target_path.split('.')
        value = targets
        for part in parts:
            value = value.get(part, {}) if isinstance(value, dict) else None

        if value is not None:
            print(f"  ✅ {target_path}: {value}")
            results.append(True)
        else:
            print(f"  ❌ {target_path}: NOT DEFINED")
            results.append(False)

    return all(results)


def validate_quality_gates(spec):
    """Validate quality gates are defined."""
    print("\n🚦 Validating Quality Gates...")
    quality_gates = spec.get('quality_gates', {})

    categories = ['build', 'runtime', 'functional']
    results = []

    for category in categories:
        gates = quality_gates.get(category, [])
        if gates:
            print(f"  ✅ {category}: {len(gates)} gates defined")
            for gate in gates:
                name = gate.get('name', 'Unknown')
                target = gate.get('target', gate.get('check', 'N/A'))
                print(f"      - {name}: {target}")
            results.append(True)
        else:
            print(f"  ❌ {category}: No gates defined")
            results.append(False)

    return all(results)


def validate_architecture(spec):
    """Validate architecture components are defined."""
    print("\n🏗️ Validating Architecture...")
    arch = spec.get('architecture', {})

    components = [
        ('ml_subsystem', 'EdgeAIInferenceService'),
        ('llm_subsystem', 'SmartOrchestrator'),
        ('coordinator', 'HybridAICoordinator'),
    ]

    results = []
    for component_key, expected_name in components:
        component = arch.get('components', {}).get(component_key, {})
        name = component.get('name', '')

        if name == expected_name:
            print(f"  ✅ {component_key}: {name}")

            # Check sub-components
            if 'models' in component:
                print(f"      Models: {len(component['models'])}")
            if 'engines' in component:
                print(f"      Engines: {len(component['engines'])}")
            if 'analysis_types' in component:
                print(f"      Analysis Types: {component['analysis_types']}")

            results.append(True)
        else:
            print(f"  ❌ {component_key}: Expected {expected_name}, got {name}")
            results.append(False)

    return all(results)


def generate_validation_report(results):
    """Generate a validation report."""
    print("\n" + "=" * 60)
    print("📋 SPEC VALIDATION REPORT")
    print("=" * 60)

    all_pass = all(results.values())

    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {check}")

    print("=" * 60)
    if all_pass:
        print("🎉 OVERALL STATUS: PASS")
        print("   All specification requirements validated successfully.")
    else:
        print("⚠️ OVERALL STATUS: FAIL")
        print("   Some specification requirements are not met.")
    print("=" * 60)

    return all_pass


def main():
    print("=" * 60)
    print("🔍 SPEC VALIDATION: hybrid-ai-coordinator_spec.yaml")
    print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load spec
    spec = load_spec()
    if spec is None:
        return 1

    print(f"\n📄 Loaded spec: {spec.get('project', 'Unknown')} v{spec.get('version', '?')}")

    # Run validations
    results = {}

    results['Architecture'] = validate_architecture(spec)
    results['Source Files'] = validate_source_files(spec)
    test_pass, test_count = validate_test_files(spec)
    results['Test Files'] = test_pass
    results['Performance Targets'] = validate_performance_targets(spec)
    results['Quality Gates'] = validate_quality_gates(spec)

    # Generate report
    overall_pass = generate_validation_report(results)

    return 0 if overall_pass else 1


if __name__ == '__main__':
    sys.exit(main())

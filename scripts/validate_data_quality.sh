#!/bin/bash
# GLEC DTG - Data Quality Validation
# Automated validation of training datasets and CAN data quality

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================="
echo "GLEC DTG - Data Quality Validator"
echo "========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Tracking
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

# Function to check file exists
check_file() {
    local file=$1
    local name=$2

    echo -n "  Checking $name... "
    if [ -f "$file" ]; then
        local size=$(du -h "$file" | cut -f1)
        echo -e "${GREEN}✅ Found${NC} ($size)"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        return 0
    else
        echo -e "${YELLOW}⚠️  Not found${NC} (will skip validation)"
        WARNING_CHECKS=$((WARNING_CHECKS + 1))
        return 1
    fi
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
}

# Function to validate dataset
validate_dataset() {
    local dataset_path=$1
    local dataset_name=$2

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Validating: $dataset_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ ! -f "$dataset_path" ]; then
        echo -e "${YELLOW}⚠️  Dataset not found: $dataset_path${NC}"
        echo "    Skipping validation"
        WARNING_CHECKS=$((WARNING_CHECKS + 1))
        TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
        return
    fi

    # Run Python validation script
    python3 -c "
import pandas as pd
import numpy as np
import sys

def validate_dataset(path):
    try:
        # Load dataset
        df = pd.read_csv(path)
        print(f'  ✓ Loaded: {len(df)} rows, {len(df.columns)} columns')

        issues = []

        # Check required columns (CAN data columns)
        required_cols = [
            'vehicle_speed', 'engine_rpm', 'throttle_position',
            'brake_position', 'fuel_level', 'coolant_temp'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f'Missing columns: {missing_cols}')
        else:
            print('  ✓ Required columns present')

        # Check for missing values
        null_counts = df.isnull().sum()
        if null_counts.any():
            null_cols = null_counts[null_counts > 0]
            pct = (null_cols / len(df) * 100).round(2)
            issues.append(f'Missing values: {dict(pct)}')
            print(f'  ⚠️  Missing values detected: {len(null_cols)} columns')
        else:
            print('  ✓ No missing values')

        # Check for duplicate rows
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            pct = (dup_count / len(df) * 100).round(2)
            issues.append(f'Duplicate rows: {dup_count} ({pct}%)')
            print(f'  ⚠️  Duplicate rows: {dup_count} ({pct}%)')
        else:
            print('  ✓ No duplicate rows')

        # Check value ranges (if columns exist)
        if 'vehicle_speed' in df.columns:
            speed_invalid = ((df['vehicle_speed'] < 0) | (df['vehicle_speed'] > 255)).sum()
            if speed_invalid > 0:
                issues.append(f'Invalid speed values: {speed_invalid}')
                print(f'  ⚠️  Invalid speed values: {speed_invalid}')
            else:
                print('  ✓ Vehicle speed in valid range [0-255]')

        if 'engine_rpm' in df.columns:
            rpm_invalid = ((df['engine_rpm'] < 0) | (df['engine_rpm'] > 16383)).sum()
            if rpm_invalid > 0:
                issues.append(f'Invalid RPM values: {rpm_invalid}')
                print(f'  ⚠️  Invalid RPM values: {rpm_invalid}')
            else:
                print('  ✓ Engine RPM in valid range [0-16383]')

        if 'throttle_position' in df.columns:
            throttle_invalid = ((df['throttle_position'] < 0) | (df['throttle_position'] > 100)).sum()
            if throttle_invalid > 0:
                issues.append(f'Invalid throttle values: {throttle_invalid}')
                print(f'  ⚠️  Invalid throttle values: {throttle_invalid}')
            else:
                print('  ✓ Throttle position in valid range [0-100]')

        if 'coolant_temp' in df.columns:
            temp_invalid = ((df['coolant_temp'] < -40) | (df['coolant_temp'] > 215)).sum()
            if temp_invalid > 0:
                issues.append(f'Invalid temperature values: {temp_invalid}')
                print(f'  ⚠️  Invalid temperature values: {temp_invalid}')
            else:
                print('  ✓ Coolant temperature in valid range [-40-215]')

        # Check statistical properties
        if 'vehicle_speed' in df.columns and len(df) > 0:
            mean_speed = df['vehicle_speed'].mean()
            std_speed = df['vehicle_speed'].std()
            print(f'  ℹ️  Speed stats: mean={mean_speed:.1f}, std={std_speed:.1f}')

        # Summary
        print()
        if issues:
            print(f'  ❌ Validation FAILED: {len(issues)} issue(s)')
            for issue in issues:
                print(f'     - {issue}')
            return 1
        else:
            print('  ✅ Validation PASSED: All quality checks passed')
            return 0

    except Exception as e:
        print(f'  ❌ Error: {e}')
        return 1

exit(validate_dataset('$dataset_path'))
" || DATASET_FAILED=$?

    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    if [ ${DATASET_FAILED:-0} -eq 0 ]; then
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
    fi
}

# Start validation
echo "🔍 Starting data quality validation..."
echo ""

# 1. Check for datasets directory
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣  Dataset Files"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

DATASETS_DIR="datasets"

if [ -d "$DATASETS_DIR" ]; then
    echo -e "  Datasets directory: ${GREEN}✅ Found${NC}"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
else
    echo -e "  Datasets directory: ${YELLOW}⚠️  Not found${NC}"
    echo "  Creating datasets/ directory..."
    mkdir -p "$DATASETS_DIR"
    WARNING_CHECKS=$((WARNING_CHECKS + 1))
fi
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

# Check for common dataset files
check_file "$DATASETS_DIR/train.csv" "Training dataset"
check_file "$DATASETS_DIR/val.csv" "Validation dataset"
check_file "$DATASETS_DIR/test.csv" "Test dataset"

# 2. Validate datasets if they exist
if [ -f "$DATASETS_DIR/train.csv" ]; then
    validate_dataset "$DATASETS_DIR/train.csv" "Training Dataset"
fi

if [ -f "$DATASETS_DIR/val.csv" ]; then
    validate_dataset "$DATASETS_DIR/val.csv" "Validation Dataset"
fi

if [ -f "$DATASETS_DIR/test.csv" ]; then
    validate_dataset "$DATASETS_DIR/test.csv" "Test Dataset"
fi

# 3. Check AI model files
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️⃣  AI Model Files"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

check_file "ai-models/trained_models/lightgbm_multi_v1_0_0.onnx" "LightGBM ONNX model"
check_file "ai-models/trained_models/tcn_fuel_v1_0_0.onnx" "TCN ONNX model (optional)"
check_file "ai-models/trained_models/lstmae_anomaly_v1_0_0.onnx" "LSTM-AE ONNX model (optional)"

# 4. Python script validation
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3️⃣  Data Generation Scripts"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -f "data-generation/synthetic_driving_simulator.py" ]; then
    echo -n "  Testing data generator... "
    if python3 data-generation/synthetic_driving_simulator.py --test-mode 2>/dev/null; then
        echo -e "${GREEN}✅ Working${NC}"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        echo -e "${RED}❌ Failed${NC}"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
    fi
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
else
    echo -e "  Data generator: ${YELLOW}⚠️  Not found${NC}"
    WARNING_CHECKS=$((WARNING_CHECKS + 1))
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
fi

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Validation Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Total Checks:  $TOTAL_CHECKS"
echo -e "Passed:        ${GREEN}$PASSED_CHECKS${NC}"
echo -e "Failed:        ${RED}$FAILED_CHECKS${NC}"
echo -e "Warnings:      ${YELLOW}$WARNING_CHECKS${NC}"
echo ""

# Quality score
if [ $TOTAL_CHECKS -gt 0 ]; then
    SCORE=$(awk "BEGIN {printf \"%.1f\", ($PASSED_CHECKS/$TOTAL_CHECKS)*100}")
    echo "Quality Score: ${SCORE}%"
else
    SCORE=0
    echo "Quality Score: N/A (no datasets found)"
fi

echo ""

# Recommendations
if [ $FAILED_CHECKS -gt 0 ]; then
    echo -e "${RED}❌ Data quality issues detected${NC}"
    echo ""
    echo "Action required:"
    echo "  1. Fix data quality issues listed above"
    echo "  2. Re-generate datasets with correct parameters"
    echo "  3. Run validation again: ./scripts/validate_data_quality.sh"
    echo ""
    exit 1
elif [ $WARNING_CHECKS -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Data quality warnings${NC}"
    echo ""
    echo "Recommendations:"
    echo "  - Generate training datasets: python data-generation/synthetic_driving_simulator.py"
    echo "  - Train models: see docs/GPU_REQUIRED_TASKS.md"
    echo ""
    exit 0
else
    echo -e "${GREEN}✅ All data quality checks passed${NC}"
    echo ""
    echo "Next steps:"
    echo "  - Train models with validated data"
    echo "  - Run performance benchmarks: ./scripts/benchmark_performance.sh"
    echo ""
    exit 0
fi

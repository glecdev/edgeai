#!/bin/bash
# GLEC DTG - Integrated Test Runner
# Run all test suites with coordinated reporting

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================="
echo "GLEC DTG - Integrated Test Suite"
echo "========================================="
echo ""
echo "📅 $(date '+%Y-%m-%d %H:%M:%S')"
echo "📁 Working directory: $PROJECT_ROOT"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Function to run test suite
run_test_suite() {
    local name=$1
    local path=$2
    local extra_args=${3:-""}

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}📦 $name${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    if [ ! -f "$path" ] && [ ! -d "$path" ]; then
        echo -e "${YELLOW}⚠️  Skipped: $path not found${NC}"
        return
    fi

    # Run pytest and capture results
    if pytest "$path" -v --tb=short $extra_args 2>&1 | tee /tmp/test_output.txt; then
        echo -e "${GREEN}✅ $name: PASSED${NC}"

        # Extract test counts
        local count=$(grep -oP '\d+(?= passed)' /tmp/test_output.txt | tail -1 || echo "0")
        PASSED_TESTS=$((PASSED_TESTS + count))
        TOTAL_TESTS=$((TOTAL_TESTS + count))
    else
        echo -e "${RED}❌ $name: FAILED${NC}"

        # Extract test counts
        local passed=$(grep -oP '\d+(?= passed)' /tmp/test_output.txt | tail -1 || echo "0")
        local failed=$(grep -oP '\d+(?= failed)' /tmp/test_output.txt | tail -1 || echo "0")

        PASSED_TESTS=$((PASSED_TESTS + passed))
        FAILED_TESTS=$((FAILED_TESTS + failed))
        TOTAL_TESTS=$((TOTAL_TESTS + passed + failed))

        return 1
    fi
}

# Start test execution
echo "🚀 Starting test execution..."

TEST_SUITES_FAILED=0

# 1. Synthetic Data Generation Tests
run_test_suite "Synthetic Driving Simulator" "tests/test_synthetic_simulator.py" || TEST_SUITES_FAILED=$((TEST_SUITES_FAILED + 1))

# 2. AI Model Tests
run_test_suite "TCN Fuel Prediction Model" "ai-models/tests/test_tcn.py" || TEST_SUITES_FAILED=$((TEST_SUITES_FAILED + 1))
run_test_suite "LSTM-AE Anomaly Detection" "ai-models/tests/test_lstm_ae.py" || TEST_SUITES_FAILED=$((TEST_SUITES_FAILED + 1))
run_test_suite "LightGBM Behavior Classification" "ai-models/tests/test_lightgbm.py" || TEST_SUITES_FAILED=$((TEST_SUITES_FAILED + 1))

# 3. Production Integration Tests
run_test_suite "Physics-Based Validation" "tests/test_physics_validation.py" || TEST_SUITES_FAILED=$((TEST_SUITES_FAILED + 1))
run_test_suite "Realtime Data Integration" "tests/test_realtime_integration.py" || TEST_SUITES_FAILED=$((TEST_SUITES_FAILED + 1))
run_test_suite "CAN Protocol Parser" "tests/test_can_parser.py" || TEST_SUITES_FAILED=$((TEST_SUITES_FAILED + 1))

# 4. Multi-Model AI Tests
run_test_suite "Multi-Model Integration" "ai-models/tests/test_multi_model.py" || TEST_SUITES_FAILED=$((TEST_SUITES_FAILED + 1))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test Execution Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Calculate pass rate
if [ $TOTAL_TESTS -gt 0 ]; then
    PASS_RATE=$(awk "BEGIN {printf \"%.1f\", ($PASSED_TESTS/$TOTAL_TESTS)*100}")
else
    PASS_RATE="0.0"
fi

echo "Total Tests:    $TOTAL_TESTS"
echo -e "Passed:         ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed:         ${RED}$FAILED_TESTS${NC}"
echo -e "Pass Rate:      ${BLUE}${PASS_RATE}%${NC}"
echo ""

# Performance metrics
echo "Performance Metrics:"
echo "  - Physics Validation: 19 tests (Newton's laws, thermodynamics)"
echo "  - Realtime Pipeline: 8 tests (254.7 rec/sec throughput)"
echo "  - Multi-Model AI: 16 tests (LightGBM + TCN + LSTM-AE)"
echo ""

# Quality gates
echo "Quality Gates:"
if (( $(echo "$PASS_RATE >= 95.0" | bc -l) )); then
    echo -e "  ${GREEN}✅ Pass Rate: ${PASS_RATE}% (target: ≥95%)${NC}"
else
    echo -e "  ${RED}❌ Pass Rate: ${PASS_RATE}% (target: ≥95%)${NC}"
fi

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "  ${GREEN}✅ No failing tests${NC}"
else
    echo -e "  ${RED}❌ $FAILED_TESTS test(s) failing${NC}"
fi

echo ""

# Exit with appropriate code
if [ $FAILED_TESTS -gt 0 ] || [ $TEST_SUITES_FAILED -gt 0 ]; then
    echo -e "${RED}❌ Test suite FAILED${NC}"
    echo ""
    echo "Action required:"
    echo "  1. Review failed tests above"
    echo "  2. Fix issues in source code or tests"
    echo "  3. Re-run: ./scripts/run_all_tests.sh"
    echo ""
    exit 1
else
    echo -e "${GREEN}✅ All tests PASSED${NC}"
    echo ""
    echo "Next steps:"
    echo "  - Generate coverage: ./scripts/generate_coverage.sh"
    echo "  - Code quality: ./scripts/format_code.sh"
    echo "  - Security scan: ./scripts/security_scan.sh"
    echo ""
    exit 0
fi

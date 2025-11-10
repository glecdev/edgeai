#!/bin/bash
# GLEC DTG - Coverage Report Generator
# Generate comprehensive test coverage reports

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================="
echo "GLEC DTG - Coverage Report Generator"
echo "========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if pytest-cov is installed
if ! python -c "import pytest_cov" 2>/dev/null; then
    echo "⚠️  pytest-cov not found. Installing..."
    pip install pytest-cov
fi

echo "📊 Generating test coverage reports..."
echo ""

# Run tests with coverage
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running tests with coverage tracking..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

pytest tests/ \
    --cov=ai-models \
    --cov=fleet-integration \
    --cov=data-generation \
    --cov-report=html \
    --cov-report=term \
    --cov-report=json \
    --cov-config=.coveragerc \
    -v \
    || COVERAGE_FAILED=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Coverage Report Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Extract coverage percentage from JSON report
if [ -f "coverage.json" ]; then
    COVERAGE_PCT=$(python3 -c "
import json
with open('coverage.json') as f:
    data = json.load(f)
    print(f\"{data['totals']['percent_covered']:.1f}\")
" 2>/dev/null || echo "0.0")

    echo "Total Coverage: ${COVERAGE_PCT}%"
    echo ""

    # Per-module coverage
    echo "Module Coverage:"
    python3 -c "
import json
with open('coverage.json') as f:
    data = json.load(f)
    files = data.get('files', {})

    # Group by module
    modules = {}
    for filepath, stats in files.items():
        if 'ai-models' in filepath:
            module = 'ai-models'
        elif 'fleet-integration' in filepath:
            module = 'fleet-integration'
        elif 'data-generation' in filepath:
            module = 'data-generation'
        else:
            continue

        if module not in modules:
            modules[module] = {'covered': 0, 'total': 0}

        modules[module]['covered'] += stats['summary']['covered_lines']
        modules[module]['total'] += stats['summary']['num_statements']

    for module, stats in sorted(modules.items()):
        if stats['total'] > 0:
            pct = (stats['covered'] / stats['total']) * 100
            print(f\"  - {module}: {pct:.1f}% ({stats['covered']}/{stats['total']} lines)\")
" 2>/dev/null

    echo ""
else
    echo -e "${YELLOW}⚠️  Coverage JSON report not found${NC}"
    COVERAGE_PCT="0.0"
fi

# Report locations
echo "📄 Reports Generated:"
echo "  - HTML: htmlcov/index.html"
echo "  - JSON: coverage.json"
echo "  - Terminal: (shown above)"
echo ""

# Quality gate check
TARGET_COVERAGE=80.0

echo "Quality Gate:"
if (( $(echo "$COVERAGE_PCT >= $TARGET_COVERAGE" | bc -l) )); then
    echo -e "  ${GREEN}✅ Coverage: ${COVERAGE_PCT}% (target: ≥${TARGET_COVERAGE}%)${NC}"
else
    echo -e "  ${RED}❌ Coverage: ${COVERAGE_PCT}% (target: ≥${TARGET_COVERAGE}%)${NC}"
fi
echo ""

# Low coverage files
echo "Low Coverage Files (<80%):"
if [ -f "coverage.json" ]; then
    python3 -c "
import json
with open('coverage.json') as f:
    data = json.load(f)
    files = data.get('files', {})

    low_coverage = []
    for filepath, stats in files.items():
        pct = stats['summary']['percent_covered']
        if pct < 80.0 and stats['summary']['num_statements'] > 0:
            low_coverage.append((filepath, pct, stats['summary']['missing_lines']))

    if low_coverage:
        low_coverage.sort(key=lambda x: x[1])  # Sort by coverage percentage
        for filepath, pct, missing in low_coverage[:10]:  # Show top 10
            # Shorten path
            short_path = filepath.replace('$PROJECT_ROOT/', '')
            print(f\"  - {short_path}: {pct:.1f}% (missing: {missing} lines)\")
    else:
        print(\"  None - All files above 80%! 🎉\")
" 2>/dev/null
fi

echo ""

# Instructions
echo "Next Steps:"
echo "  1. Open HTML report: open htmlcov/index.html"
echo "  2. Review low-coverage files and add tests"
echo "  3. Aim for ≥80% coverage on all modules"
echo ""

# Exit with appropriate code
if [ ${COVERAGE_FAILED:-0} -ne 0 ]; then
    echo -e "${RED}❌ Coverage generation failed (some tests failed)${NC}"
    exit 1
elif (( $(echo "$COVERAGE_PCT < $TARGET_COVERAGE" | bc -l) )); then
    echo -e "${YELLOW}⚠️  Coverage below target (${COVERAGE_PCT}% < ${TARGET_COVERAGE}%)${NC}"
    echo "    Add tests to increase coverage before release"
    exit 0  # Warning, but not a hard failure
else
    echo -e "${GREEN}✅ Coverage target met (${COVERAGE_PCT}% ≥ ${TARGET_COVERAGE}%)${NC}"
    exit 0
fi

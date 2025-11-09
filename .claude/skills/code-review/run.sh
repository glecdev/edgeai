#!/bin/bash
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}GLEC DTG - Code Review${NC}"
echo -e "${GREEN}========================================${NC}\n"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

TARGET=""
STRICT=false
ALL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --target)
            TARGET="$2"
            shift 2
            ;;
        --all)
            ALL=true
            shift
            ;;
        --strict)
            STRICT=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

REPORT_FILE="code-review-report.md"
echo "# Code Review Report" > "$REPORT_FILE"
echo "Date: $(date)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

ISSUES_FOUND=0
CRITICAL_ISSUES=0

# Python Code Review
review_python() {
    local dir=$1
    echo -e "${YELLOW}📝 Reviewing Python code: $dir${NC}\n"

    if [ ! -d "$dir" ]; then
        echo -e "${YELLOW}⚠️  Directory not found: $dir${NC}\n"
        return
    fi

    # Check if virtual environment is activated
    if [ -z "$VIRTUAL_ENV" ] && [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    fi

    # Pylint
    echo -e "${BLUE}Running Pylint...${NC}"
    if command -v pylint &> /dev/null; then
        PYLINT_SCORE=$(pylint "$dir" 2>/dev/null | grep "Your code has been rated" | awk '{print $7}' | cut -d'/' -f1 || echo "0")
        echo -e "Pylint Score: ${GREEN}$PYLINT_SCORE/10${NC}"
        echo "## Pylint" >> "$REPORT_FILE"
        echo "Score: $PYLINT_SCORE/10" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"

        if (( $(echo "$PYLINT_SCORE < 8.0" | bc -l) )); then
            ((ISSUES_FOUND++))
            if [ "$STRICT" = true ]; then
                ((CRITICAL_ISSUES++))
            fi
        fi
    else
        echo -e "${YELLOW}⚠️  Pylint not installed: pip install pylint${NC}"
    fi

    # Mypy (Type checking)
    echo -e "\n${BLUE}Running Mypy...${NC}"
    if command -v mypy &> /dev/null; then
        MYPY_OUTPUT=$(mypy "$dir" 2>&1 || true)
        MYPY_ERRORS=$(echo "$MYPY_OUTPUT" | grep -c "error:" || echo "0")
        echo -e "Type errors: $MYPY_ERRORS"
        echo "## Mypy" >> "$REPORT_FILE"
        echo "Errors: $MYPY_ERRORS" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"

        if [ "$MYPY_ERRORS" -gt 0 ]; then
            ((ISSUES_FOUND++))
        fi
    else
        echo -e "${YELLOW}⚠️  Mypy not installed: pip install mypy${NC}"
    fi

    # Bandit (Security)
    echo -e "\n${BLUE}Running Bandit (security scan)...${NC}"
    if command -v bandit &> /dev/null; then
        BANDIT_OUTPUT=$(bandit -r "$dir" -f json 2>/dev/null || echo '{"results": []}')
        BANDIT_HIGH=$(echo "$BANDIT_OUTPUT" | grep -o '"severity": "HIGH"' | wc -l || echo "0")
        BANDIT_MEDIUM=$(echo "$BANDIT_OUTPUT" | grep -o '"severity": "MEDIUM"' | wc -l || echo "0")

        echo -e "Security issues: High=$BANDIT_HIGH, Medium=$BANDIT_MEDIUM"
        echo "## Bandit (Security)" >> "$REPORT_FILE"
        echo "High: $BANDIT_HIGH, Medium: $BANDIT_MEDIUM" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"

        if [ "$BANDIT_HIGH" -gt 0 ]; then
            ((CRITICAL_ISSUES++))
        fi
        if [ "$BANDIT_MEDIUM" -gt 0 ]; then
            ((ISSUES_FOUND++))
        fi
    else
        echo -e "${YELLOW}⚠️  Bandit not installed: pip install bandit${NC}"
    fi

    # Coverage check
    echo -e "\n${BLUE}Checking test coverage...${NC}"
    if [ -f ".coverage" ]; then
        COVERAGE=$(python -c "
import coverage
cov = coverage.Coverage()
cov.load()
total = cov.report(show_missing=False)
print(f'{total:.1f}')
" 2>/dev/null || echo "0")
        echo -e "Coverage: ${GREEN}$COVERAGE%${NC}"
        echo "## Coverage" >> "$REPORT_FILE"
        echo "Coverage: $COVERAGE%" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"

        if (( $(echo "$COVERAGE < 80.0" | bc -l) )); then
            ((ISSUES_FOUND++))
        fi
    else
        echo -e "${YELLOW}⚠️  No coverage data found (run tests with --cov)${NC}"
    fi
}

# Android Code Review
review_android() {
    local dir=$1
    echo -e "${YELLOW}📱 Reviewing Android code: $dir${NC}\n"

    if [ ! -d "$dir" ]; then
        echo -e "${YELLOW}⚠️  Directory not found: $dir${NC}\n"
        return
    fi

    cd "$dir"

    # Android Lint
    echo -e "${BLUE}Running Android Lint...${NC}"
    if [ -f "gradlew" ]; then
        ./gradlew lint > /dev/null 2>&1 || true
        if [ -f "app/build/reports/lint-results.xml" ]; then
            LINT_ERRORS=$(grep -c 'severity="Error"' app/build/reports/lint-results.xml || echo "0")
            LINT_WARNINGS=$(grep -c 'severity="Warning"' app/build/reports/lint-results.xml || echo "0")
            echo -e "Lint: Errors=$LINT_ERRORS, Warnings=$LINT_WARNINGS"

            if [ "$LINT_ERRORS" -gt 0 ]; then
                ((CRITICAL_ISSUES++))
            fi
            if [ "$LINT_WARNINGS" -gt 5 ]; then
                ((ISSUES_FOUND++))
            fi
        fi
    fi

    cd "$PROJECT_ROOT"
}

# Main review logic
if [ "$ALL" = true ]; then
    review_python "ai-models"
    review_android "android-dtg"
    review_android "android-driver"
elif [ -n "$TARGET" ]; then
    if [[ "$TARGET" == *"android"* ]]; then
        review_android "$TARGET"
    else
        review_python "$TARGET"
    fi
else
    echo -e "${RED}❌ Please specify --target or --all${NC}"
    exit 1
fi

# Summary
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Review Summary${NC}"
echo -e "${GREEN}========================================${NC}\n"

echo "## Summary" >> "$REPORT_FILE"
echo "Issues Found: $ISSUES_FOUND" >> "$REPORT_FILE"
echo "Critical Issues: $CRITICAL_ISSUES" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

echo -e "📊 Results:"
echo -e "  • Issues Found: $ISSUES_FOUND"
echo -e "  • Critical Issues: $CRITICAL_ISSUES"
echo ""

# Quality Gate
if [ "$STRICT" = true ]; then
    if [ "$CRITICAL_ISSUES" -gt 0 ]; then
        echo -e "${RED}❌ QUALITY GATE: FAILED (Strict Mode)${NC}"
        echo "Quality Gate: FAILED" >> "$REPORT_FILE"
        exit 1
    else
        echo -e "${GREEN}✅ QUALITY GATE: PASSED (Strict Mode)${NC}"
        echo "Quality Gate: PASSED" >> "$REPORT_FILE"
    fi
else
    if [ "$CRITICAL_ISSUES" -gt 0 ] || [ "$ISSUES_FOUND" -gt 10 ]; then
        echo -e "${YELLOW}⚠️  QUALITY GATE: WARNING${NC}"
        echo -e "Consider fixing issues before merging"
        echo "Quality Gate: WARNING" >> "$REPORT_FILE"
    else
        echo -e "${GREEN}✅ QUALITY GATE: PASSED${NC}"
        echo "Quality Gate: PASSED" >> "$REPORT_FILE"
    fi
fi

echo ""
echo -e "${BLUE}📄 Full report: $REPORT_FILE${NC}"
echo ""
echo -e "${GREEN}Code Review Complete! 🎉${NC}"

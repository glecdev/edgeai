#!/bin/bash
# GLEC DTG - Environment Verification
# Verify development environment prerequisites and configuration

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================="
echo "GLEC DTG - Environment Verification"
echo "========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Tracking
CHECKS_PASSED=0
CHECKS_FAILED=0
CHECKS_WARNING=0

# Function to check command
check_command() {
    local cmd=$1
    local name=$2
    local required=${3:-true}

    echo -n "  $name... "
    if command -v $cmd &> /dev/null; then
        local version=$($cmd --version 2>&1 | head -1 || echo "unknown")
        echo -e "${GREEN}✅ Found${NC} ($version)"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
        return 0
    else
        if [ "$required" = true ]; then
            echo -e "${RED}❌ Missing (required)${NC}"
            CHECKS_FAILED=$((CHECKS_FAILED + 1))
        else
            echo -e "${YELLOW}⚠️  Missing (optional)${NC}"
            CHECKS_WARNING=$((CHECKS_WARNING + 1))
        fi
        return 1
    fi
}

# Function to check Python module
check_python_module() {
    local module=$1
    local name=$2
    local required=${3:-true}

    echo -n "  $name... "
    if python3 -c "import $module" 2>/dev/null; then
        local version=$(python3 -c "import $module; print(getattr($module, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
        echo -e "${GREEN}✅ Installed${NC} ($version)"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
        return 0
    else
        if [ "$required" = true ]; then
            echo -e "${RED}❌ Missing (required)${NC}"
            CHECKS_FAILED=$((CHECKS_FAILED + 1))
        else
            echo -e "${YELLOW}⚠️  Missing (optional)${NC}"
            CHECKS_WARNING=$((CHECKS_WARNING + 1))
        fi
        return 1
    fi
}

# 1. System Environment
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣  System Environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "  OS: $(uname -s)"
echo "  Kernel: $(uname -r)"
echo "  Architecture: $(uname -m)"
echo "  Working Directory: $PROJECT_ROOT"
echo ""

# 2. Core Tools
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️⃣  Core Development Tools"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

check_command python3 "Python 3" true
check_command pip "pip" true
check_command git "Git" true
check_command pytest "pytest" true

echo ""

# 3. Python Dependencies
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3️⃣  Python Dependencies (Core)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

check_python_module "numpy" "NumPy" true
check_python_module "pandas" "Pandas" true
check_python_module "pytest" "pytest" true
check_python_module "pytest_cov" "pytest-cov" true

echo ""

# 4. AI/ML Dependencies
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4️⃣  AI/ML Dependencies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

check_python_module "onnxruntime" "ONNX Runtime" true
check_python_module "lightgbm" "LightGBM" true
check_python_module "sklearn" "scikit-learn" true
check_python_module "torch" "PyTorch" false

echo ""

# 5. Code Quality Tools
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5️⃣  Code Quality Tools"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

check_command black "Black (formatter)" false
check_command isort "isort (imports)" false
check_command mypy "mypy (type checking)" false
check_command bandit "Bandit (security)" false
check_command safety "Safety (dependencies)" false

echo ""

# 6. Project Structure
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6️⃣  Project Structure"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

check_directory() {
    local dir=$1
    local name=$2
    echo -n "  $name... "
    if [ -d "$dir" ]; then
        local count=$(find "$dir" -type f 2>/dev/null | wc -l)
        echo -e "${GREEN}✅ Exists${NC} ($count files)"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
        echo -e "${RED}❌ Missing${NC}"
        CHECKS_FAILED=$((CHECKS_FAILED + 1))
    fi
}

check_directory "ai-models" "ai-models/"
check_directory "tests" "tests/"
check_directory "data-generation" "data-generation/"
check_directory "fleet-integration" "fleet-integration/"
check_directory "scripts" "scripts/"
check_directory "docs" "docs/"

echo ""

# 7. Git Configuration
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "7️⃣  Git Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "  Git repository: ${GREEN}✅ Initialized${NC}"
    echo "  Current branch: $(git branch --show-current)"
    echo "  Latest commit: $(git log -1 --oneline)"
    echo "  Remote: $(git remote get-url origin 2>/dev/null || echo 'Not configured')"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    echo -e "  Git repository: ${RED}❌ Not initialized${NC}"
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
fi

echo ""

# 8. Test Suite Status
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "8️⃣  Test Suite Status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

TEST_COUNT=$(find tests/ -name "test_*.py" -type f 2>/dev/null | wc -l)
echo "  Test files: $TEST_COUNT"

if [ -f ".coveragerc" ]; then
    echo -e "  Coverage config: ${GREEN}✅ Configured${NC}"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    echo -e "  Coverage config: ${YELLOW}⚠️  Missing .coveragerc${NC}"
    CHECKS_WARNING=$((CHECKS_WARNING + 1))
fi

echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Verification Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

TOTAL_CHECKS=$((CHECKS_PASSED + CHECKS_FAILED + CHECKS_WARNING))
echo "Total Checks: $TOTAL_CHECKS"
echo -e "Passed:       ${GREEN}$CHECKS_PASSED${NC}"
echo -e "Failed:       ${RED}$CHECKS_FAILED${NC}"
echo -e "Warnings:     ${YELLOW}$CHECKS_WARNING${NC}"
echo ""

# Recommendations
if [ $CHECKS_FAILED -gt 0 ]; then
    echo "❌ Environment issues detected"
    echo ""
    echo "Action required:"
    echo "  1. Install missing required dependencies:"
    echo "     pip install -r requirements.txt"
    echo "  2. Install code quality tools:"
    echo "     pip install black isort mypy bandit safety"
    echo "  3. Re-run verification: ./scripts/verify_environment.sh"
    echo ""
    exit 1
elif [ $CHECKS_WARNING -gt 0 ]; then
    echo "⚠️  Environment partially configured"
    echo ""
    echo "Recommendations:"
    echo "  - Install optional tools for better development experience"
    echo "  - See requirements.txt for full dependency list"
    echo ""
    exit 0
else
    echo -e "${GREEN}✅ Environment fully configured${NC}"
    echo ""
    echo "Ready for development! 🚀"
    echo ""
    echo "Next steps:"
    echo "  - Run tests: ./scripts/run_all_tests.sh"
    echo "  - Generate coverage: ./scripts/generate_coverage.sh"
    echo "  - Format code: ./scripts/format_code.sh"
    echo ""
    exit 0
fi

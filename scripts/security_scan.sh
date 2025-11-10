#!/bin/bash
# GLEC DTG - Security Scanning Tool
# Scan for security vulnerabilities using Bandit and Safety

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================="
echo "GLEC DTG - Security Scanner"
echo "========================================="
echo ""

# Check if tools are installed
if ! command -v bandit &> /dev/null; then
    echo "⚠️  Bandit not found. Installing..."
    pip install bandit
fi

if ! command -v safety &> /dev/null; then
    echo "⚠️  Safety not found. Installing..."
    pip install safety
fi

echo "🔒 Running security scans..."
echo ""

# Bandit: Static analysis for security issues in Python code
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣  Bandit - Code Security Analysis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

bandit -r ai-models/ tests/ data-generation/ fleet-integration/ \
    -f json -o security-report-bandit.json \
    --severity-level medium \
    --confidence-level medium \
    --exclude '*test*,*/__pycache__/*,*/venv/*' \
    || BANDIT_ISSUES=$?

bandit -r ai-models/ tests/ data-generation/ fleet-integration/ \
    --severity-level medium \
    --confidence-level medium \
    --exclude '*test*,*/__pycache__/*,*/venv/*' \
    -f screen \
    || true

echo ""
echo "📄 Bandit report saved: security-report-bandit.json"
echo ""

# Safety: Check dependencies for known vulnerabilities
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️⃣  Safety - Dependency Vulnerability Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -f "requirements.txt" ]; then
    safety check -r requirements.txt \
        --json > security-report-safety.json \
        || SAFETY_ISSUES=$?

    safety check -r requirements.txt || true

    echo ""
    echo "📄 Safety report saved: security-report-safety.json"
else
    echo "⚠️  requirements.txt not found, skipping dependency check"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Security Scan Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ ${BANDIT_ISSUES:-0} -eq 0 ] && [ ${SAFETY_ISSUES:-0} -eq 0 ]; then
    echo "✅ No security issues found!"
    echo ""
    echo "Summary:"
    echo "  - Bandit: No code vulnerabilities detected"
    echo "  - Safety: All dependencies secure"
    echo ""
else
    echo "⚠️  Security issues detected (see above)"
    echo ""

    if [ ${BANDIT_ISSUES:-0} -ne 0 ]; then
        echo "🔍 Bandit found potential security issues:"
        echo "   - Review security-report-bandit.json for details"
        echo "   - Common issues: hardcoded secrets, SQL injection, shell injection"
    fi

    if [ ${SAFETY_ISSUES:-0} -ne 0 ]; then
        echo "📦 Safety found vulnerable dependencies:"
        echo "   - Review security-report-safety.json for details"
        echo "   - Update vulnerable packages in requirements.txt"
    fi

    echo ""
    echo "Action required: Fix issues or add exceptions with justification"
    exit 1
fi

echo "Tip: Run regularly before commits and in CI/CD pipeline"

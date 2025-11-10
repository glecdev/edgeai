#!/bin/bash
# GLEC DTG - Code Formatting Tool
# Automatically format Python code using Black and isort

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================="
echo "GLEC DTG - Code Formatter"
echo "========================================="
echo ""

# Check if tools are installed
if ! command -v black &> /dev/null; then
    echo "⚠️  Black not found. Installing..."
    pip install black
fi

if ! command -v isort &> /dev/null; then
    echo "⚠️  isort not found. Installing..."
    pip install isort
fi

echo "📁 Formatting Python code in:"
echo "   - ai-models/"
echo "   - tests/"
echo "   - data-generation/"
echo "   - fleet-integration/"
echo ""

# Format with Black (PEP 8 compliant)
echo "🎨 Running Black formatter..."
black --line-length 100 \
    ai-models/ \
    tests/ \
    data-generation/ \
    fleet-integration/ \
    --exclude '/(\.git|\.pytest_cache|__pycache__|\.venv|venv|build|dist)/'

echo ""

# Sort imports with isort
echo "📦 Sorting imports with isort..."
isort --profile black --line-length 100 \
    ai-models/ \
    tests/ \
    data-generation/ \
    fleet-integration/ \
    --skip-gitignore

echo ""
echo "✅ Code formatting complete!"
echo ""
echo "Summary:"
echo "  - Black: Code style enforced (line length: 100)"
echo "  - isort: Import statements sorted"
echo ""
echo "Tip: Run 'git diff' to review changes before committing"

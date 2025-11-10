#!/bin/bash
# GLEC DTG - Type Checking Tool
# Static type checking using mypy

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================="
echo "GLEC DTG - Type Checker (mypy)"
echo "========================================="
echo ""

# Check if mypy is installed
if ! command -v mypy &> /dev/null; then
    echo "⚠️  mypy not found. Installing..."
    pip install mypy
fi

echo "🔍 Type checking Python code..."
echo ""

# Run mypy on all Python directories
mypy ai-models/ \
    tests/ \
    data-generation/ \
    fleet-integration/ \
    --ignore-missing-imports \
    --no-strict-optional \
    --warn-return-any \
    --warn-unused-configs \
    --disallow-untyped-defs \
    --show-error-codes \
    --pretty \
    || TYPE_ERRORS=$?

echo ""

if [ ${TYPE_ERRORS:-0} -eq 0 ]; then
    echo "✅ Type checking passed!"
    echo ""
    echo "Summary:"
    echo "  - No type errors found"
    echo "  - All function signatures validated"
    echo ""
else
    echo "⚠️  Type checking found issues (see above)"
    echo ""
    echo "Common fixes:"
    echo "  - Add type hints to function parameters and returns"
    echo "  - Use Optional[Type] for nullable values"
    echo "  - Import types: from typing import List, Dict, Optional"
    echo ""
    exit 1
fi

echo "Tip: Add '# type: ignore' comments to suppress specific warnings"

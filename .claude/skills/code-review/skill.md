# Code Review Skill

## Metadata
- **Name**: code-review
- **Description**: 자동 코드 품질 검사 및 리뷰 (Phase 4: Review)
- **Phase**: Phase 4 - Review
- **Dependencies**: pylint, mypy, bandit, detekt (Android)
- **Estimated Time**: 2-5 minutes

## What This Skill Does

### 1. 정적 분석 (Static Analysis)
- **Python**: pylint, mypy, bandit
- **Kotlin/Java**: detekt, ktlint
- **C/C++**: cppcheck, clang-tidy

### 2. 코드 복잡도 분석
- Cyclomatic complexity
- Cognitive complexity
- Maintainability index

### 3. 보안 취약점 스캔
- SQL injection
- XSS vulnerabilities
- Hardcoded secrets
- Insecure dependencies

### 4. Best Practices 검증
- SOLID 원칙
- DRY (Don't Repeat Yourself)
- 네이밍 컨벤션
- 문서화 완성도

## Usage

```bash
# Python 코드 리뷰
./.claude/skills/code-review/run.sh --target ai-models/

# Android 코드 리뷰
./.claude/skills/code-review/run.sh --target android-dtg/

# 전체 프로젝트 리뷰
./.claude/skills/code-review/run.sh --all

# 엄격 모드 (CI/CD용)
./.claude/skills/code-review/run.sh --strict
```

## Expected Output

```
🔍 Code Review Report
========================================

📁 Target: ai-models/training/

✅ Pylint Score: 9.2/10 (Excellent)
✅ Mypy: No type errors
⚠️  Bandit: 2 low-severity issues found

🔧 Issues Found:

1. [Medium] Complexity too high
   File: train_tcn.py:45
   Function: train_model()
   Cyclomatic Complexity: 15 (max: 10)
   Suggestion: Extract validation logic to separate function

2. [Low] Missing docstring
   File: utils.py:23
   Function: preprocess_data()
   Suggestion: Add docstring with Args/Returns

3. [Low] Hardcoded value
   File: config.py:12
   Variable: BATCH_SIZE = 64
   Suggestion: Move to configuration file

📊 Metrics:
  • Coverage: 85%
  • Complexity (avg): 6.2
  • Maintainability Index: 72.3
  • Duplicate Code: 2.1%

🎯 Quality Gate: ✅ PASSED
  ✅ Pylint score >8.0
  ✅ No critical security issues
  ✅ Coverage >80%

💡 Recommendations:
  1. Refactor train_model() to reduce complexity
  2. Add missing docstrings (3 functions)
  3. Move hardcoded values to config

Next Steps:
  ./.claude/skills/optimize-performance/run.sh
```

## Quality Gates

### Strict Mode (for CI/CD)
- Pylint score ≥ 9.0
- Mypy: 0 errors
- Bandit: 0 medium+ severity
- Coverage ≥ 80%
- Complexity ≤ 10
- No TODO/FIXME in production code

### Normal Mode (for development)
- Pylint score ≥ 8.0
- Mypy: ≤ 5 errors
- Bandit: 0 high severity
- Coverage ≥ 70%
- Complexity ≤ 15

## Integration

### Pre-commit Hook
```bash
# .git/hooks/pre-commit
./.claude/skills/code-review/run.sh --target $(git diff --cached --name-only)
```

### CI/CD
```yaml
# .github/workflows/code-review.yml
- name: Code Review
  run: ./.claude/skills/code-review/run.sh --all --strict
```

## Files Created
- `code-review-report.md` - Detailed report
- `code-review-summary.json` - Machine-readable summary
- `code-review.log` - Full analysis log

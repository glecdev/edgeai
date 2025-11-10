# GLEC DTG - Development Scripts

Code quality and automation tools for the GLEC DTG Edge AI SDK.

## Available Scripts

### 🎨 format_code.sh - Code Formatter

Automatically format Python code using Black and isort.

**Usage:**
```bash
./scripts/format_code.sh
```

**Features:**
- **Black**: PEP 8 compliant formatting (line length: 100)
- **isort**: Automatic import sorting
- **Coverage**: ai-models/, tests/, data-generation/, fleet-integration/

**When to use:**
- Before committing code
- After writing new functions
- To maintain consistent code style

---

### 🔍 type_check.sh - Static Type Checker

Validate type hints using mypy.

**Usage:**
```bash
./scripts/type_check.sh
```

**Features:**
- **mypy**: Static type analysis
- **Checks**: Function signatures, return types, variable types
- **Error codes**: Detailed error reporting with suggestions

**When to use:**
- During development to catch type errors early
- Before committing (recommended)
- As part of CI/CD pipeline

---

### 🔒 security_scan.sh - Security Scanner

Scan for vulnerabilities using Bandit and Safety.

**Usage:**
```bash
./scripts/security_scan.sh
```

**Features:**
- **Bandit**: Static code analysis for security issues
  - SQL injection, shell injection, hardcoded secrets
  - Confidence levels: HIGH, MEDIUM, LOW
- **Safety**: Dependency vulnerability checking
  - Known CVEs in Python packages
  - Update recommendations

**Outputs:**
- `security-report-bandit.json` - Detailed Bandit findings
- `security-report-safety.json` - Dependency vulnerabilities

**When to use:**
- Before releases
- After updating dependencies
- Weekly security audits
- CI/CD pipeline (mandatory)

---

## Quick Start

Run all quality checks before committing:

```bash
# 1. Format code
./scripts/format_code.sh

# 2. Check types
./scripts/type_check.sh

# 3. Security scan
./scripts/security_scan.sh

# 4. Run tests
pytest tests/ -v --cov

# 5. Commit if all pass
git add -A
git commit -m "your message"
```

## CI/CD Integration

These scripts are designed to be used in GitHub Actions:

```yaml
- name: Code Quality Checks
  run: |
    ./scripts/format_code.sh --check  # Fails if formatting needed
    ./scripts/type_check.sh           # Fails on type errors
    ./scripts/security_scan.sh        # Fails on vulnerabilities
    pytest tests/ -v --cov
```

## Configuration

### Black Configuration (.black)
- Line length: 100 characters
- Target Python: 3.9+
- Excludes: .git, __pycache__, venv, build, dist

### isort Configuration
- Profile: black (compatible)
- Line length: 100
- Skip: gitignore patterns

### mypy Configuration
- Ignore missing imports: Yes
- Strict optional: No
- Warn return any: Yes
- Disallow untyped defs: Yes

### Bandit Configuration
- Severity: MEDIUM and above
- Confidence: MEDIUM and above
- Excludes: test files, __pycache__, venv

## Troubleshooting

### "Command not found"
Scripts automatically install missing tools. If issues persist:
```bash
pip install black isort mypy bandit safety
```

### Type checking too strict
Add `# type: ignore` comment to specific lines:
```python
result = some_function()  # type: ignore[return-value]
```

### False positive security issues
Add Bandit exception with justification:
```python
# nosec B201 - Intentionally using flask-socketio for real-time data
```

## Best Practices

1. **Format before committing**: Ensures consistent code style
2. **Type check during development**: Catches bugs early
3. **Security scan before release**: Prevents vulnerabilities in production
4. **Automate in CI/CD**: Enforce quality gates automatically

## Support

For issues or improvements:
- Check tool documentation: Black, isort, mypy, Bandit, Safety
- Review CLAUDE.md for project-specific guidelines
- See GitHub Actions logs for CI/CD failures

# GLEC DTG - Development Scripts

Code quality and automation tools for the GLEC DTG Edge AI SDK.

## Available Scripts

### 🧪 run_all_tests.sh - Integrated Test Runner

Run all test suites with coordinated reporting and quality gates.

**Usage:**
```bash
./scripts/run_all_tests.sh
```

**Features:**
- **Comprehensive**: Runs all test suites (synthetic data, AI models, production integration)
- **Coordinated**: Sequential execution with aggregated results
- **Colored output**: Clear pass/fail indicators
- **Quality gates**: 95% pass rate requirement
- **Performance metrics**: Throughput, latency, validation stats

**Test Suites Included:**
1. Synthetic Driving Simulator (14 tests)
2. TCN Fuel Prediction Model
3. LSTM-AE Anomaly Detection
4. LightGBM Behavior Classification (28 tests)
5. Physics-Based Validation (19 tests)
6. Realtime Data Integration (8 tests)
7. CAN Protocol Parser (18 tests)
8. Multi-Model Integration (16 tests)

**When to use:**
- Before committing code
- After making changes to critical modules
- Daily development workflow
- CI/CD pipeline entry point

---

### 📊 generate_coverage.sh - Coverage Report Generator

Generate comprehensive test coverage reports with HTML, JSON, and terminal output.

**Usage:**
```bash
./scripts/generate_coverage.sh
```

**Features:**
- **Multi-format**: HTML (interactive), JSON (CI/CD), terminal (quick view)
- **Module breakdown**: Per-module coverage statistics
- **Low coverage detection**: Identifies files <80% coverage
- **Quality gate**: Enforces ≥80% coverage target

**Outputs:**
- `htmlcov/index.html` - Interactive HTML report (open in browser)
- `coverage.json` - Machine-readable JSON for automation
- Terminal summary - Quick overview

**When to use:**
- After running tests
- Before releases (verify coverage)
- Weekly quality reviews
- To identify untested code paths

---

### ✅ verify_environment.sh - Environment Verification

Verify development environment prerequisites and configuration.

**Usage:**
```bash
./scripts/verify_environment.sh
```

**Features:**
- **System check**: OS, kernel, architecture
- **Tool verification**: Python, pip, git, pytest
- **Dependency audit**: Core libraries, AI/ML packages
- **Code quality tools**: Black, isort, mypy, Bandit, Safety
- **Project structure**: Validates directory layout
- **Git config**: Branch, remote, commit status

**Checks Performed:**
1. Core development tools (Python 3, pip, git, pytest)
2. Python dependencies (numpy, pandas, pytest-cov)
3. AI/ML dependencies (onnxruntime, lightgbm, scikit-learn)
4. Code quality tools (optional but recommended)
5. Project directory structure
6. Git repository configuration
7. Test suite status

**When to use:**
- Initial project setup
- After pulling changes
- Troubleshooting build issues
- Onboarding new developers

---

### 📊 validate_data_quality.sh - Data Quality Validator

Automated validation of training datasets and CAN data quality.

**Usage:**
```bash
./scripts/validate_data_quality.sh
```

**Features:**
- **Dataset validation**: Checks train/val/test CSV files
- **Column validation**: Ensures required CAN data columns present
- **Range validation**: Verifies values within physical limits
  - Speed: 0-255 km/h
  - RPM: 0-16383
  - Throttle: 0-100%
  - Coolant: -40 to 215°C
- **Quality checks**: Missing values, duplicates, statistics
- **Model verification**: Checks ONNX model files exist
- **Quality score**: Percentage-based scoring system

**Validations Performed:**
1. Dataset files exist (train.csv, val.csv, test.csv)
2. Required columns present (vehicle_speed, engine_rpm, throttle, brake, fuel, coolant)
3. No missing values (or acceptable percentage)
4. No duplicate rows
5. Value ranges within physical limits
6. Statistical properties (mean, std within expected ranges)
7. Model files present in ai-models/trained_models/

**When to use:**
- After generating synthetic data
- Before training models
- After data preprocessing
- Regular data quality audits
- CI/CD pipeline for data validation

**Output:**
- Detailed validation report with pass/fail status
- Quality score (percentage)
- Recommendations for fixing issues

---

### ⚡ benchmark_performance.sh - Performance Benchmarking

Comprehensive performance testing for AI models and data pipeline.

**Usage:**
```bash
./scripts/benchmark_performance.sh
```

**Features:**
- **Model size verification**: Total size of all ONNX models
- **Inference latency**: P50, P95, P99 percentiles (1000 iterations)
- **Throughput testing**: Records processed per second
- **Memory usage**: Baseline, after load, after inference
- **Accuracy metrics**: If test dataset available
- **End-to-end pipeline**: Full CAN→AI workflow benchmark
- **JSON report**: Machine-readable results with timestamp

**Performance Targets:**
| Metric | Target | Measured |
|--------|--------|----------|
| Inference Latency (P95) | <50ms | ✓ |
| Throughput | >250 rec/sec | ✓ |
| Memory Usage | <50MB | ✓ |
| Model Size | <14MB | ✓ |

**Benchmarks Included:**
1. **Model Size**: Verify total ONNX model size
2. **Inference Latency**: Measure inference time (P50/P95/P99)
3. **Throughput**: Realtime integration pipeline speed
4. **Memory Usage**: psutil-based memory profiling
5. **Accuracy**: Classification accuracy on test set
6. **End-to-End**: Full pipeline (validation + features + inference)

**When to use:**
- After training new models
- Before deployment
- Performance regression testing
- Optimization validation
- Release criteria verification

**Output:**
- Terminal report with color-coded pass/fail
- JSON benchmark report (`benchmark-report-YYYYMMDD-HHMMSS.json`)
- Comparison against production targets

---

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

### Initial Setup

Verify your environment is properly configured:

```bash
./scripts/verify_environment.sh
```

### Development Workflow

Run all quality checks before committing:

```bash
# 1. Verify environment (first time only)
./scripts/verify_environment.sh

# 2. Validate data quality (if working with datasets)
./scripts/validate_data_quality.sh

# 3. Run all tests
./scripts/run_all_tests.sh

# 4. Generate coverage report
./scripts/generate_coverage.sh

# 5. Format code
./scripts/format_code.sh

# 6. Check types
./scripts/type_check.sh

# 7. Security scan
./scripts/security_scan.sh

# 8. Benchmark performance (before release)
./scripts/benchmark_performance.sh

# 9. Commit if all pass
git add -A
git commit -m "your message"
git push
```

### Daily Development

Quick checks during development:

```bash
# Run specific test file
pytest tests/test_physics_validation.py -v

# Quick format check
./scripts/format_code.sh

# Run all tests
./scripts/run_all_tests.sh
```

## CI/CD Integration

These scripts are designed to be used in GitHub Actions:

```yaml
name: CI Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Verify environment
        run: ./scripts/verify_environment.sh

      - name: Run all tests
        run: ./scripts/run_all_tests.sh

      - name: Generate coverage
        run: ./scripts/generate_coverage.sh

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.json

  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Code formatting check
        run: ./scripts/format_code.sh --check

      - name: Type checking
        run: ./scripts/type_check.sh

      - name: Security scan
        run: ./scripts/security_scan.sh
```

## Configuration

### Coverage Configuration (.coveragerc)
- Source: ai-models/, fleet-integration/, data-generation/
- Omit: tests/, __pycache__/, venv/, build/
- Target: ≥80% coverage
- Output formats: HTML, JSON, terminal
- Exclude lines: pragma: no cover, if __name__

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

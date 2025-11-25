@echo off
REM ============================================================
REM Qwen3-1.7B Fine-Tuned Model Evaluation - Quick Start Script
REM ============================================================
REM
REM This script evaluates the fine-tuned model against spec targets:
REM - Perplexity (<30)
REM - Domain Accuracy (>85%)
REM - Korean Fluency (>90%)
REM - Response Relevance (>80%)
REM - BLEU/ROUGE scores
REM
REM Usage:
REM   run_evaluation.bat
REM
REM Prerequisites:
REM   - Training completed (LoRA adapters exist)
REM   - GPU recommended (CPU will be very slow)
REM
REM ============================================================

echo.
echo ============================================================
echo QWEN3-1.7B MODEL EVALUATION - QUICK START
echo ============================================================
echo.

REM Check if we're in the correct directory
if not exist "evaluate_qwen3.py" (
    echo [ERROR] evaluate_qwen3.py not found!
    echo Please run this script from: edgeai-repo\ai-models\fine-tuning\scripts\
    pause
    exit /b 1
)

REM Verify model exists
if not exist "..\models\qwen3-truck-lora\adapter_model.safetensors" (
    echo [ERROR] Fine-tuned model not found!
    echo Expected location: ..\models\qwen3-truck-lora\adapter_model.safetensors
    echo.
    echo Please run training first:
    echo   run_training.bat
    echo.
    pause
    exit /b 1
) else (
    echo [1/4] Fine-tuned model found
    dir ..\models\qwen3-truck-lora\adapter_model.safetensors | findstr "adapter_model.safetensors"
)

REM Verify test dataset exists
if not exist "..\datasets\test.jsonl" (
    echo [ERROR] Test dataset not found!
    echo Expected location: ..\datasets\test.jsonl
    pause
    exit /b 1
) else (
    echo [2/4] Test dataset found
    dir ..\datasets\test.jsonl | findstr "test.jsonl"
)

REM Verify Python environment
echo.
echo [3/4] Verifying Python environment...
python -c "import torch; from transformers import AutoModelForCausalLM; from peft import PeftModel; print('  [OK] All dependencies available')" 2>nul
if errorlevel 1 (
    echo   [ERROR] Required dependencies not installed!
    echo   Please install: pip install transformers peft torch
    pause
    exit /b 1
)

REM Create evaluation directory
if not exist "..\evaluation" (
    mkdir ..\evaluation
)

REM Start evaluation
echo.
echo [4/4] Starting evaluation...
echo.
echo ============================================================
echo EVALUATION STARTED
echo ============================================================
echo   Start time: %date% %time%
echo   Expected duration: 30-60 minutes
echo   Model: ..\models\qwen3-truck-lora\
echo   Test samples: 200 (configurable with --max-samples)
echo   Output: ..\evaluation\
echo ============================================================
echo.

REM Execute evaluation
python evaluate_qwen3.py ^
    --model-path ..\models\qwen3-truck-lora ^
    --base-model ..\base-models\qwen3-1.7b ^
    --test-data ..\datasets\test.jsonl ^
    --output-dir ..\evaluation ^
    --max-samples 200 ^
    --device cuda

REM Check results
if errorlevel 1 (
    echo.
    echo ============================================================
    echo [FAIL] Evaluation failed - Some specs not met
    echo ============================================================
    echo.
    echo The model did not meet all specification targets.
    echo Please review the evaluation report for details.
    echo.
) else (
    echo.
    echo ============================================================
    echo [PASS] Evaluation passed - All specs met!
    echo ============================================================
    echo.
)

echo Next steps:
echo   1. Review JSON results in ..\evaluation\
echo   2. Read Markdown report for detailed analysis
echo   3. If PASS: Proceed to deployment (PHASE3K_GPU_EXECUTION_GUIDE.md Section 5)
echo   4. If FAIL: Analyze failures and iterate (increase LoRA rank, add data, etc.)
echo.

REM Open evaluation directory
start ..\evaluation

pause

@echo off
REM ============================================================
REM Phase 3K: Environment Check Script
REM ============================================================
REM
REM This script verifies all prerequisites for Qwen3-1.7B fine-tuning:
REM - GPU availability and specs
REM - Python version
REM - PyTorch CUDA support
REM - Required dependencies
REM - Dataset files
REM - Base model files
REM
REM ============================================================

echo.
echo ============================================================
echo PHASE 3K: ENVIRONMENT CHECK
echo ============================================================
echo.

set ERROR_COUNT=0

REM ============================================================
REM Step 0: Working Directory
REM ============================================================
echo [Step 0] Checking working directory...
cd /d "%~dp0"
if not exist "train_qwen3_lora.py" (
    echo   [ERROR] train_qwen3_lora.py not found!
    echo   Current directory: %CD%
    echo   Please run from: edgeai-repo\ai-models\fine-tuning\scripts\
    set /a ERROR_COUNT+=1
) else (
    echo   [OK] Working directory correct
)
echo.

REM ============================================================
REM Step 1: GPU Check
REM ============================================================
echo [Step 1] Checking GPU availability...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo   [ERROR] NVIDIA GPU not detected or driver not installed!
    echo   Please install NVIDIA drivers.
    set /a ERROR_COUNT+=1
) else (
    echo   [OK] NVIDIA GPU detected
    echo   GPU Information:
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | findstr /N "."
)
echo.

REM ============================================================
REM Step 2: Python Version
REM ============================================================
echo [Step 2] Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo   [ERROR] Python not found in PATH!
    set /a ERROR_COUNT+=1
) else (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
    echo   [OK] Python %PYTHON_VERSION%
    echo   Checking version compatibility...
    python -c "import sys; v=sys.version_info; assert (v.major==3 and v.minor in [9,10,11]), 'Python 3.9-3.11 required'; print(f'  [OK] Version {v.major}.{v.minor}.{v.micro} is compatible')" 2>nul
    if errorlevel 1 (
        echo   [WARNING] Python 3.12+ may have compatibility issues with bitsandbytes
    )
)
echo.

REM ============================================================
REM Step 3: PyTorch CUDA
REM ============================================================
echo [Step 3] Checking PyTorch CUDA support...
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'  [OK] PyTorch {torch.__version__} with CUDA {torch.version.cuda}'); print(f'  [OK] GPU: {torch.cuda.get_device_name(0)}')" 2>nul
if errorlevel 1 (
    echo   [ERROR] PyTorch with CUDA not available!
    echo   Please install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    set /a ERROR_COUNT+=1
)
echo.

REM ============================================================
REM Step 4: Dependencies
REM ============================================================
echo [Step 4] Checking required dependencies...
python -c "from transformers import __version__ as tf_ver; from peft import __version__ as peft_ver; from datasets import __version__ as ds_ver; print(f'  [OK] transformers {tf_ver}'); print(f'  [OK] peft {peft_ver}'); print(f'  [OK] datasets {ds_ver}')" 2>nul
if errorlevel 1 (
    echo   [ERROR] Required dependencies not installed!
    echo   Please install: pip install transformers datasets peft accelerate bitsandbytes
    set /a ERROR_COUNT+=1
) else (
    python -c "import bitsandbytes; print(f'  [OK] bitsandbytes {bitsandbytes.__version__}')" 2>nul
    if errorlevel 1 (
        echo   [WARNING] bitsandbytes not found (required for 4-bit quantization)
    )
)
echo.

REM ============================================================
REM Step 5: Dataset Files
REM ============================================================
echo [Step 5] Checking dataset files...
if exist "..\datasets\train.jsonl" (
    for %%A in ("..\datasets\train.jsonl") do echo   [OK] train.jsonl (%%~zA bytes)
) else (
    echo   [ERROR] train.jsonl not found!
    echo   Expected: ..\datasets\train.jsonl
    set /a ERROR_COUNT+=1
)

if exist "..\datasets\test.jsonl" (
    for %%A in ("..\datasets\test.jsonl") do echo   [OK] test.jsonl (%%~zA bytes)
) else (
    echo   [ERROR] test.jsonl not found!
    echo   Expected: ..\datasets\test.jsonl
    set /a ERROR_COUNT+=1
)
echo.

REM ============================================================
REM Step 6: Base Model
REM ============================================================
echo [Step 6] Checking base model files...
if exist "..\base-models\qwen3-1.7b\model.safetensors" (
    for %%A in ("..\base-models\qwen3-1.7b\model.safetensors") do echo   [OK] model.safetensors (%%~zA bytes)
) else (
    echo   [ERROR] Base model not found!
    echo   Expected: ..\base-models\qwen3-1.7b\model.safetensors
    echo   Please download from Hugging Face:
    echo     cd ..\base-models
    echo     pip install huggingface-hub
    echo     python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-1.7B', local_dir='qwen3-1.7b')"
    set /a ERROR_COUNT+=1
)

if exist "..\base-models\qwen3-1.7b\config.json" (
    echo   [OK] config.json found
) else (
    echo   [ERROR] config.json not found!
    set /a ERROR_COUNT+=1
)

if exist "..\base-models\qwen3-1.7b\tokenizer.json" (
    echo   [OK] tokenizer.json found
) else (
    echo   [ERROR] tokenizer.json not found!
    set /a ERROR_COUNT+=1
)
echo.

REM ============================================================
REM Step 7: Output Directory
REM ============================================================
echo [Step 7] Checking output directory...
if not exist "..\models" (
    echo   [INFO] Creating models directory...
    mkdir "..\models"
    echo   [OK] Models directory created
) else (
    echo   [OK] Models directory exists
)
echo.

REM ============================================================
REM Summary
REM ============================================================
echo ============================================================
echo ENVIRONMENT CHECK SUMMARY
echo ============================================================
echo.

if %ERROR_COUNT% EQU 0 (
    echo   [SUCCESS] All checks passed! Ready for training.
    echo.
    echo   Next steps:
    echo     1. Run training: run_training.bat
    echo     2. Monitor GPU: nvidia-smi -l 5 (in separate terminal)
    echo     3. After training: run_evaluation.bat
    echo.
    exit /b 0
) else (
    echo   [FAIL] %ERROR_COUNT% error(s) found!
    echo.
    echo   Please fix the errors above before proceeding.
    echo.
    exit /b 1
)


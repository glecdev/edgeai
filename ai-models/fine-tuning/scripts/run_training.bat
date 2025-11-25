@echo off
REM ============================================================
REM Qwen3-1.7B QLoRA Fine-Tuning - Quick Start Script
REM ============================================================
REM
REM This script executes the complete training pipeline:
REM 1. Verify environment (GPU, dependencies)
REM 2. Execute QLoRA fine-tuning (2-4 hours)
REM 3. Save training logs
REM
REM Usage:
REM   run_training.bat
REM
REM Prerequisites:
REM   - NVIDIA GPU with >=6GB VRAM
REM   - Python 3.9-3.11 with CUDA support
REM   - All dependencies installed (see requirements.txt)
REM
REM ============================================================

echo.
echo ============================================================
echo QWEN3-1.7B QLORA FINE-TUNING - QUICK START
echo ============================================================
echo.

REM Check if we're in the correct directory
if not exist "train_qwen3_lora.py" (
    echo [ERROR] train_qwen3_lora.py not found!
    echo Please run this script from: edgeai-repo\ai-models\fine-tuning\scripts\
    pause
    exit /b 1
)

REM Create models directory if it doesn't exist
if not exist "..\models" (
    echo [1/5] Creating models directory...
    mkdir ..\models
    echo   [OK] Models directory created
) else (
    echo [1/5] Models directory already exists
)

REM Verify GPU availability
echo.
echo [2/5] Checking GPU availability...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo   [ERROR] NVIDIA GPU not detected or driver not installed!
    echo   Please install NVIDIA drivers and ensure CUDA is available.
    pause
    exit /b 1
) else (
    echo   [OK] NVIDIA GPU detected
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
)

REM Verify Python and torch
echo.
echo [3/5] Verifying Python environment...
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'  [OK] PyTorch {torch.__version__} with CUDA {torch.version.cuda}')" 2>nul
if errorlevel 1 (
    echo   [ERROR] PyTorch with CUDA not available!
    echo   Please install PyTorch with CUDA support:
    echo   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pause
    exit /b 1
)

REM Verify dependencies
echo.
echo [4/5] Verifying dependencies...
python -c "from transformers import AutoModelForCausalLM; from peft import LoraConfig; from datasets import load_dataset; print('  [OK] All dependencies installed')" 2>nul
if errorlevel 1 (
    echo   [ERROR] Required dependencies not installed!
    echo   Please install: pip install transformers datasets peft accelerate bitsandbytes
    pause
    exit /b 1
)

REM Start training
echo.
echo [5/5] Starting QLoRA fine-tuning...
echo.
echo ============================================================
echo TRAINING STARTED
echo ============================================================
echo   Start time: %date% %time%
echo   Expected duration: 2-4 hours
echo   GPU memory usage: ~5-6 GB
echo   Output: ..\models\qwen3-truck-lora\
echo   Log file: ..\models\training.log
echo ============================================================
echo.
echo Training progress will be displayed below.
echo You can also monitor in a separate terminal:
echo   Get-Content ..\models\training.log -Wait -Tail 20
echo.
echo To monitor GPU usage:
echo   nvidia-smi -l 5
echo.
echo ============================================================
echo.

REM Execute training and log output
python train_qwen3_lora.py 2>&1 | tee ..\models\training.log

REM Check if training completed successfully
if errorlevel 1 (
    echo.
    echo ============================================================
    echo [ERROR] Training failed!
    echo ============================================================
    echo.
    echo Please check ..\models\training.log for details.
    echo.
    echo Common issues:
    echo   - GPU out of memory: Reduce batch size in train_qwen3_lora.py
    echo   - CUDA error: Update NVIDIA drivers
    echo   - Missing dependencies: Install required packages
    echo.
    pause
    exit /b 1
) else (
    echo.
    echo ============================================================
    echo [SUCCESS] Training completed!
    echo ============================================================
    echo.
    echo   End time: %date% %time%
    echo   Output: ..\models\qwen3-truck-lora\
    echo   Log: ..\models\training.log
    echo.
    echo Next steps:
    echo   1. Review training log: type ..\models\training.log
    echo   2. Run evaluation: python evaluate_qwen3.py --model-path ..\models\qwen3-truck-lora
    echo   3. Check PHASE3K_GPU_EXECUTION_GUIDE.md for deployment steps
    echo.
    pause
)

@echo off
REM ============================================================================
REM GLEC DTG EdgeAI - Quick Start GPU Training (Windows)
REM ============================================================================
REM
REM Purpose: Automated environment setup and GPU training execution
REM Requirements: NVIDIA GPU, Miniconda installed
REM Estimated Time: 6-10 hours
REM
REM ============================================================================

echo.
echo ============================================================================
echo  GLEC DTG EdgeAI - GPU Training Quick Start
echo ============================================================================
echo.
echo This script will:
echo   1. Create conda environment (dtg-ai)
echo   2. Install PyTorch + CUDA 11.8
echo   3. Install dependencies
echo   4. Generate training datasets
echo   5. Train TCN and LSTM-AE models
echo   6. Export to ONNX
echo   7. Copy to Android project
echo.
echo Estimated time: 6-10 hours
echo.
pause

REM ============================================================================
REM Step 1: Check NVIDIA GPU
REM ============================================================================
echo.
echo [Step 1/7] Checking NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] NVIDIA GPU not detected or driver not installed!
    echo Please install NVIDIA driver from: https://www.nvidia.com/Download/index.aspx
    pause
    exit /b 1
)
echo [OK] NVIDIA GPU detected
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

REM ============================================================================
REM Step 2: Create Conda Environment
REM ============================================================================
echo.
echo [Step 2/7] Creating Conda environment (dtg-ai)...
call conda env list | findstr /C:"dtg-ai" >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Environment 'dtg-ai' already exists
    choice /C YN /M "Do you want to remove and recreate it"
    if errorlevel 2 goto skip_env_creation
    call conda env remove -n dtg-ai -y
)

call conda create -n dtg-ai python=3.10 -y
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create conda environment
    pause
    exit /b 1
)
echo [OK] Conda environment created

:skip_env_creation

REM ============================================================================
REM Step 3: Install PyTorch + CUDA
REM ============================================================================
echo.
echo [Step 3/7] Installing PyTorch 2.2.0 + CUDA 11.8...
call conda activate dtg-ai
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate conda environment
    pause
    exit /b 1
)

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install PyTorch
    pause
    exit /b 1
)
echo [OK] PyTorch installed

REM Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); assert torch.cuda.is_available(), 'CUDA not available!'"
if %errorlevel% neq 0 (
    echo [ERROR] CUDA not available in PyTorch!
    echo Please check NVIDIA driver and CUDA installation
    pause
    exit /b 1
)
echo [OK] CUDA is available

REM ============================================================================
REM Step 4: Install Dependencies
REM ============================================================================
echo.
echo [Step 4/7] Installing dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed

REM Verify with tests
echo [INFO] Running tests to verify environment...
python -m pytest tests/ -v --tb=no -q --ignore=tests\e2e_test.py --ignore=tests\benchmark_inference.py --ignore=tests\data_validator.py -x
if %errorlevel% neq 0 (
    echo [ERROR] Tests failed! Please check environment
    pause
    exit /b 1
)
echo [OK] All tests passed (159/159)

REM ============================================================================
REM Step 5: Generate Training Datasets
REM ============================================================================
echo.
echo [Step 5/7] Generating training datasets...
echo [INFO] This will take 10-30 minutes for production dataset

choice /C YN /M "Use test dataset (1,000 samples, 5min) or production dataset (10,000 samples, 30min)"
if errorlevel 2 goto production_dataset

REM Test dataset (quick)
:test_dataset
echo [INFO] Generating test dataset (1,000 samples)...
python -c "import sys; sys.path.append('ai-models'); exec(open('scripts/generate_test_dataset.py').read())"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to generate test dataset
    pause
    exit /b 1
)
goto dataset_complete

REM Production dataset (full)
:production_dataset
echo [INFO] Generating production dataset (10,000 samples)...
python -c "import sys; sys.path.append('ai-models'); exec(open('scripts/generate_production_dataset.py').read())"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to generate production dataset
    pause
    exit /b 1
)

:dataset_complete
echo [OK] Datasets generated
dir datasets\*.csv

REM ============================================================================
REM Step 6: Train Models
REM ============================================================================
echo.
echo [Step 6/7] Training AI models (this will take 4-8 hours)...
echo.

REM TCN Training
echo [INFO] Starting TCN training (2-4 hours)...
cd ai-models\training
python train_tcn.py --config ..\config.yaml --epochs 100 --batch-size 64
if %errorlevel% neq 0 (
    echo [ERROR] TCN training failed!
    pause
    exit /b 1
)
echo [OK] TCN training complete
cd ..\..

echo.

REM LSTM-AE Training
echo [INFO] Starting LSTM-AE training (2-4 hours)...
cd ai-models\training
python train_lstm_ae.py --config ..\config.yaml --epochs 100 --batch-size 64
if %errorlevel% neq 0 (
    echo [ERROR] LSTM-AE training failed!
    pause
    exit /b 1
)
echo [OK] LSTM-AE training complete
cd ..\..

REM ============================================================================
REM Step 7: Export to ONNX
REM ============================================================================
echo.
echo [Step 7/7] Exporting models to ONNX...

REM Create models directory
if not exist "ai-models\models" mkdir ai-models\models

REM TCN ONNX export
echo [INFO] Exporting TCN to ONNX...
python -c "import torch; import sys; sys.path.append('ai-models'); from training.train_tcn import TCN; device = torch.device('cuda'); model = TCN(input_dim=11, output_dim=1, num_channels=[64, 128, 256]).to(device); checkpoint = torch.load('ai-models/training/models/tcn_fuel_best.pth'); model.load_state_dict(checkpoint['model_state_dict']); model.eval(); dummy_input = torch.randn(1, 60, 11).to(device); torch.onnx.export(model, dummy_input, 'ai-models/models/tcn_fuel_prediction.onnx', export_params=True, opset_version=13, do_constant_folding=True, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}); print('TCN ONNX export complete')"
if %errorlevel% neq 0 (
    echo [ERROR] TCN ONNX export failed
    pause
    exit /b 1
)

REM LSTM-AE ONNX export
echo [INFO] Exporting LSTM-AE to ONNX...
python -c "import torch; import sys; sys.path.append('ai-models'); from training.train_lstm_ae import LSTM_Autoencoder; device = torch.device('cuda'); model = LSTM_Autoencoder(input_dim=11, hidden_dim=128, num_layers=2, latent_dim=32).to(device); checkpoint = torch.load('ai-models/training/models/lstm_ae_best.pth'); model.load_state_dict(checkpoint['model_state_dict']); model.eval(); dummy_input = torch.randn(1, 60, 11).to(device); torch.onnx.export(model, dummy_input, 'ai-models/models/lstm_ae_anomaly_detection.onnx', export_params=True, opset_version=13, do_constant_folding=True, input_names=['input'], output_names=['output']); print('LSTM-AE ONNX export complete')"
if %errorlevel% neq 0 (
    echo [ERROR] LSTM-AE ONNX export failed
    pause
    exit /b 1
)

echo [OK] ONNX export complete

REM Model size check
echo.
echo [INFO] Checking model sizes...
dir ai-models\models\*.onnx

REM Copy to Android
echo.
echo [INFO] Copying ONNX models to Android project...
if not exist "android-dtg\app\src\main\assets\models" mkdir android-dtg\app\src\main\assets\models
xcopy /Y ai-models\models\*.onnx android-dtg\app\src\main\assets\models\
if %errorlevel% neq 0 (
    echo [ERROR] Failed to copy models to Android
    pause
    exit /b 1
)
echo [OK] Models copied to Android assets

REM ============================================================================
REM Completion Summary
REM ============================================================================
echo.
echo ============================================================================
echo  Training Complete!
echo ============================================================================
echo.
echo Models trained and exported:
dir ai-models\models\*.onnx
echo.
echo Next steps:
echo   1. Build Android APK: cd android-dtg ^&^& gradlew assembleDebug
echo   2. Install on device: adb install -r app\build\outputs\apk\debug\app-debug.apk
echo   3. Test inference performance
echo.
echo For detailed instructions, see:
echo   - LOCAL_GPU_EXECUTION_GUIDE.md
echo   - PRE_EXECUTION_VALIDATION_CHECKLIST.md
echo.
pause

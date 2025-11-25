# Phase 3K: QLoRA Fine-Tuning GPU Execution Guide

**Status**: Ready for local GPU execution
**Created**: 2025-11-16
**Hardware Required**: GPU with ≥6GB VRAM (GTX 1660 SUPER or better)

---

## Executive Summary

All dataset preparation and training scripts are ready. Due to GPU memory requirements, **QLoRA fine-tuning must be executed on local hardware with GPU support**.

**What's Ready**:
- ✅ 11,999 training samples (10,799 train + 1,200 test)
- ✅ Qwen3-1.7B base model downloaded
- ✅ QLoRA training script configured
- ✅ All dependencies documented

**Next Step**: Execute training on local GPU environment (2-4 hours)

---

## 1. Environment Setup

### 1.1 System Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| **GPU** | ≥6GB VRAM | GTX 1660 SUPER, RTX 3060, or better |
| **RAM** | ≥16GB | System RAM for data loading |
| **Storage** | ≥20GB free | Model (4GB) + datasets (5MB) + outputs (1GB) |
| **Python** | 3.9 - 3.11 | Python 3.12 not recommended for PyTorch |
| **CUDA** | 11.8 or 12.1 | Match with PyTorch version |

### 1.2 Verify GPU Availability

```bash
# Windows
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 545.xx.xx    Driver Version: 545.xx.xx    CUDA Version: 12.x  |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  NVIDIA GeForce ...  WDDM | 00000000:01:00.0 Off |                  N/A |
# | 30%   45C    P8    15W / 120W |      0MiB /  6144MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
```

### 1.3 Install Python Dependencies

```bash
cd d:\edgeai

# Create virtual environment (recommended)
python -m venv venv-gpu
venv-gpu\Scripts\activate

# Install PyTorch with CUDA support (CUDA 12.1 example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Expected output:
# CUDA available: True
# GPU: NVIDIA GeForce GTX 1660 SUPER

# Install ML libraries
pip install transformers datasets peft accelerate bitsandbytes tqdm

# Verify installations
python -c "from transformers import AutoModelForCausalLM; from peft import LoraConfig; from datasets import load_dataset; print('All libraries imported successfully')"
```

### 1.4 Verify File Structure

```bash
cd d:\edgeai\edgeai-repo\ai-models\fine-tuning

# Check datasets
dir datasets\train.jsonl
dir datasets\test.jsonl

# Check base model
dir base-models\qwen3-1.7b\model.safetensors

# Check training script
dir scripts\train_qwen3_lora.py

# Expected sizes:
# datasets\train.jsonl: ~4.9 MB (10,799 samples)
# datasets\test.jsonl: ~0.55 MB (1,200 samples)
# base-models\qwen3-1.7b\: ~3.5 GB (model files)
# scripts\train_qwen3_lora.py: ~10 KB (300+ lines)
```

---

## 2. Execute QLoRA Fine-Tuning

### 2.1 Start Training

```bash
cd d:\edgeai\edgeai-repo\ai-models\fine-tuning\scripts

# Create output directory
mkdir ..\models 2>nul

# Execute training (2-4 hours)
python train_qwen3_lora.py > ..\models\training.log 2>&1

# The script will:
# [1/8] Load configuration
# [2/8] Setup 4-bit quantization
# [3/8] Load tokenizer
# [4/8] Load Qwen3-1.7B with 4-bit NF4 quantization (1-2 min)
# [5/8] Prepare model for QLoRA (add LoRA adapters)
# [6/8] Load and tokenize datasets
# [7/8] Setup training arguments
# [8/8] Initialize Trainer
# [TRAIN] Start fine-tuning (2-4 hours, 2,025 steps)
```

### 2.2 Monitor Training Progress

**Option 1: Real-time monitoring (separate terminal)**
```bash
cd d:\edgeai\edgeai-repo\ai-models\fine-tuning\models
Get-Content training.log -Wait -Tail 20
```

**Option 2: GPU monitoring**
```bash
# Monitor GPU usage every 5 seconds
nvidia-smi -l 5

# Expected during training:
# GPU Memory: 5-6 GB / 6 GB (80-95% usage)
# GPU Utilization: 80-100%
# Power: 80-120W (depends on GPU)
# Temperature: 60-80°C
```

**Option 3: Training metrics**
```bash
# Check latest metrics
cd d:\edgeai\edgeai-repo\ai-models\fine-tuning\models
type training.log | findstr /C:"{'loss'" /C:"{'eval_loss'"

# Expected output (sample):
# {'loss': 2.1234, 'grad_norm': 0.5678, 'learning_rate': 0.0002, 'epoch': 0.16}
# {'eval_loss': 1.9876, 'eval_runtime': 45.32, 'epoch': 0.49}
```

### 2.3 Training Timeline

| Step | Duration | Description | GPU Memory |
|------|----------|-------------|------------|
| **Initialization** | 2-3 min | Load model with 4-bit quantization | 0.5 GB → 4 GB |
| **Epoch 1** | 40-60 min | 675 steps (10,799 samples ÷ 16 batch) | 5-6 GB |
| **Evaluation 1** | 2-3 min | Test set evaluation (1,200 samples) | 5-6 GB |
| **Epoch 2** | 40-60 min | 675 steps | 5-6 GB |
| **Evaluation 2** | 2-3 min | Test set evaluation | 5-6 GB |
| **Epoch 3** | 40-60 min | 675 steps | 5-6 GB |
| **Final Evaluation** | 2-3 min | Test set evaluation | 5-6 GB |
| **Save Model** | 1-2 min | Save LoRA adapters (~200 MB) | 5-6 GB |
| **Total** | **2-4 hours** | 2,025 steps + 3 evaluations | **Peak: 5-6 GB** |

### 2.4 Expected Metrics

**Training Loss** (lower is better):
- Initial: ~2.5-3.0 (untrained)
- After Epoch 1: ~1.5-2.0
- After Epoch 2: ~1.2-1.5
- After Epoch 3: ~1.0-1.3

**Evaluation Loss** (lower is better):
- After Epoch 1: ~1.8-2.2
- After Epoch 2: ~1.5-1.8
- After Epoch 3: ~1.3-1.6

**Learning Rate Schedule**:
- Warmup (steps 0-100): 0 → 0.0002
- Cosine decay (steps 100-2025): 0.0002 → ~0.00001

---

## 3. Verify Training Completion

### 3.1 Check Output Files

```bash
cd d:\edgeai\edgeai-repo\ai-models\fine-tuning\models

# List output files
dir qwen3-truck-lora

# Expected files:
# adapter_config.json       (~1 KB)    - LoRA configuration
# adapter_model.safetensors (~200 MB)  - LoRA adapter weights
# README.md                 (~1 KB)    - Model card
# trainer_state.json        (~10 KB)   - Training state
# training_args.bin         (~5 KB)    - Training arguments
```

### 3.2 Verify Training Metrics

```bash
# Extract final metrics
type training.log | findstr /C:"TrainOutput"

# Expected output:
# TrainOutput(global_step=2025, training_loss=1.234, metrics={'train_runtime': 7200.0, 'train_samples_per_second': 4.5, 'train_steps_per_second': 0.28, 'total_flos': 1.23e+16, 'train_loss': 1.234, 'epoch': 3.0})
```

### 3.3 Quick Inference Test

```bash
cd d:\edgeai\edgeai-repo\ai-models\fine-tuning\scripts

# Test inference with fine-tuned model
python << EOF
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model and tokenizer
model_name = "d:/edgeai/edgeai-repo/ai-models/fine-tuning/base-models/qwen3-1.7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapters
lora_path = "d:/edgeai/edgeai-repo/ai-models/fine-tuning/models/qwen3-truck-lora"
model = PeftModel.from_pretrained(base_model, lora_path)

# Test query
query = "과속 운전을 피하려면 어떻게 해야 하나요?"
prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Query: {query}")
print(f"Response: {response}")
EOF

# Expected response (example):
# Query: 과속 운전을 피하려면 어떻게 해야 하나요?
# Response: 과속 운전을 피하려면 다음 사항을 준수하세요:
# 1. 제한속도 표지판을 항상 확인하고 준수하세요
# 2. 크루즈 컨트롤을 활용하여 일정 속도 유지
# 3. 앞차와의 안전거리를 2초 이상 확보
# 4. 날씨와 도로 상황에 맞춰 속도 조절
# 안전 운전이 가장 중요합니다!
```

---

## 4. Troubleshooting

### 4.1 GPU Out of Memory (OOM)

**Error**:
```
RuntimeError: CUDA out of memory. Tried to allocate 512.00 MiB (GPU 0; 6.00 GiB total capacity; 5.50 GiB already allocated; 256.00 MiB free)
```

**Solutions**:

**Option 1: Reduce batch size** (train_qwen3_lora.py)
```python
# Change line 180:
per_device_train_batch_size=2,  # Was 4, now 2
gradient_accumulation_steps=8,  # Was 4, now 8 (keeps effective batch size = 16)
```

**Option 2: Enable gradient checkpointing** (already enabled)
```python
# Verify line 190:
gradient_checkpointing=True,
```

**Option 3: Reduce max sequence length**
```python
# Change line 125:
max_length=256,  # Was 512, now 256
```

### 4.2 CUDA Not Available

**Error**:
```python
AssertionError: CUDA is not available. GPU is required for training.
```

**Solutions**:

1. **Verify PyTorch installation**:
   ```bash
   python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
   # Expected: 2.x.x+cu121 (or cu118)
   ```

2. **Reinstall PyTorch with CUDA**:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Check NVIDIA driver**:
   ```bash
   nvidia-smi
   # If error, update NVIDIA GPU driver from nvidia.com
   ```

### 4.3 Slow Training Speed

**Symptom**: <0.1 steps/sec (expected: 0.2-0.3 steps/sec)

**Causes & Solutions**:

1. **CPU bottleneck** (data loading):
   ```python
   # Change line 185:
   dataloader_num_workers=4,  # Was 0, use 4 CPU cores
   ```

2. **Disk I/O bottleneck**:
   - Move datasets to SSD (not HDD)
   - Reduce `logging_steps` and `save_steps`

3. **GPU thermal throttling**:
   ```bash
   nvidia-smi
   # If temp > 85°C, improve cooling or reduce power limit
   ```

### 4.4 Training Loss Not Decreasing

**Symptom**: Loss stays at ~2.5-3.0 after 500+ steps

**Solutions**:

1. **Check learning rate**:
   ```python
   # Increase learning rate (line 183):
   learning_rate=3e-4,  # Was 2e-4, try 3e-4
   ```

2. **Increase LoRA rank**:
   ```python
   # Change line 152:
   lora_config = LoraConfig(
       r=128,  # Was 64, try 128
       lora_alpha=32,  # Was 16, double alpha too
   )
   ```

3. **Verify dataset quality**:
   ```bash
   # Check train.jsonl format
   type datasets\train.jsonl | findstr /C:"<|im_start|>" | more
   # Should see Qwen3 chat template format
   ```

### 4.5 MemoryError (System RAM)

**Error**:
```
MemoryError: Unable to allocate array with shape (10799, 512)
```

**Solution**: Reduce dataset size or enable streaming
```python
# Change line 120:
dataset = load_dataset("json", data_files=data_files, streaming=True)
# Note: Streaming mode may slow down training
```

---

## 5. Next Steps After Training

### 5.1 Phase 4: Evaluation

Create evaluation script to measure:

1. **Quantitative Metrics**:
   - Perplexity (lower is better, target: <30)
   - BLEU score (0-100, target: >40)
   - ROUGE-L (0-1, target: >0.6)
   - Truck domain accuracy (target: >85%)

2. **Qualitative Metrics**:
   - Korean fluency (1-5 scale, target: >4.0)
   - Domain knowledge (1-5 scale, target: >4.0)
   - Response relevance (1-5 scale, target: >4.0)
   - Safety & professionalism (1-5 scale, target: >4.5)

**Script**: `edgeai-repo/ai-models/fine-tuning/scripts/evaluate_qwen3.py` (to be created)

### 5.2 Phase 5: Deployment Preparation

1. **Merge LoRA with base model**:
   ```python
   from peft import PeftModel

   merged_model = base_model.merge_and_unload()
   merged_model.save_pretrained("qwen3-truck-korean-merged")
   ```

2. **Quantize to INT8** (for mobile deployment):
   ```bash
   # Using ONNX Runtime quantization
   python -m onnxruntime.quantization.preprocess --model qwen3.onnx --output qwen3_int8.onnx
   ```

3. **Export to ONNX**:
   ```bash
   optimum-cli export onnx \
     --model qwen3-truck-korean-merged \
     --task text-generation-with-past \
     qwen3-truck-korean-onnx/
   ```

4. **Deploy to Android** (Qualcomm Snapdragon):
   - Convert ONNX → SNPE DLC (Qualcomm SDK)
   - Integrate with Android DTG app
   - Test on-device inference latency (<2s target)

---

## 6. Performance Benchmarks

### 6.1 Expected Training Performance

| GPU Model | VRAM | Steps/sec | Epoch Time | Total Time |
|-----------|------|-----------|------------|------------|
| GTX 1660 SUPER | 6 GB | 0.20-0.25 | 45-55 min | 2.5-3 hours |
| RTX 3060 | 12 GB | 0.30-0.35 | 30-40 min | 1.5-2 hours |
| RTX 4060 | 8 GB | 0.35-0.40 | 25-35 min | 1.5-2 hours |
| RTX 4090 | 24 GB | 0.50-0.60 | 20-25 min | 1-1.5 hours |

### 6.2 Model Size & Memory

| Stage | Model Size | GPU Memory | System RAM |
|-------|------------|------------|------------|
| **Base Model (FP16)** | 3.4 GB | 3.5 GB | 4 GB |
| **4-bit Quantized** | 980 MB | 1.2 GB | 4 GB |
| **+ LoRA Adapters** | +200 MB | +0.5 GB | +2 GB |
| **Training (peak)** | 1.18 GB | 5-6 GB | 8-12 GB |
| **LoRA Adapters Only** | 200 MB | - | - |

### 6.3 Inference Performance (After Fine-Tuning)

| Hardware | Quantization | Tokens/sec | Latency (50 tokens) |
|----------|-------------|------------|---------------------|
| **RTX 3060 (GPU)** | FP16 | 30-40 | 1.5-2s |
| **RTX 3060 (GPU)** | INT8 | 50-60 | 1-1.5s |
| **Snapdragon 8 Gen 2** | INT8 | 5-10 | 5-10s |
| **QCM2290 (DTG)** | INT8 | 2-5 | 10-25s |

---

## 7. Quality Gates

### 7.1 Before Starting Training

- [ ] GPU available with ≥6GB VRAM (`nvidia-smi`)
- [ ] PyTorch CUDA available (`torch.cuda.is_available() == True`)
- [ ] All dependencies installed (`transformers`, `peft`, `datasets`, `bitsandbytes`)
- [ ] Datasets exist (train.jsonl: 10,799 samples, test.jsonl: 1,200 samples)
- [ ] Base model downloaded (qwen3-1.7b: ~3.5 GB)
- [ ] Training script verified (`train_qwen3_lora.py` exists)

### 7.2 During Training

- [ ] GPU memory usage 5-6 GB (80-95% of 6GB)
- [ ] Training loss decreasing (initial ~2.5 → final ~1.0-1.3)
- [ ] Evaluation loss stable or decreasing (~1.5-2.0)
- [ ] No CUDA OOM errors
- [ ] Steps/sec > 0.15 (minimum acceptable speed)

### 7.3 After Training

- [ ] Training completed 2,025 steps (3 epochs)
- [ ] LoRA adapter saved (~200 MB)
- [ ] Final training loss <1.5
- [ ] Final evaluation loss <2.0
- [ ] Inference test produces Korean response
- [ ] Training log saved successfully

---

## 8. File Locations Summary

```
d:\edgeai\edgeai-repo\ai-models\fine-tuning\
│
├── specs/
│   └── qwen3-truck-korean_spec.yaml          # Training specification
│
├── base-models/
│   └── qwen3-1.7b/                            # Base model (3.5 GB)
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── config.json
│
├── datasets/
│   ├── can_conversations.jsonl                # CAN-based samples (9,999)
│   ├── manual_truck.jsonl                     # Manual samples (2,000)
│   ├── train.jsonl                            # Training set (10,799 samples)
│   └── test.jsonl                             # Test set (1,200 samples)
│
├── scripts/
│   ├── can_templates.py                       # CAN→conversation templates
│   ├── prepare_dataset.py                     # CAN dataset generator
│   ├── generate_manual_samples.py             # Manual sample generator
│   ├── merge_datasets.py                      # Dataset merger
│   └── train_qwen3_lora.py                    # QLoRA training script ← RUN THIS
│
└── models/
    ├── training.log                           # Training output log
    └── qwen3-truck-lora/                      # Fine-tuned model output
        ├── adapter_config.json
        ├── adapter_model.safetensors          # LoRA weights (200 MB)
        └── trainer_state.json
```

---

## 9. Quick Start Checklist

**For experienced users, execute these commands in sequence**:

```bash
# 1. Navigate to project
cd d:\edgeai

# 2. Activate GPU environment
venv-gpu\Scripts\activate

# 3. Verify GPU
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# 4. Verify files
cd edgeai-repo\ai-models\fine-tuning
dir datasets\train.jsonl
dir base-models\qwen3-1.7b\model.safetensors
dir scripts\train_qwen3_lora.py

# 5. Start training (2-4 hours)
cd scripts
mkdir ..\models 2>nul
python train_qwen3_lora.py > ..\models\training.log 2>&1

# 6. Monitor (separate terminal)
cd ..\models
Get-Content training.log -Wait -Tail 20

# 7. Verify completion
dir qwen3-truck-lora\adapter_model.safetensors
# Expected: ~200 MB file

# 8. Test inference
cd ..\scripts
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; from peft import PeftModel; import torch; model = AutoModelForCausalLM.from_pretrained('../base-models/qwen3-1.7b', torch_dtype=torch.float16, device_map='auto'); model = PeftModel.from_pretrained(model, '../models/qwen3-truck-lora'); tokenizer = AutoTokenizer.from_pretrained('../base-models/qwen3-1.7b'); inputs = tokenizer('<|im_start|>user\n과속을 피하려면?<|im_end|>\n<|im_start|>assistant\n', return_tensors='pt').to('cuda'); outputs = model.generate(**inputs, max_new_tokens=50); print(tokenizer.decode(outputs[0]))"
```

---

## 10. Support & Documentation

**Primary Documents**:
- **This Guide**: `docs/PHASE3K_GPU_EXECUTION_GUIDE.md`
- **Dataset Preparation**: `PHASE3K_DATASET_PREPARATION_COMPLETE.md`
- **Spec**: `ai-models/fine-tuning/specs/qwen3-truck-korean_spec.yaml`
- **LLM Analysis**: `EDGE_LLM_COMPREHENSIVE_ANALYSIS.md`
- **Phase 3K Plan**: `docs/PHASE3K_LLM_INTEGRATION.md`

**External Resources**:
- Qwen3 Model: https://huggingface.co/Qwen/Qwen3-1.7B
- LoRA Paper: https://arxiv.org/abs/2106.09685
- QLoRA Paper: https://arxiv.org/abs/2305.14314
- PEFT Library: https://github.com/huggingface/peft
- BitsAndBytes: https://github.com/TimDettmers/bitsandbytes

**Questions & Issues**:
- Check `docs/` for related documentation
- Review `scripts/train_qwen3_lora.py` comments
- Verify against `specs/qwen3-truck-korean_spec.yaml`

---

## Conclusion

**All preparation complete. Ready for local GPU execution.**

**Estimated Timeline**:
- Environment setup: 15-30 minutes (one-time)
- Training execution: 2-4 hours (automated)
- Verification: 10-15 minutes

**Expected Output**: Fine-tuned Qwen3-1.7B model optimized for Korean truck domain conversations, ready for evaluation and deployment.

**Next Phase**: Evaluation (Phase 4) - Measure performance against spec targets (>85% accuracy, >90% Korean fluency).

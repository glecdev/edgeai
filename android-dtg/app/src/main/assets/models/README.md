# LLM Model Assets

## Model: Qwen2.5-0.5B-Instruct (Q4_K_M GGUF)

This directory contains the quantized LLM model for on-device inference.

### Model Details

| Property | Value |
|----------|-------|
| Name | qwen2.5-0.5b-instruct-q4_k_m.gguf |
| Size | 468.6 MB |
| Quantization | Q4_K_M (4-bit) |
| Format | GGUF |
| Parameters | 0.49B |

### Performance (Desktop GTX 1660 SUPER)

| Metric | Value |
|--------|-------|
| Avg Latency | 1422 ms |
| Token Speed | 33.0 tok/s |

### Download Instructions

Due to the large file size, the model is not included in the repository.

**Option 1: Download from Hugging Face**
```bash
# Using huggingface-cli
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF \
  qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --local-dir ./app/src/main/assets/models/
```

**Option 2: Copy from local models folder**
```bash
# Windows
copy D:\edgeai\models\qwen2.5-0.5b-instruct-q4_k_m.gguf ^
  app\src\main\assets\models\

# Linux/Mac
cp ~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  app/src/main/assets/models/
```

**Option 3: Download at runtime**
The app will automatically download the model on first launch if not present.

### Integration Notes

1. The model will be loaded by `Qwen25InferenceEngine.kt`
2. Inference uses llama.cpp via JNI
3. Target device: Qualcomm QCM2290 (2GB RAM)

### Expected Performance (QCM2290)

| Metric | Projected Value |
|--------|-----------------|
| Token Speed | ~5 tok/s |
| Latency (50 tokens) | ~10s |
| RAM Usage | ~600 MB |

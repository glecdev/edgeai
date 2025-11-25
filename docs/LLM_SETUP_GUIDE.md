# LLM Setup Guide - Qwen2.5 Integration

**Target**: Qwen2.5-0.5B-Instruct (INT4) on Android
**Runtime**: MLC-LLM (Machine Learning Compilation)
**Hardware**: GLEC DTG (ARM Cortex-A53, 2GB RAM)

---

## 📋 Prerequisites

### System Requirements

**Development Machine**:
- OS: Ubuntu 20.04+ / macOS 12+ / Windows 11 (WSL2)
- RAM: 16 GB minimum
- Storage: 50 GB free
- GPU: NVIDIA (CUDA 11.8+) optional for fine-tuning

**Android Device**:
- OS: Android 8.0+ (API 26+)
- RAM: 2 GB minimum
- Storage: 2 GB free
- CPU: ARM64-v8a (64-bit)

---

## 🔧 Environment Setup

### Step 1: Python Environment

```bash
# Create virtual environment
python3.10 -m venv venv_llm
source venv_llm/bin/activate  # Linux/macOS
# or
venv_llm\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install core dependencies
pip install mlc-llm==0.1.0
pip install mlc-ai-nightly-cu118  # CUDA 11.8 (if GPU available)
# or
pip install mlc-ai-nightly  # CPU-only

# Verify installation
python -c "import mlc_llm; print(mlc_llm.__version__)"
```

**Expected Output**:
```
0.1.0
```

---

### Step 2: Android Development Environment

#### Install Android Studio

1. Download Android Studio Hedgehog (2023.1.1+)
2. Install Android SDK:
   - Android API 26 (Oreo) minimum
   - Android API 33 (Tiramisu) recommended
3. Install Android NDK:
   ```bash
   # Via SDK Manager
   Tools → SDK Manager → SDK Tools → NDK (Side by side)
   # Install version: 26.1.10909125
   ```

4. Set environment variables:
   ```bash
   # ~/.bashrc or ~/.zshrc
   export ANDROID_HOME=$HOME/Android/Sdk
   export ANDROID_NDK=$ANDROID_HOME/ndk/26.1.10909125
   export PATH=$PATH:$ANDROID_HOME/platform-tools
   export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin
   ```

#### Verify Android Setup

```bash
# Check Android SDK
adb version
# Expected: Android Debug Bridge version 1.0.41+

# Check NDK
ls $ANDROID_NDK/
# Expected: build/ meta/ ndk-build* ...

# Check connected device
adb devices
# Expected: device serial number + "device"
```

---

### Step 3: Model Download & Preparation

```bash
# Create model directory
mkdir -p models/qwen2.5-0.5b

# Download from Hugging Face
pip install huggingface-hub

python << EOF
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    local_dir="./models/qwen2.5-0.5b",
    local_dir_use_symlinks=False
)
EOF
```

**Expected Files**:
```
models/qwen2.5-0.5b/
├── config.json
├── generation_config.json
├── model.safetensors.index.json
├── model-00001-of-00002.safetensors  (~490MB)
├── model-00002-of-00002.safetensors  (~490MB)
├── tokenizer.json
├── tokenizer_config.json
└── vocab.json
```

---

### Step 4: Model Quantization (INT4)

```bash
# Convert to MLC format with INT4 quantization
mlc_llm convert_weight \
  --model ./models/qwen2.5-0.5b \
  --quantization q4f16_1 \
  --output ./dist/Qwen2.5-0.5B-q4f16_1 \
  --device cpu

# Expected duration: 30-60 minutes (CPU)
```

**Quantization Details**:
- **q4f16_1**: 4-bit weights, FP16 activations, group size 1
- **Compression**: 980MB → ~300MB (3.3x)
- **Accuracy loss**: <1% (negligible for chat)

**Expected Output**:
```
dist/Qwen2.5-0.5B-q4f16_1/
├── params_shard_0.bin  (~150MB)
├── params_shard_1.bin  (~150MB)
├── mlc-chat-config.json
└── tokenizer/
```

---

### Step 5: Model Validation

```bash
# Test inference (CPU)
mlc_llm chat \
  --model ./dist/Qwen2.5-0.5B-q4f16_1 \
  --device cpu

# Enter prompt:
# > 안녕하세요, 저는 화물차 운전자입니다. 연비를 개선하려면 어떻게 해야 하나요?
```

**Expected Behavior**:
- Response time: 5-10 seconds (CPU, first run)
- Response quality: Coherent Korean text
- No errors/warnings

---

## 🤖 Android Build Setup

### Step 1: Clone & Setup Android Project

```bash
cd android-dtg

# Add MLC-LLM library
mkdir -p app/libs
# Download prebuilt AAR (or build from source)
wget https://github.com/mlc-ai/mlc-llm/releases/download/v0.1.0/mlc_llm_android.aar \
  -O app/libs/mlc_llm_android.aar
```

---

### Step 2: Update build.gradle

```gradle
// app/build.gradle
android {
    ...
    defaultConfig {
        ...
        ndk {
            abiFilters 'arm64-v8a'  // ARM 64-bit only
        }
    }

    packagingOptions {
        pickFirst 'lib/arm64-v8a/libc++_shared.so'
    }
}

dependencies {
    ...
    // MLC-LLM library
    implementation files('libs/mlc_llm_android.aar')

    // Coroutines (for async inference)
    implementation "org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3"
}
```

---

### Step 3: Copy Model to Android Assets

```bash
# Create assets directory
mkdir -p android-dtg/app/src/main/assets/models

# Copy INT4 model
cp -r dist/Qwen2.5-0.5B-q4f16_1 \
  android-dtg/app/src/main/assets/models/

# Compress if needed (optional, reduces APK size)
cd android-dtg/app/src/main/assets/models
zip -r qwen25_0.5b_q4.zip Qwen2.5-0.5B-q4f16_1/
rm -rf Qwen2.5-0.5B-q4f16_1/  # Keep only ZIP
```

**APK Size Impact**:
- Uncompressed: +300MB
- Compressed (ZIP): +250MB (17% reduction)

---

### Step 4: Build Android APK

```bash
cd android-dtg

# Clean previous builds
./gradlew clean

# Build debug APK
./gradlew assembleDebug

# Expected output:
# app/build/outputs/apk/debug/app-debug.apk (~350MB with model)
```

---

### Step 5: Install & Test on Device

```bash
# Install APK
adb install -r app/build/outputs/apk/debug/app-debug.apk

# Check logs
adb logcat | grep -E "Qwen25|LLM|MLC"

# Test voice command (manual)
# 1. Open app
# 2. Say "헤이 드라이버"
# 3. Ask "오늘 운행이 어땠나요?"
# 4. Check response latency (<3초)
```

---

## 🧪 Dependencies & Versions

### Python Dependencies

```txt
# requirements_llm.txt
mlc-llm==0.1.0
mlc-ai-nightly-cu118==0.15.0  # or mlc-ai-nightly for CPU
torch==2.1.0
transformers==4.36.0
huggingface-hub==0.19.4
numpy==1.24.3
sentencepiece==0.1.99
```

Install:
```bash
pip install -r requirements_llm.txt
```

---

### Android Dependencies

```gradle
// build.gradle (project)
buildscript {
    ext.kotlin_version = "1.9.20"
    ext.coroutines_version = "1.7.3"

    repositories {
        google()
        mavenCentral()
    }

    dependencies {
        classpath 'com.android.tools.build:gradle:8.2.0'
        classpath "org.jetbrains.kotlin:kotlin-gradle-plugin:$kotlin_version"
    }
}

// build.gradle (app)
dependencies {
    // Kotlin
    implementation "org.jetbrains.kotlin:kotlin-stdlib:$kotlin_version"
    implementation "org.jetbrains.kotlinx:kotlinx-coroutines-android:$coroutines_version"

    // MLC-LLM
    implementation files('libs/mlc_llm_android.aar')

    // AndroidX
    implementation 'androidx.core:core-ktx:1.12.0'
    implementation 'androidx.lifecycle:lifecycle-runtime-ktx:2.6.2'

    // Logging
    implementation 'com.jakewharton.timber:timber:5.0.1'
}
```

---

## 🔍 Troubleshooting

### Issue 1: Model Download Fails

**Error**:
```
HTTPError: 401 Unauthorized
```

**Solution**:
```bash
# Login to Hugging Face
huggingface-cli login

# Enter your token from: https://huggingface.co/settings/tokens
```

---

### Issue 2: Quantization OOM (CPU)

**Error**:
```
RuntimeError: out of memory during quantization
```

**Solution**:
```bash
# Reduce batch size
mlc_llm convert_weight \
  --model ./models/qwen2.5-0.5b \
  --quantization q4f16_1 \
  --output ./dist/Qwen2.5-0.5B-q4f16_1 \
  --device cpu \
  --max-batch-size 1  # ← Add this

# Or use swap (Linux)
sudo dd if=/dev/zero of=/swapfile bs=1G count=16
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

### Issue 3: Android Build NDK Not Found

**Error**:
```
NDK not configured
```

**Solution**:
```bash
# Set NDK path in local.properties
echo "ndk.dir=/path/to/Android/Sdk/ndk/26.1.10909125" >> local.properties

# Verify
cat local.properties
```

---

### Issue 4: MLC-LLM Runtime Error on Android

**Error**:
```
java.lang.UnsatisfiedLinkError: dlopen failed: library "libtvm_runtime.so" not found
```

**Solution**:
```gradle
// app/build.gradle
android {
    packagingOptions {
        pickFirst 'lib/arm64-v8a/libc++_shared.so'
        pickFirst 'lib/arm64-v8a/libtvm_runtime.so'  // ← Add this
    }
}
```

---

### Issue 5: Inference Hangs on Device

**Symptoms**:
- LLM inference never completes
- Logcat shows no errors
- CPU usage low

**Solution**:
```kotlin
// Add timeout
withTimeout(5000) {  // 5 seconds
    llm.inference(query, context)
}

// Check model path
val modelPath = File(context.filesDir, "models/Qwen2.5-0.5B-q4f16_1")
if (!modelPath.exists()) {
    throw IllegalStateException("Model not found: ${modelPath.absolutePath}")
}
```

---

## 📊 Performance Benchmarks

### Expected Performance (GLEC DTG)

| Metric | Target | Typical | Worst Case |
|--------|--------|---------|------------|
| First token latency | <500ms | 400ms | 600ms |
| Subsequent tokens | 200ms/token | 250ms/token | 300ms/token |
| Total (20 tokens) | <3s | 2.3s | 3.5s |
| Memory usage | <1.2GB | 1.1GB | 1.15GB |
| Power consumption | <2W | 1.7W | 2.1W |

### Optimization Tips

1. **Reduce prompt size**:
   ```kotlin
   // ❌ Bad: Long prompt
   val prompt = """
   System: You are an AI assistant...
   Context: ${vehicleData.toDetailedString()}  // 500+ tokens
   User: $query
   """

   // ✅ Good: Concise prompt
   val prompt = """
   System: 화물차 AI 어시스턴트
   Context: 중량 ${vehicleData.weight}kg, 연비 ${vehicleData.fuelEff}km/L
   User: $query
   """
   ```

2. **Cache context**:
   ```kotlin
   // Cache unchanged context (reuse across queries)
   private var cachedContextTokens: IntArray? = null

   fun buildPrompt(query: String): IntArray {
       if (cachedContextTokens == null) {
           cachedContextTokens = tokenizer.encode(systemPrompt + context)
       }
       val queryTokens = tokenizer.encode(query)
       return cachedContextTokens!! + queryTokens
   }
   ```

3. **Use streaming**:
   ```kotlin
   // Start TTS as soon as first token arrives
   llm.inferenceStreaming(query, context).collect { token ->
       if (buffer.size > 10) {  // Wait for 10 tokens
           kokoro.synthesize(buffer.toString())
           buffer.clear()
       }
       buffer.append(token)
   }
   ```

---

## 🎓 Advanced Configuration

### Custom Quantization (Optional)

```python
# For advanced users: custom quantization config
from mlc_llm import quantization

config = quantization.QuantizeConfig(
    mode="int4",  # 4-bit weights
    group_size=128,  # Larger = faster but less accurate
    sym=True,  # Symmetric quantization
    calibration_samples=512  # More = better calibration
)

quantization.quantize(
    model_path="./models/qwen2.5-0.5b",
    output_path="./dist/qwen25_custom",
    config=config
)
```

### Model Pruning (Optional)

```python
# Reduce model size further (10-20%)
from torch.nn.utils import prune

# Prune 20% of least important weights
for module in model.modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, 'weight', amount=0.2)

# Fine-tune after pruning (1-2 epochs to recover accuracy)
```

---

## 📚 Additional Resources

### Official Documentation

- **MLC-LLM**: https://mlc.ai/mlc-llm/docs/
- **Qwen2.5**: https://github.com/QwenLM/Qwen2.5
- **Android NDK**: https://developer.android.com/ndk/guides

### Community Resources

- **MLC-LLM Discord**: https://discord.gg/mlc-ai
- **Qwen GitHub Issues**: https://github.com/QwenLM/Qwen2.5/issues
- **GLEC DTG Forum**: (internal)

---

## ✅ Setup Checklist

### Pre-Implementation

- [ ] Python 3.10+ installed
- [ ] Android Studio + NDK 26 installed
- [ ] GPU (CUDA 11.8+) available (optional)
- [ ] 50GB free storage

### Model Preparation

- [ ] Qwen2.5-0.5B downloaded (980MB)
- [ ] INT4 quantization completed (300MB)
- [ ] Model validation passed (inference test)

### Android Setup

- [ ] MLC-LLM AAR added to project
- [ ] Model copied to assets/models
- [ ] APK built successfully (<500MB)
- [ ] Device installation successful

### Testing

- [ ] Basic inference test (1 query)
- [ ] Latency measurement (<3초)
- [ ] Memory profiling (<1.2GB)
- [ ] 24-hour stability test

---

**Document Version**: 1.0
**Last Updated**: 2025-01-14
**Status**: Ready for Use
**Next**: Proceed to [LLM_IMPLEMENTATION_GUIDE.md](LLM_IMPLEMENTATION_GUIDE.md)

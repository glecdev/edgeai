# LightGBM Android Integration Research

**Date**: 2025-11-10
**Author**: Claude
**Purpose**: Research deployment strategies for LightGBM behavior classification model on Android

---

## Executive Summary

Researched 5 integration approaches for deploying our **99.54% accurate, 0.064ms P95, 22KB LightGBM model** to Android. Recommended approach: **ONNX → TFLite** conversion for optimal mobile performance and hardware acceleration.

---

## Current State

### Trained Model Performance
- **Accuracy**: 99.54% (test), 96.92% (validation)
- **F1-Score**: 99.30%
- **Latency**: 0.064ms P95 (234x faster than target)
- **Model Size**: 22KB (456x smaller than target)
- **Format**: LightGBM text model (`lightgbm_behavior.txt`)

### Android App Architecture
- **ModelManager.kt** (line 440-451): Placeholder `loadLightGBMModel()` method exists
- **Runtime constant**: `RUNTIME_LIGHTGBM = "lightgbm"` defined
- **Build system**: Gradle Kotlin DSL, Min SDK 29 (Android 10), arm64-v8a
- **Existing JNI**: Already configured for SNPE native libraries

---

## Integration Options Analysis

### Option 1: ONNX → TFLite Conversion ⭐ **RECOMMENDED**

**Approach**: LightGBM → ONNX → TensorFlow → TFLite → Android

**Advantages**:
- ✅ **Best mobile optimization**: TFLite designed for mobile-first (size, latency, power)
- ✅ **Hardware acceleration**: NNAPI, GPU, EdgeTPU support via TFLite
- ✅ **Unified inference engine**: Reuse same TFLite runtime for TCN/LSTM-AE
- ✅ **Proven conversion path**: `onnxmltools` → `onnx2tf` → TFLite
- ✅ **Quantization support**: INT8/FP16 post-training quantization
- ✅ **Android ecosystem**: First-class Android support from Google

**Disadvantages**:
- ⚠️ Conversion complexity: 3-step process (LightGBM → ONNX → TF → TFLite)
- ⚠️ Potential operator incompatibility during conversion
- ⚠️ Model size may increase (~22KB → ~50-100KB after TFLite conversion)

**Implementation Steps**:
```bash
# Step 1: LightGBM → ONNX
pip install onnxmltools
python
>>> import onnxmltools
>>> import lightgbm as lgb
>>> model = lgb.Booster(model_file='lightgbm_behavior.txt')
>>> onnx_model = onnxmltools.convert_lightgbm(model,
...     initial_types=[('input', FloatTensorType([None, 18]))],
...     target_opset=13)
>>> onnxmltools.utils.save_model(onnx_model, 'lightgbm_behavior.onnx')

# Step 2: ONNX → TensorFlow
pip install onnx2tf
onnx2tf -i lightgbm_behavior.onnx -o ./tf_model

# Step 3: TensorFlow → TFLite with INT8 quantization
python
>>> import tensorflow as tf
>>> converter = tf.lite.TFLiteConverter.from_saved_model('./tf_model')
>>> converter.optimizations = [tf.lite.Optimize.DEFAULT]
>>> tflite_model = converter.convert()
>>> with open('lightgbm_behavior.tflite', 'wb') as f:
...     f.write(tflite_model)

# Step 4: Deploy to Android (reuse existing TFLite engine)
```

**Android Integration**:
```kotlin
// Add TFLite dependency to build.gradle.kts
implementation("org.tensorflow:tensorflow-lite:2.14.0")
implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
implementation("org.tensorflow:tensorflow-lite-gpu:2.14.0")  // Optional: GPU acceleration

// Inference code (similar to TCN/LSTM-AE)
class LightGBMTFLiteEngine(context: Context) {
    private val interpreter: Interpreter

    init {
        val model = loadModelFile(context, "lightgbm_behavior.tflite")
        val options = Interpreter.Options().apply {
            setNumThreads(4)
            // Enable NNAPI delegate for hardware acceleration
            addDelegate(NnApiDelegate())
        }
        interpreter = Interpreter(model, options)
    }

    fun predict(features: FloatArray): Int {
        val output = Array(1) { FloatArray(3) }  // 3 classes
        interpreter.run(features, output)
        return output[0].indexOfMax()
    }
}
```

**Estimated Performance**:
- Latency: ~0.5-2ms (slight increase from ONNX conversion overhead)
- Model Size: ~50-100KB (TFLite format overhead)
- Still meets targets: <15ms latency, <10MB size

---

### Option 2: lightgbm4j (Metarank) Library

**Approach**: Use `io.github.metarank:lightgbm4j` Java library on Android

**GitHub**: https://github.com/metarank/lightgbm4j
**Maven**: `io.github.metarank:lightgbm4j:4.6.0-2`

**Advantages**:
- ✅ **Zero dependencies**: Self-contained JAR with bundled native libraries
- ✅ **Direct LightGBM API**: 1:1 mapping to native LightGBM methods
- ✅ **Optimized inference**: `predictForMatSingleRowFast()` for low latency
- ✅ **Tested platforms**: Linux, Windows, macOS, Mac M1
- ✅ **Memory management**: Auto-cleanup of JNI resources

**Disadvantages**:
- ❌ **No official Android support**: No Android binaries or documentation
- ❌ **Native library distribution**: Requires bundling `lib_lightgbm.so` and `lib_lightgbm_swig.so`
- ❌ **Large dependency**: ~10-20MB for native libraries (vs 22KB model)
- ❌ **Untested on ARM64**: Tested on x86/x64, not arm64-v8a
- ❌ **Requires custom build**: Need to compile LightGBM for Android NDK

**Android Compatibility Assessment**:
- **Architecture mismatch**: lightgbm4j provides x86/x64 binaries, we need arm64-v8a
- **Build required**: Must compile LightGBM with Android NDK to get `lib_lightgbm.so`
- **Risk**: High - unproven on Android, significant integration effort

**Verdict**: ⚠️ **Not recommended** without Android NDK build verification

---

### Option 3: PMML Export + JPMML Evaluator

**Approach**: LightGBM → PMML XML → JPMML evaluator on Android

**Libraries**:
- Export: `jpmml-lightgbm` (https://github.com/jpmml/jpmml-lightgbm)
- Runtime: JPMML evaluator for Android

**Advantages**:
- ✅ **Platform-agnostic**: XML-based model format
- ✅ **Java-native**: Pure Java runtime, no JNI required
- ✅ **Proven conversion**: JPMML-LightGBM is mature
- ✅ **High fidelity**: Results match Python to 5th decimal

**Disadvantages**:
- ⚠️ **Larger model size**: XML format verbose (~100-500KB for 22KB model)
- ⚠️ **Slower inference**: Pure Java vs native code (10-50x slower)
- ⚠️ **Limited optimization**: No hardware acceleration support
- ⚠️ **Extra dependency**: JPMML evaluator library (~5MB)

**Implementation**:
```bash
# Export to PMML
java -jar jpmml-lightgbm-executable-1.5.0.jar \
    --lgbm-input lightgbm_behavior.txt \
    --pmml-output lightgbm_behavior.pmml
```

```kotlin
// Android inference
implementation("org.jpmml:pmml-evaluator:1.6.5")

class LightGBMPMMLEngine(context: Context) {
    private val evaluator: Evaluator

    fun predict(features: Map<String, Float>): String {
        val results = evaluator.evaluate(features)
        return results["prediction"] as String
    }
}
```

**Verdict**: ⚠️ **Fallback option** if ONNX/TFLite conversion fails

---

### Option 4: Native JNI Wrapper (Custom Build)

**Approach**: Compile LightGBM C API for Android, wrap with JNI

**Advantages**:
- ✅ **Maximum control**: Direct access to LightGBM C API
- ✅ **Minimal overhead**: No conversion, use original 22KB model
- ✅ **Best performance**: Native code, no intermediate layers
- ✅ **Smallest size**: Only bundle what's needed

**Disadvantages**:
- ❌ **High complexity**: Requires Android NDK, CMake build configuration
- ❌ **Maintenance burden**: Must update with LightGBM releases
- ❌ **Cross-compilation**: Build for arm64-v8a, test on device
- ❌ **JNI boilerplate**: Write Java↔C++ bridge code

**Implementation Outline**:
```cmake
# CMakeLists.txt
add_library(lightgbm SHARED IMPORTED)
set_target_properties(lightgbm PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/lib_lightgbm.so)

add_library(lightgbm_jni SHARED
    lightgbm_jni.cpp)
target_link_libraries(lightgbm_jni lightgbm)
```

```cpp
// lightgbm_jni.cpp
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_glec_dtg_inference_LightGBMNative_predict(
    JNIEnv* env, jobject, jlong handle, jfloatArray features) {

    // Call LightGBM C API: LGBM_BoosterPredictForMat()
    // ...
}
```

**Build Steps**:
1. Cross-compile LightGBM with Android NDK:
   ```bash
   cd LightGBM
   mkdir build && cd build
   cmake .. \
       -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
       -DANDROID_ABI=arm64-v8a \
       -DANDROID_PLATFORM=android-29
   make -j$(nproc)
   ```

2. Copy `lib_lightgbm.so` to `android-dtg/app/src/main/jniLibs/arm64-v8a/`

3. Write JNI wrapper and integrate

**Verdict**: ✅ **Best performance**, but ⚠️ **highest effort** - consider for optimization phase

---

### Option 5: ONNX Runtime for Android

**Approach**: LightGBM → ONNX → ONNX Runtime Mobile

**Advantages**:
- ✅ **Cross-platform**: Same model for iOS/Android
- ✅ **ONNX ecosystem**: Standard interchange format
- ✅ **Hardware acceleration**: NNAPI, GPU support
- ✅ **Simpler than TFLite**: Skip TensorFlow conversion

**Disadvantages**:
- ⚠️ **Larger runtime**: ONNX Runtime AAR ~20-30MB (vs TFLite ~4MB)
- ⚠️ **Less mobile-optimized**: TFLite has more mobile-specific optimizations
- ⚠️ **Conversion complexity**: May hit operator compatibility issues

**Implementation**:
```kotlin
// build.gradle.kts
implementation("com.microsoft.onnxruntime:onnxruntime-android:1.16.3")

class LightGBMONNXEngine(context: Context) {
    private val session: OrtSession

    init {
        val env = OrtEnvironment.getEnvironment()
        val options = SessionOptions().apply {
            setIntraOpNumThreads(4)
            addNnapi()  // Hardware acceleration
        }
        session = env.createSession(loadModel(), options)
    }

    fun predict(features: FloatArray): Int {
        val input = OnnxTensor.createTensor(env, features.reshape(1, 18))
        val output = session.run(mapOf("input" to input))
        return output[0].value as Int
    }
}
```

**Verdict**: ✅ **Solid alternative** to TFLite if targeting cross-platform

---

## Recommendation Matrix

| Option | Performance | Effort | Android Support | Recommendation |
|--------|------------|--------|-----------------|----------------|
| **ONNX → TFLite** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ **PRIMARY** |
| lightgbm4j | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ | ❌ No Android binaries |
| PMML + JPMML | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⚠️ Fallback only |
| Native JNI | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ✅ Optimization phase |
| ONNX Runtime | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ Alternative |

---

## Implementation Plan

### Phase 1: Proof of Concept (Web-Compatible) ✅ **DO NOW**

**Goal**: Validate ONNX → TFLite conversion pipeline

```bash
# 1. Install dependencies
pip install onnxmltools onnx2tf tensorflow

# 2. Convert LightGBM → ONNX
cd ai-models/conversion
python convert_lightgbm_to_onnx.py \
    --input ../models/lightgbm_behavior.txt \
    --output ../models/lightgbm_behavior.onnx

# 3. Convert ONNX → TensorFlow
onnx2tf -i ../models/lightgbm_behavior.onnx \
    -o ../models/lightgbm_tf

# 4. Convert TensorFlow → TFLite with quantization
python convert_tf_to_tflite.py \
    --input ../models/lightgbm_tf \
    --output ../models/lightgbm_behavior.tflite \
    --quantize int8

# 5. Validate TFLite model accuracy
python validate_tflite_model.py \
    --model ../models/lightgbm_behavior.tflite \
    --test-data ../../datasets/test.csv \
    --original ../models/lightgbm_behavior.txt
```

**Success Criteria**:
- ✅ TFLite model produced without errors
- ✅ Accuracy delta < 1% (99.54% → >98.54%)
- ✅ Model size < 500KB (acceptable for TFLite)
- ✅ Inference latency < 5ms on CPU

---

### Phase 2: Android Integration (Local Environment) ⏸️ **DEFER**

**Requires**: Android Studio, device/emulator, build tools

```kotlin
// 1. Add TFLite dependency to build.gradle.kts
dependencies {
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.14.0")
}

// 2. Create LightGBMTFLiteEngine.kt
package com.glec.dtg.inference

class LightGBMTFLiteEngine(context: Context) : AutoCloseable {
    private val interpreter: Interpreter
    private val inputShape = intArrayOf(1, 18)  // [batch, features]
    private val outputShape = intArrayOf(1, 3)  // [batch, classes]

    init {
        val model = loadModelFile(context, "lightgbm_behavior.tflite")
        val options = Interpreter.Options().apply {
            setNumThreads(4)
            addDelegate(NnApiDelegate())  // Hardware acceleration
        }
        interpreter = Interpreter(model, options)
    }

    /**
     * Predict driving behavior from 60-second window features
     *
     * @param features 18-dimensional feature vector (mean, std, max, min of speed, rpm, etc.)
     * @return Predicted class: 0=normal, 1=eco_driving, 2=aggressive
     */
    fun predict(features: FloatArray): Int {
        require(features.size == 18) { "Expected 18 features, got ${features.size}" }

        val inputBuffer = ByteBuffer.allocateDirect(4 * 18).apply {
            order(ByteOrder.nativeOrder())
            asFloatBuffer().put(features)
        }

        val outputBuffer = ByteBuffer.allocateDirect(4 * 3).apply {
            order(ByteOrder.nativeOrder())
        }

        interpreter.run(inputBuffer, outputBuffer)

        outputBuffer.rewind()
        val probabilities = FloatArray(3) { outputBuffer.float }
        return probabilities.indexOfMax()
    }

    override fun close() {
        interpreter.close()
    }
}

// 3. Integrate with ModelManager.kt (line 440-451)
private fun loadLightGBMModel(file: File, metadata: ModelMetadata): LoadedModel {
    val engine = LightGBMTFLiteEngine(context)

    return LoadedModel(
        name = metadata.name,
        version = metadata.version,
        runtime = RUNTIME_LIGHTGBM,
        filePath = file.absolutePath,
        handle = engine  // Store TFLite interpreter
    )
}
```

**Testing**:
```bash
# Run on device
./gradlew connectedAndroidTest

# Benchmark latency
adb shell am instrument -w -e class \
    com.glec.dtg.inference.LightGBMBenchmarkTest \
    com.glec.dtg.test/androidx.test.runner.AndroidJUnitRunner
```

---

### Phase 3: Optimization (If Needed) ⏸️ **DEFER**

If TFLite latency exceeds 5ms or size exceeds 500KB:

**Option A: Aggressive Quantization**
```python
# INT8 quantization with representative dataset
def representative_dataset():
    for sample in test_features[:1000]:
        yield [sample.astype(np.float32)]

converter.representative_dataset = representative_dataset
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
```

**Option B: Switch to Native JNI**
- Build LightGBM C API for Android (Option 4 above)
- Achieve original 0.064ms latency with 22KB model
- Trade simplicity for maximum performance

---

## Risk Analysis

### Risk 1: ONNX Conversion Operator Incompatibility
**Probability**: Medium
**Impact**: High
**Mitigation**: Test conversion immediately (Phase 1). Fallback to PMML if fails.

### Risk 2: TFLite Model Size Bloat
**Probability**: Low
**Impact**: Low
**Mitigation**: 22KB LightGBM → likely <500KB TFLite (still << 10MB target)

### Risk 3: Accuracy Degradation After Conversion
**Probability**: Low
**Impact**: High
**Mitigation**: Validate with `validate_tflite_model.py`. Acceptable: >98% accuracy

### Risk 4: Android TFLite Latency Regression
**Probability**: Medium
**Impact**: Medium
**Mitigation**: 0.064ms → expected ~1-5ms (still << 15ms target). NNAPI acceleration helps.

---

## Next Steps

### Immediate Actions (Web Environment Compatible) ✅

1. **Create conversion scripts** (`ai-models/conversion/`):
   - `convert_lightgbm_to_onnx.py`
   - `convert_tf_to_tflite.py`
   - `validate_tflite_model.py`

2. **Run Phase 1 PoC**:
   - Convert LightGBM → ONNX → TFLite
   - Validate accuracy and size
   - Document results

3. **Update TODO list**:
   - Mark "Research LightGBM4j" as completed
   - Add "Convert LightGBM to TFLite and validate" as new task

### Deferred Actions (Local Environment Required) ⏸️

4. **Android integration** (requires build tools)
5. **Device benchmarking** (requires hardware)
6. **Optimization** (if needed)

---

## References

### Official Documentation
- **LightGBM**: https://lightgbm.readthedocs.io/
- **ONNX**: https://onnx.ai/
- **TFLite**: https://www.tensorflow.org/lite
- **ONNX Runtime**: https://onnxruntime.ai/

### Libraries & Tools
- **onnxmltools**: https://github.com/onnx/onnxmltools
- **onnx2tf**: https://github.com/PINTO0309/onnx2tf
- **lightgbm4j**: https://github.com/metarank/lightgbm4j
- **JPMML**: https://github.com/jpmml/jpmml-lightgbm

### Integration Guides
- **Deploying LightGBM on JVM**: https://openscoring.io/blog/2019/12/03/deploying_lightgbm_java/
- **ONNX to TFLite**: https://stackoverflow.com/questions/53182177/how-do-you-convert-a-onnx-to-tflite
- **TFLite Android**: https://www.tensorflow.org/lite/android

---

**Status**: Research complete ✅
**Next**: Implement Phase 1 conversion scripts
**Owner**: AI Models Team

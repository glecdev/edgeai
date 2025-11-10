# Troubleshooting Guide - Phase 1: LightGBM Behavior Classification

**GLEC DTG Edge AI SDK - Common Issues and Solutions**

---

## Table of Contents

- [Installation Issues](#installation-issues)
- [Runtime Errors](#runtime-errors)
- [Performance Issues](#performance-issues)
- [Data Quality Issues](#data-quality-issues)
- [Integration Issues](#integration-issues)
- [Testing Issues](#testing-issues)
- [Diagnostic Tools](#diagnostic-tools)

---

## Installation Issues

### Issue 1: ONNX Model Not Found

**Error**:
```
java.io.FileNotFoundException: lightgbm_behavior.onnx
  at com.glec.dtg.inference.LightGBMONNXEngine.<init>
```

**Cause**: ONNX model missing from Android assets directory

**Solution**:

```bash
# 1. Check if model exists
ls -lh android-dtg/app/src/main/assets/models/lightgbm_behavior.onnx

# Expected output:
# -rw-r--r-- 1 user user 12.6K Jan 10 10:00 lightgbm_behavior.onnx

# 2. If missing, copy from conversion output
cd ai-models/conversion
python convert_lightgbm_to_onnx.py \
    --model-path ../../models/lightgbm/lightgbm_model.txt \
    --output-path ../../android-dtg/app/src/main/assets/models/lightgbm_behavior.onnx

# 3. Rebuild APK
cd android-dtg
./gradlew clean assembleDebug
```

**Prevention**: Add to `.gitignore` exceptions

```gitignore
# .gitignore
*.onnx
!android-dtg/app/src/main/assets/models/*.onnx
```

---

### Issue 2: ONNX Runtime Dependency Missing

**Error**:
```
java.lang.NoClassDefFoundError: ai.onnxruntime.OrtSession
  at com.glec.dtg.inference.LightGBMONNXEngine.loadModel
```

**Cause**: ONNX Runtime Android library not included in dependencies

**Solution**:

```gradle
// app/build.gradle.kts
dependencies {
    // Add ONNX Runtime Mobile
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.0")
}
```

```bash
# Sync Gradle
./gradlew --refresh-dependencies

# Clean and rebuild
./gradlew clean assembleDebug
```

---

### Issue 3: Gradle Build Fails

**Error**:
```
A problem occurred evaluating project ':app'.
> Failed to apply plugin 'com.android.application'.
```

**Cause**: Gradle version incompatibility

**Solution**:

```properties
# gradle/wrapper/gradle-wrapper.properties
distributionUrl=https\://services.gradle.org/distributions/gradle-8.2-bin.zip
```

```kotlin
// build.gradle.kts (project level)
plugins {
    id("com.android.application") version "8.2.0" apply false
    id("org.jetbrains.kotlin.android") version "1.9.20" apply false
}
```

---

## Runtime Errors

### Issue 4: Inference Returns Null

**Symptom**: `runInferenceWithConfidence()` returns `null`

**Diagnostic**:

```kotlin
val result = inferenceService.runInferenceWithConfidence()

if (result == null) {
    // Debug information
    Log.d(TAG, "isReady: ${inferenceService.isReady()}")
    Log.d(TAG, "sampleCount: ${inferenceService.getSampleCount()}")
}
```

**Possible Causes**:

#### Cause 1: Window Not Ready (< 60 samples)

**Solution**:

```kotlin
// Always check isReady() before inference
if (inferenceService.isReady()) {
    val result = inferenceService.runInferenceWithConfidence()
    // Handle result
} else {
    Log.d(TAG, "Window not ready: ${inferenceService.getSampleCount()}/60")
}
```

#### Cause 2: Feature Extraction Failure

**Solution**:

```kotlin
// Manually test feature extraction
val features = featureExtractor.extractFeatures()

if (features == null) {
    Log.e(TAG, "Feature extraction failed")
} else {
    Log.d(TAG, "Features: ${features.joinToString()}")

    // Check for NaN or Infinity
    if (features.any { it.isNaN() || it.isInfinite() }) {
        Log.e(TAG, "Invalid feature values detected")
    }
}
```

#### Cause 3: ONNX Runtime Error

**Solution**: Check logcat for ONNX errors

```bash
adb logcat -s LightGBMONNXEngine:E

# Look for:
# E/LightGBMONNXEngine: ONNX Runtime error: ...
```

---

### Issue 5: App Crashes on Inference

**Error**:
```
Process: com.glec.dtg, PID: 12345
java.lang.OutOfMemoryError: Failed to allocate memory
```

**Cause**: Memory leak or insufficient heap

**Solution**:

#### 1. Increase Heap Size

```xml
<!-- AndroidManifest.xml -->
<application
    android:name=".MyApplication"
    android:largeHeap="true"
    ... >
```

#### 2. Profile Memory Usage

```bash
# Memory profiler
adb shell am dumpheap com.glec.dtg /data/local/tmp/heap.hprof
adb pull /data/local/tmp/heap.hprof

# Analyze with Android Studio Profiler
```

#### 3. Release Resources

```kotlin
override fun onDestroy() {
    super.onDestroy()

    // Release ONNX session
    lightGBMEngine.close()

    // Clear buffers
    inferenceService.reset()
}
```

---

### Issue 6: Invalid CAN Data Values

**Error**: Inference produces unrealistic results

**Diagnostic**:

```kotlin
// Validate CAN data before adding
if (canData.isValid()) {
    inferenceService.addSample(canData)
} else {
    Log.w(TAG, "Invalid CAN data detected:")
    Log.w(TAG, "  Speed: ${canData.vehicleSpeed} (expected: 0-255)")
    Log.w(TAG, "  RPM: ${canData.engineRPM} (expected: 0-16383)")
    Log.w(TAG, "  Throttle: ${canData.throttlePosition} (expected: 0-100)")
}
```

**Solution**: Add data validation layer

```kotlin
fun sanitizeCANData(raw: CANData): CANData {
    return raw.copy(
        vehicleSpeed = raw.vehicleSpeed.coerceIn(0.0f, 255.0f),
        engineRPM = raw.engineRPM.coerceIn(0, 16383),
        throttlePosition = raw.throttlePosition.coerceIn(0.0f, 100.0f),
        brakePosition = raw.brakePosition.coerceIn(0.0f, 100.0f),
        accelerationX = raw.accelerationX.coerceIn(-20.0f, 20.0f),
        accelerationY = raw.accelerationY.coerceIn(-20.0f, 20.0f)
    )
}

// Usage
val sanitized = sanitizeCANData(rawCanData)
if (sanitized.isValid()) {
    inferenceService.addSample(sanitized)
}
```

---

## Performance Issues

### Issue 7: Inference Latency Too High

**Symptom**: Inference takes > 50ms (target: < 50ms P95)

**Diagnostic**:

```kotlin
val startTime = System.nanoTime()
val result = inferenceService.runInferenceWithConfidence()
val latencyMs = (System.nanoTime() - startTime) / 1_000_000.0

Log.i(TAG, "Inference latency: $latencyMs ms")

if (latencyMs > 50.0) {
    Log.w(TAG, "Latency exceeds target (50ms)")
}
```

**Possible Causes**:

#### Cause 1: Running on Main Thread

**Solution**: Use background thread

```kotlin
// Bad: Main thread
val result = inferenceService.runInferenceWithConfidence()  // Blocks UI!

// Good: Background thread
lifecycleScope.launch(Dispatchers.Default) {
    val result = inferenceService.runInferenceWithConfidence()

    withContext(Dispatchers.Main) {
        updateUI(result)
    }
}
```

#### Cause 2: CPU Governor in Power-Save Mode

**Solution**: Check CPU governor

```bash
# Check current governor
adb shell cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# If "powersave", switch to "performance" for testing
adb shell "echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
```

#### Cause 3: Thermal Throttling

**Solution**: Monitor device temperature

```bash
# Check temperature
adb shell cat /sys/class/thermal/thermal_zone0/temp

# If > 60000 (60°C), device may throttle
# Allow device to cool down
```

---

### Issue 8: High Memory Usage

**Symptom**: App uses > 100MB memory

**Diagnostic**:

```bash
# Check memory usage
adb shell dumpsys meminfo com.glec.dtg

# Focus on:
# - Native Heap
# - Dalvik Heap
# - Graphics
```

**Solution**:

#### 1. Reduce Window Size (if acceptable)

```kotlin
// Reduce from 60 to 30 samples (30-second window)
val extractor = FeatureExtractor(windowSize = 30)
```

#### 2. Use Weak References

```kotlin
class MyService : Service() {
    private var inferenceServiceRef: WeakReference<EdgeAIInferenceService>? = null

    override fun onCreate() {
        val service = EdgeAIInferenceService(this)
        inferenceServiceRef = WeakReference(service)
    }
}
```

---

## Data Quality Issues

### Issue 9: Low Confidence Predictions

**Symptom**: Confidence consistently < 0.7

**Diagnostic**:

```kotlin
val result = inferenceService.runInferenceWithConfidence()

if (result != null && result.confidence < 0.7f) {
    Log.w(TAG, "Low confidence: ${result.confidence}")
    Log.w(TAG, "Behavior: ${result.behavior}")

    // Inspect feature values
    val features = featureExtractor.extractFeatures()
    features?.forEachIndexed { idx, value ->
        Log.d(TAG, "Feature $idx: $value")
    }
}
```

**Possible Causes**:

#### Cause 1: Noisy CAN Data

**Solution**: Apply smoothing filter

```kotlin
class ExponentialMovingAverage(private val alpha: Float = 0.3f) {
    private var value: Float = 0.0f

    fun update(newValue: Float): Float {
        value = alpha * newValue + (1 - alpha) * value
        return value
    }
}

// Usage
val speedEMA = ExponentialMovingAverage()

val smoothedSpeed = speedEMA.update(rawCanData.vehicleSpeed)
```

#### Cause 2: Driving Pattern Not in Training Data

**Solution**: Retrain model with more diverse data

```bash
cd ai-models/training

# Generate more diverse synthetic data
python ../data-generation/synthetic_driving_simulator.py \
    --samples 100000 \
    --output-dir ../../datasets_v2

# Retrain
python train_lightgbm.py \
    --train-data ../../datasets_v2/train.csv \
    --test-data ../../datasets_v2/test.csv
```

---

### Issue 10: Fuel Consumption Returns 0

**Symptom**: `calculateFuelConsumption()` always returns 0

**Diagnostic**:

```kotlin
val fuel = canData.calculateFuelConsumption()

if (fuel == 0.0f) {
    Log.d(TAG, "Fuel consumption is 0")
    Log.d(TAG, "  vehicleSpeed: ${canData.vehicleSpeed}")
    Log.d(TAG, "  mafRate: ${canData.mafRate}")
}
```

**Cause**: Speed or MAF rate below threshold

**Solution**:

```kotlin
// Check thresholds
if (canData.vehicleSpeed < 1.0f) {
    Log.d(TAG, "Vehicle stopped, fuel consumption = 0")
}

if (canData.mafRate < 0.1f) {
    Log.w(TAG, "MAF rate too low, check sensor")
}
```

---

## Integration Issues

### Issue 11: UART Communication Fails

**Error**: CAN data not received from STM32

**Diagnostic**:

```bash
# Check UART device exists
adb shell ls -l /dev/ttyUSB0

# If missing:
# - Check USB cable
# - Check USB permissions
```

**Solution**:

```xml
<!-- AndroidManifest.xml -->
<uses-permission android:name="android.permission.USB_PERMISSION" />

<application>
    <uses-library android:name="com.android.future.usb.accessory" />
</application>
```

```kotlin
// Request USB permission
val usbManager = getSystemService(Context.USB_SERVICE) as UsbManager
val device = usbManager.deviceList.values.firstOrNull()

if (device != null) {
    val permissionIntent = PendingIntent.getBroadcast(
        this, 0, Intent(ACTION_USB_PERMISSION), 0
    )
    usbManager.requestPermission(device, permissionIntent)
}
```

---

### Issue 12: MQTT Connection Fails

**Error**: Cannot publish inference results to MQTT broker

**Diagnostic**:

```kotlin
try {
    mqttClient.connect()
} catch (e: MqttException) {
    Log.e(TAG, "MQTT connection failed", e)
    // Check:
    // 1. Broker URL correct?
    // 2. Network available?
    // 3. Credentials valid?
}
```

**Solution**:

```kotlin
// Add connection retry logic
suspend fun connectWithRetry(maxRetries: Int = 3) {
    repeat(maxRetries) { attempt ->
        try {
            mqttClient.connect()
            Log.i(TAG, "MQTT connected")
            return
        } catch (e: MqttException) {
            Log.w(TAG, "MQTT connection attempt ${attempt + 1} failed")
            if (attempt < maxRetries - 1) {
                delay(2000 * (attempt + 1))  // Exponential backoff
            }
        }
    }
    Log.e(TAG, "MQTT connection failed after $maxRetries attempts")
}
```

---

## Testing Issues

### Issue 13: Python Tests Fail

**Error**: `pytest tests/ -v` fails

**Diagnostic**:

```bash
# Run with verbose output
pytest tests/test_edge_ai_inference_integration.py -v -s

# Check specific test
pytest tests/test_edge_ai_inference_integration.py::TestEdgeAIInferenceIntegration::test_onnx_model_loading -v
```

**Common Issues**:

#### Issue 13a: ONNX Model Not Found (Python)

```bash
# Check model path
ls -lh android-dtg/app/src/main/assets/models/lightgbm_behavior.onnx

# If missing, run conversion
cd ai-models/conversion
python convert_lightgbm_to_onnx.py
```

#### Issue 13b: onnxruntime Not Installed

```bash
# Install dependencies
pip install -r requirements.txt

# Or install directly
pip install onnxruntime==1.17.0
```

#### Issue 13c: Synthetic Dataset Missing

```bash
# Generate synthetic data
cd data-generation
python synthetic_driving_simulator.py \
    --output-dir ../datasets \
    --samples 35000

# Should create:
# datasets/train.csv (28,000 samples)
# datasets/val.csv (3,500 samples)
# datasets/test.csv (3,500 samples)
```

---

### Issue 14: Android Unit Tests Fail

**Error**: `./gradlew testDebugUnitTest` fails

**Diagnostic**:

```bash
# Run tests with stacktrace
./gradlew testDebugUnitTest --stacktrace

# Run specific test class
./gradlew testDebugUnitTest --tests "com.glec.dtg.inference.FeatureExtractorTest"
```

**Solution**: Check test dependencies

```gradle
// app/build.gradle.kts
dependencies {
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.mockito:mockito-core:5.3.1")
    testImplementation("org.robolectric:robolectric:4.11.1")
}
```

---

## Diagnostic Tools

### Tool 1: Inference Diagnostics Script

```kotlin
fun diagnoseInference(inferenceService: EdgeAIInferenceService) {
    Log.d(TAG, "=== Inference Diagnostics ===")

    // 1. Check window status
    Log.d(TAG, "Window ready: ${inferenceService.isReady()}")
    Log.d(TAG, "Sample count: ${inferenceService.getSampleCount()}/60")

    // 2. Check feature extraction
    val features = inferenceService.featureExtractor.extractFeatures()
    if (features != null) {
        Log.d(TAG, "Feature extraction: OK")
        Log.d(TAG, "Features: ${features.joinToString()}")

        // Check for invalid values
        if (features.any { it.isNaN() || it.isInfinite() }) {
            Log.e(TAG, "Invalid feature values detected!")
        }
    } else {
        Log.e(TAG, "Feature extraction: FAILED")
    }

    // 3. Run inference
    val start = System.nanoTime()
    val result = inferenceService.runInferenceWithConfidence()
    val latency = (System.nanoTime() - start) / 1_000_000.0

    if (result != null) {
        Log.d(TAG, "Inference: OK")
        Log.d(TAG, "  Behavior: ${result.behavior}")
        Log.d(TAG, "  Confidence: ${result.confidence}")
        Log.d(TAG, "  Latency: ${latency}ms")
    } else {
        Log.e(TAG, "Inference: FAILED")
    }

    Log.d(TAG, "=== End Diagnostics ===")
}
```

### Tool 2: CAN Data Validator

```kotlin
fun validateCANDataStream(samples: List<CANData>) {
    Log.d(TAG, "=== CAN Data Validation ===")
    Log.d(TAG, "Total samples: ${samples.size}")

    var validCount = 0
    var invalidCount = 0

    samples.forEach { sample ->
        if (sample.isValid()) {
            validCount++
        } else {
            invalidCount++
            Log.w(TAG, "Invalid sample at ${sample.timestamp}")
        }
    }

    Log.d(TAG, "Valid: $validCount (${validCount * 100 / samples.size}%)")
    Log.d(TAG, "Invalid: $invalidCount (${invalidCount * 100 / samples.size}%)")

    // Check for missing timestamps
    val timestamps = samples.map { it.timestamp }.sorted()
    val gaps = timestamps.zipWithNext().filter { (a, b) -> b - a > 1500 }

    if (gaps.isNotEmpty()) {
        Log.w(TAG, "Found ${gaps.size} timestamp gaps > 1.5s")
    }

    Log.d(TAG, "=== End Validation ===")
}
```

### Tool 3: Performance Profiler

```kotlin
class InferenceProfiler {
    private val latencies = mutableListOf<Float>()

    fun profile(inferenceService: EdgeAIInferenceService): InferenceResult? {
        val start = System.nanoTime()
        val result = inferenceService.runInferenceWithConfidence()
        val latency = (System.nanoTime() - start) / 1_000_000.0f

        latencies.add(latency)

        return result
    }

    fun printStatistics() {
        if (latencies.isEmpty()) return

        val sorted = latencies.sorted()
        val mean = latencies.average()
        val p50 = sorted[latencies.size / 2]
        val p95 = sorted[(latencies.size * 0.95).toInt()]
        val p99 = sorted[(latencies.size * 0.99).toInt()]

        Log.i(TAG, "=== Inference Performance ===")
        Log.i(TAG, "Samples: ${latencies.size}")
        Log.i(TAG, "Mean: $mean ms")
        Log.i(TAG, "P50: $p50 ms")
        Log.i(TAG, "P95: $p95 ms (target: <50ms)")
        Log.i(TAG, "P99: $p99 ms")
    }
}
```

---

## Getting Help

### Step 1: Collect Diagnostic Information

```bash
# 1. App version
adb shell dumpsys package com.glec.dtg | grep versionName

# 2. Device info
adb shell getprop ro.product.model
adb shell getprop ro.build.version.release

# 3. Logs
adb logcat -s DTGForegroundService EdgeAIInferenceService LightGBMONNXEngine > logs.txt

# 4. Memory dump
adb shell am dumpheap com.glec.dtg /data/local/tmp/heap.hprof
adb pull /data/local/tmp/heap.hprof

# 5. ANR traces (if app freezes)
adb pull /data/anr/traces.txt
```

### Step 2: Check Documentation

- [Phase 1 Deployment Guide](./PHASE1_DEPLOYMENT_GUIDE.md)
- [API Reference](./API_REFERENCE.md)
- [Main README](../README.md)

### Step 3: Contact Support

**GitHub Issues**: https://github.com/glecdev/edgeai/issues

Include in your issue report:
1. Error message (full stacktrace)
2. Device info (model, Android version)
3. App version
4. Steps to reproduce
5. Diagnostic logs

---

## FAQ

### Q1: How do I update the ONNX model?

**A**: Retrain LightGBM, convert to ONNX, copy to assets, rebuild APK

```bash
# 1. Retrain
cd ai-models/training
python train_lightgbm.py

# 2. Convert
cd ../conversion
python convert_lightgbm_to_onnx.py

# 3. Model automatically copied to assets

# 4. Rebuild
cd ../../android-dtg
./gradlew clean assembleDebug
```

### Q2: Can I use a different window size?

**A**: Yes, but model must be retrained with same window size

```kotlin
// Change window size
val extractor = FeatureExtractor(windowSize = 30)  // 30-second windows

// IMPORTANT: Retrain model with 30-second windows!
```

### Q3: How do I enable NNAPI acceleration?

**A**: ONNX Runtime Mobile supports NNAPI by default on Android 8.1+

```kotlin
// NNAPI automatically enabled if available
// No code changes needed

// To disable NNAPI (for debugging):
val sessionOptions = OrtSession.SessionOptions()
sessionOptions.addConfigEntry("session.disable_nnapi", "1")
```

### Q4: Can I run multiple inferences in parallel?

**A**: No, use single instance per service

```kotlin
// Bad: Multiple instances
val service1 = EdgeAIInferenceService(context)
val service2 = EdgeAIInferenceService(context)  // Wastes memory!

// Good: Single instance, sequential inference
val service = EdgeAIInferenceService(context)
```

---

**Last Updated**: 2025-01-10
**Version**: 1.0.0
**Author**: GLEC DTG Team

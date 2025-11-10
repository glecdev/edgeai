# API Reference - Phase 1: LightGBM Behavior Classification

**GLEC DTG Edge AI SDK - Production API Documentation**

---

## Table of Contents

- [EdgeAIInferenceService](#edgeaiinferenceservice)
- [FeatureExtractor](#featureextractor)
- [LightGBMONNXEngine](#lightgbmonnxengine)
- [Data Models](#data-models)
  - [CANData](#candata)
  - [DrivingBehavior](#drivingbehavior)
  - [InferenceResult](#inferenceresult)
- [Usage Examples](#usage-examples)

---

## EdgeAIInferenceService

**Package**: `com.glec.dtg.inference`
**File**: `android-dtg/app/src/main/java/com/glec/dtg/inference/EdgeAIInferenceService.kt`

### Description

Orchestrates the complete AI inference pipeline for driving behavior classification.
Manages 60-second sliding windows, feature extraction, and ONNX model inference.

### Constructor

```kotlin
class EdgeAIInferenceService(context: Context)
```

**Parameters**:
- `context: Context` - Android application context (required for loading ONNX model from assets)

**Example**:
```kotlin
val inferenceService = EdgeAIInferenceService(applicationContext)
```

### Methods

#### `addSample(sample: CANData)`

Add a CAN data sample to the 60-second sliding window.

**Parameters**:
- `sample: CANData` - CAN bus data sample collected at 1Hz

**Returns**: `Unit`

**Behavior**:
- Adds sample to internal buffer managed by `FeatureExtractor`
- Automatically removes oldest sample if window > 60 (FIFO)
- Thread-safe (uses `@Synchronized`)

**Example**:
```kotlin
val canData = CANData(
    timestamp = System.currentTimeMillis(),
    vehicleSpeed = 60.0f,
    engineRPM = 1800,
    throttlePosition = 25.0f,
    // ... other fields
)

inferenceService.addSample(canData)
```

---

#### `isReady(): Boolean`

Check if 60-second window is complete and ready for inference.

**Returns**: `Boolean`
- `true` - Window has exactly 60 samples, ready for inference
- `false` - Window has < 60 samples, not ready

**Example**:
```kotlin
if (inferenceService.isReady()) {
    val result = inferenceService.runInferenceWithConfidence()
    println("Behavior: ${result.behavior}")
} else {
    println("Window not ready: ${inferenceService.getSampleCount()}/60")
}
```

---

#### `getSampleCount(): Int`

Get current number of samples in sliding window.

**Returns**: `Int` - Number of samples (0-60)

**Example**:
```kotlin
val count = inferenceService.getSampleCount()
Log.d(TAG, "Window progress: $count/60")
```

---

#### `runInferenceWithConfidence(): InferenceResult?`

Run LightGBM inference on current 60-second window.

**Returns**: `InferenceResult?`
- `InferenceResult` - Success (window ready, inference completed)
- `null` - Failure (window not ready or inference error)

**Preconditions**:
- `isReady()` must return `true` (60 samples collected)

**Performance**:
- P95 latency: **0.0119ms**
- Includes feature extraction + ONNX inference

**Example**:
```kotlin
if (inferenceService.isReady()) {
    val result = inferenceService.runInferenceWithConfidence()

    if (result != null) {
        Log.i(TAG, "Behavior: ${result.behavior}")
        Log.i(TAG, "Confidence: ${result.confidence}")
        Log.i(TAG, "Latency: ${result.latencyMs}ms")

        when (result.behavior) {
            DrivingBehavior.ECO_DRIVING -> showEcoReward()
            DrivingBehavior.AGGRESSIVE -> showWarning()
            // ... handle other behaviors
        }
    } else {
        Log.e(TAG, "Inference failed")
    }
}
```

---

#### `reset()`

Reset the inference service (clear 60-second window).

**Returns**: `Unit`

**Use Cases**:
- Trip end
- Driver change
- System reset

**Example**:
```kotlin
// Trip ended, reset for next trip
inferenceService.reset()
Log.i(TAG, "Inference service reset")
```

---

## FeatureExtractor

**Package**: `com.glec.dtg.inference`
**File**: `android-dtg/app/src/main/java/com/glec/dtg/inference/FeatureExtractor.kt`

### Description

Converts 60-second windows of raw CAN data into 18-dimensional feature vectors
for LightGBM model input.

### Feature Vector (18 dimensions)

| Index | Feature Name | Description |
|-------|--------------|-------------|
| [0] | speed_mean | Mean vehicle speed (km/h) |
| [1] | speed_std | Std deviation of speed |
| [2] | speed_max | Maximum speed |
| [3] | speed_min | Minimum speed |
| [4] | rpm_mean | Mean engine RPM |
| [5] | rpm_std | Std deviation of RPM |
| [6] | throttle_mean | Mean throttle position (%) |
| [7] | throttle_std | Std deviation of throttle |
| [8] | throttle_max | Maximum throttle |
| [9] | brake_mean | Mean brake position (%) |
| [10] | brake_std | Std deviation of brake |
| [11] | brake_max | Maximum brake |
| [12] | accel_x_mean | Mean longitudinal accel (m/s²) |
| [13] | accel_x_std | Std deviation of accel X |
| [14] | accel_x_max | Maximum accel X |
| [15] | accel_y_mean | Mean lateral accel (m/s²) |
| [16] | accel_y_std | Std deviation of accel Y |
| [17] | fuel_consumption | Mean fuel consumption (L/100km) |

### Constructor

```kotlin
class FeatureExtractor(windowSize: Int = 60)
```

**Parameters**:
- `windowSize: Int` - Sliding window size (default: 60 for 60-second windows @ 1Hz)

**Example**:
```kotlin
val extractor = FeatureExtractor(windowSize = 60)
```

### Methods

#### `addSample(sample: CANData)`

Add CAN sample to sliding window.

**Parameters**:
- `sample: CANData` - Raw CAN data

**Example**:
```kotlin
extractor.addSample(canData)
```

---

#### `isWindowReady(): Boolean`

Check if window is full (60 samples).

**Returns**: `Boolean`

**Example**:
```kotlin
if (extractor.isWindowReady()) {
    val features = extractor.extractFeatures()
}
```

---

#### `extractFeatures(): FloatArray?`

Extract 18-dimensional feature vector from current window.

**Returns**: `FloatArray?`
- `FloatArray[18]` - Feature vector (if window ready)
- `null` - Window not ready

**Example**:
```kotlin
val features = extractor.extractFeatures()

if (features != null) {
    println("Speed mean: ${features[0]} km/h")
    println("RPM mean: ${features[4]} RPM")
    println("Fuel consumption: ${features[17]} L/100km")
}
```

---

## LightGBMONNXEngine

**Package**: `com.glec.dtg.inference`
**File**: `android-dtg/app/src/main/java/com/glec/dtg/inference/LightGBMONNXEngine.kt`

### Description

ONNX Runtime Mobile wrapper for LightGBM behavior classification model.

### Constructor

```kotlin
class LightGBMONNXEngine(context: Context, modelPath: String = "models/lightgbm_behavior.onnx")
```

**Parameters**:
- `context: Context` - Android context (for asset loading)
- `modelPath: String` - Relative path in assets directory (default: "models/lightgbm_behavior.onnx")

**Example**:
```kotlin
val engine = LightGBMONNXEngine(context)
```

### Methods

#### `predict(features: FloatArray): DrivingBehavior?`

Predict driving behavior from 18-dimensional feature vector.

**Parameters**:
- `features: FloatArray` - Feature vector (must be size 18)

**Returns**: `DrivingBehavior?`
- `DrivingBehavior` - Predicted class
- `null` - Inference error

**Example**:
```kotlin
val features = floatArrayOf(
    60.0f, 2.5f, 80.0f, 50.0f,  // Speed stats
    1800.0f, 100.0f,             // RPM stats
    25.0f, 5.0f, 40.0f,          // Throttle stats
    0.0f, 0.0f, 0.0f,            // Brake stats
    0.5f, 0.3f, 1.5f,            // Accel X stats
    0.0f, 0.2f,                  // Accel Y stats
    5.0f                         // Fuel consumption
)

val behavior = engine.predict(features)
println("Predicted: $behavior")
```

---

#### `predictWithProbabilities(features: FloatArray): Pair<DrivingBehavior, FloatArray>?`

Predict behavior with probability distribution for all classes.

**Parameters**:
- `features: FloatArray` - Feature vector (must be size 18)

**Returns**: `Pair<DrivingBehavior, FloatArray>?`
- `Pair.first: DrivingBehavior` - Predicted class
- `Pair.second: FloatArray[7]` - Probability distribution for all 7 classes
- `null` - Inference error

**Example**:
```kotlin
val result = engine.predictWithProbabilities(features)

if (result != null) {
    val (behavior, probabilities) = result

    println("Predicted: $behavior")
    println("Probabilities:")
    DrivingBehavior.values().forEachIndexed { idx, behavior ->
        println("  $behavior: ${probabilities[idx] * 100}%")
    }

    // Get confidence as max probability
    val confidence = probabilities.maxOrNull() ?: 0.0f
    println("Confidence: ${confidence * 100}%")
}
```

---

#### `close()`

Release ONNX Runtime session resources.

**Returns**: `Unit`

**Example**:
```kotlin
override fun onDestroy() {
    super.onDestroy()
    engine.close()
}
```

---

## Data Models

### CANData

**Package**: `com.glec.dtg.models`
**File**: `android-dtg/app/src/main/java/com/glec/dtg/models/CANData.kt`

#### Description

Represents a single CAN bus data sample collected at 1Hz.

#### Properties

```kotlin
data class CANData(
    val timestamp: Long,                // Unix timestamp (milliseconds)
    val vehicleSpeed: Float,            // km/h (0-255)
    val engineRPM: Int,                 // RPM (0-16383)
    val throttlePosition: Float,        // % (0-100)
    val brakePosition: Float,           // % (0-100)
    val fuelLevel: Float,               // % (0-100)
    val coolantTemp: Float,             // °C (-40 to 215)
    val engineLoad: Float,              // % (0-100)
    val intakeAirTemp: Float,           // °C (-40 to 215)
    val mafRate: Float,                 // g/s (0-655.35)
    val batteryVoltage: Float,          // V (0-65.535)
    val accelerationX: Float,           // m/s² (-20 to 20)
    val accelerationY: Float,           // m/s² (-20 to 20)
    val accelerationZ: Float,           // m/s² (-20 to 20)
    val gyroX: Float,                   // °/s (-250 to 250)
    val gyroY: Float,                   // °/s (-250 to 250)
    val gyroZ: Float,                   // °/s (-250 to 250)
    val gpsLatitude: Double,            // Decimal degrees
    val gpsLongitude: Double,           // Decimal degrees
    val gpsAltitude: Float,             // meters
    val gpsSpeed: Float,                // km/h
    val gpsHeading: Float               // degrees (0-360)
)
```

#### Methods

##### `calculateFuelConsumption(): Float`

Calculate instantaneous fuel consumption based on MAF rate.

**Formula**:
```
fuel_flow_rate = maf_rate / 14.7 (stoichiometric ratio)
fuel_L_per_h = (fuel_flow_rate * 3600) / 750 (gasoline density)
fuel_L_per_100km = (fuel_L_per_h / vehicle_speed) * 100
```

**Returns**: `Float` - Fuel consumption in L/100km

**Example**:
```kotlin
val canData = CANData(
    vehicleSpeed = 80.0f,
    mafRate = 15.0f,
    // ... other fields
)

val fuelConsumption = canData.calculateFuelConsumption()
println("Fuel: $fuelConsumption L/100km")  // ~6.12 L/100km
```

##### `isValid(): Boolean`

Validate CAN data ranges.

**Returns**: `Boolean` - `true` if all values within valid ranges

**Example**:
```kotlin
if (canData.isValid()) {
    inferenceService.addSample(canData)
} else {
    Log.w(TAG, "Invalid CAN data, skipping")
}
```

---

### DrivingBehavior

**Package**: `com.glec.dtg.models`
**File**: `android-dtg/app/src/main/java/com/glec/dtg/models/DrivingBehavior.kt`

#### Description

Enum representing driving behavior classifications.

#### Values

```kotlin
enum class DrivingBehavior {
    ECO_DRIVING,         // 0: Smooth, fuel-efficient driving
    NORMAL,              // 1: Standard driving behavior
    AGGRESSIVE,          // 2: Harsh acceleration/braking, high variance
    HARSH_BRAKING,       // 3: Sudden braking events
    HARSH_ACCELERATION,  // 4: Rapid acceleration events
    SPEEDING,            // 5: Excessive speed
    ANOMALY              // 6: Unusual patterns (potential issue)
}
```

#### Example

```kotlin
when (behavior) {
    DrivingBehavior.ECO_DRIVING -> {
        rewardPoints += 10
        showNotification("Great eco driving! +10 points")
    }
    DrivingBehavior.AGGRESSIVE -> {
        safetyScore -= 25
        showWarning("Aggressive driving detected")
    }
    DrivingBehavior.HARSH_BRAKING -> {
        logEvent("harsh_braking", timestamp)
        alertFleetManager()
    }
    // ... other cases
}
```

---

### InferenceResult

**Package**: `com.glec.dtg.inference`
**File**: `android-dtg/app/src/main/java/com/glec/dtg/inference/EdgeAIInferenceService.kt`

#### Description

Result of AI inference containing behavior classification and metadata.

#### Properties

```kotlin
data class InferenceResult(
    val behavior: DrivingBehavior,    // Predicted driving behavior
    val confidence: Float,            // Confidence score (0.0-1.0)
    val latencyMs: Float,             // Inference latency in milliseconds
    val timestamp: Long               // Unix timestamp of inference
)
```

#### Example

```kotlin
val result = inferenceService.runInferenceWithConfidence()

if (result != null) {
    println("Behavior: ${result.behavior}")
    println("Confidence: ${(result.confidence * 100).toInt()}%")
    println("Latency: ${result.latencyMs}ms")
    println("Time: ${Date(result.timestamp)}")

    // High confidence eco driving bonus
    if (result.behavior == DrivingBehavior.ECO_DRIVING && result.confidence > 0.9f) {
        giveBonus(points = 15)
    }

    // Low confidence warning
    if (result.confidence < 0.7f) {
        Log.w(TAG, "Low confidence prediction: ${result.confidence}")
    }
}
```

---

## Usage Examples

### Example 1: Simple Inference Loop

```kotlin
class SimpleDrivingService : Service() {
    private lateinit var inferenceService: EdgeAIInferenceService

    override fun onCreate() {
        super.onCreate()
        inferenceService = EdgeAIInferenceService(this)
    }

    fun collectAndInfer(canDataStream: Flow<CANData>) {
        lifecycleScope.launch {
            canDataStream.collect { sample ->
                // Add sample
                inferenceService.addSample(sample)

                // Check if ready every sample
                if (inferenceService.isReady()) {
                    val result = inferenceService.runInferenceWithConfidence()

                    if (result != null) {
                        handleInferenceResult(result)
                    }
                }
            }
        }
    }

    private fun handleInferenceResult(result: InferenceResult) {
        Log.i(TAG, "Behavior: ${result.behavior}, Confidence: ${result.confidence}")
    }
}
```

### Example 2: Scheduled Inference (Every 60 Seconds)

```kotlin
class ScheduledInferenceService : Service() {
    private lateinit var inferenceService: EdgeAIInferenceService
    private val scope = CoroutineScope(Dispatchers.Default)

    override fun onCreate() {
        super.onCreate()
        inferenceService = EdgeAIInferenceService(this)
        startScheduledInference()
    }

    private fun startScheduledInference() {
        // Data collection job (1Hz)
        scope.launch {
            while (isActive) {
                val canData = readCANData()
                if (canData != null) {
                    inferenceService.addSample(canData)
                }
                delay(1000) // 1Hz
            }
        }

        // Inference job (every 60 seconds)
        scope.launch {
            while (isActive) {
                delay(60_000) // 60 seconds

                if (inferenceService.isReady()) {
                    val result = inferenceService.runInferenceWithConfidence()

                    if (result != null) {
                        // Publish to backend
                        publishToBackend(result)

                        // Update UI
                        broadcastResult(result)
                    }
                }
            }
        }
    }
}
```

### Example 3: Confidence-Based Actions

```kotlin
fun handleInferenceWithConfidence(result: InferenceResult) {
    val confidencePercent = (result.confidence * 100).toInt()

    when {
        result.confidence > 0.9f -> {
            // High confidence - take action
            Log.i(TAG, "High confidence ($confidencePercent%): ${result.behavior}")

            when (result.behavior) {
                DrivingBehavior.ECO_DRIVING -> giveEcoBonus()
                DrivingBehavior.AGGRESSIVE -> issueWarning()
                DrivingBehavior.HARSH_BRAKING -> logSafetyEvent()
                // ... other cases
            }
        }

        result.confidence > 0.7f -> {
            // Medium confidence - log only
            Log.i(TAG, "Medium confidence ($confidencePercent%): ${result.behavior}")
            analyticsLogger.log("medium_confidence_prediction", result)
        }

        else -> {
            // Low confidence - investigate
            Log.w(TAG, "Low confidence ($confidencePercent%): ${result.behavior}")
            flagForReview(result)
        }
    }
}
```

### Example 4: Trip-Based Inference

```kotlin
class TripInferenceService : Service() {
    private lateinit var inferenceService: EdgeAIInferenceService
    private val tripResults = mutableListOf<InferenceResult>()

    fun startTrip() {
        inferenceService.reset()
        tripResults.clear()
        Log.i(TAG, "Trip started")
    }

    fun onInferenceComplete(result: InferenceResult) {
        tripResults.add(result)

        // Calculate trip statistics
        val avgConfidence = tripResults.map { it.confidence }.average()
        val ecoDrivingRatio = tripResults.count {
            it.behavior == DrivingBehavior.ECO_DRIVING
        }.toFloat() / tripResults.size

        Log.i(TAG, "Trip stats: ${tripResults.size} inferences")
        Log.i(TAG, "Avg confidence: ${(avgConfidence * 100).toInt()}%")
        Log.i(TAG, "Eco driving: ${(ecoDrivingRatio * 100).toInt()}%")
    }

    fun endTrip() {
        // Generate trip summary
        val summary = TripSummary(
            totalInferences = tripResults.size,
            ecoPercentage = tripResults.count { it.behavior == DrivingBehavior.ECO_DRIVING }.toFloat() / tripResults.size,
            safetyScore = calculateSafetyScore(tripResults)
        )

        uploadTripSummary(summary)
        Log.i(TAG, "Trip ended: $summary")
    }
}
```

---

## Error Handling

### Common Errors

#### 1. Model Not Found

```kotlin
try {
    val inferenceService = EdgeAIInferenceService(context)
} catch (e: FileNotFoundException) {
    Log.e(TAG, "ONNX model not found in assets", e)
    // Fallback: download model or use default behavior
}
```

#### 2. Inference Returns Null

```kotlin
val result = inferenceService.runInferenceWithConfidence()

if (result == null) {
    if (!inferenceService.isReady()) {
        Log.w(TAG, "Inference failed: window not ready (${inferenceService.getSampleCount()}/60)")
    } else {
        Log.e(TAG, "Inference failed: unknown error")
    }
}
```

#### 3. Invalid CAN Data

```kotlin
if (!canData.isValid()) {
    Log.w(TAG, "Invalid CAN data: $canData")
    // Skip this sample
    return
}

inferenceService.addSample(canData)
```

---

## Performance Considerations

### Memory Management

```kotlin
class MyService : Service() {
    private var inferenceService: EdgeAIInferenceService? = null

    override fun onCreate() {
        super.onCreate()
        inferenceService = EdgeAIInferenceService(this)
    }

    override fun onDestroy() {
        super.onDestroy()
        // Release resources
        inferenceService?.reset()
        inferenceService = null
    }
}
```

### Latency Optimization

- **Inference latency**: 0.0119ms (P95) - negligible overhead
- **Feature extraction**: ~0.01ms - minimal CPU usage
- **Total overhead**: < 0.03ms per 60 seconds → 0.0005% CPU

**Recommendation**: Run inference on background thread (Dispatchers.Default)

```kotlin
scope.launch(Dispatchers.Default) {
    val result = inferenceService.runInferenceWithConfidence()
    // Process result
}
```

---

## Testing

### Unit Testing

```kotlin
@Test
fun testInferenceService() {
    val inferenceService = EdgeAIInferenceService(context)

    // Add 60 samples
    repeat(60) { i ->
        val sample = createMockCANData(timestamp = i * 1000L)
        inferenceService.addSample(sample)
    }

    // Verify ready
    assertTrue(inferenceService.isReady())

    // Run inference
    val result = inferenceService.runInferenceWithConfidence()

    assertNotNull(result)
    assertTrue(result.confidence >= 0.0f && result.confidence <= 1.0f)
    assertTrue(result.latencyMs > 0.0f)
}
```

---

## See Also

- [Phase 1 Deployment Guide](./PHASE1_DEPLOYMENT_GUIDE.md) - Full deployment instructions
- [Troubleshooting Guide](./TROUBLESHOOTING.md) - Common issues and fixes
- [Main README](../README.md) - Project overview

---

**Last Updated**: 2025-01-10
**Version**: 1.0.0
**Author**: GLEC DTG Team

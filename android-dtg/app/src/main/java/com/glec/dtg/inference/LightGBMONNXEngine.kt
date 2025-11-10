package com.glec.dtg.inference

import android.content.Context
import android.util.Log
import ai.onnxruntime.*
import java.nio.FloatBuffer
import kotlin.collections.Map

/**
 * GLEC DTG - LightGBM ONNX Runtime Engine
 *
 * Deploys LightGBM behavior classification model via ONNX Runtime Mobile
 *
 * Model Performance (Validated in Phase 1 PoC):
 * - Model Size: 12.62 KB
 * - P95 Latency: 0.0119ms (CPU)
 * - Accuracy: 99.54% (test set)
 * - F1-Score: 99.30%
 *
 * Advantages over TFLite:
 * - No conversion needed (12.62KB ONNX direct deployment)
 * - Proven performance (0.0119ms P95)
 * - NNAPI hardware acceleration support
 * - Cross-platform (same model for iOS)
 *
 * Input: 18-dimensional feature vector
 * - Features extracted from 60-second window:
 *   speed_mean, speed_std, speed_max, speed_min,
 *   rpm_mean, rpm_std, throttle_mean, throttle_std, throttle_max,
 *   brake_mean, brake_std, brake_max,
 *   accel_x_mean, accel_x_std, accel_x_max,
 *   accel_y_mean, accel_y_std,
 *   fuel_consumption
 *
 * Output: Predicted driving behavior class
 * - 0: normal
 * - 1: eco_driving
 * - 2: aggressive
 *
 * Usage:
 * ```kotlin
 * val engine = LightGBMONNXEngine(context)
 * val features = floatArrayOf(...)  // 18 features
 * val behavior = engine.predict(features)  // 0, 1, or 2
 * engine.close()
 * ```
 */
class LightGBMONNXEngine(
    private val context: Context,
    modelAssetPath: String = "models/lightgbm_behavior.onnx"
) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    // Performance tracking
    private var inferenceCount = 0L
    private var totalLatencyMs = 0.0
    private var maxLatencyMs = 0.0
    private var minLatencyMs = Double.MAX_VALUE

    companion object {
        private const val TAG = "LightGBMONNXEngine"

        // Model constants
        const val INPUT_SIZE = 18
        const val NUM_CLASSES = 3

        // Behavior class names
        val BEHAVIOR_NAMES = arrayOf("normal", "eco_driving", "aggressive")

        // Performance thresholds (from PoC)
        const val TARGET_LATENCY_MS = 5.0  // P95 target
        const val POC_LATENCY_MS = 0.0119  // Proven P95 latency
    }

    init {
        Log.d(TAG, "Initializing LightGBM ONNX Engine")
        Log.d(TAG, "  Model: $modelAssetPath")

        try {
            // Load model from assets
            val modelBytes = context.assets.open(modelAssetPath).use { it.readBytes() }
            Log.d(TAG, "  Model loaded: ${modelBytes.size} bytes")

            // Create session options
            val options = SessionOptions().apply {
                // Use 4 threads for inference
                setIntraOpNumThreads(4)

                // Enable NNAPI for hardware acceleration
                try {
                    addNnapi()
                    Log.d(TAG, "  NNAPI acceleration: ENABLED")
                } catch (e: Exception) {
                    Log.w(TAG, "  NNAPI acceleration: DISABLED (not available)")
                }

                // Set optimization level
                setOptimizationLevel(OptLevel.ALL_OPT)

                // Set execution mode
                setExecutionMode(ExecutionMode.SEQUENTIAL)
            }

            // Create ONNX Runtime session
            session = env.createSession(modelBytes, options)

            // Log input/output info
            val inputInfo = session.inputInfo
            val outputInfo = session.outputInfo

            Log.d(TAG, "  Inputs: ${inputInfo.keys}")
            Log.d(TAG, "  Outputs: ${outputInfo.keys}")

            // Validate input shape
            inputInfo.values.firstOrNull()?.let { info ->
                val shape = (info.info as? TensorInfo)?.shape
                Log.d(TAG, "  Expected input shape: ${shape?.contentToString()}")
            }

            Log.d(TAG, "✅ LightGBM ONNX Engine initialized successfully")

        } catch (e: Exception) {
            Log.e(TAG, "❌ Failed to initialize ONNX engine", e)
            throw RuntimeException("Failed to initialize LightGBM ONNX Engine", e)
        }
    }

    /**
     * Predict driving behavior from feature vector
     *
     * @param features 18-dimensional feature vector (mean, std, max, min of sensors)
     * @return Predicted class: 0=normal, 1=eco_driving, 2=aggressive
     * @throws IllegalArgumentException if features.size != 18
     */
    fun predict(features: FloatArray): Int {
        require(features.size == INPUT_SIZE) {
            "Expected $INPUT_SIZE features, got ${features.size}"
        }

        val startTime = System.nanoTime()

        try {
            // Create input tensor (batch_size=1, features=18)
            val inputName = session.inputNames.first()
            val inputTensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(features),
                longArrayOf(1, INPUT_SIZE.toLong())
            )

            // Run inference
            val outputs = session.run(mapOf(inputName to inputTensor))

            // Parse output
            // ONNX LightGBM returns: [labels (int64), probabilities (sequence<map<int64,float>>)]
            val predictedClass = when {
                outputs.size() >= 2 -> {
                    // Get probabilities from second output
                    val probsOutput = outputs[1].value

                    // Handle different output formats
                    when (probsOutput) {
                        is Array<*> -> {
                            // Array of maps: [{0: 0.1, 1: 0.8, 2: 0.1}]
                            @Suppress("UNCHECKED_CAST")
                            val probs = probsOutput[0] as? Map<Long, Float>
                            probs?.maxByOrNull { it.value }?.key?.toInt() ?: 0
                        }
                        is List<*> -> {
                            // List of maps
                            @Suppress("UNCHECKED_CAST")
                            val probs = probsOutput[0] as? Map<Long, Float>
                            probs?.maxByOrNull { it.value }?.key?.toInt() ?: 0
                        }
                        else -> {
                            // Fallback: use label output (first output)
                            val labels = outputs[0].value as? LongArray
                            labels?.get(0)?.toInt() ?: 0
                        }
                    }
                }
                else -> {
                    // Single output: labels
                    val labels = outputs[0].value as? LongArray
                    labels?.get(0)?.toInt() ?: 0
                }
            }

            // Track performance
            val latencyMs = (System.nanoTime() - startTime) / 1_000_000.0
            updatePerformanceMetrics(latencyMs)

            // Release resources
            inputTensor.close()
            outputs.close()

            Log.d(TAG, "Prediction: class=$predictedClass (${BEHAVIOR_NAMES[predictedClass]}), latency=${String.format("%.4f", latencyMs)}ms")

            return predictedClass

        } catch (e: Exception) {
            Log.e(TAG, "Prediction failed", e)
            throw RuntimeException("ONNX inference failed", e)
        }
    }

    /**
     * Predict with probability scores
     *
     * @param features 18-dimensional feature vector
     * @return Pair of (predicted_class, probability_map)
     */
    fun predictWithProbabilities(features: FloatArray): Pair<Int, Map<Int, Float>> {
        require(features.size == INPUT_SIZE) {
            "Expected $INPUT_SIZE features, got ${features.size}"
        }

        try {
            // Create input tensor
            val inputName = session.inputNames.first()
            val inputTensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(features),
                longArrayOf(1, INPUT_SIZE.toLong())
            )

            // Run inference
            val outputs = session.run(mapOf(inputName to inputTensor))

            // Get probabilities
            val probabilities: Map<Int, Float> = if (outputs.size() >= 2) {
                val probsOutput = outputs[1].value
                when (probsOutput) {
                    is Array<*> -> {
                        @Suppress("UNCHECKED_CAST")
                        val probs = probsOutput[0] as? Map<Long, Float> ?: emptyMap()
                        probs.mapKeys { it.key.toInt() }
                    }
                    is List<*> -> {
                        @Suppress("UNCHECKED_CAST")
                        val probs = probsOutput[0] as? Map<Long, Float> ?: emptyMap()
                        probs.mapKeys { it.key.toInt() }
                    }
                    else -> emptyMap()
                }
            } else {
                emptyMap()
            }

            // Get predicted class
            val predictedClass = probabilities.maxByOrNull { it.value }?.key ?: 0

            // Release resources
            inputTensor.close()
            outputs.close()

            return Pair(predictedClass, probabilities)

        } catch (e: Exception) {
            Log.e(TAG, "Prediction with probabilities failed", e)
            throw RuntimeException("ONNX inference failed", e)
        }
    }

    /**
     * Get performance metrics
     */
    fun getPerformanceMetrics(): PerformanceMetrics {
        return PerformanceMetrics(
            inferenceCount = inferenceCount,
            avgLatencyMs = if (inferenceCount > 0) totalLatencyMs / inferenceCount else 0.0,
            minLatencyMs = if (inferenceCount > 0) minLatencyMs else 0.0,
            maxLatencyMs = maxLatencyMs,
            meetsTarget = (totalLatencyMs / inferenceCount.coerceAtLeast(1)) < TARGET_LATENCY_MS
        )
    }

    /**
     * Reset performance metrics
     */
    fun resetPerformanceMetrics() {
        inferenceCount = 0L
        totalLatencyMs = 0.0
        maxLatencyMs = 0.0
        minLatencyMs = Double.MAX_VALUE
    }

    private fun updatePerformanceMetrics(latencyMs: Double) {
        inferenceCount++
        totalLatencyMs += latencyMs
        maxLatencyMs = maxOf(maxLatencyMs, latencyMs)
        minLatencyMs = minOf(minLatencyMs, latencyMs)

        // Log performance warning if exceeding target
        if (latencyMs > TARGET_LATENCY_MS) {
            Log.w(TAG, "⚠️ Latency ${String.format("%.4f", latencyMs)}ms exceeds target ${TARGET_LATENCY_MS}ms")
        }
    }

    override fun close() {
        try {
            session.close()
            env.close()

            val metrics = getPerformanceMetrics()
            Log.d(TAG, "LightGBM ONNX Engine closed")
            Log.d(TAG, "  Total inferences: ${metrics.inferenceCount}")
            Log.d(TAG, "  Avg latency: ${String.format("%.4f", metrics.avgLatencyMs)}ms")
            Log.d(TAG, "  Min latency: ${String.format("%.4f", metrics.minLatencyMs)}ms")
            Log.d(TAG, "  Max latency: ${String.format("%.4f", metrics.maxLatencyMs)}ms")
            Log.d(TAG, "  Meets target (<${TARGET_LATENCY_MS}ms): ${metrics.meetsTarget}")

        } catch (e: Exception) {
            Log.e(TAG, "Error closing ONNX engine", e)
        }
    }

    /**
     * Performance metrics data class
     */
    data class PerformanceMetrics(
        val inferenceCount: Long,
        val avgLatencyMs: Double,
        val minLatencyMs: Double,
        val maxLatencyMs: Double,
        val meetsTarget: Boolean
    )
}

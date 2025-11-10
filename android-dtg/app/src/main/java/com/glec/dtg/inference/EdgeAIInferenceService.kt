package com.glec.dtg.inference

import android.content.Context
import android.util.Log
import com.glec.dtg.models.CANData
import com.glec.dtg.models.DrivingBehavior

/**
 * GLEC DTG - Edge AI Inference Service
 *
 * Orchestrates AI inference for driving behavior classification using LightGBM model.
 * Manages the complete inference pipeline:
 * 1. Collect CAN data samples (1Hz, 60-second windows)
 * 2. Extract statistical features (18-dimensional vectors)
 * 3. Run LightGBM ONNX inference
 * 4. Return behavior classification with confidence
 *
 * Future Extensions:
 * - TCN for fuel consumption prediction
 * - LSTM-AE for anomaly detection
 * - Multi-model ensemble predictions
 *
 * Usage:
 * ```kotlin
 * val service = EdgeAIInferenceService(context)
 *
 * // Collect samples at 1Hz
 * canDataStream.forEach { sample ->
 *     service.addSample(sample)
 *
 *     if (service.isReady()) {
 *         val result = service.runInference()
 *         Log.i(TAG, "Behavior: ${result?.behavior}")
 *     }
 * }
 * ```
 *
 * Performance:
 * - Feature Extraction: < 1ms
 * - LightGBM Inference: 0.0119ms P95 (validated)
 * - Total Pipeline: < 5ms target
 *
 * Thread Safety:
 * - All public methods are thread-safe
 * - Can be called from coroutines or background threads
 *
 * @param context Android application context
 * @param lightGBMEngine ONNX Runtime inference engine (default: created from context)
 * @param featureExtractor Feature extraction utility (default: 60-sample window)
 */
class EdgeAIInferenceService(
    private val context: Context,
    private val lightGBMEngine: LightGBMONNXEngine = LightGBMONNXEngine(context),
    private val featureExtractor: FeatureExtractor = FeatureExtractor()
) : AutoCloseable {

    // Performance tracking
    private var inferenceCount = 0
    private var totalLatencyMs = 0L
    private var maxLatencyMs = 0L
    private var minLatencyMs = Long.MAX_VALUE

    // Thread safety
    private val lock = Any()

    companion object {
        private const val TAG = "EdgeAIInferenceService"

        /**
         * Map LightGBM class indices to behavior enum
         */
        private val CLASS_TO_BEHAVIOR = mapOf(
            0 to DrivingBehavior.NORMAL,
            1 to DrivingBehavior.ECO_DRIVING,
            2 to DrivingBehavior.AGGRESSIVE
        )
    }

    /**
     * Add a CAN data sample to the feature extraction window
     *
     * Thread-safe: Can be called from multiple threads
     *
     * @param sample CAN data sample from 1Hz stream
     */
    fun addSample(sample: CANData) {
        synchronized(lock) {
            featureExtractor.addSample(sample)
        }
    }

    /**
     * Check if service has enough samples for inference
     *
     * @return true if 60-sample window is ready
     */
    fun isReady(): Boolean {
        synchronized(lock) {
            return featureExtractor.isWindowReady()
        }
    }

    /**
     * Get current number of samples in window
     *
     * @return Sample count (0-60)
     */
    fun getSampleCount(): Int {
        synchronized(lock) {
            return featureExtractor.getSampleCount()
        }
    }

    /**
     * Run inference on current 60-second window
     *
     * Extracts features and performs LightGBM classification.
     * Returns null if window is not ready (< 60 samples).
     *
     * Thread-safe: Can be called from coroutines
     *
     * @return Inference result with behavior and latency, or null if not ready
     */
    fun runInference(): InferenceResult? {
        synchronized(lock) {
            if (!isReady()) {
                Log.w(TAG, "Inference called but window not ready (${getSampleCount()}/60 samples)")
                return null
            }

            val startTime = System.currentTimeMillis()

            try {
                // Extract 18-dimensional feature vector
                val features = featureExtractor.extractFeatures()
                    ?: return null.also {
                        Log.e(TAG, "Feature extraction returned null")
                    }

                Log.d(TAG, "Features extracted: ${features.size} dimensions")

                // Run LightGBM inference
                val predictedClass = lightGBMEngine.predict(features)

                // Map class to behavior
                val behavior = CLASS_TO_BEHAVIOR[predictedClass] ?: DrivingBehavior.NORMAL

                // Calculate latency
                val latency = System.currentTimeMillis() - startTime

                // Update performance metrics
                updatePerformanceMetrics(latency)

                Log.i(TAG, "Inference completed: behavior=$behavior, latency=${latency}ms")

                return InferenceResult(
                    behavior = behavior,
                    latencyMs = latency,
                    confidence = 1.0f,  // Default confidence
                    timestamp = System.currentTimeMillis()
                )

            } catch (e: Exception) {
                Log.e(TAG, "Inference failed", e)
                return null
            }
        }
    }

    /**
     * Run inference with confidence scores from probability distribution
     *
     * Uses predictWithProbabilities() to get class probabilities.
     * Confidence = max(probabilities)
     *
     * @return Inference result with behavior, latency, and confidence, or null if not ready
     */
    fun runInferenceWithConfidence(): InferenceResult? {
        synchronized(lock) {
            if (!isReady()) {
                Log.w(TAG, "Inference called but window not ready")
                return null
            }

            val startTime = System.currentTimeMillis()

            try {
                // Extract features
                val features = featureExtractor.extractFeatures() ?: return null

                // Run inference with probabilities
                val (predictedClass, probabilities) = lightGBMEngine.predictWithProbabilities(features)

                // Map class to behavior
                val behavior = CLASS_TO_BEHAVIOR[predictedClass] ?: DrivingBehavior.NORMAL

                // Get confidence (max probability)
                val confidence = probabilities[predictedClass] ?: 1.0f

                // Calculate latency
                val latency = System.currentTimeMillis() - startTime

                // Update performance metrics
                updatePerformanceMetrics(latency)

                Log.i(TAG, "Inference completed: behavior=$behavior, confidence=$confidence, latency=${latency}ms")
                Log.d(TAG, "Probabilities: $probabilities")

                return InferenceResult(
                    behavior = behavior,
                    latencyMs = latency,
                    confidence = confidence,
                    timestamp = System.currentTimeMillis()
                )

            } catch (e: Exception) {
                Log.e(TAG, "Inference with confidence failed", e)
                return null
            }
        }
    }

    /**
     * Reset the feature extraction window
     *
     * Clears all collected samples. Use when starting a new session or
     * when data stream is interrupted.
     */
    fun reset() {
        synchronized(lock) {
            featureExtractor.reset()
            Log.d(TAG, "Feature extractor reset")
        }
    }

    /**
     * Reset performance metrics
     */
    fun resetPerformanceMetrics() {
        synchronized(lock) {
            inferenceCount = 0
            totalLatencyMs = 0L
            maxLatencyMs = 0L
            minLatencyMs = Long.MAX_VALUE
            Log.d(TAG, "Performance metrics reset")
        }
    }

    /**
     * Get performance metrics for monitoring
     *
     * @return Performance metrics (inference count, latencies)
     */
    fun getPerformanceMetrics(): InferencePerformanceMetrics {
        synchronized(lock) {
            return InferencePerformanceMetrics(
                inferenceCount = inferenceCount,
                avgLatencyMs = if (inferenceCount > 0) totalLatencyMs.toDouble() / inferenceCount else 0.0,
                maxLatencyMs = if (inferenceCount > 0) maxLatencyMs.toDouble() else 0.0,
                minLatencyMs = if (inferenceCount > 0) minLatencyMs.toDouble() else 0.0
            )
        }
    }

    /**
     * Update performance metrics (thread-safe)
     */
    private fun updatePerformanceMetrics(latencyMs: Long) {
        inferenceCount++
        totalLatencyMs += latencyMs
        maxLatencyMs = maxOf(maxLatencyMs, latencyMs)
        minLatencyMs = minOf(minLatencyMs, latencyMs)

        // Log warning if latency exceeds target
        if (latencyMs > 5) {
            Log.w(TAG, "⚠️ Latency ${latencyMs}ms exceeds 5ms target")
        }
    }

    /**
     * Close resources (ONNX Runtime engine)
     */
    override fun close() {
        try {
            lightGBMEngine.close()

            val metrics = getPerformanceMetrics()
            Log.i(TAG, "EdgeAIInferenceService closed")
            Log.i(TAG, "  Total inferences: ${metrics.inferenceCount}")
            Log.i(TAG, "  Avg latency: ${String.format("%.2f", metrics.avgLatencyMs)}ms")
            Log.i(TAG, "  Max latency: ${String.format("%.2f", metrics.maxLatencyMs)}ms")
            Log.i(TAG, "  Min latency: ${String.format("%.2f", metrics.minLatencyMs)}ms")

        } catch (e: Exception) {
            Log.e(TAG, "Error closing EdgeAIInferenceService", e)
        }
    }
}

/**
 * Inference result data class
 *
 * Represents the output of a single inference run on a 60-second window
 *
 * @property behavior Predicted driving behavior class
 * @property latencyMs Total inference latency (feature extraction + model inference)
 * @property confidence Confidence score (0.0-1.0) from probability distribution
 * @property timestamp Unix timestamp (milliseconds) when inference completed
 */
data class InferenceResult(
    val behavior: DrivingBehavior,
    val latencyMs: Long,
    val confidence: Float = 1.0f,
    val timestamp: Long = System.currentTimeMillis()
) {
    /**
     * Check if inference meets latency target (<5ms)
     */
    fun meetsLatencyTarget(): Boolean {
        return latencyMs < 5
    }

    /**
     * Check if prediction is high confidence (>0.7)
     */
    fun isHighConfidence(): Boolean {
        return confidence > 0.7f
    }

    override fun toString(): String {
        return "InferenceResult(behavior=$behavior, latency=${latencyMs}ms, confidence=${String.format("%.2f", confidence)})"
    }
}

/**
 * Performance metrics data class
 *
 * Tracks aggregate performance across multiple inference runs
 *
 * @property inferenceCount Total number of inferences performed
 * @property avgLatencyMs Average inference latency
 * @property maxLatencyMs Maximum inference latency observed
 * @property minLatencyMs Minimum inference latency observed
 */
data class InferencePerformanceMetrics(
    val inferenceCount: Int,
    val avgLatencyMs: Double,
    val maxLatencyMs: Double,
    val minLatencyMs: Double
) {
    /**
     * Check if average latency meets target (<5ms)
     */
    fun meetsTarget(): Boolean {
        return avgLatencyMs < 5.0
    }

    override fun toString(): String {
        return "InferencePerformanceMetrics(count=$inferenceCount, " +
                "avg=${String.format("%.2f", avgLatencyMs)}ms, " +
                "max=${String.format("%.2f", maxLatencyMs)}ms, " +
                "min=${String.format("%.2f", minLatencyMs)}ms)"
    }
}

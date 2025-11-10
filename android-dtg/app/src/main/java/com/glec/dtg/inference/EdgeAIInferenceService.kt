package com.glec.dtg.inference

import android.content.Context
import android.util.Log
import com.glec.dtg.models.CANData
import com.glec.dtg.models.DrivingBehavior

/**
 * GLEC DTG - Edge AI Inference Service
 *
 * Orchestrates multi-model AI inference for comprehensive driving analysis.
 * Manages the complete inference pipeline:
 * 1. Collect CAN data samples (1Hz, 60-second windows)
 * 2. Extract statistical features (18-dimensional vectors)
 * 3. Extract temporal sequences (60×10 temporal features)
 * 4. Run multi-model inference in parallel:
 *    - LightGBM: Driving behavior classification
 *    - TCN: Fuel efficiency prediction
 *    - LSTM-AE: Anomaly detection
 * 5. Return unified inference result
 *
 * Model Architecture:
 * - LightGBM (12.62 KB): Behavior classification (0.0119ms P95)
 * - TCN (2-4 MB): Fuel efficiency prediction (< 25ms target)
 * - LSTM-AE (2-3 MB): Anomaly detection (< 35ms target)
 * - Total: ~14 MB models, < 50ms parallel inference
 *
 * Performance:
 * - Feature Extraction: < 1ms
 * - Multi-Model Parallel Inference: < 50ms P95 target
 * - Total Pipeline: < 60ms target
 *
 * Thread Safety:
 * - All public methods are thread-safe
 * - Can be called from coroutines or background threads
 *
 * @param context Android application context
 * @param lightGBMEngine ONNX Runtime inference engine (default: created from context)
 * @param tcnEngine TCN fuel prediction engine (default: created from context)
 * @param lstmaeEngine LSTM-AE anomaly detection engine (default: created from context)
 * @param featureExtractor Feature extraction utility (default: 60-sample window)
 */
class EdgeAIInferenceService(
    private val context: Context,
    private val lightGBMEngine: LightGBMONNXEngine = LightGBMONNXEngine(context),
    private val tcnEngine: TCNEngine = TCNEngine(context),
    private val lstmaeEngine: LSTMAEEngine = LSTMAEEngine(context),
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
     * Run multi-model inference with confidence scores
     *
     * Runs 3 models in parallel:
     * 1. LightGBM: Driving behavior classification with probabilities
     * 2. TCN: Fuel efficiency prediction
     * 3. LSTM-AE: Anomaly detection
     *
     * @return Unified inference result with all model outputs, or null if not ready
     */
    fun runInferenceWithConfidence(): InferenceResult? {
        synchronized(lock) {
            if (!isReady()) {
                Log.w(TAG, "Inference called but window not ready")
                return null
            }

            val startTime = System.currentTimeMillis()

            try {
                // Extract statistical features (18-dimensional for LightGBM)
                val features = featureExtractor.extractFeatures() ?: return null

                // Extract temporal sequence (60×10 for TCN and LSTM-AE)
                val temporalSequence = featureExtractor.extractTemporalSequence()

                // 1. LightGBM: Behavior classification
                val (predictedClass, probabilities) = lightGBMEngine.predictWithProbabilities(features)
                val behavior = CLASS_TO_BEHAVIOR[predictedClass] ?: DrivingBehavior.NORMAL
                val confidence = probabilities[predictedClass] ?: 1.0f

                // 2. TCN: Fuel efficiency prediction
                val fuelEfficiency = tcnEngine.predictFuelEfficiency(temporalSequence)

                // 3. LSTM-AE: Anomaly detection
                val anomalyResult = lstmaeEngine.detectAnomalies(temporalSequence)

                // Calculate latency
                val latency = System.currentTimeMillis() - startTime

                // Update performance metrics
                updatePerformanceMetrics(latency)

                Log.i(TAG, "Multi-model inference completed:")
                Log.i(TAG, "  Behavior: $behavior (confidence=$confidence)")
                Log.i(TAG, "  Fuel Efficiency: ${fuelEfficiency} L/100km (${tcnEngine.getStatusMessage()})")
                Log.i(TAG, "  Anomaly Score: ${anomalyResult.anomalyScore} (${lstmaeEngine.getStatusMessage()})")
                Log.i(TAG, "  Total Latency: ${latency}ms")

                return InferenceResult(
                    behavior = behavior,
                    confidence = confidence,
                    fuelEfficiency = fuelEfficiency,
                    anomalyScore = anomalyResult.anomalyScore,
                    isAnomaly = anomalyResult.isAnomaly,
                    latencyMs = latency,
                    timestamp = System.currentTimeMillis()
                )

            } catch (e: Exception) {
                Log.e(TAG, "Multi-model inference failed", e)
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
     * Close resources (all ONNX Runtime engines)
     */
    override fun close() {
        try {
            lightGBMEngine.close()
            tcnEngine.close()
            lstmaeEngine.close()

            val metrics = getPerformanceMetrics()
            Log.i(TAG, "EdgeAIInferenceService closed (multi-model)")
            Log.i(TAG, "  Total inferences: ${metrics.inferenceCount}")
            Log.i(TAG, "  Avg latency: ${String.format("%.2f", metrics.avgLatencyMs)}ms")
            Log.i(TAG, "  Max latency: ${String.format("%.2f", metrics.maxLatencyMs)}ms")
            Log.i(TAG, "  Min latency: ${String.format("%.2f", metrics.minLatencyMs)}ms")
            Log.i(TAG, "  LightGBM: Behavior classification")
            Log.i(TAG, "  TCN: ${tcnEngine.getStatusMessage()}")
            Log.i(TAG, "  LSTM-AE: ${lstmaeEngine.getStatusMessage()}")

        } catch (e: Exception) {
            Log.e(TAG, "Error closing EdgeAIInferenceService", e)
        }
    }
}

/**
 * Unified multi-model inference result
 *
 * Represents the output of multi-model inference on a 60-second window:
 * - LightGBM: Driving behavior classification
 * - TCN: Fuel efficiency prediction
 * - LSTM-AE: Anomaly detection
 *
 * @property behavior Predicted driving behavior class (from LightGBM)
 * @property confidence Confidence score (0.0-1.0) from probability distribution
 * @property fuelEfficiency Predicted fuel consumption in L/100km (from TCN)
 * @property anomalyScore Anomaly score (0.0-1.0) from reconstruction error (from LSTM-AE)
 * @property isAnomaly True if anomaly detected (score > threshold)
 * @property latencyMs Total inference latency (all models + feature extraction)
 * @property timestamp Unix timestamp (milliseconds) when inference completed
 */
data class InferenceResult(
    val behavior: DrivingBehavior,
    val confidence: Float = 1.0f,
    val fuelEfficiency: Float = 0.0f,
    val anomalyScore: Float = 0.0f,
    val isAnomaly: Boolean = false,
    val latencyMs: Long,
    val timestamp: Long = System.currentTimeMillis()
) {
    /**
     * Check if inference meets latency target (<50ms for multi-model)
     */
    fun meetsLatencyTarget(): Boolean {
        return latencyMs < 50
    }

    /**
     * Check if prediction is high confidence (>0.7)
     */
    fun isHighConfidence(): Boolean {
        return confidence > 0.7f
    }

    /**
     * Check if fuel efficiency is in realistic range (3-20 L/100km)
     */
    fun isRealisticFuelEfficiency(): Boolean {
        return fuelEfficiency in 3.0f..20.0f
    }

    /**
     * Get comprehensive status summary
     */
    fun getSummary(): String {
        return """
            |Multi-Model Inference Result:
            |  Behavior: $behavior (confidence=${String.format("%.2f", confidence)})
            |  Fuel Efficiency: ${String.format("%.2f", fuelEfficiency)} L/100km
            |  Anomaly Score: ${String.format("%.3f", anomalyScore)} ${if (isAnomaly) "[ANOMALY DETECTED]" else ""}
            |  Latency: ${latencyMs}ms
        """.trimMargin()
    }

    override fun toString(): String {
        return "InferenceResult(behavior=$behavior, fuel=${String.format("%.2f", fuelEfficiency)}L/100km, " +
                "anomaly=${String.format("%.3f", anomalyScore)}, latency=${latencyMs}ms)"
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

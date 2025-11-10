package com.glec.dtg.inference

import android.content.Context
import android.util.Log
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * LSTM-AE (LSTM Autoencoder) Engine for Anomaly Detection
 *
 * Detects anomalous driving patterns using reconstruction error from autoencoder.
 * Uses ONNX Runtime Mobile for inference.
 *
 * Model Architecture:
 * - Encoder: LSTM(64) → LSTM(32) → Latent(16)
 * - Decoder: LSTM(32) → LSTM(64) → Output
 * - Input/Output: 60 timesteps × 10 features
 *
 * Anomaly Detection:
 * - Reconstruction error > threshold → Anomaly
 * - Threshold: 95th percentile from training data
 *
 * Performance Target:
 * - Latency: < 35ms (P95)
 * - Model Size: 2-3 MB
 * - F1 Score: > 0.85
 *
 * @param context Android application context
 */
class LSTMAEEngine(private val context: Context) : AutoCloseable {

    companion object {
        private const val TAG = "LSTMAEEngine"
        private const val MODEL_PATH = "models/lstm_ae_anomaly.onnx"

        // Feature configuration
        private const val SEQUENCE_LENGTH = 60  // 60 seconds
        private const val FEATURE_DIM = 10       // 10 temporal features

        // Anomaly threshold (will be calibrated with real model)
        private const val ANOMALY_THRESHOLD = 0.15f  // Normalized reconstruction error
    }

    // ONNX Runtime session (placeholder - will be initialized when model is available)
    private var ortSession: Any? = null

    // Model loaded flag
    private var isModelLoaded = false

    // Statistics for stub mode
    private var recentErrors = mutableListOf<Float>()
    private val maxHistorySize = 100

    init {
        try {
            // TODO: Initialize ONNX Runtime when model is available
            // val modelBytes = context.assets.open(MODEL_PATH).readBytes()
            // ortSession = OrtSession.create(modelBytes)

            Log.i(TAG, "LSTM-AE Engine initialized (stub mode - waiting for ONNX model)")
            isModelLoaded = false
        } catch (e: Exception) {
            Log.w(TAG, "LSTM-AE model not found, using stub detection", e)
            isModelLoaded = false
        }
    }

    /**
     * Detect anomalies in temporal sequence
     *
     * @param sequence 60×10 temporal feature array
     * @return AnomalyDetectionResult with score and anomaly flag
     */
    fun detectAnomalies(sequence: Array<FloatArray>): AnomalyDetectionResult {
        if (!isModelLoaded) {
            // Stub detection based on statistical deviation
            return calculateStubDetection(sequence)
        }

        // TODO: Real ONNX inference when model is available
        // val inputTensor = OrtTensor.createTensor(sequence)
        // val outputs = ortSession.run(mapOf("input" to inputTensor))
        // val reconstructed = outputs["output"].floatArray
        // val error = calculateReconstructionError(sequence, reconstructed)
        // return AnomalyDetectionResult(error, error > ANOMALY_THRESHOLD)

        return calculateStubDetection(sequence)
    }

    /**
     * Calculate stub anomaly detection (statistical method)
     *
     * Uses standard deviation and sudden changes to detect anomalies
     * until real LSTM-AE model is available.
     */
    private fun calculateStubDetection(sequence: Array<FloatArray>): AnomalyDetectionResult {
        if (sequence.isEmpty() || sequence[0].size < 3) {
            return AnomalyDetectionResult(0.0f, false)
        }

        var anomalyScore = 0.0f
        var anomalyCount = 0

        // Check for sudden changes in key features
        for (i in 1 until sequence.size) {
            val prev = sequence[i - 1]
            val curr = sequence[i]

            // Speed deviation
            val speedChange = abs(curr.getOrNull(0) ?: 0f - prev.getOrNull(0) ?: 0f)
            if (speedChange > 30.0f) {  // > 30 km/h sudden change
                anomalyScore += 0.3f
                anomalyCount++
            }

            // RPM deviation
            val rpmChange = abs(curr.getOrNull(1) ?: 0f - prev.getOrNull(1) ?: 0f)
            if (rpmChange > 1000.0f) {  // > 1000 RPM sudden change
                anomalyScore += 0.2f
                anomalyCount++
            }

            // Throttle spike
            val throttle = curr.getOrNull(2) ?: 0f
            if (throttle > 95.0f) {  // Full throttle
                anomalyScore += 0.1f
            }
        }

        // Calculate variance across sequence
        val speedVariance = calculateVariance(sequence, 0)
        if (speedVariance > 200.0f) {  // High speed variance
            anomalyScore += 0.2f
            anomalyCount++
        }

        // Normalize score (divide by max reasonable score to keep in [0, 1] range)
        // Max score estimation: ~3-5 anomalies × 0.3 each = ~1.5
        val normalizedScore = (anomalyScore / 2.0f).coerceIn(0.0f, 1.0f)

        // Update history
        recentErrors.add(normalizedScore)
        if (recentErrors.size > maxHistorySize) {
            recentErrors.removeAt(0)
        }

        val isAnomaly = normalizedScore > ANOMALY_THRESHOLD || anomalyCount > 3

        return AnomalyDetectionResult(normalizedScore, isAnomaly)
    }

    /**
     * Calculate variance of a feature across sequence
     */
    private fun calculateVariance(sequence: Array<FloatArray>, featureIndex: Int): Float {
        val values = sequence.mapNotNull { it.getOrNull(featureIndex) }
        if (values.isEmpty()) return 0.0f

        val mean = values.average().toFloat()
        val variance = values.map { (it - mean) * (it - mean) }.average().toFloat()

        return variance
    }

    /**
     * Calculate reconstruction error (for real ONNX model)
     */
    private fun calculateReconstructionError(
        original: Array<FloatArray>,
        reconstructed: FloatArray
    ): Float {
        var totalError = 0.0f
        var count = 0

        for (i in original.indices) {
            for (j in original[i].indices) {
                val idx = i * original[i].size + j
                if (idx < reconstructed.size) {
                    val error = (original[i][j] - reconstructed[idx])
                    totalError += error * error
                    count++
                }
            }
        }

        // Root mean squared error
        return if (count > 0) sqrt(totalError / count) else 0.0f
    }

    /**
     * Check if real ONNX model is loaded
     */
    fun isModelAvailable(): Boolean = isModelLoaded

    /**
     * Get model status message
     */
    fun getStatusMessage(): String {
        return if (isModelLoaded) {
            "LSTM-AE model loaded (ONNX Runtime)"
        } else {
            "LSTM-AE stub mode (statistical detection)"
        }
    }

    /**
     * Get average anomaly score from recent history
     */
    fun getAverageAnomalyScore(): Float {
        return if (recentErrors.isNotEmpty()) {
            recentErrors.average().toFloat()
        } else {
            0.0f
        }
    }

    override fun close() {
        try {
            // TODO: ortSession?.close()
            Log.i(TAG, "LSTM-AE Engine closed")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing LSTM-AE engine", e)
        }
    }
}

/**
 * Anomaly detection result
 *
 * @property anomalyScore Reconstruction error score (0.0 - 1.0)
 * @property isAnomaly True if score exceeds threshold
 */
data class AnomalyDetectionResult(
    val anomalyScore: Float,
    val isAnomaly: Boolean
)

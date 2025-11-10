package com.glec.dtg.inference

import com.glec.dtg.models.CANData
import kotlin.math.sqrt

/**
 * GLEC DTG - Feature Extractor for LightGBM Behavior Classification
 *
 * Converts 60-second windows of raw CAN data into 18-dimensional feature vectors
 * for LightGBM ONNX model inference.
 *
 * Feature Vector (18 dimensions):
 * [0-3]:   speed_mean, speed_std, speed_max, speed_min
 * [4-5]:   rpm_mean, rpm_std
 * [6-8]:   throttle_mean, throttle_std, throttle_max
 * [9-11]:  brake_mean, brake_std, brake_max
 * [12-14]: accel_x_mean, accel_x_std, accel_x_max
 * [15-16]: accel_y_mean, accel_y_std
 * [17]:    fuel_consumption (mean L/100km)
 *
 * Usage:
 * ```kotlin
 * val extractor = FeatureExtractor(windowSize = 60)
 *
 * // Add samples at 1Hz
 * canDataStream.forEach { sample ->
 *     extractor.addSample(sample)
 *
 *     if (extractor.isWindowReady()) {
 *         val features = extractor.extractFeatures()
 *         val behavior = lightGBMEngine.predict(features!!)
 *     }
 * }
 * ```
 *
 * @param windowSize Number of samples in sliding window (default: 60 for 60-second windows at 1Hz)
 */
class FeatureExtractor(
    private val windowSize: Int = 60
) {
    // Sliding window of CAN samples (FIFO queue)
    private val window = ArrayDeque<CANData>(windowSize)

    /**
     * Add a new CAN data sample to the sliding window
     *
     * @param sample CAN data sample to add
     */
    fun addSample(sample: CANData) {
        // Add new sample
        window.add(sample)

        // Remove oldest sample if window exceeds size (sliding window)
        if (window.size > windowSize) {
            window.removeFirst()
        }
    }

    /**
     * Check if window has enough samples for feature extraction
     *
     * @return true if window has windowSize samples
     */
    fun isWindowReady(): Boolean {
        return window.size == windowSize
    }

    /**
     * Get current number of samples in window
     *
     * @return Sample count
     */
    fun getSampleCount(): Int {
        return window.size
    }

    /**
     * Extract 18-dimensional feature vector from current window
     *
     * @return Feature vector (FloatArray[18]) or null if window not ready
     */
    fun extractFeatures(): FloatArray? {
        if (!isWindowReady()) {
            return null
        }

        // Extract raw values from window
        val speeds = window.map { it.vehicleSpeed }
        val rpms = window.map { it.engineRPM.toFloat() }
        val throttles = window.map { it.throttlePosition }
        val brakes = window.map { it.brakePosition }
        val accelsX = window.map { it.accelerationX }
        val accelsY = window.map { it.accelerationY }
        val fuelConsumptions = window.map { it.calculateFuelConsumption() }

        // Calculate features (18 dimensions)
        return floatArrayOf(
            // [0-3] Speed statistics
            speeds.mean(),
            speeds.std(),
            speeds.maxOrNull() ?: 0.0f,
            speeds.minOrNull() ?: 0.0f,

            // [4-5] RPM statistics
            rpms.mean(),
            rpms.std(),

            // [6-8] Throttle statistics
            throttles.mean(),
            throttles.std(),
            throttles.maxOrNull() ?: 0.0f,

            // [9-11] Brake statistics
            brakes.mean(),
            brakes.std(),
            brakes.maxOrNull() ?: 0.0f,

            // [12-14] Acceleration X statistics
            accelsX.mean(),
            accelsX.std(),
            accelsX.maxOrNull() ?: 0.0f,

            // [15-16] Acceleration Y statistics
            accelsY.mean(),
            accelsY.std(),

            // [17] Fuel consumption (mean)
            fuelConsumptions.mean()
        )
    }

    /**
     * Reset the feature extractor (clear window)
     */
    fun reset() {
        window.clear()
    }

    /**
     * Extension function: Calculate mean of Float list
     */
    private fun List<Float>.mean(): Float {
        if (isEmpty()) return 0.0f
        return sum() / size
    }

    /**
     * Extension function: Calculate standard deviation of Float list
     */
    private fun List<Float>.std(): Float {
        if (size < 2) return 0.0f

        val mean = mean()
        val variance = map { (it - mean) * (it - mean) }.sum() / size
        return sqrt(variance)
    }

    companion object {
        /**
         * Feature vector dimension (must match LightGBM model input)
         */
        const val FEATURE_DIMENSION = 18

        /**
         * Recommended window size for 60-second windows at 1Hz sampling
         */
        const val DEFAULT_WINDOW_SIZE = 60

        /**
         * Feature names (for logging/debugging)
         */
        val FEATURE_NAMES = arrayOf(
            "speed_mean", "speed_std", "speed_max", "speed_min",
            "rpm_mean", "rpm_std",
            "throttle_mean", "throttle_std", "throttle_max",
            "brake_mean", "brake_std", "brake_max",
            "accel_x_mean", "accel_x_std", "accel_x_max",
            "accel_y_mean", "accel_y_std",
            "fuel_consumption"
        )
    }
}

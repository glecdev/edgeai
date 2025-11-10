package com.glec.dtg.inference

import com.glec.dtg.models.CANData
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

/**
 * GLEC DTG - Feature Extractor Unit Tests
 *
 * Tests the conversion of 60-second CAN data windows into 18-dimensional
 * feature vectors for LightGBM behavior classification.
 *
 * Feature Vector (18 dimensions):
 * [0-3]:   speed_mean, speed_std, speed_max, speed_min
 * [4-5]:   rpm_mean, rpm_std
 * [6-8]:   throttle_mean, throttle_std, throttle_max
 * [9-11]:  brake_mean, brake_std, brake_max
 * [12-14]: accel_x_mean, accel_x_std, accel_x_max
 * [15-16]: accel_y_mean, accel_y_std
 * [17]:    fuel_consumption
 */
class FeatureExtractorTest {

    private lateinit var extractor: FeatureExtractor

    @Before
    fun setUp() {
        extractor = FeatureExtractor(windowSize = 60)
    }

    @Test
    fun `test feature extractor initializes with empty window`() {
        assertFalse("Window should not be ready with 0 samples", extractor.isWindowReady())
        assertEquals("Sample count should be 0", 0, extractor.getSampleCount())
    }

    @Test
    fun `test feature extractor accepts samples`() {
        val sample = createTestSample(
            speed = 60.0f,
            rpm = 2000,
            throttle = 30.0f,
            brake = 0.0f,
            accelX = 0.5f,
            accelY = 0.1f,
            maf = 5.0f
        )

        extractor.addSample(sample)
        assertEquals("Sample count should be 1", 1, extractor.getSampleCount())
    }

    @Test
    fun `test window becomes ready after 60 samples`() {
        // Add 59 samples - should not be ready
        repeat(59) { i ->
            extractor.addSample(createTestSample(speed = 60.0f + i))
        }
        assertFalse("Window should not be ready with 59 samples", extractor.isWindowReady())

        // Add 60th sample - should be ready
        extractor.addSample(createTestSample(speed = 60.0f))
        assertTrue("Window should be ready with 60 samples", extractor.isWindowReady())
    }

    @Test
    fun `test extract features from uniform speed data`() {
        // Add 60 samples with constant speed
        val constantSpeed = 80.0f
        repeat(60) {
            extractor.addSample(
                createTestSample(
                    speed = constantSpeed,
                    rpm = 2500,
                    throttle = 40.0f,
                    brake = 0.0f,
                    accelX = 0.0f,
                    accelY = 0.0f,
                    maf = 6.0f
                )
            )
        }

        val features = extractor.extractFeatures()
        assertNotNull("Features should not be null", features)
        assertEquals("Feature vector should have 18 dimensions", 18, features!!.size)

        // Check speed statistics (constant value)
        assertEquals("Speed mean should equal constant speed", constantSpeed, features[0], 0.01f)
        assertEquals("Speed std should be 0 for constant values", 0.0f, features[1], 0.01f)
        assertEquals("Speed max should equal constant speed", constantSpeed, features[2], 0.01f)
        assertEquals("Speed min should equal constant speed", constantSpeed, features[3], 0.01f)
    }

    @Test
    fun `test extract features from varying speed data`() {
        // Add 60 samples with varying speed: 50-70 km/h
        repeat(60) { i ->
            val speed = 50.0f + (i % 21)  // Varies from 50 to 70
            extractor.addSample(
                createTestSample(
                    speed = speed,
                    rpm = 2000 + (i * 10),
                    throttle = 30.0f,
                    brake = 0.0f,
                    accelX = 0.2f,
                    accelY = 0.1f,
                    maf = 5.0f
                )
            )
        }

        val features = extractor.extractFeatures()
        assertNotNull("Features should not be null", features)

        // Check speed statistics
        assertTrue("Speed mean should be around 60", features!![0] in 55.0f..65.0f)
        assertTrue("Speed std should be > 0 for varying values", features[1] > 0.0f)
        assertTrue("Speed max should be around 70", features[2] >= 65.0f)
        assertTrue("Speed min should be around 50", features[3] <= 55.0f)

        // Check RPM statistics
        assertTrue("RPM mean should be > 0", features[4] > 0.0f)
        assertTrue("RPM std should be > 0 for varying values", features[5] > 0.0f)
    }

    @Test
    fun `test extract features from harsh braking scenario`() {
        // Simulate harsh braking: high speed → low speed with negative acceleration
        repeat(30) { i ->
            extractor.addSample(
                createTestSample(
                    speed = 90.0f,
                    rpm = 3000,
                    throttle = 0.0f,
                    brake = 0.0f,
                    accelX = 0.0f,
                    accelY = 0.0f,
                    maf = 3.0f
                )
            )
        }

        // Harsh braking
        repeat(30) { i ->
            extractor.addSample(
                createTestSample(
                    speed = 90.0f - (i * 2.5f),  // Decelerate
                    rpm = 3000 - (i * 50),
                    throttle = 0.0f,
                    brake = 80.0f,  // Heavy braking
                    accelX = -5.0f,  // Harsh deceleration
                    accelY = 0.0f,
                    maf = 2.0f
                )
            )
        }

        val features = extractor.extractFeatures()
        assertNotNull("Features should not be null", features)

        // Check brake statistics
        assertTrue("Brake mean should be > 30% (braking for half window)", features!![9] > 30.0f)
        assertTrue("Brake max should be around 80%", features[11] >= 75.0f)

        // Check acceleration statistics
        assertTrue("AccelX mean should be negative (deceleration)", features[12] < 0.0f)
        assertTrue("AccelX max should be negative (harsh braking)", features[14] < -3.0f)
    }

    @Test
    fun `test extract features from aggressive driving scenario`() {
        // Simulate aggressive driving: rapid acceleration, high throttle
        repeat(60) { i ->
            val isAccelerating = i < 30
            extractor.addSample(
                createTestSample(
                    speed = if (isAccelerating) 40.0f + (i * 2.0f) else 100.0f,
                    rpm = if (isAccelerating) 2000 + (i * 80) else 4500,
                    throttle = if (isAccelerating) 90.0f else 80.0f,
                    brake = 0.0f,
                    accelX = if (isAccelerating) 4.0f else 0.5f,  // Harsh acceleration
                    accelY = 0.2f,
                    maf = if (isAccelerating) 15.0f else 12.0f
                )
            )
        }

        val features = extractor.extractFeatures()
        assertNotNull("Features should not be null", features)

        // Check throttle statistics
        assertTrue("Throttle mean should be high (>70%)", features!![6] > 70.0f)
        assertTrue("Throttle max should be around 90%", features[8] >= 85.0f)

        // Check acceleration statistics
        assertTrue("AccelX mean should be positive (acceleration)", features[12] > 0.0f)
        assertTrue("AccelX max should be high (harsh acceleration)", features[14] > 3.0f)
    }

    @Test
    fun `test fuel consumption calculation`() {
        // Add samples with known MAF and speed
        repeat(60) {
            extractor.addSample(
                createTestSample(
                    speed = 80.0f,
                    rpm = 2500,
                    throttle = 40.0f,
                    brake = 0.0f,
                    accelX = 0.0f,
                    accelY = 0.0f,
                    maf = 8.0f  // g/s
                )
            )
        }

        val features = extractor.extractFeatures()
        assertNotNull("Features should not be null", features)

        // Fuel consumption should be calculated
        // Formula: (maf / 14.7 * 3600 / 750) / speed * 100
        // Expected: (8.0 / 14.7 * 3600 / 750) / 80.0 * 100 ≈ 3.27 L/100km
        assertTrue("Fuel consumption should be > 0", features!![17] > 0.0f)
        assertTrue("Fuel consumption should be reasonable (<20 L/100km)", features[17] < 20.0f)
    }

    @Test
    fun `test sliding window behavior`() {
        // Add 70 samples (10 more than window size)
        repeat(70) { i ->
            extractor.addSample(createTestSample(speed = 60.0f + i))
        }

        // Window should be ready
        assertTrue("Window should be ready", extractor.isWindowReady())

        // Sample count should be capped at window size
        assertEquals("Sample count should be window size (60)", 60, extractor.getSampleCount())

        // Extract features - should use most recent 60 samples
        val features = extractor.extractFeatures()
        assertNotNull("Features should not be null", features)

        // Speed should reflect recent samples (60-129 km/h range)
        assertTrue("Speed mean should reflect recent samples", features!![0] > 85.0f)
    }

    @Test
    fun `test reset functionality`() {
        // Add samples
        repeat(60) {
            extractor.addSample(createTestSample(speed = 60.0f))
        }

        assertTrue("Window should be ready", extractor.isWindowReady())

        // Reset
        extractor.reset()

        assertFalse("Window should not be ready after reset", extractor.isWindowReady())
        assertEquals("Sample count should be 0 after reset", 0, extractor.getSampleCount())
    }

    @Test
    fun `test extract features before window ready returns null`() {
        // Add only 30 samples (less than window size)
        repeat(30) {
            extractor.addSample(createTestSample(speed = 60.0f))
        }

        val features = extractor.extractFeatures()
        assertNull("Features should be null when window not ready", features)
    }

    @Test
    fun `test feature vector matches LightGBM input format`() {
        // Verify feature order matches training data
        repeat(60) {
            extractor.addSample(
                createTestSample(
                    speed = 70.0f,
                    rpm = 2500,
                    throttle = 35.0f,
                    brake = 5.0f,
                    accelX = 0.3f,
                    accelY = 0.1f,
                    maf = 6.5f
                )
            )
        }

        val features = extractor.extractFeatures()
        assertNotNull("Features should not be null", features)
        assertEquals("Feature vector should have exactly 18 dimensions", 18, features!!.size)

        // Verify all features are finite numbers
        features.forEach { feature ->
            assertTrue("All features should be finite", feature.isFinite())
        }
    }

    // Helper function to create test CANData samples
    private fun createTestSample(
        speed: Float = 60.0f,
        rpm: Int = 2000,
        throttle: Float = 30.0f,
        brake: Float = 0.0f,
        accelX: Float = 0.0f,
        accelY: Float = 0.0f,
        maf: Float = 5.0f
    ): CANData {
        return CANData(
            timestamp = System.currentTimeMillis(),
            vehicleSpeed = speed,
            engineRPM = rpm,
            throttlePosition = throttle,
            brakePosition = brake,
            fuelLevel = 50.0f,
            coolantTemp = 90.0f,
            engineLoad = 50.0f,
            intakeAirTemp = 25.0f,
            mafRate = maf,
            batteryVoltage = 12.5f,
            accelerationX = accelX,
            accelerationY = accelY,
            accelerationZ = 0.0f,
            gyroX = 0.0f,
            gyroY = 0.0f,
            gyroZ = 0.0f,
            gpsLatitude = 37.5665,
            gpsLongitude = 126.9780,
            gpsAltitude = 100.0f,
            gpsSpeed = speed,
            gpsHeading = 90.0f
        )
    }
}

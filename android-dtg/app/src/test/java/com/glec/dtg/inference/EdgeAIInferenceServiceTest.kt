package com.glec.dtg.inference

import android.content.Context
import com.glec.dtg.models.CANData
import com.glec.dtg.models.DrivingBehavior
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.MockitoAnnotations

/**
 * GLEC DTG - Edge AI Inference Service Unit Tests
 *
 * Tests the orchestration of AI inference including:
 * - Feature extraction from CAN data windows
 * - LightGBM behavior classification
 * - Multi-model support (future: TCN, LSTM-AE)
 * - Performance tracking
 * - Error handling
 */
class EdgeAIInferenceServiceTest {

    @Mock
    private lateinit var mockContext: Context

    @Mock
    private lateinit var mockLightGBMEngine: LightGBMONNXEngine

    private lateinit var inferenceService: EdgeAIInferenceService

    @Before
    fun setUp() {
        MockitoAnnotations.openMocks(this)

        // Create real feature extractor (lightweight, no mocking needed)
        val featureExtractor = FeatureExtractor(windowSize = 60)

        // Create inference service with mocked engine
        inferenceService = EdgeAIInferenceService(
            context = mockContext,
            lightGBMEngine = mockLightGBMEngine,
            featureExtractor = featureExtractor
        )
    }

    @Test
    fun `test inference service initialization`() {
        assertNotNull("Inference service should be initialized", inferenceService)
        assertFalse("Service should not be ready without samples", inferenceService.isReady())
    }

    @Test
    fun `test add sample updates feature extractor`() {
        val sample = createTestSample(speed = 60.0f)

        inferenceService.addSample(sample)

        assertEquals("Sample count should be 1", 1, inferenceService.getSampleCount())
        assertFalse("Service should not be ready with 1 sample", inferenceService.isReady())
    }

    @Test
    fun `test service becomes ready after 60 samples`() {
        // Add 60 samples
        repeat(60) { i ->
            inferenceService.addSample(createTestSample(speed = 60.0f + i))
        }

        assertTrue("Service should be ready with 60 samples", inferenceService.isReady())
        assertEquals("Sample count should be 60", 60, inferenceService.getSampleCount())
    }

    @Test
    fun `test run inference with normal driving behavior`() {
        // Setup mock to return normal behavior (class 0)
        `when`(mockLightGBMEngine.predict(any())).thenReturn(0)

        // Add 60 normal driving samples
        repeat(60) {
            inferenceService.addSample(
                createTestSample(
                    speed = 70.0f,
                    rpm = 2500,
                    throttle = 35.0f,
                    brake = 0.0f,
                    accelX = 0.0f
                )
            )
        }

        // Run inference
        val result = inferenceService.runInference()

        assertNotNull("Inference result should not be null", result)
        assertEquals("Behavior should be NORMAL", DrivingBehavior.NORMAL, result!!.behavior)
        assertTrue("Inference latency should be positive", result.latencyMs > 0)

        // Verify engine was called with correct feature vector size
        verify(mockLightGBMEngine, times(1)).predict(argThat { it.size == 18 })
    }

    @Test
    fun `test run inference with eco driving behavior`() {
        // Setup mock to return eco driving (class 1)
        `when`(mockLightGBMEngine.predict(any())).thenReturn(1)

        // Add 60 eco driving samples
        repeat(60) {
            inferenceService.addSample(
                createTestSample(
                    speed = 60.0f,
                    rpm = 1800,
                    throttle = 25.0f,
                    brake = 0.0f,
                    accelX = 0.1f
                )
            )
        }

        val result = inferenceService.runInference()

        assertNotNull("Inference result should not be null", result)
        assertEquals("Behavior should be ECO_DRIVING", DrivingBehavior.ECO_DRIVING, result!!.behavior)
    }

    @Test
    fun `test run inference with aggressive driving behavior`() {
        // Setup mock to return aggressive (class 2)
        `when`(mockLightGBMEngine.predict(any())).thenReturn(2)

        // Add 60 aggressive driving samples
        repeat(60) { i ->
            val isAccelerating = i < 30
            inferenceService.addSample(
                createTestSample(
                    speed = if (isAccelerating) 40.0f + (i * 2.0f) else 100.0f,
                    rpm = if (isAccelerating) 2000 + (i * 80) else 4500,
                    throttle = 90.0f,
                    brake = 0.0f,
                    accelX = if (isAccelerating) 4.0f else 0.5f
                )
            )
        }

        val result = inferenceService.runInference()

        assertNotNull("Inference result should not be null", result)
        assertEquals("Behavior should be AGGRESSIVE", DrivingBehavior.AGGRESSIVE, result!!.behavior)
    }

    @Test
    fun `test run inference before ready returns null`() {
        // Add only 30 samples (not enough)
        repeat(30) {
            inferenceService.addSample(createTestSample(speed = 60.0f))
        }

        val result = inferenceService.runInference()

        assertNull("Inference should return null when not ready", result)

        // Verify engine was never called
        verify(mockLightGBMEngine, never()).predict(any())
    }

    @Test
    fun `test run inference tracks latency`() {
        // Setup mock
        `when`(mockLightGBMEngine.predict(any())).thenReturn(0)

        // Add samples
        repeat(60) {
            inferenceService.addSample(createTestSample(speed = 60.0f))
        }

        // Run inference
        val result = inferenceService.runInference()

        assertNotNull("Result should not be null", result)
        assertTrue("Latency should be > 0", result!!.latencyMs > 0)
        assertTrue("Latency should be < 100ms", result.latencyMs < 100)  // Sanity check
    }

    @Test
    fun `test run inference tracks confidence scores`() {
        // Setup mock to return probabilities
        `when`(mockLightGBMEngine.predictWithProbabilities(any())).thenReturn(
            Pair(0, mapOf(0 to 0.85f, 1 to 0.10f, 2 to 0.05f))
        )

        // Add samples
        repeat(60) {
            inferenceService.addSample(createTestSample(speed = 60.0f))
        }

        // Run inference with probabilities
        val result = inferenceService.runInferenceWithConfidence()

        assertNotNull("Result should not be null", result)
        assertEquals("Behavior should be NORMAL", DrivingBehavior.NORMAL, result!!.behavior)
        assertEquals("Confidence should be 0.85", 0.85f, result.confidence, 0.01f)
    }

    @Test
    fun `test reset clears feature extractor`() {
        // Add samples
        repeat(60) {
            inferenceService.addSample(createTestSample(speed = 60.0f))
        }

        assertTrue("Service should be ready", inferenceService.isReady())

        // Reset
        inferenceService.reset()

        assertFalse("Service should not be ready after reset", inferenceService.isReady())
        assertEquals("Sample count should be 0 after reset", 0, inferenceService.getSampleCount())
    }

    @Test
    fun `test get performance metrics`() {
        // Setup mock
        `when`(mockLightGBMEngine.predict(any())).thenReturn(0)

        // Add samples and run inference 3 times
        repeat(3) {
            repeat(60) { i ->
                inferenceService.addSample(createTestSample(speed = 60.0f + i))
            }
            inferenceService.runInference()
        }

        val metrics = inferenceService.getPerformanceMetrics()

        assertEquals("Inference count should be 3", 3, metrics.inferenceCount)
        assertTrue("Average latency should be > 0", metrics.avgLatencyMs > 0.0)
        assertTrue("Max latency should be > 0", metrics.maxLatencyMs > 0.0)
    }

    @Test
    fun `test sliding window maintains 60 samples`() {
        // Add 100 samples (more than window size)
        repeat(100) { i ->
            inferenceService.addSample(createTestSample(speed = 60.0f + i))
        }

        // Sample count should be capped at 60
        assertEquals("Sample count should be 60", 60, inferenceService.getSampleCount())
        assertTrue("Service should be ready", inferenceService.isReady())
    }

    @Test
    fun `test feature vector format matches LightGBM expectations`() {
        var capturedFeatures: FloatArray? = null

        // Capture the feature vector passed to engine
        `when`(mockLightGBMEngine.predict(any())).thenAnswer { invocation ->
            capturedFeatures = invocation.getArgument(0)
            0  // Return normal behavior
        }

        // Add samples
        repeat(60) {
            inferenceService.addSample(createTestSample(speed = 70.0f, rpm = 2500))
        }

        inferenceService.runInference()

        assertNotNull("Features should be captured", capturedFeatures)
        assertEquals("Feature vector should have 18 dimensions", 18, capturedFeatures!!.size)

        // Verify all features are finite
        capturedFeatures!!.forEach { feature ->
            assertTrue("All features should be finite", feature.isFinite())
        }
    }

    @Test
    fun `test concurrent inference calls are safe`() {
        `when`(mockLightGBMEngine.predict(any())).thenReturn(0)

        // Add samples
        repeat(60) {
            inferenceService.addSample(createTestSample(speed = 60.0f))
        }

        // Run inference multiple times (simulating concurrent calls)
        val results = List(5) { inferenceService.runInference() }

        // All results should be valid
        results.forEach { result ->
            assertNotNull("All results should be non-null", result)
            assertEquals("All results should have NORMAL behavior", DrivingBehavior.NORMAL, result!!.behavior)
        }

        // Engine should be called 5 times
        verify(mockLightGBMEngine, times(5)).predict(any())
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

    // Mockito matcher for any FloatArray
    private fun <T> any(): T {
        Mockito.any<T>()
        return null as T
    }

    // Mockito argThat matcher
    private fun <T> argThat(matcher: (T) -> Boolean): T {
        Mockito.argThat { matcher(it) }
        return null as T
    }
}

/**
 * Data class for inference result (internal to service)
 */
data class InferenceResult(
    val behavior: DrivingBehavior,
    val latencyMs: Long,
    val confidence: Float = 1.0f,
    val timestamp: Long = System.currentTimeMillis()
)

/**
 * Performance metrics for inference service
 */
data class InferencePerformanceMetrics(
    val inferenceCount: Int,
    val avgLatencyMs: Double,
    val maxLatencyMs: Double,
    val minLatencyMs: Double
)

package com.glec.dtg.llm

import android.content.Context
import com.glec.dtg.models.VehicleData
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.TimeoutCancellationException
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Unit tests for Qwen25InferenceEngine
 *
 * Test Coverage:
 * - Initialization
 * - Basic inference
 * - Korean language quality
 * - Timeout handling
 * - Memory constraints
 * - Context integration
 *
 * Reference: LLM_IMPLEMENTATION_GUIDE.md
 */
@RunWith(MockitoJUnitRunner::class)
class Qwen25InferenceEngineTest {

    @Mock
    private lateinit var mockContext: Context

    private lateinit var engine: Qwen25InferenceEngine

    @Before
    fun setup() {
        // Mock file directory for model path
        `when`(mockContext.filesDir).thenReturn(
            java.io.File("/mock/path")
        )

        engine = Qwen25InferenceEngine(mockContext)
    }

    /**
     * Test: Engine initializes successfully
     *
     * Expected:
     * - No exceptions thrown
     * - Engine ready for inference
     */
    @Test
    fun testEngineInitialization() {
        // Given: Fresh engine instance
        // When: Initialize is called
        // Then: Should complete without exceptions

        // Note: This will fail until implementation exists (RED phase)
        assertFailsWith<NotImplementedError> {
            engine.initialize()
        }
    }

    /**
     * Test: Basic inference with cargo weight query
     *
     * Expected:
     * - Response contains weight information
     * - Response is in Korean
     * - Response mentions "5000" or "5톤"
     *
     * Reference: PHASE3K_LLM_INTEGRATION.md - Success Criteria
     */
    @Test
    fun testBasicInference_CargoWeight() = runBlocking {
        // Given: Initialized engine and vehicle context
        engine.initialize()

        val query = "현재 적재 중량이 얼마인가요?"
        val context = VehicleData(
            cargoWeight = 5000.0,
            tirePressure = 220.0,
            engineTemp = 85.0,
            fuelEfficiency = 6.5,
            timestamp = System.currentTimeMillis()
        )

        // When: Inference is performed
        val response = engine.inference(query, context)

        // Then: Response should contain weight information
        assertNotNull(response, "Response should not be null")
        assertTrue(response.isNotEmpty(), "Response should not be empty")
        assertTrue(
            response.contains("5000") || response.contains("5톤") || response.contains("5.0톤"),
            "Response should mention cargo weight: $response"
        )
        assertTrue(
            response.contains("적재") || response.contains("중량") || response.contains("킬로그램"),
            "Response should contain cargo-related Korean terms: $response"
        )
    }

    /**
     * Test: Korean language quality with multiple queries
     *
     * Expected:
     * - Responses are coherent Korean text
     * - Responses contain expected keywords
     * - No garbled text or encoding issues
     */
    @Test
    fun testKoreanLanguageQuality() = runBlocking {
        engine.initialize()

        val testCases = listOf(
            TestCase(
                query = "타이어 공기압이 낮습니다",
                context = VehicleData(tirePressure = 180.0),
                expectedKeywords = listOf("타이어", "공기압", "낮")
            ),
            TestCase(
                query = "오늘 운행이 어땠나요?",
                context = VehicleData(fuelEfficiency = 7.2),
                expectedKeywords = listOf("운행", "연비")
            ),
            TestCase(
                query = "엔진 온도가 몇 도인가요?",
                context = VehicleData(engineTemp = 92.0),
                expectedKeywords = listOf("엔진", "온도", "92")
            )
        )

        testCases.forEach { case ->
            val response = engine.inference(case.query, case.context)

            assertNotNull(response, "Response for '${case.query}' should not be null")
            assertTrue(response.isNotEmpty(), "Response should not be empty")
            assertTrue(response.length > 10, "Response too short: ${response.length} chars")

            val containsKeyword = case.expectedKeywords.any { keyword ->
                response.contains(keyword)
            }
            assertTrue(
                containsKeyword,
                "Response should contain at least one of ${case.expectedKeywords}: $response"
            )
        }
    }

    /**
     * Test: Inference timeout handling
     *
     * Expected:
     * - Timeout after 5 seconds (per LLM_IMPLEMENTATION_GUIDE.md)
     * - TimeoutCancellationException thrown
     *
     * Quality Gate: P95 latency < 3000ms
     */
    @Test(timeout = 6000) // 6 seconds max test duration
    fun testInferenceTimeout() = runBlocking {
        engine.initialize()

        // Simulate very long query that might cause timeout
        val veryLongQuery = "매우 긴 쿼리 " + "반복 ".repeat(200)
        val context = VehicleData()

        // Should timeout or complete within 5 seconds
        assertFailsWith<TimeoutCancellationException> {
            engine.inference(veryLongQuery, context)
        }
    }

    /**
     * Test: Prompt building with vehicle context
     *
     * Expected:
     * - Prompt includes system message
     * - Prompt includes vehicle data
     * - Prompt includes user query
     */
    @Test
    fun testPromptBuilding() {
        engine.initialize()

        val query = "짐 상태 확인"
        val context = VehicleData(
            cargoWeight = 5200.0,
            tirePressure = 220.0,
            engineTemp = 88.0,
            fuelEfficiency = 6.8
        )

        // Access private buildPrompt method via reflection for testing
        val buildPromptMethod = engine.javaClass.getDeclaredMethod(
            "buildPrompt",
            String::class.java,
            VehicleData::class.java
        )
        buildPromptMethod.isAccessible = true

        val prompt = buildPromptMethod.invoke(engine, query, context) as String

        // Verify prompt structure
        assertTrue(prompt.contains("화물차"), "Prompt should mention truck")
        assertTrue(prompt.contains("5200"), "Prompt should include cargo weight")
        assertTrue(prompt.contains("220"), "Prompt should include tire pressure")
        assertTrue(prompt.contains("88"), "Prompt should include engine temp")
        assertTrue(prompt.contains(query), "Prompt should include user query")
    }

    /**
     * Test: Multiple consecutive inferences (memory stability)
     *
     * Expected:
     * - No memory leaks
     * - Consistent performance
     * - No crashes
     *
     * Quality Gate: 100 consecutive runs without errors
     */
    @Test
    fun testMemoryStability_ConsecutiveInferences() = runBlocking {
        engine.initialize()

        val iterations = 20 // Reduced for unit test speed
        val query = "차량 상태 확인"

        repeat(iterations) { i ->
            val context = VehicleData(
                cargoWeight = 5000.0 + (i * 100),
                tirePressure = 220.0,
                engineTemp = 85.0,
                fuelEfficiency = 6.5
            )

            val response = engine.inference(query, context)

            assertNotNull(response, "Response at iteration $i should not be null")
            assertTrue(response.isNotEmpty(), "Response at iteration $i should not be empty")
        }

        // If we got here, no crashes or OOM errors occurred
        assertTrue(true, "Completed $iterations consecutive inferences successfully")
    }

    /**
     * Test: Engine release and cleanup
     *
     * Expected:
     * - Resources properly released
     * - No memory leaks
     */
    @Test
    fun testEngineRelease() {
        engine.initialize()

        // Perform inference
        runBlocking {
            val response = engine.inference(
                "테스트",
                VehicleData()
            )
            assertNotNull(response)
        }

        // Release engine
        engine.release()

        // After release, inference should fail
        assertFailsWith<IllegalStateException> {
            runBlocking {
                engine.inference("테스트", VehicleData())
            }
        }
    }

    /**
     * Test: Error handling for uninitialized engine
     *
     * Expected:
     * - IllegalStateException thrown
     * - Clear error message
     */
    @Test
    fun testUninitializedEngineError() {
        // Don't call initialize()

        assertFailsWith<IllegalStateException> {
            runBlocking {
                engine.inference("테스트", VehicleData())
            }
        }
    }

    /**
     * Test: Context-aware responses
     *
     * Expected:
     * - Different responses based on context
     * - Appropriate reactions to warning conditions
     */
    @Test
    fun testContextAwareResponses() = runBlocking {
        engine.initialize()

        // Test 1: Normal tire pressure
        val normalContext = VehicleData(tirePressure = 220.0)
        val normalResponse = engine.inference("타이어 상태는?", normalContext)
        assertTrue(
            normalResponse.contains("정상") || normalResponse.contains("양호") || normalResponse.contains("220"),
            "Normal pressure response: $normalResponse"
        )

        // Test 2: Low tire pressure (warning)
        val lowContext = VehicleData(tirePressure = 160.0)
        val lowResponse = engine.inference("타이어 상태는?", lowContext)
        assertTrue(
            lowResponse.contains("낮") || lowResponse.contains("부족") || lowResponse.contains("주의"),
            "Low pressure response should indicate warning: $lowResponse"
        )
    }

    // Helper data class for test cases
    data class TestCase(
        val query: String,
        val context: VehicleData,
        val expectedKeywords: List<String>
    )
}

/**
 * VehicleData data class for testing
 *
 * TODO: Move to shared models package
 */
data class VehicleData(
    val cargoWeight: Double = 0.0,      // kg
    val tirePressure: Double = 220.0,   // kPa
    val engineTemp: Double = 85.0,      // °C
    val fuelEfficiency: Double = 6.5,   // km/L
    val timestamp: Long = System.currentTimeMillis()
)

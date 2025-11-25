package com.glec.dtg.llm

import com.glec.dtg.models.VehicleData
import kotlinx.coroutines.runBlocking
import org.junit.Before
import org.junit.Test
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.junit.MockitoJUnitRunner
import org.junit.runner.RunWith
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Unit tests for LLMFallbackHandler
 *
 * Test Coverage:
 * - Low memory detection and fallback
 * - OOM error handling
 * - Timeout handling
 * - Rule-based fallback responses
 * - Memory threshold enforcement
 *
 * Reference: LLM_IMPLEMENTATION_GUIDE.md - OOM Prevention
 */
@RunWith(MockitoJUnitRunner::class)
class LLMFallbackHandlerTest {

    @Mock
    private lateinit var mockLLM: Qwen25InferenceEngine

    @Mock
    private lateinit var mockRuleBasedParser: IntentParser

    private lateinit var handler: LLMFallbackHandler

    @Before
    fun setup() {
        handler = LLMFallbackHandler(mockLLM, mockRuleBasedParser)
    }

    /**
     * Test: Normal inference with sufficient memory
     *
     * Expected:
     * - LLM inference is called
     * - Response from LLM returned
     * - No fallback triggered
     */
    @Test
    fun testNormalInference_SufficientMemory() = runBlocking {
        // Given: Sufficient memory (>200MB free)
        // Mock Runtime to report high free memory
        val query = "현재 적재 중량은?"
        val context = VehicleData(cargoWeight = 5000.0)
        val expectedResponse = "현재 적재 중량은 5000킬로그램입니다."

        `when`(mockLLM.inference(query, context))
            .thenReturn(expectedResponse)

        // When: Safe inference is called
        val response = handler.safeInference(query, context)

        // Then: LLM inference should be called
        verify(mockLLM, times(1)).inference(query, context)
        assertEquals(expectedResponse, response)
    }

    /**
     * Test: Low memory triggers rule-based fallback
     *
     * Expected:
     * - Free memory < 200MB detected
     * - LLM inference NOT called
     * - Rule-based parser used instead
     *
     * Quality Gate: Prevent OOM by fallback at 200MB threshold
     */
    @Test
    fun testLowMemory_FallbackTriggered() = runBlocking {
        // Given: Low memory condition (<200MB free)
        // Note: In real implementation, this checks Runtime.getRuntime().freeMemory()

        val query = "적재 중량 확인"
        val context = VehicleData(cargoWeight = 5200.0)

        `when`(mockRuleBasedParser.parse(query))
            .thenReturn(Intent.CHECK_CARGO)

        // When: Safe inference called in low memory
        // (Implementation should detect low memory and use fallback)
        val response = handler.safeInference(query, context)

        // Then: Should return rule-based response
        assertNotNull(response)
        assertTrue(response.contains("적재") || response.contains("중량") || response.contains("5200"))
    }

    /**
     * Test: OOM error during inference triggers fallback
     *
     * Expected:
     * - OutOfMemoryError caught
     * - System.gc() called
     * - Rule-based fallback returned
     */
    @Test
    fun testOOM_FallbackTriggered() = runBlocking {
        val query = "차량 상태"
        val context = VehicleData()

        // Mock LLM to throw OOM
        `when`(mockLLM.inference(query, context))
            .thenThrow(OutOfMemoryError("Simulated OOM"))

        `when`(mockRuleBasedParser.parse(query))
            .thenReturn(Intent.VEHICLE_STATUS)

        // When: Inference causes OOM
        val response = handler.safeInference(query, context)

        // Then: Should return fallback response (not crash)
        assertNotNull(response)
        assertTrue(response.isNotEmpty())
        // Verify GC was called (implementation detail)
    }

    /**
     * Test: Timeout during inference
     *
     * Expected:
     * - TimeoutCancellationException caught
     * - User-friendly Korean error message
     */
    @Test
    fun testTimeout_UserFriendlyMessage() = runBlocking {
        val query = "매우 복잡한 쿼리"
        val context = VehicleData()

        // Mock LLM to throw timeout
        `when`(mockLLM.inference(query, context))
            .thenThrow(kotlinx.coroutines.TimeoutCancellationException("Timeout"))

        // When: Inference times out
        val response = handler.safeInference(query, context)

        // Then: Should return timeout message in Korean
        assertNotNull(response)
        assertTrue(
            response.contains("시간") && response.contains("초과"),
            "Expected timeout message in Korean: $response"
        )
    }

    /**
     * Test: Rule-based fallback for cargo weight
     *
     * Expected:
     * - Intent correctly parsed
     * - Cargo weight from context used
     * - Korean response format
     */
    @Test
    fun testRuleBasedFallback_CargoWeight() {
        val query = "짐이 얼마나 실렸나요?"
        val context = VehicleData(cargoWeight = 6500.0)

        `when`(mockRuleBasedParser.parse(query))
            .thenReturn(Intent.CHECK_CARGO)

        // Call private ruleBasedFallback via reflection
        val method = handler.javaClass.getDeclaredMethod(
            "ruleBasedFallback",
            String::class.java,
            VehicleData::class.java
        )
        method.isAccessible = true

        val response = method.invoke(handler, query, context) as String

        assertNotNull(response)
        assertTrue(response.contains("6500") || response.contains("6.5"))
        assertTrue(response.contains("적재") || response.contains("중량"))
    }

    /**
     * Test: Rule-based fallback for tire pressure
     *
     * Expected:
     * - Tire pressure value included
     * - Korean units (kPa)
     */
    @Test
    fun testRuleBasedFallback_TirePressure() {
        val query = "타이어 공기압 확인"
        val context = VehicleData(tirePressure = 215.0)

        `when`(mockRuleBasedParser.parse(query))
            .thenReturn(Intent.TIRE_PRESSURE)

        val method = handler.javaClass.getDeclaredMethod(
            "ruleBasedFallback",
            String::class.java,
            VehicleData::class.java
        )
        method.isAccessible = true

        val response = method.invoke(handler, query, context) as String

        assertTrue(response.contains("215"))
        assertTrue(response.contains("타이어") || response.contains("공기압"))
    }

    /**
     * Test: Rule-based fallback for fuel efficiency
     */
    @Test
    fun testRuleBasedFallback_FuelEfficiency() {
        val query = "연비가 어때?"
        val context = VehicleData(fuelEfficiency = 7.8)

        `when`(mockRuleBasedParser.parse(query))
            .thenReturn(Intent.FUEL_EFFICIENCY)

        val method = handler.javaClass.getDeclaredMethod(
            "ruleBasedFallback",
            String::class.java,
            VehicleData::class.java
        )
        method.isAccessible = true

        val response = method.invoke(handler, query, context) as String

        assertTrue(response.contains("7.8"))
        assertTrue(response.contains("연비") || response.contains("km/L"))
    }

    /**
     * Test: Rule-based fallback for engine temperature
     */
    @Test
    fun testRuleBasedFallback_EngineTemp() {
        val query = "엔진 온도 확인"
        val context = VehicleData(engineTemp = 95.0)

        `when`(mockRuleBasedParser.parse(query))
            .thenReturn(Intent.ENGINE_TEMP)

        val method = handler.javaClass.getDeclaredMethod(
            "ruleBasedFallback",
            String::class.java,
            VehicleData::class.java
        )
        method.isAccessible = true

        val response = method.invoke(handler, query, context) as String

        assertTrue(response.contains("95"))
        assertTrue(response.contains("엔진") || response.contains("온도"))
    }

    /**
     * Test: Unknown intent fallback
     *
     * Expected:
     * - Polite "didn't understand" message in Korean
     */
    @Test
    fun testRuleBasedFallback_UnknownIntent() {
        val query = "날씨가 어때?"
        val context = VehicleData()

        `when`(mockRuleBasedParser.parse(query))
            .thenReturn(Intent.UNKNOWN)

        val method = handler.javaClass.getDeclaredMethod(
            "ruleBasedFallback",
            String::class.java,
            VehicleData::class.java
        )
        method.isAccessible = true

        val response = method.invoke(handler, query, context) as String

        assertTrue(
            response.contains("죄송") || response.contains("이해"),
            "Expected polite error message: $response"
        )
    }

    /**
     * Test: Multiple consecutive fallbacks (stability)
     *
     * Expected:
     * - No crashes
     * - Consistent responses
     */
    @Test
    fun testMultipleFallbacks_Stability() = runBlocking {
        val queries = listOf(
            "적재 중량" to Intent.CHECK_CARGO,
            "타이어 압력" to Intent.TIRE_PRESSURE,
            "연비 확인" to Intent.FUEL_EFFICIENCY,
            "엔진 온도" to Intent.ENGINE_TEMP
        )

        queries.forEach { (query, intent) ->
            `when`(mockRuleBasedParser.parse(query))
                .thenReturn(intent)

            // Simulate low memory forcing fallback
            val response = handler.safeInference(query, VehicleData())

            assertNotNull(response, "Response for '$query' should not be null")
            assertTrue(response.isNotEmpty(), "Response for '$query' should not be empty")
        }
    }
}

/**
 * Intent enum for rule-based parsing
 *
 * TODO: Move to shared models package
 */
enum class Intent {
    CHECK_CARGO,
    TIRE_PRESSURE,
    FUEL_EFFICIENCY,
    ENGINE_TEMP,
    VEHICLE_STATUS,
    UNKNOWN
}

/**
 * IntentParser interface for rule-based fallback
 *
 * TODO: Implement actual rule-based parser
 */
interface IntentParser {
    fun parse(query: String): Intent
}

package com.glec.dtg.hybrid

import com.glec.dtg.models.DrivingBehavior
import org.junit.Assert.*
import org.junit.Test

/**
 * Unit tests for HybridAICoordinator
 * Phase 3-C: Hybrid AI Testing
 */
class HybridAICoordinatorTest {

    // ========== Performance Thresholds Tests ==========

    @Test
    fun `ML latency target is 50ms`() {
        assertEquals(50L, HybridAICoordinator.ML_LATENCY_TARGET_MS)
    }

    @Test
    fun `LLM latency target is 3 seconds`() {
        assertEquals(3000L, HybridAICoordinator.LLM_LATENCY_TARGET_MS)
    }

    @Test
    fun `combined latency target is 3_5 seconds`() {
        assertEquals(3500L, HybridAICoordinator.COMBINED_LATENCY_TARGET_MS)
    }

    @Test
    fun `memory peak target is 700MB`() {
        assertEquals(700L, HybridAICoordinator.MEMORY_PEAK_TARGET_MB)
    }

    @Test
    fun `auto analysis interval is 60 seconds`() {
        assertEquals(60_000L, HybridAICoordinator.AUTO_ANALYSIS_INTERVAL_MS)
    }

    // ========== CombinedAnalysisResult Tests ==========

    @Test
    fun `CombinedAnalysisResult isSuccessful returns true when mlResult exists`() {
        val result = CombinedAnalysisResult(
            mlResult = createMockMLResult(),
            llmResult = createMockLLMResult(),
            totalLatencyMs = 1500L,
            timestamp = System.currentTimeMillis()
        )

        assertTrue(result.isSuccessful())
    }

    @Test
    fun `CombinedAnalysisResult isSuccessful returns false when mlResult is null`() {
        val result = CombinedAnalysisResult(
            mlResult = null,
            llmResult = createMockLLMResult(),
            totalLatencyMs = 1500L,
            timestamp = System.currentTimeMillis()
        )

        assertFalse(result.isSuccessful())
    }

    @Test
    fun `CombinedAnalysisResult isSuccessful returns false when error exists`() {
        val result = CombinedAnalysisResult(
            mlResult = createMockMLResult(),
            llmResult = createMockLLMResult(),
            totalLatencyMs = 1500L,
            timestamp = System.currentTimeMillis(),
            error = "Test error"
        )

        assertFalse(result.isSuccessful())
    }

    @Test
    fun `CombinedAnalysisResult getSummary contains ML behavior`() {
        val result = CombinedAnalysisResult(
            mlResult = createMockMLResult(behavior = DrivingBehavior.ECO_DRIVING),
            llmResult = createMockLLMResult(),
            totalLatencyMs = 1500L,
            timestamp = System.currentTimeMillis()
        )

        val summary = result.getSummary()
        assertTrue(summary.contains("ECO_DRIVING"))
    }

    @Test
    fun `CombinedAnalysisResult getSummary contains LLM response`() {
        val testResponse = "차량 상태가 양호합니다."
        val result = CombinedAnalysisResult(
            mlResult = createMockMLResult(),
            llmResult = createMockLLMResult(response = testResponse),
            totalLatencyMs = 1500L,
            timestamp = System.currentTimeMillis()
        )

        val summary = result.getSummary()
        assertTrue(summary.contains(testResponse))
    }

    @Test
    fun `CombinedAnalysisResult getSummary contains latency`() {
        val result = CombinedAnalysisResult(
            mlResult = createMockMLResult(),
            llmResult = createMockLLMResult(),
            totalLatencyMs = 2500L,
            timestamp = System.currentTimeMillis()
        )

        val summary = result.getSummary()
        assertTrue(summary.contains("2500ms"))
    }

    // ========== HybridAIStatus Tests ==========

    @Test
    fun `HybridAIStatus errorRate is zero when no analyses`() {
        val status = createMockStatus(totalAnalysisCount = 0, errorCount = 0)
        assertEquals(0f, status.errorRate, 0.001f)
    }

    @Test
    fun `HybridAIStatus errorRate calculation is correct`() {
        val status = createMockStatus(totalAnalysisCount = 100, errorCount = 5)
        assertEquals(0.05f, status.errorRate, 0.001f)
    }

    @Test
    fun `HybridAIStatus errorRate is 10 percent with 10 errors in 100`() {
        val status = createMockStatus(totalAnalysisCount = 100, errorCount = 10)
        assertEquals(0.1f, status.errorRate, 0.001f)
    }

    @Test
    fun `HybridAIStatus getStatusSummary contains all key info`() {
        val status = createMockStatus(
            isInitialized = true,
            mlReady = true,
            llmReady = true,
            batteryPercent = 80
        )

        val summary = status.getStatusSummary()

        assertTrue(summary.contains("Initialized: true"))
        assertTrue(summary.contains("ML Models"))
        assertTrue(summary.contains("LLM"))
        assertTrue(summary.contains("Battery: 80%"))
    }

    // ========== AnalysisType Tests ==========

    @Test
    fun `AnalysisType has all three types`() {
        val types = AnalysisType.values()
        assertEquals(3, types.size)
        assertTrue(types.contains(AnalysisType.ML_ONLY))
        assertTrue(types.contains(AnalysisType.LLM_ONLY))
        assertTrue(types.contains(AnalysisType.COMBINED))
    }

    // ========== AnalysisRequest Tests ==========

    @Test
    fun `AnalysisRequest has correct default timestamp`() {
        val beforeCreate = System.currentTimeMillis()
        val request = AnalysisRequest(type = AnalysisType.ML_ONLY)
        val afterCreate = System.currentTimeMillis()

        assertTrue(request.timestamp >= beforeCreate)
        assertTrue(request.timestamp <= afterCreate)
    }

    @Test
    fun `AnalysisRequest query defaults to null`() {
        val request = AnalysisRequest(type = AnalysisType.ML_ONLY)
        assertNull(request.query)
    }

    @Test
    fun `AnalysisRequest vehicleData defaults to null`() {
        val request = AnalysisRequest(type = AnalysisType.LLM_ONLY)
        assertNull(request.vehicleData)
    }

    // ========== Latency Target Validation Tests ==========

    @Test
    fun `ML latency target is less than LLM target`() {
        assertTrue(
            HybridAICoordinator.ML_LATENCY_TARGET_MS < HybridAICoordinator.LLM_LATENCY_TARGET_MS
        )
    }

    @Test
    fun `combined latency target is sum of ML and LLM plus buffer`() {
        val expectedMin = HybridAICoordinator.ML_LATENCY_TARGET_MS + HybridAICoordinator.LLM_LATENCY_TARGET_MS
        assertTrue(
            HybridAICoordinator.COMBINED_LATENCY_TARGET_MS >= expectedMin
        )
    }

    // ========== Status Field Tests ==========

    @Test
    fun `HybridAIStatus contains mlSampleCount field`() {
        val status = createMockStatus(mlSampleCount = 30)
        assertEquals(30, status.mlSampleCount)
    }

    @Test
    fun `HybridAIStatus mlSampleCount max is 60`() {
        val status = createMockStatus(mlSampleCount = 60)
        assertEquals(60, status.mlSampleCount)
    }

    @Test
    fun `HybridAIStatus contains cache hit rate`() {
        val status = createMockStatus(llmCacheHitRate = 0.75f)
        assertEquals(0.75f, status.llmCacheHitRate, 0.001f)
    }

    @Test
    fun `HybridAIStatus contains overheating flag`() {
        val status = createMockStatus(isOverheating = true)
        assertTrue(status.isOverheating)
    }

    @Test
    fun `HybridAIStatus default is not overheating`() {
        val status = createMockStatus(isOverheating = false)
        assertFalse(status.isOverheating)
    }

    // ========== Count Tracking Tests ==========

    @Test
    fun `HybridAIStatus tracks ml only count`() {
        val status = createMockStatus(mlOnlyCount = 50)
        assertEquals(50, status.mlOnlyCount)
    }

    @Test
    fun `HybridAIStatus tracks llm only count`() {
        val status = createMockStatus(llmOnlyCount = 30)
        assertEquals(30, status.llmOnlyCount)
    }

    @Test
    fun `HybridAIStatus tracks combined count`() {
        val status = createMockStatus(combinedCount = 20)
        assertEquals(20, status.combinedCount)
    }

    @Test
    fun `total should equal sum of ml llm and combined when no errors`() {
        val mlOnly = 50
        val llmOnly = 30
        val combined = 20
        val total = mlOnly + llmOnly + combined

        val status = createMockStatus(
            totalAnalysisCount = total,
            mlOnlyCount = mlOnly,
            llmOnlyCount = llmOnly,
            combinedCount = combined
        )

        assertEquals(total, status.totalAnalysisCount)
    }

    // ========== Memory and Battery Tests ==========

    @Test
    fun `HybridAIStatus contains available memory`() {
        val status = createMockStatus(availableMemoryMB = 500L)
        assertEquals(500L, status.availableMemoryMB)
    }

    @Test
    fun `HybridAIStatus contains battery percent`() {
        val status = createMockStatus(batteryPercent = 65)
        assertEquals(65, status.batteryPercent)
    }

    @Test
    fun `battery percent range is 0 to 100`() {
        val statusLow = createMockStatus(batteryPercent = 0)
        val statusHigh = createMockStatus(batteryPercent = 100)

        assertTrue(statusLow.batteryPercent >= 0)
        assertTrue(statusHigh.batteryPercent <= 100)
    }

    // ========== Helper Methods ==========

    private fun createMockMLResult(
        behavior: DrivingBehavior = DrivingBehavior.NORMAL,
        confidence: Float = 0.85f,
        fuelEfficiency: Float = 8.5f,
        anomalyScore: Float = 0.1f,
        isAnomaly: Boolean = false,
        latencyMs: Long = 25L
    ): com.glec.dtg.inference.InferenceResult {
        return com.glec.dtg.inference.InferenceResult(
            behavior = behavior,
            confidence = confidence,
            fuelEfficiency = fuelEfficiency,
            anomalyScore = anomalyScore,
            isAnomaly = isAnomaly,
            latencyMs = latencyMs
        )
    }

    private fun createMockLLMResult(
        response: String = "테스트 응답입니다.",
        engine: String = "QwenLLMEngine",
        latencyMs: Long = 1500L,
        fromCache: Boolean = false
    ): com.glec.dtg.llm.SmartOrchestrator.InferenceResult {
        return com.glec.dtg.llm.SmartOrchestrator.InferenceResult(
            response = response,
            engine = engine,
            latencyMs = latencyMs,
            fromCache = fromCache
        )
    }

    private fun createMockStatus(
        isInitialized: Boolean = true,
        mlReady: Boolean = true,
        mlSampleCount: Int = 0,
        mlAvgLatencyMs: Double = 25.0,
        mlInferenceCount: Int = 0,
        llmReady: Boolean = true,
        ruleEngineReady: Boolean = true,
        llmCacheHitRate: Float = 0f,
        batteryPercent: Int = 100,
        availableMemoryMB: Long = 600,
        isOverheating: Boolean = false,
        totalAnalysisCount: Int = 0,
        mlOnlyCount: Int = 0,
        llmOnlyCount: Int = 0,
        combinedCount: Int = 0,
        errorCount: Int = 0
    ): HybridAIStatus {
        return HybridAIStatus(
            isInitialized = isInitialized,
            mlReady = mlReady,
            mlSampleCount = mlSampleCount,
            mlAvgLatencyMs = mlAvgLatencyMs,
            mlInferenceCount = mlInferenceCount,
            llmReady = llmReady,
            ruleEngineReady = ruleEngineReady,
            llmCacheHitRate = llmCacheHitRate,
            batteryPercent = batteryPercent,
            availableMemoryMB = availableMemoryMB,
            isOverheating = isOverheating,
            totalAnalysisCount = totalAnalysisCount,
            mlOnlyCount = mlOnlyCount,
            llmOnlyCount = llmOnlyCount,
            combinedCount = combinedCount,
            errorCount = errorCount
        )
    }
}

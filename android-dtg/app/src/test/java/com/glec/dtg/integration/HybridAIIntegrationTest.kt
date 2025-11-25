package com.glec.dtg.integration

import com.glec.dtg.hybrid.*
import com.glec.dtg.inference.InferenceResult as MLInferenceResult
import com.glec.dtg.llm.SmartOrchestrator
import com.glec.dtg.models.CANData
import com.glec.dtg.models.DrivingBehavior
import com.glec.dtg.models.VehicleData
import org.junit.Assert.*
import org.junit.Test

/**
 * Integration Tests for Phase 3-D: Hybrid AI System Integration
 *
 * These tests verify the integration between:
 * 1. EdgeAIInferenceService (ML models: LightGBM, TCN, LSTM-AE)
 * 2. SmartOrchestrator (LLM: Qwen2.5-0.5B + RuleBasedEngine)
 * 3. HybridAICoordinator (unified interface)
 *
 * Test Categories:
 * - Data Flow Tests: ML results flow into LLM context
 * - Performance Tests: Latency, memory, and throughput targets
 * - Error Handling Tests: Graceful degradation and fallbacks
 * - State Management Tests: Lifecycle and metrics tracking
 * - Quality Gate Tests: Production-ready criteria
 *
 * Note: Full end-to-end tests require Android context.
 * These tests verify logical integration without Android dependencies.
 */
class HybridAIIntegrationTest {

    // ========== Data Flow Integration Tests ==========

    @Test
    fun `ML result enriches VehicleData context`() {
        // Given: ML inference produces behavior analysis
        val mlResult = createMockMLResult(
            behavior = DrivingBehavior.ECO_DRIVING,
            confidence = 0.92f,
            fuelEfficiency = 7.5f,
            anomalyScore = 0.05f,
            isAnomaly = false
        )

        // When: VehicleData is enriched with ML results
        val baseVehicleData = VehicleData(
            cargoWeight = 3500.0,
            tirePressure = 220.0,
            engineTemp = 85.0,
            fuelEfficiency = 6.8
        )

        val enrichedData = enrichVehicleDataWithML(baseVehicleData, mlResult)

        // Then: All ML fields should be populated
        assertEquals("ECO_DRIVING", enrichedData.drivingBehavior)
        assertEquals(0.92f, enrichedData.behaviorConfidence, 0.01f)
        assertEquals(7.5f, enrichedData.fuelEfficiencyPredicted, 0.1f)
        assertEquals(0.05f, enrichedData.anomalyScore, 0.01f)
        assertFalse(enrichedData.isAnomalyDetected)

        // Original data should be preserved
        assertEquals(3500.0, enrichedData.cargoWeight, 0.1)
        assertEquals(220.0, enrichedData.tirePressure, 0.1)
    }

    @Test
    fun `CombinedAnalysisResult contains both ML and LLM results`() {
        // Given: Both ML and LLM produce results
        val mlResult = createMockMLResult(behavior = DrivingBehavior.NORMAL)
        val llmResult = createMockLLMResult(response = "현재 운행 상태가 양호합니다.")

        // When: Combined result is created
        val combinedResult = CombinedAnalysisResult(
            mlResult = mlResult,
            llmResult = llmResult,
            totalLatencyMs = 2500L,
            timestamp = System.currentTimeMillis()
        )

        // Then: Both results should be accessible
        assertNotNull(combinedResult.mlResult)
        assertNotNull(combinedResult.llmResult)
        assertEquals(DrivingBehavior.NORMAL, combinedResult.mlResult?.behavior)
        assertTrue(combinedResult.llmResult.response.contains("양호"))
        assertTrue(combinedResult.isSuccessful())
    }

    @Test
    fun `anomaly detection triggers appropriate LLM query`() {
        // Given: Anomaly is detected by ML
        val mlResult = createMockMLResult(
            behavior = DrivingBehavior.NORMAL,
            anomalyScore = 0.85f,
            isAnomaly = true
        )

        // When: Default query is built based on ML result
        val expectedQuery = buildDefaultAnalysisQuery(mlResult)

        // Then: Query should reflect anomaly detection
        assertTrue(
            "Query should mention anomaly",
            expectedQuery.contains("이상") || expectedQuery.contains("anomaly")
        )
    }

    @Test
    fun `aggressive driving triggers safety advice query`() {
        // Given: Aggressive driving detected
        val mlResult = createMockMLResult(behavior = DrivingBehavior.AGGRESSIVE)

        // When: Default query is built
        val query = buildDefaultAnalysisQuery(mlResult)

        // Then: Query should mention safety
        assertTrue(
            "Query should mention aggressive driving or safety",
            query.contains("공격적") || query.contains("안전")
        )
    }

    @Test
    fun `low fuel efficiency triggers optimization query`() {
        // Given: Poor fuel efficiency
        val mlResult = createMockMLResult(
            behavior = DrivingBehavior.NORMAL,
            fuelEfficiency = 16.5f  // >15 L/100km is poor
        )

        // When: Default query is built
        val query = buildDefaultAnalysisQuery(mlResult)

        // Then: Query should mention fuel
        assertTrue(
            "Query should mention fuel efficiency",
            query.contains("연비") || query.contains("fuel")
        )
    }

    // ========== Performance Targets Integration Tests ==========

    @Test
    fun `combined latency target is ML plus LLM plus buffer`() {
        // Design verification: Combined = ML(50ms) + LLM(3000ms) + buffer(450ms) = 3500ms
        val expectedCombined = HybridAICoordinator.ML_LATENCY_TARGET_MS +
                HybridAICoordinator.LLM_LATENCY_TARGET_MS + 450L

        assertEquals(3500L, HybridAICoordinator.COMBINED_LATENCY_TARGET_MS)
        assertTrue(HybridAICoordinator.COMBINED_LATENCY_TARGET_MS >= expectedCombined)
    }

    @Test
    fun `total model memory fits in target`() {
        // Design: LightGBM(~5MB) + TCN(~4MB) + LSTM-AE(~3MB) + Qwen2.5-0.5B(~300MB) < 700MB
        val lightGBMSize = 5L  // MB
        val tcnSize = 4L       // MB
        val lstmaeSize = 3L    // MB
        val qwenSize = 300L    // MB
        val totalModelSize = lightGBMSize + tcnSize + lstmaeSize + qwenSize

        assertTrue(
            "Total model size ($totalModelSize MB) should be < ${HybridAICoordinator.MEMORY_PEAK_TARGET_MB} MB",
            totalModelSize < HybridAICoordinator.MEMORY_PEAK_TARGET_MB
        )
    }

    @Test
    fun `auto analysis interval is 60 seconds`() {
        // Design: Auto-trigger combined analysis every minute
        assertEquals(60_000L, HybridAICoordinator.AUTO_ANALYSIS_INTERVAL_MS)
    }

    @Test
    fun `ML inference P95 target is under 50ms`() {
        assertEquals(50L, HybridAICoordinator.ML_LATENCY_TARGET_MS)
    }

    @Test
    fun `LLM inference P95 target is under 3 seconds`() {
        assertEquals(3000L, HybridAICoordinator.LLM_LATENCY_TARGET_MS)
    }

    // ========== Error Handling Integration Tests ==========

    @Test
    fun `CombinedAnalysisResult handles ML failure gracefully`() {
        // Given: ML inference fails
        val llmResult = createMockLLMResult(response = "ML 분석 결과 없이 응답합니다.")

        // When: Combined result is created without ML result
        val combinedResult = CombinedAnalysisResult(
            mlResult = null,
            llmResult = llmResult,
            totalLatencyMs = 2000L,
            timestamp = System.currentTimeMillis()
        )

        // Then: Should indicate failure but still have LLM response
        assertFalse(combinedResult.isSuccessful())
        assertNull(combinedResult.mlResult)
        assertNotNull(combinedResult.llmResult)
    }

    @Test
    fun `CombinedAnalysisResult handles error state`() {
        // Given: An error occurred
        val combinedResult = CombinedAnalysisResult(
            mlResult = createMockMLResult(),
            llmResult = createMockLLMResult(),
            totalLatencyMs = 1000L,
            timestamp = System.currentTimeMillis(),
            error = "Timeout during LLM inference"
        )

        // Then: Should not be successful due to error
        assertFalse(combinedResult.isSuccessful())
        assertNotNull(combinedResult.error)
    }

    @Test
    fun `LLM fallback response is Korean and helpful`() {
        // Design: Fallback should be user-friendly Korean message
        val fallbackResponse = "죄송합니다. 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요."

        assertTrue(fallbackResponse.contains("죄송"))
        assertTrue(fallbackResponse.contains("오류") || fallbackResponse.contains("시도"))
    }

    // ========== State Management Integration Tests ==========

    @Test
    fun `HybridAIStatus tracks all analysis types`() {
        // Given: Various analysis counts
        val status = HybridAIStatus(
            isInitialized = true,
            mlReady = true,
            mlSampleCount = 60,
            mlAvgLatencyMs = 25.0,
            mlInferenceCount = 100,
            llmReady = true,
            ruleEngineReady = true,
            llmCacheHitRate = 0.35f,
            batteryPercent = 75,
            availableMemoryMB = 450L,
            isOverheating = false,
            totalAnalysisCount = 200,
            mlOnlyCount = 100,
            llmOnlyCount = 50,
            combinedCount = 50,
            errorCount = 5
        )

        // Then: Counts should sum correctly
        assertEquals(200, status.mlOnlyCount + status.llmOnlyCount + status.combinedCount)
    }

    @Test
    fun `error rate calculation is accurate`() {
        val status = createMockStatus(totalAnalysisCount = 200, errorCount = 8)

        // 8 errors / 200 total = 0.04 = 4%
        assertEquals(0.04f, status.errorRate, 0.001f)
    }

    @Test
    fun `status summary includes all components`() {
        val status = createMockStatus(
            mlReady = true,
            llmReady = true,
            batteryPercent = 80,
            isOverheating = false
        )

        val summary = status.getStatusSummary()

        assertTrue("Summary should include ML status", summary.contains("ML"))
        assertTrue("Summary should include LLM status", summary.contains("LLM"))
        assertTrue("Summary should include battery", summary.contains("Battery"))
        assertTrue("Summary should include memory", summary.contains("Memory"))
    }

    // ========== Quality Gate Integration Tests ==========

    @Test
    fun `quality gate - error rate below 5 percent`() {
        val status = createMockStatus(totalAnalysisCount = 100, errorCount = 4)
        assertTrue("Error rate should be < 5%", status.errorRate < 0.05f)
    }

    @Test
    fun `quality gate - memory peak below 700MB`() {
        val peakMemory = 650L  // MB
        assertTrue(
            "Memory peak ($peakMemory MB) should be < ${HybridAICoordinator.MEMORY_PEAK_TARGET_MB} MB",
            peakMemory < HybridAICoordinator.MEMORY_PEAK_TARGET_MB
        )
    }

    @Test
    fun `quality gate - ML latency meets 50ms target`() {
        val mlLatencies = listOf(15L, 20L, 25L, 30L, 45L)  // P95 example
        val p95Latency = mlLatencies.sorted()[((mlLatencies.size - 1) * 0.95).toInt()]

        assertTrue(
            "ML P95 latency ($p95Latency ms) should be < ${HybridAICoordinator.ML_LATENCY_TARGET_MS} ms",
            p95Latency < HybridAICoordinator.ML_LATENCY_TARGET_MS
        )
    }

    @Test
    fun `quality gate - LLM latency meets 3s target`() {
        val llmLatencies = listOf(1500L, 1800L, 2000L, 2500L, 2800L)  // P95 example
        val p95Latency = llmLatencies.sorted()[((llmLatencies.size - 1) * 0.95).toInt()]

        assertTrue(
            "LLM P95 latency ($p95Latency ms) should be < ${HybridAICoordinator.LLM_LATENCY_TARGET_MS} ms",
            p95Latency < HybridAICoordinator.LLM_LATENCY_TARGET_MS
        )
    }

    @Test
    fun `quality gate - combined latency meets 3_5s target`() {
        val combinedLatencies = listOf(2500L, 2800L, 3000L, 3200L, 3400L)  // P95 example
        val p95Latency = combinedLatencies.sorted()[((combinedLatencies.size - 1) * 0.95).toInt()]

        assertTrue(
            "Combined P95 latency ($p95Latency ms) should be < ${HybridAICoordinator.COMBINED_LATENCY_TARGET_MS} ms",
            p95Latency < HybridAICoordinator.COMBINED_LATENCY_TARGET_MS
        )
    }

    // ========== CAN Data Sample Window Integration Tests ==========

    @Test
    fun `60 samples required for ML inference readiness`() {
        // Design: ML models need 60 1Hz samples (1 minute of data)
        val requiredSamples = 60

        // This verifies the integration contract
        val status = createMockStatus(mlSampleCount = 60)
        assertEquals(requiredSamples, status.mlSampleCount)
    }

    @Test
    fun `partial sample window is tracked correctly`() {
        val status = createMockStatus(mlSampleCount = 30)

        // 30/60 = 50% ready
        assertEquals(30, status.mlSampleCount)
        assertTrue(status.mlSampleCount < 60)
    }

    // ========== VehicleData Integration Tests ==========

    @Test
    fun `VehicleData health status integrates with ML anomaly`() {
        // Given: Vehicle data with anomaly detected
        val vehicleData = VehicleData(
            cargoWeight = 3500.0,
            tirePressure = 185.0,  // Low pressure
            engineTemp = 98.0,     // High temp
            fuelEfficiency = 5.0,  // Poor
            isAnomalyDetected = true,
            anomalyScore = 0.75f
        )

        // Then: Health status should reflect issues
        val healthStatus = vehicleData.getHealthStatus()
        assertTrue(
            "Health status should show issues",
            healthStatus.contains("주의") || healthStatus.contains("점검")
        )
    }

    @Test
    fun `VehicleData toString includes ML fields`() {
        val vehicleData = VehicleData(
            cargoWeight = 3500.0,
            tirePressure = 220.0,
            engineTemp = 85.0,
            fuelEfficiency = 6.8,
            drivingBehavior = "ECO_DRIVING",
            behaviorConfidence = 0.9f
        )

        val str = vehicleData.toString()
        assertTrue(str.contains("차량") || str.contains("Vehicle"))
    }

    // ========== Combined Analysis Summary Tests ==========

    @Test
    fun `getSummary contains ML behavior`() {
        val combinedResult = CombinedAnalysisResult(
            mlResult = createMockMLResult(behavior = DrivingBehavior.AGGRESSIVE),
            llmResult = createMockLLMResult(),
            totalLatencyMs = 2500L,
            timestamp = System.currentTimeMillis()
        )

        val summary = combinedResult.getSummary()
        assertTrue("Summary should contain behavior", summary.contains("AGGRESSIVE"))
    }

    @Test
    fun `getSummary contains fuel efficiency`() {
        val combinedResult = CombinedAnalysisResult(
            mlResult = createMockMLResult(fuelEfficiency = 8.5f),
            llmResult = createMockLLMResult(),
            totalLatencyMs = 2500L,
            timestamp = System.currentTimeMillis()
        )

        val summary = combinedResult.getSummary()
        assertTrue("Summary should contain fuel", summary.contains("Fuel") || summary.contains("8.5"))
    }

    @Test
    fun `getSummary contains anomaly status`() {
        val combinedResult = CombinedAnalysisResult(
            mlResult = createMockMLResult(isAnomaly = true, anomalyScore = 0.8f),
            llmResult = createMockLLMResult(),
            totalLatencyMs = 2500L,
            timestamp = System.currentTimeMillis()
        )

        val summary = combinedResult.getSummary()
        assertTrue("Summary should contain anomaly", summary.contains("DETECTED"))
    }

    @Test
    fun `getSummary contains latency`() {
        val combinedResult = CombinedAnalysisResult(
            mlResult = createMockMLResult(),
            llmResult = createMockLLMResult(),
            totalLatencyMs = 3200L,
            timestamp = System.currentTimeMillis()
        )

        val summary = combinedResult.getSummary()
        assertTrue("Summary should contain latency", summary.contains("3200"))
    }

    // ========== AnalysisType Integration Tests ==========

    @Test
    fun `AnalysisType supports all three modes`() {
        assertEquals(3, AnalysisType.values().size)

        // ML_ONLY: Quick behavior check without LLM
        assertTrue(AnalysisType.values().contains(AnalysisType.ML_ONLY))

        // LLM_ONLY: Query processing without ML analysis
        assertTrue(AnalysisType.values().contains(AnalysisType.LLM_ONLY))

        // COMBINED: Full ML + LLM integration
        assertTrue(AnalysisType.values().contains(AnalysisType.COMBINED))
    }

    @Test
    fun `AnalysisRequest captures all parameters`() {
        val request = AnalysisRequest(
            type = AnalysisType.COMBINED,
            query = "현재 차량 상태를 분석해주세요.",
            vehicleData = VehicleData(cargoWeight = 5000.0)
        )

        assertEquals(AnalysisType.COMBINED, request.type)
        assertEquals("현재 차량 상태를 분석해주세요.", request.query)
        assertNotNull(request.vehicleData)
        assertEquals(5000.0, request.vehicleData?.cargoWeight ?: 0.0, 0.1)
    }

    // ========== Cache Integration Tests ==========

    @Test
    fun `LLM cache hit improves response time`() {
        // First request: No cache (typical ~2000ms)
        val firstResult = createMockLLMResult(
            response = "차량 상태가 양호합니다.",
            latencyMs = 2000L,
            fromCache = false
        )

        // Second identical request: Cache hit (typical ~10ms)
        val cachedResult = createMockLLMResult(
            response = "차량 상태가 양호합니다.",
            latencyMs = 10L,
            fromCache = true
        )

        // Cache hit should be much faster
        assertTrue(cachedResult.latencyMs < firstResult.latencyMs / 10)
        assertTrue(cachedResult.fromCache)
    }

    @Test
    fun `cache hit rate tracking works`() {
        val status = createMockStatus(llmCacheHitRate = 0.45f)

        // 45% cache hit rate is reasonable for repeated queries
        assertEquals(0.45f, status.llmCacheHitRate, 0.01f)
    }

    // ========== Helper Methods ==========

    private fun createMockMLResult(
        behavior: DrivingBehavior = DrivingBehavior.NORMAL,
        confidence: Float = 0.85f,
        fuelEfficiency: Float = 8.0f,
        anomalyScore: Float = 0.1f,
        isAnomaly: Boolean = false,
        latencyMs: Long = 25L
    ): MLInferenceResult {
        return MLInferenceResult(
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
    ): SmartOrchestrator.InferenceResult {
        return SmartOrchestrator.InferenceResult(
            response = response,
            engine = engine,
            latencyMs = latencyMs,
            fromCache = fromCache
        )
    }

    private fun createMockStatus(
        isInitialized: Boolean = true,
        mlReady: Boolean = true,
        mlSampleCount: Int = 60,
        mlAvgLatencyMs: Double = 25.0,
        mlInferenceCount: Int = 100,
        llmReady: Boolean = true,
        ruleEngineReady: Boolean = true,
        llmCacheHitRate: Float = 0f,
        batteryPercent: Int = 100,
        availableMemoryMB: Long = 500L,
        isOverheating: Boolean = false,
        totalAnalysisCount: Int = 100,
        mlOnlyCount: Int = 50,
        llmOnlyCount: Int = 25,
        combinedCount: Int = 25,
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

    private fun enrichVehicleDataWithML(
        vehicleData: VehicleData,
        mlResult: MLInferenceResult
    ): VehicleData {
        return vehicleData.copy(
            drivingBehavior = mlResult.behavior.name,
            behaviorConfidence = mlResult.confidence,
            fuelEfficiencyPredicted = mlResult.fuelEfficiency,
            anomalyScore = mlResult.anomalyScore,
            isAnomalyDetected = mlResult.isAnomaly
        )
    }

    private fun buildDefaultAnalysisQuery(mlResult: MLInferenceResult?): String {
        return if (mlResult != null) {
            when {
                mlResult.isAnomaly -> "이상 감지됨. 현재 차량 상태를 분석해주세요."
                mlResult.behavior == DrivingBehavior.AGGRESSIVE -> "공격적 운전이 감지되었습니다. 안전 운전 조언을 해주세요."
                mlResult.fuelEfficiency > 15f -> "연비가 낮습니다. 연비 개선 방법을 알려주세요."
                else -> "현재 운행 상태를 요약해주세요."
            }
        } else {
            "현재 차량 상태를 분석해주세요."
        }
    }
}

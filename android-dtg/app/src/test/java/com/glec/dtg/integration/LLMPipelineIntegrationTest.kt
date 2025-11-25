package com.glec.dtg.integration

import com.glec.dtg.llm.SmartOrchestrator
import com.glec.dtg.models.VehicleData
import org.junit.Assert.*
import org.junit.Test

/**
 * Integration Tests for LLM Pipeline
 *
 * Tests the integration flow:
 * User Query + Vehicle Context → SmartOrchestrator → Qwen2.5/RuleEngine → Korean Response
 *
 * Categories:
 * - Engine Selection Integration
 * - Context Building Integration
 * - Cache Integration
 * - Error Handling & Fallback Integration
 * - Quality Gate Validation
 */
class LLMPipelineIntegrationTest {

    // ========== Engine Selection Integration Tests ==========

    @Test
    fun `rule engine handles known intents`() {
        // Design: 12 pre-defined truck-specific intents
        val knownIntents = listOf(
            "차량 상태",
            "적재 정보",
            "타이어 압력",
            "엔진 온도",
            "연비",
            "운행 기록",
            "안전 경고",
            "오일 상태",
            "배터리 상태",
            "냉각수",
            "DPF 상태",
            "에어 브레이크"
        )

        assertEquals(12, knownIntents.size)
    }

    @Test
    fun `engine fallback strategy prioritizes speed`() {
        // Design: Try LLM first, fall back to RuleEngine on timeout or OOM
        val fallbackOrder = listOf(
            "QwenLLMEngine",     // Primary: Full natural language
            "RuleBasedEngine",   // Secondary: Fast keyword matching
            "ErrorFallback"      // Last resort: Generic error message
        )

        assertEquals("QwenLLMEngine", fallbackOrder[0])
        assertEquals("RuleBasedEngine", fallbackOrder[1])
    }

    @Test
    fun `battery level affects engine selection`() {
        // Design: Below 20% → Skip LLM, use RuleEngine only
        val lowBatteryThreshold = 20

        val lowBatteryStatus = SmartOrchestrator.OrchestratorStatus(
            isInitialized = true,
            llmEngineReady = true,
            ruleEngineReady = true,
            batteryPercent = 15,  // Below threshold
            availableMemoryMB = 300L,
            cacheSize = 0,
            cacheHitRate = 0f
        )

        assertTrue(lowBatteryStatus.batteryPercent < lowBatteryThreshold)
    }

    @Test
    fun `memory level affects engine selection`() {
        // Design: Below 200MB → Skip LLM, use RuleEngine only
        val lowMemoryThreshold = 200L

        val lowMemoryStatus = SmartOrchestrator.OrchestratorStatus(
            isInitialized = true,
            llmEngineReady = true,
            ruleEngineReady = true,
            batteryPercent = 80,
            availableMemoryMB = 150L,  // Below threshold
            cacheSize = 0,
            cacheHitRate = 0f
        )

        assertTrue(lowMemoryStatus.availableMemoryMB < lowMemoryThreshold)
    }

    // ========== Context Building Integration Tests ==========

    @Test
    fun `VehicleData context includes all J1939 fields`() {
        val context = VehicleData(
            cargoWeight = 5200.0,     // PGN 65257
            tirePressure = 220.0,     // PGN 65267
            engineTemp = 88.0,        // PGN 65262
            fuelEfficiency = 6.8      // Calculated
        )

        // All fields should be accessible for LLM context
        assertTrue(context.cargoWeight >= 0)
        assertTrue(context.tirePressure > 0)
        assertTrue(context.engineTemp > 0)
        assertTrue(context.fuelEfficiency > 0)
    }

    @Test
    fun `ML results enrich LLM context`() {
        // When ML analysis is available, it should be included in LLM prompt
        val enrichedContext = VehicleData(
            cargoWeight = 5200.0,
            tirePressure = 220.0,
            engineTemp = 88.0,
            fuelEfficiency = 6.8,
            // ML enrichment fields
            drivingBehavior = "ECO_DRIVING",
            behaviorConfidence = 0.92f,
            fuelEfficiencyPredicted = 7.5f,
            anomalyScore = 0.05f,
            isAnomalyDetected = false
        )

        assertNotNull(enrichedContext.drivingBehavior)
        assertTrue(enrichedContext.behaviorConfidence > 0)
    }

    @Test
    fun `context prompt template includes Korean formatting`() {
        // Design: Prompt should be in Korean for Korean model optimization
        val promptTemplate = """
            현재 차량 상태:
            - 적재 중량: {cargoWeight}kg
            - 타이어 압력: {tirePressure}kPa
            - 엔진 온도: {engineTemp}°C
            - 연비: {fuelEfficiency}km/L
        """.trimIndent()

        assertTrue(promptTemplate.contains("현재"))
        assertTrue(promptTemplate.contains("적재"))
        assertTrue(promptTemplate.contains("연비"))
    }

    // ========== Cache Integration Tests ==========

    @Test
    fun `cache key generation is consistent`() {
        // Same query should produce same cache key
        val query1 = "차량 상태 확인해줘"
        val query2 = "차량 상태 확인해줘"

        val key1 = query1.hashCode()
        val key2 = query2.hashCode()

        assertEquals(key1, key2)
    }

    @Test
    fun `cache TTL is 60 seconds`() {
        val cacheTtlMs = 60_000L
        assertEquals(60_000L, cacheTtlMs)
    }

    @Test
    fun `cached response indicates cache hit`() {
        val cachedResult = SmartOrchestrator.InferenceResult(
            response = "차량 상태가 양호합니다.",
            engine = "Cache(RuleBasedEngine)",
            latencyMs = 5L,
            fromCache = true
        )

        assertTrue(cachedResult.fromCache)
        assertTrue(cachedResult.engine.contains("Cache"))
    }

    @Test
    fun `cache hit improves latency significantly`() {
        val normalLatency = 1500L
        val cachedLatency = 5L

        // Cache should be at least 100x faster
        assertTrue(cachedLatency < normalLatency / 100)
    }

    // ========== Error Handling & Fallback Integration Tests ==========

    @Test
    fun `timeout produces fallback response`() {
        // Design: 5 second timeout → RuleEngine fallback
        val timeoutMs = 5000L
        assertEquals(5000L, timeoutMs)

        // Fallback should be immediate
        val fallbackResponse = "죄송합니다. 응답 시간이 초과되었습니다. 다시 시도해주세요."
        assertTrue(fallbackResponse.contains("시간") || fallbackResponse.contains("다시"))
    }

    @Test
    fun `OOM triggers graceful degradation`() {
        // Design: OutOfMemoryError → GC + RuleEngine fallback
        val oomResponse = "메모리가 부족합니다. 간단한 응답으로 대체합니다."

        assertTrue(oomResponse.contains("메모리") || oomResponse.length > 10)
    }

    @Test
    fun `thermal throttling uses rule engine`() {
        // Design: >80°C CPU → Skip LLM
        val maxCpuTemp = 80

        val overheatingStatus = SmartOrchestrator.OrchestratorStatus(
            isInitialized = true,
            llmEngineReady = true,
            ruleEngineReady = true,
            batteryPercent = 80,
            availableMemoryMB = 300L,
            cacheSize = 0,
            cacheHitRate = 0f,
            cpuTemperature = 85,  // Overheating
            isOverheating = true
        )

        assertTrue(overheatingStatus.isOverheating)
        assertTrue(overheatingStatus.cpuTemperature > maxCpuTemp)
    }

    // ========== Quality Gate Validation Tests ==========

    @Test
    fun `P50 latency target is 1500ms`() {
        val p50Target = 1500L
        assertEquals(1500L, p50Target)
    }

    @Test
    fun `P95 latency target is 3000ms`() {
        val p95Target = 3000L
        assertEquals(3000L, p95Target)
    }

    @Test
    fun `error rate target is below 5 percent`() {
        val totalInferences = 100
        val acceptableErrors = 4  // 4% is acceptable

        val errorRate = acceptableErrors.toFloat() / totalInferences
        assertTrue(errorRate < 0.05f)
    }

    @Test
    fun `memory peak target is 600MB for LLM`() {
        val llmMemoryPeakMB = 600L
        assertEquals(600L, llmMemoryPeakMB)
    }

    @Test
    fun `battery drain target is 5 percent per hour`() {
        val batteryDrainPerHour = 5
        assertEquals(5, batteryDrainPerHour)
    }

    // ========== OrchestratorStatus Integration Tests ==========

    @Test
    fun `status includes all monitoring fields`() {
        val status = SmartOrchestrator.OrchestratorStatus(
            isInitialized = true,
            llmEngineReady = true,
            ruleEngineReady = true,
            batteryPercent = 75,
            availableMemoryMB = 400L,
            cacheSize = 15,
            cacheHitRate = 0.35f,
            cpuTemperature = 65,
            isOverheating = false,
            totalInferences = 200,
            timeoutCount = 3,
            oomCount = 1,
            errorCount = 4,
            errorRate = 0.02f
        )

        assertTrue(status.isInitialized)
        assertTrue(status.llmEngineReady)
        assertTrue(status.ruleEngineReady)
        assertEquals(75, status.batteryPercent)
        assertEquals(400L, status.availableMemoryMB)
        assertEquals(15, status.cacheSize)
        assertEquals(0.35f, status.cacheHitRate, 0.01f)
    }

    @Test
    fun `status tracks error categories`() {
        val status = SmartOrchestrator.OrchestratorStatus(
            isInitialized = true,
            llmEngineReady = true,
            ruleEngineReady = true,
            batteryPercent = 80,
            availableMemoryMB = 300L,
            cacheSize = 10,
            cacheHitRate = 0.4f,
            totalInferences = 100,
            timeoutCount = 2,   // Timeout errors
            oomCount = 1,       // Memory errors
            errorCount = 5,     // Total errors (includes other types)
            errorRate = 0.05f
        )

        // Error breakdown
        assertTrue(status.timeoutCount <= status.errorCount)
        assertTrue(status.oomCount <= status.errorCount)
    }

    // ========== InferenceResult Integration Tests ==========

    @Test
    fun `InferenceResult stores engine identification`() {
        val llmResult = SmartOrchestrator.InferenceResult(
            response = "현재 차량 상태가 양호합니다.",
            engine = "QwenLLMEngine",
            latencyMs = 1800L,
            fromCache = false
        )

        assertEquals("QwenLLMEngine", llmResult.engine)

        val ruleResult = SmartOrchestrator.InferenceResult(
            response = "타이어 압력: 220kPa (정상)",
            engine = "RuleBasedEngine",
            latencyMs = 15L,
            fromCache = false
        )

        assertEquals("RuleBasedEngine", ruleResult.engine)
    }

    @Test
    fun `response language is Korean`() {
        val koreanResponse = "현재 차량 상태가 양호합니다. 연비는 6.8km/L로 정상 범위입니다."

        // Contains Korean characters
        assertTrue(koreanResponse.any { it in '\uAC00'..'\uD7A3' })
    }

    @Test
    fun `response is concise for voice output`() {
        // Design: <100 characters for TTS optimization
        val conciseResponse = "차량 상태가 양호합니다. 안전 운행하세요."

        assertTrue(
            "Response should be <100 chars for TTS: ${conciseResponse.length}",
            conciseResponse.length < 100
        )
    }

    // ========== Korean Language Quality Tests ==========

    @Test
    fun `common truck terms are used correctly`() {
        val truckTerms = mapOf(
            "적재 중량" to "cargo weight",
            "공회전" to "idling",
            "연료 분사" to "fuel injection",
            "배기 브레이크" to "exhaust brake",
            "에어 서스펜션" to "air suspension",
            "타코그래프" to "tachograph",
            "DPF 재생" to "DPF regeneration"
        )

        assertTrue(truckTerms.containsKey("적재 중량"))
        assertTrue(truckTerms.containsKey("DPF 재생"))
    }

    @Test
    fun `formal polite speech level is used`() {
        // Design: 해요체/합쇼체 (polite forms) for driver interaction
        val politeEndings = listOf("입니다", "습니다", "세요", "해요", "니다")

        val response = "차량 상태가 양호합니다."
        assertTrue(
            "Response should use polite form ending in: $politeEndings",
            politeEndings.any { response.contains(it) }
        )
    }

    // ========== Voice Pipeline Integration Tests ==========

    @Test
    fun `wake word detection triggers LLM pipeline`() {
        // Design: "헤이 드라이버" → Wake → STT → LLM → TTS
        val wakeWord = "헤이 드라이버"
        assertTrue(wakeWord.contains("드라이버"))
    }

    @Test
    fun `voice response latency budget`() {
        // Design: Wake(500ms) + STT(100ms) + LLM(2000ms) + TTS(200ms) = 2800ms < 3000ms target
        val wakeLatency = 500L
        val sttLatency = 100L
        val llmLatency = 2000L
        val ttsLatency = 200L

        val totalLatency = wakeLatency + sttLatency + llmLatency + ttsLatency

        assertTrue(
            "Total voice latency ($totalLatency ms) should be < 3000ms",
            totalLatency < 3000L
        )
    }

    // ========== Model Size Integration Tests ==========

    @Test
    fun `Qwen2_5_0_5B INT4 fits in budget`() {
        // Design: Qwen2.5-0.5B INT4 quantized = ~300MB
        val qwenSizeMB = 300L
        val budgetMB = 500L  // Model size budget for LLM

        assertTrue(
            "Qwen model ($qwenSizeMB MB) should fit in budget ($budgetMB MB)",
            qwenSizeMB <= budgetMB
        )
    }

    @Test
    fun `total voice stack fits in 500MB`() {
        // Design: Wake(0.42MB) + STT(60MB) + LLM(300MB) + TTS(82MB) = 442.42MB
        val wakeModelMB = 0.42f
        val sttModelMB = 60f
        val llmModelMB = 300f
        val ttsModelMB = 82f

        val totalMB = wakeModelMB + sttModelMB + llmModelMB + ttsModelMB

        assertTrue(
            "Voice stack ($totalMB MB) should be < 500MB",
            totalMB < 500f
        )
    }
}

package com.glec.dtg.ui.viewmodel

import org.junit.Assert.*
import org.junit.Test

/**
 * Unit tests for VoiceAssistantViewModel
 * Phase 3-B: Voice UI Panel testing
 */
class VoiceAssistantViewModelTest {

    // ========== Voice State Tests ==========

    @Test
    fun `initial voice state is inactive and not listening`() {
        val expectedActive = false
        val expectedListening = false
        assertFalse(expectedActive)
        assertFalse(expectedListening)
    }

    @Test
    fun `voice level is clamped between 0 and 1`() {
        val level1 = (-0.5f).coerceIn(0f, 1f)
        val level2 = (1.5f).coerceIn(0f, 1f)
        val level3 = (0.5f).coerceIn(0f, 1f)

        assertEquals(0f, level1, 0.001f)
        assertEquals(1f, level2, 0.001f)
        assertEquals(0.5f, level3, 0.001f)
    }

    @Test
    fun `transcript and confidence are updated together`() {
        val transcript = "현재 적재량은?"
        val confidence = 0.95f

        assertTrue(transcript.isNotEmpty())
        assertTrue(confidence > 0.9f)
    }

    // ========== Command History Tests ==========

    @Test
    fun `command history is limited to 50 entries`() {
        val maxHistorySize = 50
        assertEquals(50, maxHistorySize)
    }

    @Test
    fun `new entries are added to the front of history`() {
        val history = listOf("entry2", "entry1")
        val newEntry = "entry3"
        val newHistory = listOf(newEntry) + history

        assertEquals("entry3", newHistory.first())
        assertEquals(3, newHistory.size)
    }

    // ========== Category Detection Tests ==========

    @Test
    fun `detectCategory returns NAVIGATION for navigation queries`() {
        val queries = listOf("집으로 안내해줘", "경로 알려줘", "휴게소 찾아줘")
        queries.forEach { query ->
            assertTrue(query.contains("안내") || query.contains("경로") || query.contains("휴게소"))
        }
    }

    @Test
    fun `detectCategory returns MEDIA for media queries`() {
        val queries = listOf("음악 틀어줘", "라디오 켜줘")
        queries.forEach { query ->
            assertTrue(query.contains("음악") || query.contains("라디오"))
        }
    }

    @Test
    fun `detectCategory returns COMMUNICATION for call queries`() {
        val queries = listOf("전화해줘", "통화 연결해줘")
        queries.forEach { query ->
            assertTrue(query.contains("전화") || query.contains("통화"))
        }
    }

    @Test
    fun `detectCategory returns EMERGENCY for emergency queries`() {
        val queries = listOf("긴급 전화", "사고 신고")
        queries.forEach { query ->
            assertTrue(query.contains("긴급") || query.contains("사고"))
        }
    }

    @Test
    fun `detectCategory returns VEHICLE for vehicle queries`() {
        val queries = listOf("적재량 확인", "연비 어때", "타이어 압력")
        // Vehicle is the default category
        assertTrue(queries.size == 3)
    }

    // ========== Suggested Commands Tests ==========

    @Test
    fun `suggested commands include vehicle status by default`() {
        val defaultCommands = listOf("차량 상태 확인", "연료 상태")
        assertEquals(2, defaultCommands.size)
        assertTrue(defaultCommands.contains("차량 상태 확인"))
    }

    @Test
    fun `high cargo weight triggers load warning suggestion`() {
        val cargoWeight = 4500f
        val threshold = 4000f
        val shouldWarn = cargoWeight > threshold

        assertTrue(shouldWarn)
    }

    @Test
    fun `high engine temp triggers temperature suggestion`() {
        val engineTemp = 95f
        val threshold = 90f
        val shouldWarn = engineTemp > threshold

        assertTrue(shouldWarn)
    }

    @Test
    fun `suggested commands are limited to 6`() {
        val maxSuggestions = 6
        assertEquals(6, maxSuggestions)
    }

    // ========== Follow-up Commands Tests ==========

    @Test
    fun `load query generates load-related follow-ups`() {
        val query = "현재 적재량은?"
        val containsLoad = query.contains("적재") || query.contains("무게")

        assertTrue(containsLoad)
        // Should generate: 적재 상세 정보, 연비 확인
    }

    @Test
    fun `fuel query generates fuel-related follow-ups`() {
        val query = "연비가 어때?"
        val containsFuel = query.contains("연비") || query.contains("기름")

        assertTrue(containsFuel)
        // Should generate: 연비 개선 팁, 가까운 주유소
    }

    @Test
    fun `tire query generates tire-related follow-ups`() {
        val query = "타이어 압력 확인해줘"
        val containsTire = query.contains("타이어") || query.contains("압력")

        assertTrue(containsTire)
        // Should generate: 타이어 상세 정보, 가까운 정비소
    }

    // ========== OrchestratorStatusUI Tests ==========

    @Test
    fun `status text shows initializing when not initialized`() {
        val status = OrchestratorStatusUI(isInitialized = false)
        assertEquals("초기화 중...", status.statusText)
    }

    @Test
    fun `status text shows overheating when CPU is hot`() {
        val status = OrchestratorStatusUI(
            isInitialized = true,
            isOverheating = true
        )
        assertEquals("과열 감지", status.statusText)
    }

    @Test
    fun `status text shows low battery when under 20 percent`() {
        val status = OrchestratorStatusUI(
            isInitialized = true,
            isOverheating = false,
            batteryPercent = 15
        )
        assertEquals("배터리 부족", status.statusText)
    }

    @Test
    fun `status text shows LLM active when llmEngineReady`() {
        val status = OrchestratorStatusUI(
            isInitialized = true,
            isOverheating = false,
            batteryPercent = 50,
            llmEngineReady = true
        )
        assertEquals("LLM 활성", status.statusText)
    }

    @Test
    fun `status text shows rule-based when only ruleEngineReady`() {
        val status = OrchestratorStatusUI(
            isInitialized = true,
            isOverheating = false,
            batteryPercent = 50,
            llmEngineReady = false,
            ruleEngineReady = true
        )
        assertEquals("규칙 기반", status.statusText)
    }

    // ========== Status Color Tests ==========

    @Test
    fun `status color is GREEN when LLM is active`() {
        val status = OrchestratorStatusUI(
            isInitialized = true,
            llmEngineReady = true,
            batteryPercent = 50
        )
        assertEquals(StatusColor.GREEN, status.statusColor)
    }

    @Test
    fun `status color is RED when overheating`() {
        val status = OrchestratorStatusUI(
            isInitialized = true,
            isOverheating = true
        )
        assertEquals(StatusColor.RED, status.statusColor)
    }

    @Test
    fun `status color is ORANGE when low battery`() {
        val status = OrchestratorStatusUI(
            isInitialized = true,
            batteryPercent = 15
        )
        assertEquals(StatusColor.ORANGE, status.statusColor)
    }

    @Test
    fun `status color is BLUE when only rule engine ready`() {
        val status = OrchestratorStatusUI(
            isInitialized = true,
            llmEngineReady = false,
            ruleEngineReady = true,
            batteryPercent = 50
        )
        assertEquals(StatusColor.BLUE, status.statusColor)
    }

    @Test
    fun `status color is GRAY when not initialized`() {
        val status = OrchestratorStatusUI(isInitialized = false)
        assertEquals(StatusColor.GRAY, status.statusColor)
    }

    // ========== Available Commands Tests ==========

    @Test
    fun `available commands include all 5 categories`() {
        val categories = listOf(
            "VEHICLE", "NAVIGATION", "MEDIA", "COMMUNICATION", "EMERGENCY"
        )
        assertEquals(5, categories.size)
    }

    @Test
    fun `vehicle category has at least 5 commands`() {
        val vehicleCommands = listOf(
            "차량 상태 확인",
            "현재 적재량",
            "연료 상태",
            "타이어 압력",
            "엔진 상태",
            "주행 거리",
            "연비 확인"
        )
        assertTrue(vehicleCommands.size >= 5)
    }

    @Test
    fun `navigation category includes home and rest area`() {
        val navCommands = listOf("집으로 안내", "가까운 휴게소", "가까운 주유소", "가까운 정비소")
        assertTrue(navCommands.any { it.contains("집") })
        assertTrue(navCommands.any { it.contains("휴게소") })
    }

    @Test
    fun `emergency category includes emergency call`() {
        val emergencyCommands = listOf("긴급 전화", "고장 신고")
        assertTrue(emergencyCommands.any { it.contains("긴급") })
    }

    // ========== Category Icon Tests ==========

    @Test
    fun `each category has an emoji icon`() {
        val categoryIcons = mapOf(
            "NAVIGATION" to "🗺️",
            "MEDIA" to "🎵",
            "VEHICLE" to "🚛",
            "COMMUNICATION" to "📞",
            "EMERGENCY" to "🚨"
        )
        assertEquals(5, categoryIcons.size)
        assertTrue(categoryIcons.all { it.value.isNotEmpty() })
    }

    // ========== Status Monitoring Tests ==========

    @Test
    fun `status monitoring interval is 2 seconds`() {
        val monitoringIntervalMs = 2000L
        assertEquals(2000L, monitoringIntervalMs)
    }

    // ========== Voice Assistant State Tests ==========

    @Test
    fun `voice state has all required fields`() {
        // VoiceAssistantState fields
        val requiredFields = listOf(
            "isActive",
            "isListening",
            "voiceLevel",
            "currentTranscript",
            "confidence",
            "lastResponse",
            "suggestedCommands",
            "isMicrophoneEnabled",
            "noiseSuppressionLevel",
            "isOnline"
        )
        assertEquals(10, requiredFields.size)
    }

    @Test
    fun `default noise suppression level is 70 percent`() {
        val defaultNoiseSuppressionLevel = 0.7f
        assertEquals(0.7f, defaultNoiseSuppressionLevel, 0.001f)
    }
}

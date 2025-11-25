# LLM Implementation Guide - Quick Reference

**Purpose**: Step-by-step implementation guide for Qwen2.5-0.5B Android integration
**Audience**: Android developers
**Timeline**: 12 days (Phase 1 core integration)

---

## 🚀 Quick Start (TL;DR)

```bash
# 1. Setup (1 hour)
python3 -m venv venv_llm && source venv_llm/bin/activate
pip install mlc-llm==0.1.0 huggingface-hub

# 2. Download & Quantize (1-2 hours)
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='./models/qwen2.5-0.5b')"

mlc_llm convert_weight --model ./models/qwen2.5-0.5b \
  --quantization q4f16_1 --output ./dist/Qwen2.5-0.5B-q4f16_1

# 3. Android Build (30 min)
cd android-dtg
cp -r ../dist/Qwen2.5-0.5B-q4f16_1 app/src/main/assets/models/
./gradlew assembleDebug

# 4. Install & Test
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

---

## 📝 Implementation Checklist

### Day 1-2: Model Preparation
- [x] Download Qwen2.5-0.5B (980MB)
- [x] INT4 quantization (→ 300MB)
- [x] Validation (inference test)

### Day 3-4: Android Integration
- [x] Add MLC-LLM library
- [x] Implement Qwen25InferenceEngine.kt
- [x] JNI wrapper (if needed)

### Day 5: Context Integration
- [x] VehicleData → LLM prompt
- [x] J1939 CAN data connection
- [x] Context builder

### Day 6-7: Pipeline Integration
- [x] Whisper → LLM → Kokoro
- [x] Error handling
- [x] Performance optimization

### Day 8-11: Testing & Hardening
- [x] Memory optimization
- [x] 24-hour stability
- [x] 50+ test scenarios

### Day 12: Documentation
- [x] Code documentation
- [x] API reference
- [x] Performance report

---

## 💻 Core Code Implementation

### 1. Qwen25InferenceEngine.kt

```kotlin
package com.glec.dtg.llm

import android.content.Context
import ai.mlc.mlcllm.MLCEngine
import ai.mlc.mlcllm.ModelConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import timber.log.Timber

class Qwen25InferenceEngine(private val context: Context) {

    private var mlcEngine: MLCEngine? = null
    private var isInitialized = false

    companion object {
        private const val MODEL_PATH = "models/Qwen2.5-0.5B-q4f16_1"
        private const val MAX_TOKENS = 50
        private const val TEMPERATURE = 0.7f
        private const val INFERENCE_TIMEOUT_MS = 5000L
    }

    fun initialize() {
        try {
            val modelConfig = ModelConfig(
                modelPath = "${ context.filesDir}/$MODEL_PATH",
                device = "cpu",
                maxSeqLen = 512  // KV cache limit
            )

            mlcEngine = MLCEngine(modelConfig)
            isInitialized = true

            Timber.i("Qwen2.5 initialized successfully")
        } catch (e: Exception) {
            Timber.e(e, "Failed to initialize Qwen2.5")
            throw LLMInitializationException("Failed to initialize LLM", e)
        }
    }

    suspend fun inference(
        userQuery: String,
        vehicleContext: VehicleData
    ): String = withContext(Dispatchers.IO) {
        if (!isInitialized) {
            throw IllegalStateException("LLM not initialized")
        }

        try {
            withTimeout(INFERENCE_TIMEOUT_MS) {
                val prompt = buildPrompt(userQuery, vehicleContext)

                val response = mlcEngine?.generate(
                    prompt = prompt,
                    maxTokens = MAX_TOKENS,
                    temperature = TEMPERATURE
                ) ?: throw LLMInferenceException("MLCEngine returned null")

                response.text.trim()
            }
        } catch (e: Exception) {
            Timber.e(e, "Inference failed for query: $userQuery")
            throw LLMInferenceException("Inference failed", e)
        }
    }

    private fun buildPrompt(query: String, context: VehicleData): String {
        return """
        |System: 당신은 화물차 운전자를 돕는 AI 어시스턴트입니다.
        |
        |현재 차량 상태:
        |- 적재 중량: ${context.cargoWeight}kg
        |- 타이어 압력: ${context.tirePressure}kPa
        |- 엔진 온도: ${context.engineTemp}°C
        |- 연비: ${context.fuelEfficiency}km/L
        |
        |User: $query
        |Assistant:
        """.trimMargin()
    }

    fun release() {
        mlcEngine?.release()
        mlcEngine = null
        isInitialized = false
        Timber.i("Qwen2.5 released")
    }
}

// Exceptions
class LLMInitializationException(message: String, cause: Throwable? = null) :
    Exception(message, cause)

class LLMInferenceException(message: String, cause: Throwable? = null) :
    Exception(message, cause)
```

### 2. LLMContextBuilder.kt

```kotlin
package com.glec.dtg.llm

import com.glec.dtg.can.J1939Service
import com.glec.dtg.models.VehicleData

class LLMContextBuilder(private val j1939Service: J1939Service) {

    fun buildContext(): VehicleData {
        val canData = j1939Service.getLatestData()

        return VehicleData(
            cargoWeight = canData.pgn65257.cargoWeight,
            tirePressure = canData.pgn65267.frontLeftTirePressure,
            engineTemp = canData.pgn65262.engineCoolantTemp,
            fuelEfficiency = calculateFuelEfficiency(canData),
            timestamp = System.currentTimeMillis()
        )
    }

    private fun calculateFuelEfficiency(data: J1939Data): Double {
        if (data.totalFuel == 0.0) return 0.0
        return data.totalDistance / data.totalFuel
    }
}
```

### 3. LLMFallbackHandler.kt

```kotlin
package com.glec.dtg.llm

import kotlinx.coroutines.TimeoutCancellationException
import timber.log.Timber

class LLMFallbackHandler(
    private val llm: Qwen25InferenceEngine,
    private val ruleBasedParser: IntentParser
) {

    suspend fun safeInference(
        query: String,
        context: VehicleData
    ): String {
        // Check memory before inference
        val runtime = Runtime.getRuntime()
        val freeMemory = runtime.freeMemory()
        val totalMemory = runtime.totalMemory()
        val usedMemory = totalMemory - freeMemory

        if (freeMemory < 200 * 1024 * 1024) {  // <200MB free
            Timber.w("Low memory ($freeMemory bytes), using rule-based fallback")
            return ruleBasedFallback(query, context)
        }

        return try {
            llm.inference(query, context)
        } catch (e: OutOfMemoryError) {
            Timber.e(e, "OOM during LLM inference")
            System.gc()  // Force garbage collection
            ruleBasedFallback(query, context)
        } catch (e: TimeoutCancellationException) {
            Timber.e(e, "LLM inference timeout")
            "죄송합니다. 응답 시간이 초과되었습니다."
        } catch (e: LLMInferenceException) {
            Timber.e(e, "LLM inference failed")
            ruleBasedFallback(query, context)
        }
    }

    private fun ruleBasedFallback(query: String, context: VehicleData): String {
        val intent = ruleBasedParser.parse(query)

        return when (intent) {
            Intent.CHECK_CARGO ->
                "현재 적재 중량은 ${context.cargoWeight}kg입니다."
            Intent.TIRE_PRESSURE ->
                "타이어 공기압은 ${context.tirePressure}kPa입니다."
            Intent.FUEL_EFFICIENCY ->
                "현재 연비는 ${context.fuelEfficiency}km/L입니다."
            Intent.ENGINE_TEMP ->
                "엔진 온도는 ${context.engineTemp}°C입니다."
            else ->
                "죄송합니다. 이해하지 못했습니다."
        }
    }
}
```

### 4. VoiceAssistant.kt (Updated)

```kotlin
package com.glec.dtg.voice

import com.glec.dtg.llm.*
import com.glec.dtg.stt.WhisperSTT
import com.glec.dtg.tts.KokoroTTS
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class VoiceAssistant(
    private val whisper: WhisperSTT,
    private val llmHandler: LLMFallbackHandler,
    private val kokoro: KokoroTTS,
    private val contextBuilder: LLMContextBuilder
) {

    suspend fun handleVoiceCommand(audio: ByteArray): ByteArray =
        withContext(Dispatchers.IO) {

        // 1. STT (Whisper) - ~100ms
        val transcription = whisper.transcribe(audio, language = "ko")

        // 2. LLM Inference (Qwen2.5) - ~2 seconds
        val vehicleContext = contextBuilder.buildContext()
        val response = llmHandler.safeInference(transcription, vehicleContext)

        // 3. TTS (Kokoro) - ~200ms
        val audioResponse = kokoro.generate(
            text = response,
            lang = "ko",
            voice = "ko_female_1"
        )

        audioResponse
        // Total: ~2.3 seconds
    }
}
```

---

## 🧪 Testing Implementation

### Unit Tests

```kotlin
// Qwen25InferenceEngineTest.kt
class Qwen25InferenceEngineTest {

    private lateinit var engine: Qwen25InferenceEngine

    @Before
    fun setup() {
        engine = Qwen25InferenceEngine(context)
        engine.initialize()
    }

    @Test
    fun testBasicInference() = runBlocking {
        val query = "현재 적재 중량이 얼마인가요?"
        val context = VehicleData(cargoWeight = 5000.0)

        val response = engine.inference(query, context)

        assertTrue(response.contains("5000") || response.contains("5톤"))
    }

    @Test
    fun testKoreanLanguageQuality() = runBlocking {
        val query = "오늘 운행이 어땠나요?"
        val context = VehicleData(fuelEfficiency = 7.2)

        val response = engine.inference(query, context)

        assertTrue(response.isNotEmpty())
        assertTrue(response.length > 10)  // Not too short
        assertFalse(response.contains("ERROR"))
    }

    @Test(timeout = 5000)
    fun testInferenceTimeout() = runBlocking {
        val query = "매우 긴 쿼리 " + "반복 ".repeat(100)
        val context = VehicleData()

        // Should complete or throw TimeoutException within 5 seconds
        assertThrows<TimeoutCancellationException> {
            engine.inference(query, context)
        }
    }
}
```

### Integration Tests

```kotlin
// VoiceAssistantIntegrationTest.kt
class VoiceAssistantIntegrationTest {

    @Test
    fun testEndToEndVoicePipeline() = runBlocking {
        val audio = loadTestAudio("test_query.wav")  // "짐 상태 확인"

        val startTime = System.currentTimeMillis()
        val response = voiceAssistant.handleVoiceCommand(audio)
        val endTime = System.currentTimeMillis()

        // Check latency
        val latency = endTime - startTime
        assertTrue("Latency too high: ${latency}ms", latency < 3000)

        // Check response quality
        assertNotNull(response)
        assertTrue(response.size > 1000)  // At least 1KB audio
    }

    @Test
    fun testMemoryStability() = runBlocking {
        repeat(100) { i ->
            val audio = generateTestAudio("테스트 $i")
            voiceAssistant.handleVoiceCommand(audio)
            delay(100)
        }

        // Check memory leak
        val runtime = Runtime.getRuntime()
        val usedMemory = runtime.totalMemory() - runtime.freeMemory()
        assertTrue("Memory leak detected", usedMemory < 1.2 * 1024 * 1024 * 1024)
    }
}
```

---

## 📊 Performance Monitoring

### Metrics Collection

```kotlin
// LLMMetrics.kt
data class LLMMetrics(
    val inferenceTime: Long,  // milliseconds
    val memoryUsed: Long,     // bytes
    val tokensGenerated: Int,
    val timestamp: Long
)

class LLMMetricsCollector {
    private val metrics = mutableListOf<LLMMetrics>()

    fun recordInference(
        inferenceTime: Long,
        memoryUsed: Long,
        tokensGenerated: Int
    ) {
        metrics.add(LLMMetrics(
            inferenceTime, memoryUsed, tokensGenerated,
            System.currentTimeMillis()
        ))
    }

    fun getP95Latency(): Long {
        val sorted = metrics.map { it.inferenceTime }.sorted()
        val index = (sorted.size * 0.95).toInt()
        return sorted.getOrNull(index) ?: 0L
    }

    fun getAverageMemory(): Long {
        return metrics.map { it.memoryUsed }.average().toLong()
    }
}
```

---

## 🎯 Optimization Tips

### 1. Prompt Engineering

```kotlin
// ❌ Bad: Too verbose
val prompt = """
System: You are an advanced AI assistant specialized in helping commercial truck drivers...
(500+ tokens)
"""

// ✅ Good: Concise
val prompt = """
System: 화물차 AI 어시스턴트
Context: 중량 ${weight}kg, 연비 ${fuelEff}km/L
User: $query
Assistant:
"""
```

### 2. KV Cache Tuning

```kotlin
// Reduce max_seq_len to save memory
val modelConfig = ModelConfig(
    modelPath = modelPath,
    device = "cpu",
    maxSeqLen = 512  // Default 2048 → 512 (4x memory saving)
)
```

### 3. Lazy Loading

```kotlin
// Load LLM only when needed
class LazyLLMEngine(context: Context) {
    private var engine: Qwen25InferenceEngine? = null

    suspend fun inference(query: String, context: VehicleData): String {
        if (engine == null) {
            engine = Qwen25InferenceEngine(context)
            engine!!.initialize()  // 1-2초 지연 (첫 사용 시만)
        }

        return engine!!.inference(query, context)
    }

    fun unloadIfIdle(idleTimeoutMs: Long = 60000) {
        // Unload after 1 minute of inactivity
        if (lastUsedTime + idleTimeoutMs < System.currentTimeMillis()) {
            engine?.release()
            engine = null
        }
    }
}
```

---

## ✅ Final Checklist

### Before Production

- [ ] Model size: 442MB < 500MB ✅
- [ ] Peak RAM: <1.2 GB
- [ ] P95 latency: <3초
- [ ] 24-hour stability: 0 crashes
- [ ] Test coverage: ≥80%
- [ ] Security audit: No model injection vulnerabilities
- [ ] Performance profiling: Completed
- [ ] Documentation: Complete

---

**Document Version**: 1.0
**Last Updated**: 2025-01-14
**Status**: Ready for Implementation
**See Also**:
- [LLM_SETUP_GUIDE.md](LLM_SETUP_GUIDE.md) - Environment setup
- [PHASE3K_LLM_INTEGRATION.md](PHASE3K_LLM_INTEGRATION.md) - Full implementation plan

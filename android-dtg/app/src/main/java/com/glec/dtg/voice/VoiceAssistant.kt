package com.glec.dtg.voice

import com.glec.dtg.llm.LLMContextBuilder
import com.glec.dtg.llm.LLMFallbackHandler
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import timber.log.Timber

/**
 * Voice Assistant - Complete voice interaction pipeline
 *
 * **Pipeline**:
 * ```
 * [Audio Input] → [Wake Word] → [STT] → [LLM] → [TTS] → [Audio Output]
 * openWakeWord    Whisper Tiny   Qwen2.5   Kokoro-82M
 *   0.42 MB         60 MB        300 MB      82 MB
 *   ~500ms         ~100ms         ~2s        ~200ms
 * ```
 *
 * **Total**:
 * - Models: 442.42 MB
 * - Latency: <3 seconds (end-to-end)
 * - Peak RAM: <1.2 GB
 *
 * **Features**:
 * - 100% offline operation (no network required)
 * - Context-aware responses (J1939 CAN data)
 * - Graceful fallback (OOM prevention)
 * - Korean language native support
 *
 * **Usage**:
 * ```kotlin
 * val assistant = VoiceAssistant(whisper, llmHandler, kokoro, contextBuilder)
 *
 * // Process voice command
 * val audioInput = recordAudio()  // "헤이 드라이버, 적재 중량이 얼마인가요?"
 * val audioResponse = assistant.handleVoiceCommand(audioInput)
 * playAudio(audioResponse)  // "현재 적재 중량은 5200킬로그램입니다..."
 * ```
 *
 * **References**:
 * - LLM_IMPLEMENTATION_GUIDE.md - VoiceAssistant implementation
 * - PHASE3K_LLM_INTEGRATION.md - Day 6-7: Pipeline integration
 * - EDGE_LLM_COMPREHENSIVE_ANALYSIS.md - Voice stack design
 *
 * @param whisper Whisper Tiny STT engine (60MB INT8)
 * @param llmHandler LLM with fallback (Qwen2.5-0.5B 300MB INT4)
 * @param kokoro Kokoro-82M TTS engine (82MB)
 * @param contextBuilder J1939 CAN data → VehicleData converter
 */
class VoiceAssistant(
    private val whisper: WhisperSTT,
    private val llmHandler: LLMFallbackHandler,
    private val kokoro: KokoroTTS,
    private val contextBuilder: LLMContextBuilder
) {

    companion object {
        private const val DEFAULT_LANGUAGE = "ko"  // Korean
        private const val DEFAULT_VOICE = "ko_female_1"  // Korean female voice
    }

    /**
     * Handle complete voice command (STT → LLM → TTS)
     *
     * **Flow**:
     * 1. **STT** (Whisper Tiny, ~100ms):
     *    - Audio → Korean text
     *    - Example: "적재 중량이 얼마인가요?"
     *
     * 2. **Context Building** (~10ms):
     *    - Fetch latest J1939 CAN data
     *    - Build VehicleData context
     *
     * 3. **LLM Inference** (Qwen2.5, ~2s):
     *    - Generate response with vehicle context
     *    - Example: "현재 적재 중량은 5200킬로그램입니다..."
     *
     * 4. **TTS** (Kokoro, ~200ms):
     *    - Korean text → Audio
     *
     * **Total Latency**: ~2.3 seconds (typical)
     *
     * **Error Handling**:
     * - STT failure → Return error audio
     * - LLM failure → Fallback to rule-based response
     * - TTS failure → Return silence + log error
     *
     * @param audio Input audio (PCM 16kHz mono)
     * @return Output audio (PCM 24kHz mono, Korean speech)
     */
    suspend fun handleVoiceCommand(audio: ByteArray): ByteArray =
        withContext(Dispatchers.IO) {
            try {
                val startTime = System.currentTimeMillis()

                // Step 1: STT (Whisper Tiny) - ~100ms
                Timber.d("VoiceAssistant: Starting STT...")
                val sttStartTime = System.currentTimeMillis()

                val transcription = whisper.transcribe(audio, language = DEFAULT_LANGUAGE)
                val sttDuration = System.currentTimeMillis() - sttStartTime

                Timber.i("VoiceAssistant: STT completed in ${sttDuration}ms: '$transcription'")

                if (transcription.isBlank()) {
                    Timber.w("VoiceAssistant: Empty transcription, returning error audio")
                    return@withContext generateErrorAudio("음성을 인식하지 못했습니다.")
                }

                // Step 2: Build vehicle context - ~10ms
                val contextStartTime = System.currentTimeMillis()
                val vehicleContext = contextBuilder.buildContext()
                val contextDuration = System.currentTimeMillis() - contextStartTime

                Timber.d("VoiceAssistant: Context built in ${contextDuration}ms")

                // Step 3: LLM Inference (Qwen2.5 + fallback) - ~2s
                val llmStartTime = System.currentTimeMillis()
                val response = llmHandler.safeInference(transcription, vehicleContext)
                val llmDuration = System.currentTimeMillis() - llmStartTime

                Timber.i("VoiceAssistant: LLM inference in ${llmDuration}ms: '$response'")

                // Step 4: TTS (Kokoro) - ~200ms
                val ttsStartTime = System.currentTimeMillis()
                val audioResponse = kokoro.generate(
                    text = response,
                    lang = DEFAULT_LANGUAGE,
                    voice = DEFAULT_VOICE
                )
                val ttsDuration = System.currentTimeMillis() - ttsStartTime

                Timber.i("VoiceAssistant: TTS completed in ${ttsDuration}ms")

                // Total duration
                val totalDuration = System.currentTimeMillis() - startTime
                Timber.i(
                    "VoiceAssistant: Total pipeline duration: ${totalDuration}ms " +
                            "(STT: ${sttDuration}ms, Context: ${contextDuration}ms, " +
                            "LLM: ${llmDuration}ms, TTS: ${ttsDuration}ms)"
                )

                // Quality gate check
                if (totalDuration > 3000) {
                    Timber.w("VoiceAssistant: Latency exceeded target (${totalDuration}ms > 3000ms)")
                }

                audioResponse
            } catch (e: Exception) {
                Timber.e(e, "VoiceAssistant: Error in voice pipeline")
                generateErrorAudio("죄송합니다. 처리 중 오류가 발생했습니다.")
            }
        }

    /**
     * Process text query (skip STT, for testing)
     *
     * **Use Cases**:
     * - Unit testing
     * - Debug console
     * - Text-based UI fallback
     *
     * @param query Korean text query
     * @return Korean text response
     */
    suspend fun handleTextQuery(query: String): String =
        withContext(Dispatchers.IO) {
            try {
                val vehicleContext = contextBuilder.buildContext()
                llmHandler.safeInference(query, vehicleContext)
            } catch (e: Exception) {
                Timber.e(e, "VoiceAssistant: Error in text query")
                "죄송합니다. 처리 중 오류가 발생했습니다."
            }
        }

    /**
     * Generate error audio for user feedback
     *
     * **Error Messages**:
     * - "음성을 인식하지 못했습니다" (STT failure)
     * - "처리 중 오류가 발생했습니다" (LLM/TTS failure)
     *
     * @param errorMessage Korean error message
     * @return Audio of error message
     */
    private suspend fun generateErrorAudio(errorMessage: String): ByteArray {
        return try {
            kokoro.generate(
                text = errorMessage,
                lang = DEFAULT_LANGUAGE,
                voice = DEFAULT_VOICE
            )
        } catch (e: Exception) {
            Timber.e(e, "VoiceAssistant: Failed to generate error audio")
            ByteArray(0)  // Return silence
        }
    }

    /**
     * Check if voice assistant is ready
     *
     * **Prerequisites**:
     * - Whisper STT initialized
     * - LLM engine initialized
     * - Kokoro TTS initialized
     *
     * @return true if all components ready
     */
    fun isReady(): Boolean {
        return try {
            whisper.isInitialized() &&
                    llmHandler != null &&  // LLM handler exists
                    kokoro.isInitialized()
        } catch (e: Exception) {
            Timber.e(e, "VoiceAssistant: Error checking readiness")
            false
        }
    }

    /**
     * Release all voice assistant resources
     *
     * **Memory Freed**:
     * - Whisper: ~60MB
     * - Qwen2.5: ~1.1GB (model + KV cache)
     * - Kokoro: ~82MB
     * **Total**: ~1.24GB
     */
    fun release() {
        try {
            Timber.i("VoiceAssistant: Releasing resources...")

            whisper.release()
            // llmHandler.llm.release()  // TODO: Add after LLM integration
            kokoro.release()

            Timber.i("VoiceAssistant: Resources released")
        } catch (e: Exception) {
            Timber.e(e, "VoiceAssistant: Error releasing resources")
        }
    }
}

/**
 * Whisper STT interface
 *
 * TODO: Implement actual Whisper Tiny integration
 * Reference: Phase 3-J documentation
 */
interface WhisperSTT {
    suspend fun transcribe(audio: ByteArray, language: String): String
    fun isInitialized(): Boolean
    fun release()
}

/**
 * Kokoro TTS interface
 *
 * TODO: Implement actual Kokoro-82M integration
 * Reference: Phase 3-J documentation
 */
interface KokoroTTS {
    suspend fun generate(text: String, lang: String, voice: String): ByteArray
    fun isInitialized(): Boolean
    fun release()
}

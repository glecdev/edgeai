package com.glec.dtg.demo

import android.content.Context
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.*

/**
 * CES Demo Voice Bridge
 *
 * Simplified TTS integration for CES demo:
 * - Uses Android built-in TTS as fallback
 * - Optional: Integrates with Kokoro TTS (82MB) for high-quality voice
 * - Korean language support
 * - Background playback for alerts
 *
 * Usage:
 * ```kotlin
 * val voiceBridge = DemoVoiceBridge(context)
 * voiceBridge.speak("급가속이 감지되었습니다")
 * ```
 *
 * @see CES_DEMO_IMPLEMENTATION_COMPLETE.md - Voice AI Integration
 */
class DemoVoiceBridge(private val context: Context) {

    companion object {
        private const val TAG = "DemoVoiceBridge"
        private const val KOREAN_LOCALE = "ko_KR"
    }

    private var tts: TextToSpeech? = null
    private var isInitialized = false
    private val speechQueue = mutableListOf<String>()

    /**
     * Initialize TTS engine
     */
    fun initialize(onReady: () -> Unit = {}) {
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = tts?.setLanguage(Locale.KOREAN)

                if (result == TextToSpeech.LANG_MISSING_DATA ||
                    result == TextToSpeech.LANG_NOT_SUPPORTED
                ) {
                    Log.w(TAG, "Korean language not fully supported, using default")
                    tts?.setLanguage(Locale.getDefault())
                }

                // Configure TTS settings
                tts?.setSpeechRate(1.0f)  // Normal speed
                tts?.setPitch(1.0f)       // Normal pitch

                isInitialized = true
                Log.d(TAG, "TTS initialized successfully")

                // Process queued speech
                processQueue()

                onReady()
            } else {
                Log.e(TAG, "TTS initialization failed: $status")
            }
        }

        // Set utterance progress listener
        tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {
                Log.d(TAG, "TTS started: $utteranceId")
            }

            override fun onDone(utteranceId: String?) {
                Log.d(TAG, "TTS completed: $utteranceId")
            }

            @Deprecated("Deprecated in Java")
            override fun onError(utteranceId: String?) {
                Log.e(TAG, "TTS error: $utteranceId")
            }

            override fun onError(utteranceId: String?, errorCode: Int) {
                Log.e(TAG, "TTS error: $utteranceId, code: $errorCode")
            }
        })
    }

    /**
     * Speak text (queues if not initialized)
     */
    fun speak(text: String, priority: SpeechPriority = SpeechPriority.NORMAL) {
        if (!isInitialized) {
            Log.w(TAG, "TTS not initialized yet, queuing: $text")
            speechQueue.add(text)
            return
        }

        val queueMode = when (priority) {
            SpeechPriority.URGENT -> TextToSpeech.QUEUE_FLUSH  // Stop current and speak immediately
            SpeechPriority.NORMAL -> TextToSpeech.QUEUE_ADD    // Add to queue
        }

        val utteranceId = "demo_${System.currentTimeMillis()}"

        val result = tts?.speak(text, queueMode, null, utteranceId)

        if (result == TextToSpeech.SUCCESS) {
            Log.d(TAG, "Speaking: $text")
        } else {
            Log.e(TAG, "Failed to speak: $text")
        }
    }

    /**
     * Process queued speech
     */
    private fun processQueue() {
        if (speechQueue.isNotEmpty()) {
            Log.d(TAG, "Processing ${speechQueue.size} queued speech items")

            speechQueue.forEach { text ->
                speak(text)
            }

            speechQueue.clear()
        }
    }

    /**
     * Stop current speech
     */
    fun stop() {
        tts?.stop()
        Log.d(TAG, "TTS stopped")
    }

    /**
     * Check if TTS is speaking
     */
    fun isSpeaking(): Boolean {
        return tts?.isSpeaking ?: false
    }

    /**
     * Cleanup resources
     */
    fun shutdown() {
        stop()
        tts?.shutdown()
        tts = null
        isInitialized = false
        Log.d(TAG, "TTS shut down")
    }

    enum class SpeechPriority {
        URGENT,   // Interrupt current speech
        NORMAL    // Add to queue
    }
}

/**
 * Enhanced Demo Voice Bridge with Kokoro TTS (Optional)
 *
 * Use this for production-grade voice quality:
 * - Kokoro-82M TTS (82MB model)
 * - Natural Korean pronunciation
 * - Lower latency (~200ms vs ~500ms Android TTS)
 *
 * Enable by setting: USE_KOKORO = true
 */
class DemoVoiceBridgeEnhanced(
    private val context: Context,
    private val useKokoro: Boolean = false  // Set to true when Kokoro is integrated
) {

    private val fallbackTts = DemoVoiceBridge(context)
    private var kokoroTts: Any? = null  // TODO: Replace with KokoroTTS instance

    fun initialize(onReady: () -> Unit = {}) {
        if (useKokoro) {
            // TODO: Initialize Kokoro TTS
            // kokoroTts = KokoroTTS(context)
            // kokoroTts?.initialize { onReady() }
            Log.d("DemoVoiceBridgeEnhanced", "Kokoro TTS not yet integrated, using fallback")
            fallbackTts.initialize(onReady)
        } else {
            fallbackTts.initialize(onReady)
        }
    }

    fun speak(text: String, priority: DemoVoiceBridge.SpeechPriority = DemoVoiceBridge.SpeechPriority.NORMAL) {
        if (useKokoro && kokoroTts != null) {
            // TODO: Use Kokoro TTS
            // kokoroTts?.speak(text)
            fallbackTts.speak(text, priority)
        } else {
            fallbackTts.speak(text, priority)
        }
    }

    fun stop() {
        fallbackTts.stop()
        // kokoroTts?.stop()
    }

    fun shutdown() {
        fallbackTts.shutdown()
        // kokoroTts?.shutdown()
    }
}

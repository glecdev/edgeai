package com.glec.dtg.voice

import kotlinx.coroutines.delay
import timber.log.Timber

/**
 * Mock Whisper STT Implementation
 *
 * For testing voice pipeline without actual Whisper model.
 * Returns predefined Korean transcriptions based on audio input hash.
 */
class MockWhisperSTT : WhisperSTT {
    private var initialized = true

    override suspend fun transcribe(audio: ByteArray, language: String): String {
        // Simulate STT processing time (~100ms)
        delay(100)

        // Mock transcriptions (rotate based on audio hash)
        val mockTranscriptions = listOf(
            "적재 중량이 얼마인가요",
            "타이어 공기압 확인해줘",
            "엔진 온도 어때",
            "연비가 어떻게 돼",
            "차량 상태 알려줘"
        )

        val index = (audio.sum() % mockTranscriptions.size).toInt()
        val transcription = mockTranscriptions[index]

        Timber.d("MockWhisper: Transcribed -> '$transcription'")
        return transcription
    }

    override fun isInitialized(): Boolean = initialized

    override fun release() {
        initialized = false
        Timber.d("MockWhisper: Released")
    }
}

/**
 * Mock Kokoro TTS Implementation
 *
 * For testing voice pipeline without actual Kokoro model.
 * Returns dummy audio data (silence).
 */
class MockKokoroTTS : KokoroTTS {
    private var initialized = true

    override suspend fun generate(text: String, lang: String, voice: String): ByteArray {
        // Simulate TTS processing time (~200ms)
        delay(200)

        Timber.d("MockKokoro: Generated audio for text: '$text'")

        // Return dummy audio (16-bit PCM, 24kHz, 1 second)
        val sampleRate = 24000
        val durationSeconds = 1
        val audioData = ByteArray(sampleRate * 2 * durationSeconds)  // 2 bytes per sample

        return audioData
    }

    override fun isInitialized(): Boolean = initialized

    override fun release() {
        initialized = false
        Timber.d("MockKokoro: Released")
    }
}

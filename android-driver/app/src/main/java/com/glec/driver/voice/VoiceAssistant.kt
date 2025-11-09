package com.glec.driver.voice

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.speech.tts.TextToSpeech
import android.util.Log
import ai.picovoice.porcupine.Porcupine
import ai.picovoice.porcupine.PorcupineException
import org.vosk.Model
import org.vosk.Recognizer
import org.vosk.android.RecognitionListener
import org.vosk.android.SpeechService
import org.vosk.android.StorageService
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.util.*

/**
 * GLEC Driver - Voice Assistant
 *
 * Workflow:
 * 1. Wait for wake word "헤이 드라이버" (Porcupine)
 * 2. Respond "네, 말씀하세요" (TTS)
 * 3. Listen for voice command (Vosk STT)
 * 4. Parse intent and execute action
 * 5. Confirm with TTS
 *
 * Supported commands:
 * - "배차 수락" → ACCEPT_DISPATCH
 * - "배차 거부" → REJECT_DISPATCH
 * - "긴급 상황" → EMERGENCY_ALERT
 * - "현재 위치" → SHOW_LOCATION
 * - "연료 정보" → SHOW_FUEL_INFO
 * - "안전 점수" → SHOW_SAFETY_SCORE
 */
class VoiceAssistant(private val context: Context) {

    private var porcupine: Porcupine? = null
    private var voskModel: Model? = null
    private var speechService: SpeechService? = null
    private var tts: TextToSpeech? = null

    private var audioRecord: AudioRecord? = null
    private var isListeningForWakeWord = false
    private var isListeningForCommand = false

    private val audioSource = MediaRecorder.AudioSource.MIC
    private val sampleRate = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT

    var commandCallback: CommandCallback? = null

    /**
     * Initialize voice assistant
     */
    fun initialize() {
        Log.i(TAG, "Initializing Voice Assistant...")

        // Initialize Porcupine wake word detection
        initializePorcupine()

        // Initialize Vosk speech recognition
        initializeVosk()

        // Initialize Text-to-Speech
        initializeTTS()

        Log.i(TAG, "Voice Assistant initialized")
    }

    /**
     * Initialize Porcupine wake word engine
     */
    private fun initializePorcupine() {
        try {
            // Load custom wake word model from assets
            val wakeWordPath = File(context.filesDir, "wake_word.ppn").absolutePath

            // Copy from assets if not exists
            if (!File(wakeWordPath).exists()) {
                context.assets.open("wake_word.ppn").use { input ->
                    File(wakeWordPath).outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
            }

            // Create Porcupine instance
            porcupine = Porcupine.Builder()
                .setAccessKey(PORCUPINE_ACCESS_KEY)  // TODO: Add your Picovoice access key
                .setKeywordPath(wakeWordPath)
                .setSensitivity(0.7f)  // 0.0 to 1.0 (higher = more sensitive)
                .build(context)

            Log.i(TAG, "Porcupine wake word engine initialized")
        } catch (e: PorcupineException) {
            Log.e(TAG, "Failed to initialize Porcupine", e)
        } catch (e: IOException) {
            Log.e(TAG, "Failed to load wake word model", e)
        }
    }

    /**
     * Initialize Vosk speech recognition
     */
    private fun initializeVosk() {
        try {
            // Unpack Vosk model from assets
            StorageService.unpack(context, "vosk-model-ko", "model",
                { model ->
                    voskModel = model
                    Log.i(TAG, "Vosk model loaded successfully")
                },
                { exception ->
                    Log.e(TAG, "Failed to unpack Vosk model", exception)
                })
        } catch (e: IOException) {
            Log.e(TAG, "Failed to initialize Vosk", e)
        }
    }

    /**
     * Initialize Text-to-Speech
     */
    private fun initializeTTS() {
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = tts?.setLanguage(Locale.KOREAN)
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.e(TAG, "Korean language not supported by TTS")
                } else {
                    Log.i(TAG, "TTS initialized with Korean language")
                }
            } else {
                Log.e(TAG, "TTS initialization failed")
            }
        }
    }

    /**
     * Start listening for wake word
     */
    fun startListeningForWakeWord() {
        if (porcupine == null) {
            Log.e(TAG, "Porcupine not initialized")
            return
        }

        if (isListeningForWakeWord) {
            Log.w(TAG, "Already listening for wake word")
            return
        }

        Log.i(TAG, "Starting wake word detection...")

        isListeningForWakeWord = true

        Thread {
            val bufferSize = porcupine!!.frameLength
            val buffer = ShortArray(bufferSize)

            audioRecord = AudioRecord(
                audioSource,
                porcupine!!.sampleRate,
                channelConfig,
                audioFormat,
                bufferSize * 2
            )

            audioRecord?.startRecording()

            while (isListeningForWakeWord) {
                val numRead = audioRecord?.read(buffer, 0, bufferSize) ?: 0

                if (numRead > 0) {
                    try {
                        val keywordIndex = porcupine?.process(buffer)
                        if (keywordIndex != null && keywordIndex >= 0) {
                            Log.i(TAG, "Wake word detected!")
                            handleWakeWordDetected()
                        }
                    } catch (e: PorcupineException) {
                        Log.e(TAG, "Error processing audio", e)
                    }
                }
            }

            audioRecord?.stop()
            audioRecord?.release()
            audioRecord = null
        }.start()

        Log.i(TAG, "Wake word detection started")
    }

    /**
     * Stop listening for wake word
     */
    fun stopListeningForWakeWord() {
        isListeningForWakeWord = false
        Log.i(TAG, "Wake word detection stopped")
    }

    /**
     * Handle wake word detected
     */
    private fun handleWakeWordDetected() {
        stopListeningForWakeWord()

        // Respond with TTS
        speak("네, 말씀하세요") {
            // Start listening for voice command after TTS completes
            startListeningForCommand()
        }
    }

    /**
     * Start listening for voice command
     */
    private fun startListeningForCommand() {
        if (voskModel == null) {
            Log.e(TAG, "Vosk model not loaded")
            return
        }

        if (isListeningForCommand) {
            Log.w(TAG, "Already listening for command")
            return
        }

        Log.i(TAG, "Starting command recognition...")

        isListeningForCommand = true

        try {
            val recognizer = Recognizer(voskModel, sampleRate.toFloat())

            speechService = SpeechService(recognizer, sampleRate.toFloat())
            speechService?.startListening(object : RecognitionListener {
                override fun onResult(hypothesis: String) {
                    Log.d(TAG, "Recognition result: $hypothesis")
                    handleVoiceCommandResult(hypothesis)
                }

                override fun onPartialResult(hypothesis: String) {
                    Log.d(TAG, "Partial result: $hypothesis")
                }

                override fun onError(exception: Exception) {
                    Log.e(TAG, "Recognition error", exception)
                    stopListeningForCommand()
                    startListeningForWakeWord()
                }

                override fun onTimeout() {
                    Log.w(TAG, "Recognition timeout")
                    stopListeningForCommand()
                    speak("명령을 인식하지 못했습니다") {
                        startListeningForWakeWord()
                    }
                }
            })

            // Auto-stop after 5 seconds
            android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                if (isListeningForCommand) {
                    stopListeningForCommand()
                    startListeningForWakeWord()
                }
            }, 5000)

        } catch (e: IOException) {
            Log.e(TAG, "Failed to start command recognition", e)
            isListeningForCommand = false
        }

        Log.i(TAG, "Command recognition started")
    }

    /**
     * Stop listening for command
     */
    private fun stopListeningForCommand() {
        speechService?.stop()
        speechService?.shutdown()
        speechService = null
        isListeningForCommand = false
        Log.i(TAG, "Command recognition stopped")
    }

    /**
     * Handle voice command result
     */
    private fun handleVoiceCommandResult(hypothesis: String) {
        stopListeningForCommand()

        try {
            val json = JSONObject(hypothesis)
            val text = json.optString("text", "").lowercase()

            Log.i(TAG, "Recognized text: $text")

            val intent = parseIntent(text)
            if (intent != null) {
                executeCommand(intent)
            } else {
                speak("명령을 이해하지 못했습니다") {
                    startListeningForWakeWord()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing recognition result", e)
            startListeningForWakeWord()
        }
    }

    /**
     * Parse intent from recognized text
     */
    private fun parseIntent(text: String): VoiceIntent? {
        return when {
            text.contains("배차") && text.contains("수락") -> VoiceIntent.ACCEPT_DISPATCH
            text.contains("배차") && text.contains("거부") -> VoiceIntent.REJECT_DISPATCH
            text.contains("긴급") -> VoiceIntent.EMERGENCY_ALERT
            text.contains("위치") -> VoiceIntent.SHOW_LOCATION
            text.contains("연료") -> VoiceIntent.SHOW_FUEL_INFO
            text.contains("안전") && text.contains("점수") -> VoiceIntent.SHOW_SAFETY_SCORE
            text.contains("출발") -> VoiceIntent.START_NAVIGATION
            text.contains("도착") -> VoiceIntent.ARRIVE_DESTINATION
            else -> null
        }
    }

    /**
     * Execute voice command
     */
    private fun executeCommand(intent: VoiceIntent) {
        Log.i(TAG, "Executing command: $intent")

        val responseMessage = when (intent) {
            VoiceIntent.ACCEPT_DISPATCH -> "배차를 수락했습니다"
            VoiceIntent.REJECT_DISPATCH -> "배차를 거부했습니다"
            VoiceIntent.EMERGENCY_ALERT -> "긴급 상황을 알렸습니다"
            VoiceIntent.SHOW_LOCATION -> "현재 위치를 표시합니다"
            VoiceIntent.SHOW_FUEL_INFO -> "연료 정보를 표시합니다"
            VoiceIntent.SHOW_SAFETY_SCORE -> "안전 점수를 표시합니다"
            VoiceIntent.START_NAVIGATION -> "내비게이션을 시작합니다"
            VoiceIntent.ARRIVE_DESTINATION -> "도착 처리했습니다"
        }

        // Execute command callback
        commandCallback?.onCommandReceived(intent)

        // Confirm with TTS
        speak(responseMessage) {
            startListeningForWakeWord()
        }
    }

    /**
     * Speak text using TTS
     */
    private fun speak(text: String, onComplete: (() -> Unit)? = null) {
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "utteranceId")

        if (onComplete != null) {
            tts?.setOnUtteranceProgressListener(object : android.speech.tts.UtteranceProgressListener() {
                override fun onStart(utteranceId: String) {}

                override fun onDone(utteranceId: String) {
                    onComplete()
                }

                override fun onError(utteranceId: String) {
                    onComplete()
                }
            })
        }
    }

    /**
     * Shutdown voice assistant
     */
    fun shutdown() {
        Log.i(TAG, "Shutting down Voice Assistant...")

        stopListeningForWakeWord()
        stopListeningForCommand()

        porcupine?.delete()
        porcupine = null

        tts?.stop()
        tts?.shutdown()
        tts = null

        Log.i(TAG, "Voice Assistant shut down")
    }

    /**
     * Voice intents
     */
    enum class VoiceIntent {
        ACCEPT_DISPATCH,
        REJECT_DISPATCH,
        EMERGENCY_ALERT,
        SHOW_LOCATION,
        SHOW_FUEL_INFO,
        SHOW_SAFETY_SCORE,
        START_NAVIGATION,
        ARRIVE_DESTINATION
    }

    /**
     * Command callback interface
     */
    interface CommandCallback {
        fun onCommandReceived(intent: VoiceIntent)
    }

    companion object {
        private const val TAG = "VoiceAssistant"

        // TODO: Add your Picovoice access key
        // Get free key from: https://console.picovoice.ai/
        private const val PORCUPINE_ACCESS_KEY = "YOUR_PICOVOICE_ACCESS_KEY"
    }
}

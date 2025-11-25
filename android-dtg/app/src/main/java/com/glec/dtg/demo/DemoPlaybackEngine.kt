package com.glec.dtg.demo

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.webkit.WebView
import com.google.gson.Gson
import com.google.gson.JsonObject
import com.google.gson.annotations.SerializedName
import java.io.InputStreamReader
import kotlin.math.abs

/**
 * CES Demo Playback Engine
 *
 * Plays pre-recorded demo scenarios for CES exhibition:
 * - Reads JSON timeline from assets
 * - Interpolates vehicle data between keyframes
 * - Triggers events and alerts at precise timestamps
 * - Updates 3D viewer and dashboard in real-time
 * - Loops automatically for continuous booth display
 *
 * @see CES_DEMO_VIEWER_OPTIMIZED.md
 */
class DemoPlaybackEngine(
    private val context: Context,
    private val alertManager: AlertManager,
    private val webView: WebView
) {

    companion object {
        private const val TAG = "DemoPlaybackEngine"
        private const val PLAYBACK_FPS = 30  // 30 frames per second
        private const val FRAME_INTERVAL_MS = 1000L / PLAYBACK_FPS
    }

    data class DemoTimeline(
        val demoId: String,
        val title: String,
        val description: String,
        val duration: Int,
        val timeline: List<TimelineEvent>
    )

    data class TimelineEvent(
        val timestamp: Int,
        val comment: String? = null,
        val vehicleSpeed: Float = 0f,
        val engineRPM: Int = 0,
        val accelerationX: Float = 0f,
        val safetyScore: Int = 100,
        val fuelLevel: Float = 100f,
        val fuelConsumption: Float? = null,
        val tirePressureFrontLeft: Float? = null,
        val vehicleColor: String = "#00FF00",
        val event: Event? = null
    )

    data class Event(
        val type: String,
        val alert: AlertData? = null,
        val voiceConversation: VoiceConversation? = null,
        @SerializedName("3dViewer")
        val viewer3d: Viewer3D? = null,
        val action: String? = null
    )

    data class AlertData(
        val id: String,
        val title: String,
        val message: String,
        val voiceAlert: String? = null,
        val severity: String,
        val autoDismissSeconds: Int = 5,
        val icon: String = "warning"
    )

    data class VoiceConversation(
        val query: String,
        val response: String
    )

    data class Viewer3D(
        val highlightPart: String? = null,
        val color: String? = null,
        val animation: String? = null,
        val cameraAngle: String? = null,
        val duration: Int? = null
    )

    // Playback state
    private var isPlaying = false
    private var isPaused = false
    private var currentTime = 0L  // Elapsed time in milliseconds
    private var demoTimeline: DemoTimeline? = null
    private var lastEventIndex = -1

    // Handlers
    private val handler = Handler(Looper.getMainLooper())
    private val playbackRunnable = object : Runnable {
        override fun run() {
            if (isPlaying && !isPaused) {
                updatePlayback()
                handler.postDelayed(this, FRAME_INTERVAL_MS)
            }
        }
    }

    /**
     * Load demo timeline from assets
     */
    fun loadDemo(assetPath: String): Boolean {
        return try {
            val inputStream = context.assets.open(assetPath)
            val reader = InputStreamReader(inputStream)
            demoTimeline = Gson().fromJson(reader, DemoTimeline::class.java)
            reader.close()

            Log.d(TAG, "Loaded demo: ${demoTimeline?.title}")
            Log.d(TAG, "Duration: ${demoTimeline?.duration}s, Events: ${demoTimeline?.timeline?.size}")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load demo: ${e.message}", e)
            false
        }
    }

    /**
     * Start demo playback
     */
    fun start() {
        if (demoTimeline == null) {
            Log.e(TAG, "No demo loaded!")
            return
        }

        isPlaying = true
        isPaused = false
        currentTime = 0L
        lastEventIndex = -1

        Log.d(TAG, "Starting demo playback...")
        handler.post(playbackRunnable)
    }

    /**
     * Pause demo playback
     */
    fun pause() {
        isPaused = true
        Log.d(TAG, "Demo paused at ${currentTime / 1000}s")
    }

    /**
     * Resume demo playback
     */
    fun resume() {
        if (isPlaying) {
            isPaused = false
            Log.d(TAG, "Demo resumed at ${currentTime / 1000}s")
            handler.post(playbackRunnable)
        }
    }

    /**
     * Stop demo playback
     */
    fun stop() {
        isPlaying = false
        isPaused = false
        currentTime = 0L
        lastEventIndex = -1

        handler.removeCallbacks(playbackRunnable)
        alertManager.dismissAllAlerts()

        Log.d(TAG, "Demo stopped")
    }

    /**
     * Update playback - called every frame
     */
    private fun updatePlayback() {
        val timeline = demoTimeline ?: return
        val currentSeconds = (currentTime / 1000).toInt()

        // Check if demo finished -> loop restart
        if (currentSeconds >= timeline.duration) {
            Log.d(TAG, "Demo finished, restarting...")
            currentTime = 0L
            lastEventIndex = -1
            alertManager.dismissAllAlerts()
            return
        }

        // Get current and next keyframes
        val currentKeyframe = findKeyframe(currentSeconds)
        val nextKeyframe = findNextKeyframe(currentSeconds)

        if (currentKeyframe != null && nextKeyframe != null) {
            // Interpolate vehicle data
            val interpolatedData = interpolateData(
                currentKeyframe,
                nextKeyframe,
                currentSeconds
            )

            // Update dashboard and 3D viewer
            updateViewer(interpolatedData)
        }

        // Check for events at current timestamp
        checkAndTriggerEvents(currentSeconds)

        // Advance time
        currentTime += FRAME_INTERVAL_MS
    }

    /**
     * Find keyframe at or before timestamp
     */
    private fun findKeyframe(seconds: Int): TimelineEvent? {
        val timeline = demoTimeline?.timeline ?: return null

        var result: TimelineEvent? = null
        for (event in timeline) {
            if (event.timestamp <= seconds) {
                result = event
            } else {
                break
            }
        }
        return result
    }

    /**
     * Find next keyframe after timestamp
     */
    private fun findNextKeyframe(seconds: Int): TimelineEvent? {
        val timeline = demoTimeline?.timeline ?: return null

        for (event in timeline) {
            if (event.timestamp > seconds) {
                return event
            }
        }
        return null
    }

    /**
     * Interpolate vehicle data between two keyframes
     */
    private fun interpolateData(
        current: TimelineEvent,
        next: TimelineEvent,
        currentSeconds: Int
    ): TimelineEvent {
        val timeDiff = next.timestamp - current.timestamp
        if (timeDiff == 0) return current

        val progress = (currentSeconds - current.timestamp).toFloat() / timeDiff

        return TimelineEvent(
            timestamp = currentSeconds,
            vehicleSpeed = lerp(current.vehicleSpeed, next.vehicleSpeed, progress),
            engineRPM = lerp(current.engineRPM.toFloat(), next.engineRPM.toFloat(), progress).toInt(),
            accelerationX = lerp(current.accelerationX, next.accelerationX, progress),
            safetyScore = lerp(current.safetyScore.toFloat(), next.safetyScore.toFloat(), progress).toInt(),
            fuelLevel = lerp(current.fuelLevel, next.fuelLevel, progress),
            vehicleColor = if (progress < 0.5f) current.vehicleColor else next.vehicleColor
        )
    }

    /**
     * Linear interpolation
     */
    private fun lerp(start: Float, end: Float, progress: Float): Float {
        return start + (end - start) * progress.coerceIn(0f, 1f)
    }

    /**
     * Update WebView dashboard and 3D viewer
     */
    private fun updateViewer(data: TimelineEvent) {
        val json = JsonObject().apply {
            addProperty("type", "update_vehicle_data")
            addProperty("vehicleSpeed", data.vehicleSpeed)
            addProperty("engineRPM", data.engineRPM)
            addProperty("accelerationX", data.accelerationX)
            addProperty("safetyScore", data.safetyScore)
            addProperty("fuelLevel", data.fuelLevel)
            addProperty("vehicleColor", data.vehicleColor)
        }

        val jsCode = "window.updateVehicleData && window.updateVehicleData(${json})"

        handler.post {
            webView.evaluateJavascript(jsCode, null)
        }
    }

    /**
     * Check and trigger events at current timestamp
     */
    private fun checkAndTriggerEvents(currentSeconds: Int) {
        val timeline = demoTimeline?.timeline ?: return

        // Find events at current timestamp
        for ((index, timelineEvent) in timeline.withIndex()) {
            if (timelineEvent.timestamp == currentSeconds && index > lastEventIndex) {
                lastEventIndex = index

                timelineEvent.event?.let { event ->
                    triggerEvent(event, timelineEvent)
                }
            }
        }
    }

    /**
     * Trigger event (alert, voice, 3D animation)
     */
    private fun triggerEvent(event: Event, timelineEvent: TimelineEvent) {
        Log.d(TAG, "Triggering event: ${event.type} at ${timelineEvent.timestamp}s")

        // Show alert
        event.alert?.let { alertData ->
            val alert = AlertManager.Alert(
                id = alertData.id,
                type = mapAlertType(event.type),
                title = alertData.title,
                message = alertData.message,
                severity = mapSeverity(alertData.severity),
                autoDismissSeconds = alertData.autoDismissSeconds,
                voiceAlert = alertData.voiceAlert,
                icon = mapIcon(alertData.icon)
            )

            alertManager.showAlert(alert)

            // Trigger voice alert if specified
            alertData.voiceAlert?.let { voiceText ->
                triggerVoiceAlert(voiceText)
            }
        }

        // Voice conversation
        event.voiceConversation?.let { conversation ->
            Log.d(TAG, "Voice conversation: ${conversation.query} -> ${conversation.response}")
            // TODO: Integrate with TTS system
        }

        // 3D viewer animation
        event.viewer3d?.let { viewer ->
            trigger3DAnimation(viewer)
        }

        // Handle special actions
        when (event.action) {
            "loop_restart" -> {
                Log.d(TAG, "Demo end - will loop")
            }
        }
    }

    /**
     * Trigger voice alert via TTS
     */
    private fun triggerVoiceAlert(text: String) {
        val jsCode = "window.speakText && window.speakText('${text.replace("'", "\\'")}')"

        handler.post {
            webView.evaluateJavascript(jsCode, null)
        }
    }

    /**
     * Trigger 3D viewer animation
     */
    private fun trigger3DAnimation(viewer: Viewer3D) {
        val json = JsonObject().apply {
            addProperty("type", "animate_3d")
            viewer.highlightPart?.let { addProperty("highlightPart", it) }
            viewer.color?.let { addProperty("color", it) }
            viewer.animation?.let { addProperty("animation", it) }
            viewer.cameraAngle?.let { addProperty("cameraAngle", it) }
            viewer.duration?.let { addProperty("duration", it) }
        }

        val jsCode = "window.animate3DViewer && window.animate3DViewer(${json})"

        handler.post {
            webView.evaluateJavascript(jsCode, null)
        }
    }

    /**
     * Map event type to AlertType
     */
    private fun mapAlertType(type: String): AlertManager.AlertType {
        return when (type) {
            "harsh_acceleration" -> AlertManager.AlertType.HARSH_ACCELERATION
            "harsh_braking" -> AlertManager.AlertType.HARSH_BRAKING
            "speeding" -> AlertManager.AlertType.SPEEDING
            "drowsy_driving" -> AlertManager.AlertType.DROWSY_DRIVING
            "fuel_efficiency" -> AlertManager.AlertType.FUEL_EFFICIENCY
            "tire_pressure" -> AlertManager.AlertType.TIRE_PRESSURE
            "rest_area_guidance" -> AlertManager.AlertType.MANDATORY_REST
            "route_optimization" -> AlertManager.AlertType.ROUTE_OPTIMIZATION
            "oil_change" -> AlertManager.AlertType.OIL_CHANGE
            "fatigue_detection" -> AlertManager.AlertType.FATIGUE_DETECTION
            else -> AlertManager.AlertType.WARNING
        }
    }

    /**
     * Map severity string to Severity enum
     */
    private fun mapSeverity(severity: String): AlertManager.Severity {
        return when (severity.lowercase()) {
            "low" -> AlertManager.Severity.LOW
            "medium" -> AlertManager.Severity.MEDIUM
            "high" -> AlertManager.Severity.HIGH
            "critical" -> AlertManager.Severity.CRITICAL
            else -> AlertManager.Severity.MEDIUM
        }
    }

    /**
     * Map icon string to AlertIcon enum
     */
    private fun mapIcon(icon: String): AlertManager.AlertIcon {
        return when (icon.lowercase()) {
            "warning" -> AlertManager.AlertIcon.WARNING
            "danger" -> AlertManager.AlertIcon.DANGER
            "info" -> AlertManager.AlertIcon.INFO
            "success" -> AlertManager.AlertIcon.SUCCESS
            "maintenance" -> AlertManager.AlertIcon.MAINTENANCE
            else -> AlertManager.AlertIcon.WARNING
        }
    }

    /**
     * Get current playback progress (0.0 to 1.0)
     */
    fun getProgress(): Float {
        val duration = demoTimeline?.duration ?: return 0f
        return (currentTime / 1000f) / duration
    }

    /**
     * Cleanup resources
     */
    fun cleanup() {
        stop()
        demoTimeline = null
    }
}

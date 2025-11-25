package com.glec.dtg.ui

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import kotlin.math.roundToInt

/**
 * Demo Data Source for CES 150-second looping demo
 * Loads timeline from demo-ultimate-150s.json and provides interpolated data
 */
class DemoDataSource(private val context: Context) {

    data class DemoData(
        val vehicleSpeed: Int,
        val engineRPM: Int,
        val fuelEfficiency: Float,
        val drivingStatus: String,
        val confidence: Float,
        val safetyScore: Int,
        val accelerationX: Float
    )

    private var timeline: List<TimelineEvent> = emptyList()
    private val demoDurationSeconds = 150
    private var demoStartTime: Long = 0
    private var isLoaded = false

    data class TimelineEvent(
        val timestamp: Int,
        val vehicleSpeed: Int,
        val engineRPM: Int,
        val accelerationX: Float,
        val safetyScore: Int,
        val fuelLevel: Int
    )

    /**
     * Load demo timeline from assets
     */
    suspend fun loadDemo(): Boolean = withContext(Dispatchers.IO) {
        try {
            val jsonString = context.assets.open("demo-ultimate-150s.json").bufferedReader().use { it.readText() }
            val jsonObject = JSONObject(jsonString)
            val timelineArray = jsonObject.getJSONArray("timeline")

            val events = mutableListOf<TimelineEvent>()
            for (i in 0 until timelineArray.length()) {
                val event = timelineArray.getJSONObject(i)
                events.add(
                    TimelineEvent(
                        timestamp = event.getInt("timestamp"),
                        vehicleSpeed = event.getInt("vehicleSpeed"),
                        engineRPM = event.getInt("engineRPM"),
                        accelerationX = event.getDouble("accelerationX").toFloat(),
                        safetyScore = event.getInt("safetyScore"),
                        fuelLevel = event.optInt("fuelLevel", 85) // Default to 85% if missing
                    )
                )
            }

            timeline = events.sortedBy { it.timestamp }
            demoStartTime = System.currentTimeMillis()
            isLoaded = true
            android.util.Log.d("DemoDataSource", "Timeline loaded successfully: ${timeline.size} events")
            true
        } catch (e: Exception) {
            e.printStackTrace()
            android.util.Log.e("DemoDataSource", "Failed to load demo timeline", e)
            false
        }
    }

    /**
     * Get current demo data with interpolation
     */
    fun getCurrentData(): DemoData {
        if (!isLoaded || timeline.isEmpty()) {
            return getDefaultData()
        }

        // Calculate current position in 150-second loop
        val elapsedMs = System.currentTimeMillis() - demoStartTime
        val currentSecond = ((elapsedMs / 1000) % demoDurationSeconds).toInt()

        // Find surrounding events for interpolation
        val currentEvent = timeline.lastOrNull { it.timestamp <= currentSecond }
        val nextEvent = timeline.firstOrNull { it.timestamp > currentSecond }

        return if (currentEvent != null && nextEvent != null) {
            interpolateData(currentEvent, nextEvent, currentSecond)
        } else if (currentEvent != null) {
            // At end of timeline, interpolate to first event
            val firstEvent = timeline.first()
            interpolateData(currentEvent, firstEvent, currentSecond)
        } else {
            getDefaultData()
        }
    }

    /**
     * Linear interpolation between two timeline events
     */
    private fun interpolateData(
        current: TimelineEvent,
        next: TimelineEvent,
        currentSecond: Int
    ): DemoData {
        val timeDiff = if (next.timestamp > current.timestamp) {
            next.timestamp - current.timestamp
        } else {
            // Wrap around case (end → start)
            (demoDurationSeconds - current.timestamp) + next.timestamp
        }

        if (timeDiff == 0) return eventToDemoData(current)

        val progress = (currentSecond - current.timestamp).toFloat() / timeDiff

        val vehicleSpeed = lerp(current.vehicleSpeed.toFloat(), next.vehicleSpeed.toFloat(), progress).roundToInt()
        val engineRPM = lerp(current.engineRPM.toFloat(), next.engineRPM.toFloat(), progress).roundToInt()
        val accelerationX = lerp(current.accelerationX, next.accelerationX, progress)
        val safetyScore = lerp(current.safetyScore.toFloat(), next.safetyScore.toFloat(), progress).roundToInt()

        // Calculate fuel efficiency (simple approximation based on speed)
        val fuelEfficiency = if (vehicleSpeed > 0) {
            (5.0f + (vehicleSpeed / 15.0f)).coerceIn(5.0f, 10.0f)
        } else {
            0.0f
        }

        // Determine driving status based on acceleration and safety score
        val drivingStatus = when {
            accelerationX > 3.0f -> "Aggressive"
            accelerationX < -3.0f -> "Hard Braking"
            safetyScore < 80 -> "Caution"
            safetyScore >= 90 -> "Eco-Friendly"
            else -> "Normal"
        }

        val confidence = when {
            safetyScore >= 90 -> 92f + (safetyScore - 90) * 0.5f
            safetyScore >= 80 -> 85f + (safetyScore - 80) * 0.7f
            else -> 75f + safetyScore * 0.1f
        }.coerceIn(75f, 99f)

        return DemoData(
            vehicleSpeed = vehicleSpeed,
            engineRPM = engineRPM,
            fuelEfficiency = fuelEfficiency,
            drivingStatus = drivingStatus,
            confidence = confidence,
            safetyScore = safetyScore,
            accelerationX = accelerationX
        )
    }

    /**
     * Convert timeline event to demo data
     */
    private fun eventToDemoData(event: TimelineEvent): DemoData {
        val fuelEfficiency = if (event.vehicleSpeed > 0) {
            (5.0f + (event.vehicleSpeed / 15.0f)).coerceIn(5.0f, 10.0f)
        } else {
            0.0f
        }

        val drivingStatus = when {
            event.accelerationX > 3.0f -> "Aggressive"
            event.accelerationX < -3.0f -> "Hard Braking"
            event.safetyScore < 80 -> "Caution"
            event.safetyScore >= 90 -> "Eco-Friendly"
            else -> "Normal"
        }

        val confidence = when {
            event.safetyScore >= 90 -> 92f
            event.safetyScore >= 80 -> 85f
            else -> 78f
        }

        return DemoData(
            vehicleSpeed = event.vehicleSpeed,
            engineRPM = event.engineRPM,
            fuelEfficiency = fuelEfficiency,
            drivingStatus = drivingStatus,
            confidence = confidence,
            safetyScore = event.safetyScore,
            accelerationX = event.accelerationX
        )
    }

    /**
     * Get default data for fallback
     */
    private fun getDefaultData(): DemoData {
        return DemoData(
            vehicleSpeed = 80,
            engineRPM = 2000,
            fuelEfficiency = 7.5f,
            drivingStatus = "Normal",
            confidence = 92.0f,
            safetyScore = 95,
            accelerationX = 0.0f
        )
    }

    /**
     * Linear interpolation helper
     */
    private fun lerp(start: Float, end: Float, progress: Float): Float {
        return start + (end - start) * progress.coerceIn(0f, 1f)
    }

    /**
     * Get current demo progress (0-150 seconds)
     */
    fun getCurrentDemoSecond(): Int {
        if (!isLoaded) return 0
        val elapsedMs = System.currentTimeMillis() - demoStartTime
        return ((elapsedMs / 1000) % demoDurationSeconds).toInt()
    }
}

package com.glec.dtg.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import com.glec.dtg.MainActivity
import com.glec.dtg.R
import com.glec.dtg.inference.SNPEEngine
import com.glec.dtg.models.CANData
import com.glec.dtg.models.AIInferenceResult
import com.glec.dtg.models.DrivingBehavior
import kotlinx.coroutines.*
import org.eclipse.paho.client.mqttv3.MqttClient
import org.eclipse.paho.client.mqttv3.MqttConnectOptions
import timber.log.Timber
import java.util.concurrent.ConcurrentLinkedQueue

/**
 * GLEC DTG Foreground Service
 *
 * Responsibilities:
 * - Read vehicle CAN data via UART (1Hz)
 * - Collect 60 samples (60 seconds)
 * - Run AI inference every 60 seconds
 * - Send results to Fleet AI platform via MQTT
 * - BLE peripheral for driver app connection
 *
 * Performance Targets:
 * - AI Inference: < 50ms (parallel execution)
 * - Power Consumption: < 2W
 * - Memory Usage: < 500MB
 */
class DTGForegroundService : Service() {

    private val serviceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    // CAN data buffer (60 samples at 1Hz)
    private val canDataBuffer = ConcurrentLinkedQueue<CANData>()
    private val maxBufferSize = 60

    // AI Inference Engine
    private lateinit var snpeEngine: SNPEEngine

    // MQTT Client
    private var mqttClient: MqttClient? = null

    // Jobs
    private var canReaderJob: Job? = null
    private var inferenceJob: Job? = null
    private var mqttJob: Job? = null

    companion object {
        private const val NOTIFICATION_ID = 1
        private const val CHANNEL_ID = "DTGServiceChannel"
        private const val CHANNEL_NAME = "DTG Service"

        private const val CAN_SAMPLE_RATE_HZ = 1  // 1Hz
        private const val INFERENCE_INTERVAL_MS = 60_000L  // 60 seconds
    }

    override fun onCreate() {
        super.onCreate()
        Timber.i("DTGForegroundService onCreate")

        // Initialize SNPE engine (Android 13 Fix: graceful fallback)
        snpeEngine = SNPEEngine(this)
        try {
            snpeEngine.loadModels()
        } catch (e: Exception) {
            Timber.w(e, "Failed to load SNPE models - running in data collection mode")
            // Continue without AI models - service can still collect CAN data
        }

        // Create notification channel
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Timber.i("DTGForegroundService onStartCommand")

        // Start foreground service
        startForeground(NOTIFICATION_ID, createNotification())

        // Start CAN data collection
        startCANReader()

        // Start AI inference scheduler
        startInferenceScheduler()

        // Start MQTT connection
        startMQTTClient()

        return START_STICKY  // Restart service if killed
    }

    override fun onBind(intent: Intent?): IBinder? {
        return null
    }

    override fun onDestroy() {
        Timber.i("DTGForegroundService onDestroy")

        // Cancel all jobs
        canReaderJob?.cancel()
        inferenceJob?.cancel()
        mqttJob?.cancel()
        serviceScope.cancel()

        // Cleanup
        snpeEngine.release()
        mqttClient?.disconnect()

        super.onDestroy()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                CHANNEL_NAME,
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "DTG service is running"
                setShowBadge(false)
            }

            val notificationManager = getSystemService(NotificationManager::class.java)
            notificationManager.createNotificationChannel(channel)
        }
    }

    private fun createNotification(): Notification {
        val intent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("GLEC DTG Service")
            .setContentText("Monitoring vehicle data and running AI inference")
            .setSmallIcon(R.drawable.ic_notification)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }

    private fun startCANReader() {
        canReaderJob = serviceScope.launch(Dispatchers.IO) {
            Timber.i("Starting CAN reader (1Hz)")

            while (isActive) {
                try {
                    // Try to read CAN data via JNI (graceful fallback if hardware not available)
                    val canData = try {
                        readCANDataFromUART()
                    } catch (e: UnsatisfiedLinkError) {
                        // JNI library not available - use mock data for testing
                        Timber.w("UART JNI not available, using mock CAN data")
                        createMockCANData()
                    }

                    // Add to buffer
                    canDataBuffer.offer(canData)

                    // Limit buffer size
                    while (canDataBuffer.size > maxBufferSize) {
                        canDataBuffer.poll()
                    }

                    Timber.d("CAN data collected. Buffer size: ${canDataBuffer.size}")

                } catch (e: Exception) {
                    Timber.e(e, "Error reading CAN data")
                }

                // 1Hz sampling rate
                delay(1000L / CAN_SAMPLE_RATE_HZ)
            }
        }
    }

    private fun startInferenceScheduler() {
        inferenceJob = serviceScope.launch(Dispatchers.IO) {
            Timber.i("Starting AI inference scheduler (60-second interval)")

            // Wait for initial data collection
            delay(INFERENCE_INTERVAL_MS)

            while (isActive) {
                try {
                    if (canDataBuffer.size >= maxBufferSize) {
                        // Run AI inference
                        val results = runAIInference(canDataBuffer.toList())

                        Timber.i("AI Inference results: $results")

                        // Send to Fleet AI platform
                        sendToFleetPlatform(results)

                        // Broadcast to BLE peripheral
                        broadcastToBLE(results)
                    } else {
                        Timber.w("Not enough data for inference. Buffer size: ${canDataBuffer.size}")
                    }

                } catch (e: Exception) {
                    Timber.e(e, "Error during AI inference")
                }

                delay(INFERENCE_INTERVAL_MS)
            }
        }
    }

    // TODO: Implement runAIInference - stub for compilation
    private suspend fun runAIInference(canDataList: List<CANData>): AIInferenceResult {
        // Placeholder implementation
        return AIInferenceResult(
            timestamp = System.currentTimeMillis(),
            fuelEfficiencyPrediction = 0.0f,
            anomalyScore = 0.0f,
            behaviorClass = DrivingBehavior.NORMAL.toClassification(0.95f),
            safetyScore = 0,
            carbonEmission = 0.0f,
            anomalies = emptyList(),
            inferenceLatency = 0L
        )
    }

    private fun startMQTTClient() {
        mqttJob = serviceScope.launch(Dispatchers.IO) {
            try {
                // TODO: Implement MQTT connection
                Timber.i("MQTT client started")

                val options = MqttConnectOptions().apply {
                    isAutomaticReconnect = true
                    isCleanSession = false
                    connectionTimeout = 30
                    keepAliveInterval = 60
                }

                // mqttClient.connect(options)

            } catch (e: Exception) {
                Timber.e(e, "Error starting MQTT client")
            }
        }
    }

    // TODO: Implement sendToFleetPlatform - stub for compilation
    private suspend fun sendToFleetPlatform(results: AIInferenceResult) {
        // Placeholder
    }

    // TODO: Implement broadcastToBLE - stub for compilation
    private fun broadcastToBLE(results: AIInferenceResult) {
        // Placeholder
    }

    /**
     * Create mock CAN data for testing when hardware is not available
     * Returns realistic vehicle telemetry data
     */
    private fun createMockCANData(): CANData {
        return CANData(
            timestamp = System.currentTimeMillis(),
            vehicleSpeed = 60.0f,      // 60 km/h
            engineRPM = 1800.0f,        // 1800 rpm (cruising)
            throttlePosition = 35.0f,   // 35% throttle
            brakePosition = 0.0f,       // No braking
            fuelLevel = 75.0f,          // 75% fuel
            coolantTemp = 90.0f,        // 90°C (normal operating temp)
            batteryVoltage = 13.8f,     // 13.8V (charging)
            accelerationX = 0.0f,
            accelerationY = 0.0f,
            accelerationZ = 0.0f,
            gyroX = 0.0f,
            gyroY = 0.0f,
            gyroZ = 0.0f,
            steeringAngle = 0.0f,
            gpsLat = 37.5665,           // Seoul coordinates
            gpsLon = 126.9780
        )
    }

    // TODO: Native methods (implement in C++)
    external fun readCANDataFromUART(): CANData
    external fun preprocessCANData(canDataList: List<CANData>): FloatArray

    // Note: Native library initialization - moved to avoid duplicate companion object
    // companion object {
    //     init {
    //         System.loadLibrary("uart_reader")
    //     }
    // }
}

// NOTE: CANData and AIInferenceResult models are now in com.glec.dtg.models package
// These duplicate definitions have been removed to avoid conflicts

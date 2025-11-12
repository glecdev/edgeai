package com.glec.dtg.models

import java.io.Serializable

/**
 * GLEC DTG - CAN Data Model
 * Represents a single CAN bus data sample collected at 1Hz
 */
data class CANData(
    val timestamp: Long,                    // Unix timestamp (milliseconds)
    val vehicleSpeed: Float,                // km/h (0-255)
    val engineRPM: Int,                     // RPM (0-16383)
    val throttlePosition: Float,            // % (0-100)
    val brakePosition: Float,               // % (0-100)
    val fuelLevel: Float,                   // % (0-100)
    val coolantTemp: Float,                 // °C (-40 to 215)
    val engineLoad: Float,                  // % (0-100)
    val intakeAirTemp: Float,               // °C (-40 to 215)
    val mafRate: Float,                     // g/s (0-655.35)
    val batteryVoltage: Float,              // V (0-65.535)
    val accelerationX: Float,               // m/s² (-20 to 20)
    val accelerationY: Float,               // m/s² (-20 to 20)
    val accelerationZ: Float,               // m/s² (-20 to 20)
    val gyroX: Float,                       // °/s (-250 to 250)
    val gyroY: Float,                       // °/s (-250 to 250)
    val gyroZ: Float,                       // °/s (-250 to 250)
    val gpsLatitude: Double,                // Decimal degrees
    val gpsLongitude: Double,               // Decimal degrees
    val gpsAltitude: Float,                 // meters
    val gpsSpeed: Float,                    // km/h
    val gpsHeading: Float                   // degrees (0-360)
) : Serializable {

    /**
     * Convert to float array for AI model input
     * Order matches training data format
     */
    fun toFloatArray(): FloatArray {
        return floatArrayOf(
            vehicleSpeed,
            engineRPM.toFloat(),
            throttlePosition,
            brakePosition,
            fuelLevel,
            coolantTemp,
            engineLoad,
            intakeAirTemp,
            mafRate,
            batteryVoltage,
            accelerationX,
            accelerationY,
            accelerationZ,
            gyroX,
            gyroY,
            gyroZ,
            gpsSpeed,
            gpsHeading
        )
    }

    /**
     * Calculate instantaneous fuel consumption (L/100km)
     * Based on MAF (Mass Air Flow) rate and stoichiometric ratio
     */
    fun calculateFuelConsumption(): Float {
        if (vehicleSpeed < 1.0f || mafRate < 0.1f) {
            return 0.0f
        }

        // Stoichiometric air-fuel ratio for gasoline: 14.7:1
        val fuelFlowRate = mafRate / 14.7f  // g/s

        // Convert to L/h: (g/s) * 3600 / (density of gasoline ~750 g/L)
        val fuelFlowLiterPerHour = (fuelFlowRate * 3600.0f) / 750.0f

        // Convert to L/100km: (L/h) / (km/h) * 100
        return (fuelFlowLiterPerHour / vehicleSpeed) * 100.0f
    }

    /**
     * Detect harsh braking event
     * Threshold: deceleration < -4 m/s²
     */
    fun isHarshBraking(): Boolean {
        return accelerationX < -4.0f && brakePosition > 50.0f
    }

    /**
     * Detect harsh acceleration event
     * Threshold: acceleration > 3 m/s²
     */
    fun isHarshAcceleration(): Boolean {
        return accelerationX > 3.0f && throttlePosition > 70.0f
    }

    /**
     * Validate data integrity
     */
    fun isValid(): Boolean {
        return vehicleSpeed in 0.0f..255.0f &&
                engineRPM in 0..16383 &&
                throttlePosition in 0.0f..100.0f &&
                brakePosition in 0.0f..100.0f &&
                fuelLevel in 0.0f..100.0f &&
                coolantTemp in -40.0f..215.0f &&
                batteryVoltage in 10.0f..16.0f &&
                gpsLatitude in -90.0..90.0 &&
                gpsLongitude in -180.0..180.0
    }

    companion object {
        /**
         * Create from raw byte array (received via UART)
         * Protocol: [HEADER(1)] [TIMESTAMP(8)] [VALUES...] [CRC(2)] [FOOTER(1)]
         */
        fun fromByteArray(data: ByteArray): CANData? {
            if (data.size < 80) return null

            try {
                var offset = 1  // Skip header

                val timestamp = data.getLong(offset)
                offset += 8

                return CANData(
                    timestamp = timestamp,
                    vehicleSpeed = data.getFloat(offset).also { offset += 4 },
                    engineRPM = data.getInt(offset).also { offset += 4 },
                    throttlePosition = data.getFloat(offset).also { offset += 4 },
                    brakePosition = data.getFloat(offset).also { offset += 4 },
                    fuelLevel = data.getFloat(offset).also { offset += 4 },
                    coolantTemp = data.getFloat(offset).also { offset += 4 },
                    engineLoad = data.getFloat(offset).also { offset += 4 },
                    intakeAirTemp = data.getFloat(offset).also { offset += 4 },
                    mafRate = data.getFloat(offset).also { offset += 4 },
                    batteryVoltage = data.getFloat(offset).also { offset += 4 },
                    accelerationX = data.getFloat(offset).also { offset += 4 },
                    accelerationY = data.getFloat(offset).also { offset += 4 },
                    accelerationZ = data.getFloat(offset).also { offset += 4 },
                    gyroX = data.getFloat(offset).also { offset += 4 },
                    gyroY = data.getFloat(offset).also { offset += 4 },
                    gyroZ = data.getFloat(offset).also { offset += 4 },
                    gpsLatitude = data.getDouble(offset).also { offset += 8 },
                    gpsLongitude = data.getDouble(offset).also { offset += 8 },
                    gpsAltitude = data.getFloat(offset).also { offset += 4 },
                    gpsSpeed = data.getFloat(offset).also { offset += 4 },
                    gpsHeading = data.getFloat(offset)
                )
            } catch (e: Exception) {
                return null
            }
        }
    }
}

/**
 * AI Inference Results
 */
data class AIInferenceResult(
    val timestamp: Long,
    val fuelEfficiencyPrediction: Float,    // L/100km
    val anomalyScore: Float,                // 0.0-1.0 (higher = more anomalous)
    val behaviorClass: DrivingBehavior,
    val safetyScore: Int,                   // 0-100
    val carbonEmission: Float,              // g CO₂/km
    val anomalies: List<AnomalyType>,
    val inferenceLatency: Long              // milliseconds
)

/**
 * Driving Behavior Classification
 */
enum class DrivingBehavior(val label: String) {
    NORMAL("normal"),
    ECO_DRIVING("eco_driving"),
    HARSH_BRAKING("harsh_braking"),
    HARSH_ACCELERATION("harsh_acceleration"),
    SPEEDING("speeding"),
    AGGRESSIVE("aggressive"),
    ANOMALY("anomaly")
}

/**
 * Anomaly Types
 */
enum class AnomalyType(val description: String) {
    HARSH_BRAKING("Harsh braking detected"),
    HARSH_ACCELERATION("Harsh acceleration detected"),
    SPEEDING("Speeding detected"),
    SENSOR_FAULT("Sensor fault detected"),
    CAN_INTRUSION("Potential CAN bus intrusion"),
    ABNORMAL_PATTERN("Abnormal driving pattern"),
    ENGINE_OVERHEATING("Engine overheating"),
    LOW_BATTERY("Low battery voltage")
}

/**
 * Extension functions for ByteArray parsing
 */
private fun ByteArray.getFloat(offset: Int): Float {
    return java.nio.ByteBuffer.wrap(this, offset, 4).order(java.nio.ByteOrder.LITTLE_ENDIAN).float
}

private fun ByteArray.getInt(offset: Int): Int {
    return java.nio.ByteBuffer.wrap(this, offset, 4).order(java.nio.ByteOrder.LITTLE_ENDIAN).int
}

private fun ByteArray.getLong(offset: Int): Long {
    return java.nio.ByteBuffer.wrap(this, offset, 8).order(java.nio.ByteOrder.LITTLE_ENDIAN).long
}

private fun ByteArray.getDouble(offset: Int): Double {
    return java.nio.ByteBuffer.wrap(this, offset, 8).order(java.nio.ByteOrder.LITTLE_ENDIAN).double
}

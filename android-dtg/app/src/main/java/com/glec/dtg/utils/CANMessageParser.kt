package com.glec.dtg.utils

import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * GLEC DTG - CAN Message Parser
 * Parses CAN messages from UART protocol
 *
 * Protocol format:
 * [START(0xAA)] [ID_H] [ID_L] [DLC] [DATA(8)] [CRC16_H] [CRC16_L] [END(0x55)]
 * Total: 15 bytes per frame
 */
object CANMessageParser {

    private const val TAG = "CANMessageParser"

    private const val START_BYTE: Byte = 0xAA.toByte()
    private const val END_BYTE: Byte = 0x55.toByte()
    private const val FRAME_SIZE = 15

    /**
     * Parse single CAN frame from byte array
     */
    fun parseFrame(data: ByteArray): CANFrame? {
        if (data.size < FRAME_SIZE) {
            Log.w(TAG, "Frame too short: ${data.size} bytes")
            return null
        }

        // Check start byte
        if (data[0] != START_BYTE) {
            Log.w(TAG, "Invalid start byte: 0x${data[0].toString(16)}")
            return null
        }

        // Check end byte
        if (data[14] != END_BYTE) {
            Log.w(TAG, "Invalid end byte: 0x${data[14].toString(16)}")
            return null
        }

        // Extract fields
        val idHigh = data[1].toInt() and 0xFF
        val idLow = data[2].toInt() and 0xFF
        val canId = (idHigh shl 8) or idLow

        val dlc = data[3].toInt() and 0xFF

        val payload = ByteArray(8)
        System.arraycopy(data, 4, payload, 0, 8)

        val crcHigh = data[12].toInt() and 0xFF
        val crcLow = data[13].toInt() and 0xFF
        val receivedCRC = (crcHigh shl 8) or crcLow

        // Verify CRC
        val calculatedCRC = calculateCRC16(data, 1, 11)  // CRC over ID, DLC, DATA
        if (calculatedCRC != receivedCRC) {
            Log.w(TAG, "CRC mismatch: expected=0x${calculatedCRC.toString(16)}, received=0x${receivedCRC.toString(16)}")
            return null
        }

        return CANFrame(canId, dlc, payload)
    }

    /**
     * Parse OBD-II PID response
     * Format: [Mode + 0x40] [PID] [Data bytes]
     */
    fun parseOBDPID(frame: CANFrame): OBDData? {
        if (frame.dlc < 3) return null

        val mode = frame.data[0].toInt() and 0xFF
        val pid = frame.data[1].toInt() and 0xFF

        // Check if response mode (request mode + 0x40)
        if (mode != 0x41) return null

        return when (pid) {
            0x0C -> {
                // Engine RPM = ((A*256)+B)/4
                val a = frame.data[2].toInt() and 0xFF
                val b = frame.data[3].toInt() and 0xFF
                val rpm = ((a * 256) + b) / 4
                OBDData.EngineRPM(rpm)
            }

            0x0D -> {
                // Vehicle Speed = A (km/h)
                val speed = (frame.data[2].toInt() and 0xFF).toFloat()
                OBDData.VehicleSpeed(speed)
            }

            0x11 -> {
                // Throttle Position = A*100/255 (%)
                val a = frame.data[2].toInt() and 0xFF
                val throttle = (a * 100.0f) / 255.0f
                OBDData.ThrottlePosition(throttle)
            }

            0x2F -> {
                // Fuel Level = A*100/255 (%)
                val a = frame.data[2].toInt() and 0xFF
                val fuelLevel = (a * 100.0f) / 255.0f
                OBDData.FuelLevel(fuelLevel)
            }

            0x05 -> {
                // Coolant Temperature = A-40 (°C)
                val a = frame.data[2].toInt() and 0xFF
                val temp = (a - 40).toFloat()
                OBDData.CoolantTemp(temp)
            }

            0x04 -> {
                // Engine Load = A*100/255 (%)
                val a = frame.data[2].toInt() and 0xFF
                val load = (a * 100.0f) / 255.0f
                OBDData.EngineLoad(load)
            }

            0x0F -> {
                // Intake Air Temperature = A-40 (°C)
                val a = frame.data[2].toInt() and 0xFF
                val temp = (a - 40).toFloat()
                OBDData.IntakeAirTemp(temp)
            }

            0x10 -> {
                // MAF Air Flow Rate = ((A*256)+B)/100 (g/s)
                val a = frame.data[2].toInt() and 0xFF
                val b = frame.data[3].toInt() and 0xFF
                val maf = ((a * 256) + b) / 100.0f
                OBDData.MAFRate(maf)
            }

            0x42 -> {
                // Control Module Voltage = ((A*256)+B)/1000 (V)
                val a = frame.data[2].toInt() and 0xFF
                val b = frame.data[3].toInt() and 0xFF
                val voltage = ((a * 256) + b) / 1000.0f
                OBDData.BatteryVoltage(voltage)
            }

            else -> {
                Log.d(TAG, "Unknown PID: 0x${pid.toString(16)}")
                null
            }
        }
    }

    /**
     * Parse J1939 PGN (for commercial vehicles)
     *
     * Ported from production: dtg_can_bus_system.py
     * Source: GLEC_DTG_INTEGRATED_v20.0.0/03_sensors_integration/can_bus/
     *
     * J1939 is the SAE standard for commercial vehicle communication.
     * Uses 29-bit extended CAN IDs with embedded PGN (Parameter Group Number).
     */
    fun parseJ1939PGN(frame: CANFrame): J1939Data? {
        // J1939 uses 29-bit extended CAN ID
        // PGN is extracted from bits 8-25
        val pgn = (frame.canId shr 8) and 0x3FFFF

        return when (pgn) {
            // --- Engine Data ---

            61444 -> {  // 0xF004
                // Electronic Engine Controller 1 (EEC1)
                // Engine speed, torque, driver demand
                // Production-verified: Most critical engine data

                // Engine Speed (bytes 3-4): RPM = value * 0.125
                val speedLow = frame.data[3].toInt() and 0xFF
                val speedHigh = frame.data[4].toInt() and 0xFF
                val engineSpeed = ((speedHigh shl 8) or speedLow) * 0.125f

                // Engine Torque Mode (byte 2): Current torque as % of reference
                val torqueMode = frame.data[2].toInt() and 0xFF
                val engineTorquePercent = (torqueMode * 100.0f) / 250.0f - 125.0f

                // Driver Demand Torque (byte 1): Driver's requested torque
                val driverDemand = frame.data[1].toInt() and 0xFF
                val driverDemandPercent = (driverDemand * 100.0f) / 250.0f - 125.0f

                // Actual Engine Torque (byte 2): Current actual torque
                val actualTorque = frame.data[2].toInt() and 0xFF
                val actualTorquePercent = (actualTorque * 100.0f) / 250.0f - 125.0f

                J1939Data.EngineController1(
                    engineSpeed = engineSpeed,
                    driverDemandTorque = driverDemandPercent,
                    actualTorque = actualTorquePercent
                )
            }

            61443 -> {  // 0xF003
                // Electronic Engine Controller 2 (EEC2)
                // Accelerator pedal position, engine load

                val accelPedal = frame.data[1].toInt() and 0xFF
                val accelPosition = (accelPedal * 100.0f) / 250.0f

                val engineLoad = frame.data[2].toInt() and 0xFF
                val loadPercent = (engineLoad * 100.0f) / 250.0f

                J1939Data.EngineController2(
                    acceleratorPedal = accelPosition,
                    engineLoad = loadPercent
                )
            }

            // --- Vehicle Speed & Cruise Control ---

            65265 -> {  // 0xFEF1
                // Cruise Control/Vehicle Speed (CCVS)
                // Production-verified: Primary vehicle speed source

                // Wheel-based vehicle speed (bytes 1-2): km/h = value / 256
                val speedLow = frame.data[1].toInt() and 0xFF
                val speedHigh = frame.data[2].toInt() and 0xFF
                val wheelSpeed = ((speedHigh shl 8) or speedLow) / 256.0f

                // Cruise control speed (bytes 3-4)
                val cruiseLow = frame.data[3].toInt() and 0xFF
                val cruiseHigh = frame.data[4].toInt() and 0xFF
                val cruiseSpeed = ((cruiseHigh shl 8) or cruiseLow) / 256.0f

                // Brake pedal position (byte 5)
                val brakeSwitch = (frame.data[5].toInt() and 0x0C) shr 2
                val parkingBrake = (frame.data[5].toInt() and 0x30) shr 4

                J1939Data.CruiseControlSpeed(
                    wheelBasedSpeed = wheelSpeed,
                    cruiseControlSpeed = cruiseSpeed,
                    brakeSwitch = brakeSwitch,
                    parkingBrake = parkingBrake
                )
            }

            // --- Fuel System ---

            65262 -> {  // 0xFEEE
                // Engine Fluid Level/Pressure (EFL/P)
                // Production-verified: Fuel level and rate

                // Fuel level (bytes 0-1): % = value * 0.4
                val fuelLevelLow = frame.data[0].toInt() and 0xFF
                val fuelLevelHigh = frame.data[1].toInt() and 0xFF
                val fuelLevel = ((fuelLevelHigh shl 8) or fuelLevelLow) * 0.4f

                // Fuel rate (bytes 2-3): L/h = value * 0.05
                val fuelRateLow = frame.data[2].toInt() and 0xFF
                val fuelRateHigh = frame.data[3].toInt() and 0xFF
                val fuelRate = ((fuelRateHigh shl 8) or fuelRateLow) * 0.05f

                J1939Data.FuelData(
                    fuelLevel = fuelLevel,
                    fuelRate = fuelRate
                )
            }

            65266 -> {  // 0xFEF2
                // Fuel Economy (LFE)
                // Production: Instantaneous and average fuel economy

                val instantLow = frame.data[0].toInt() and 0xFF
                val instantHigh = frame.data[1].toInt() and 0xFF
                val instantFuelEconomy = ((instantHigh shl 8) or instantLow) / 512.0f  // km/L

                val averageLow = frame.data[2].toInt() and 0xFF
                val averageHigh = frame.data[3].toInt() and 0xFF
                val averageFuelEconomy = ((averageHigh shl 8) or averageLow) / 512.0f  // km/L

                J1939Data.FuelEconomy(
                    instantaneous = instantFuelEconomy,
                    average = averageFuelEconomy
                )
            }

            // --- Transmission ---

            61445 -> {  // 0xF005
                // Electronic Transmission Controller 1 (ETC1)
                // Gear selection, output shaft speed

                val currentGear = frame.data[3].toInt() and 0xFF
                val selectedGear = frame.data[4].toInt() and 0xFF

                val shaftSpeedLow = frame.data[5].toInt() and 0xFF
                val shaftSpeedHigh = frame.data[6].toInt() and 0xFF
                val outputShaftSpeed = ((shaftSpeedHigh shl 8) or shaftSpeedLow) * 0.125f

                J1939Data.TransmissionController(
                    currentGear = currentGear - 125,  // Offset
                    selectedGear = selectedGear - 125,
                    outputShaftSpeed = outputShaftSpeed
                )
            }

            // --- Braking System ---

            65215 -> {  // 0xFEBF
                // Electronic Brake Controller 1 (EBC1)
                // Production: Air brake pressure monitoring

                val serviceBrakeLow = frame.data[0].toInt() and 0xFF
                val serviceBrakeHigh = frame.data[1].toInt() and 0xFF
                val serviceBrakePressure = ((serviceBrakeHigh shl 8) or serviceBrakeLow) * 4.0f  // kPa

                val parkingBrakeLow = frame.data[4].toInt() and 0xFF
                val parkingBrakeHigh = frame.data[5].toInt() and 0xFF
                val parkingBrakePressure = ((parkingBrakeHigh shl 8) or parkingBrakeLow) * 4.0f  // kPa

                J1939Data.BrakeController(
                    serviceBrakePressure = serviceBrakePressure,
                    parkingBrakePressure = parkingBrakePressure
                )
            }

            // --- Tire Pressure Monitoring (TPMS) ---

            65268 -> {  // 0xFEF4
                // Tire Condition (TC1)
                // Production: Critical for cargo trucks

                // Front left tire (bytes 0-1): bar = value * 0.5
                val frontLeftLow = frame.data[0].toInt() and 0xFF
                val frontLeftHigh = frame.data[1].toInt() and 0xFF
                val frontLeftPressure = ((frontLeftHigh shl 8) or frontLeftLow) * 0.5f / 1000.0f

                // Front right tire (bytes 2-3)
                val frontRightLow = frame.data[2].toInt() and 0xFF
                val frontRightHigh = frame.data[3].toInt() and 0xFF
                val frontRightPressure = ((frontRightHigh shl 8) or frontRightLow) * 0.5f / 1000.0f

                // Rear left tire (bytes 4-5)
                val rearLeftLow = frame.data[4].toInt() and 0xFF
                val rearLeftHigh = frame.data[5].toInt() and 0xFF
                val rearLeftPressure = ((rearLeftHigh shl 8) or rearLeftLow) * 0.5f / 1000.0f

                // Rear right tire (bytes 6-7)
                val rearRightLow = frame.data[6].toInt() and 0xFF
                val rearRightHigh = frame.data[7].toInt() and 0xFF
                val rearRightPressure = ((rearRightHigh shl 8) or rearRightLow) * 0.5f / 1000.0f

                J1939Data.TireCondition(
                    frontLeftPressure = frontLeftPressure,
                    frontRightPressure = frontRightPressure,
                    rearLeftPressure = rearLeftPressure,
                    rearRightPressure = rearRightPressure
                )
            }

            // --- Weight & Load ---

            65257 -> {  // 0xFEE9
                // Vehicle Weight (VW)
                // Production: Cargo weight monitoring for compliance

                val frontAxleLow = frame.data[0].toInt() and 0xFF
                val frontAxleHigh = frame.data[1].toInt() and 0xFF
                val frontAxleWeight = ((frontAxleHigh shl 8) or frontAxleLow) * 0.5f  // kg

                val rearAxleLow = frame.data[2].toInt() and 0xFF
                val rearAxleHigh = frame.data[3].toInt() and 0xFF
                val rearAxleWeight = ((rearAxleHigh shl 8) or rearAxleLow) * 0.5f  // kg

                val totalWeight = frontAxleWeight + rearAxleWeight

                J1939Data.VehicleWeight(
                    frontAxle = frontAxleWeight,
                    rearAxle = rearAxleWeight,
                    totalWeight = totalWeight
                )
            }

            // --- Emissions & Diagnostics ---

            61442 -> {  // 0xF002
                // Electronic Engine Controller 3 (EEC3)
                // NOx levels, DPF status (diesel particulate filter)

                val noxLevel = frame.data[0].toInt() and 0xFF
                val dpfStatus = frame.data[2].toInt() and 0xFF
                val engineCoolantTemp = frame.data[3].toInt() and 0xFF

                J1939Data.EngineController3(
                    noxLevel = noxLevel.toFloat(),
                    dpfStatus = dpfStatus,
                    coolantTemp = (engineCoolantTemp - 40).toFloat()
                )
            }

            // --- Ambient Conditions ---

            65269 -> {  // 0xFEF5
                // Ambient Conditions (AMB)
                // Outside air temperature, barometric pressure

                val ambientTemp = frame.data[3].toInt() and 0xFF
                val temperature = (ambientTemp * 0.03125f) - 273.0f  // Convert to Celsius

                val baroLow = frame.data[0].toInt() and 0xFF
                val baroHigh = frame.data[1].toInt() and 0xFF
                val baroPressure = ((baroHigh shl 8) or baroLow) * 0.125f  // kPa

                J1939Data.AmbientConditions(
                    airTemperature = temperature,
                    barometricPressure = baroPressure
                )
            }

            else -> {
                Log.d(TAG, "Unknown J1939 PGN: $pgn (0x${pgn.toString(16)})")
                null
            }
        }
    }

    /**
     * Calculate CRC-16 (CCITT)
     * Polynomial: 0x1021
     */
    private fun calculateCRC16(data: ByteArray, offset: Int, length: Int): Int {
        var crc = 0xFFFF

        for (i in offset until offset + length) {
            val byte = data[i].toInt() and 0xFF
            crc = crc xor (byte shl 8)

            for (j in 0 until 8) {
                if ((crc and 0x8000) != 0) {
                    crc = (crc shl 1) xor 0x1021
                } else {
                    crc = crc shl 1
                }
            }
        }

        return crc and 0xFFFF
    }

    /**
     * Find start of next frame in byte stream
     */
    fun findFrameStart(buffer: ByteArray, startOffset: Int = 0): Int {
        for (i in startOffset until buffer.size) {
            if (buffer[i] == START_BYTE) {
                return i
            }
        }
        return -1
    }

    /**
     * Extract complete frame from buffer
     */
    fun extractFrame(buffer: ByteArray, startOffset: Int): Pair<CANFrame?, Int>? {
        if (startOffset + FRAME_SIZE > buffer.size) {
            return null  // Not enough data
        }

        val frameData = ByteArray(FRAME_SIZE)
        System.arraycopy(buffer, startOffset, frameData, 0, FRAME_SIZE)

        val frame = parseFrame(frameData)
        return Pair(frame, startOffset + FRAME_SIZE)
    }
}

/**
 * CAN Frame representation
 */
data class CANFrame(
    val canId: Int,        // CAN identifier (11-bit or 29-bit)
    val dlc: Int,          // Data Length Code (0-8)
    val data: ByteArray    // Payload (8 bytes)
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as CANFrame

        if (canId != other.canId) return false
        if (dlc != other.dlc) return false
        if (!data.contentEquals(other.data)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = canId
        result = 31 * result + dlc
        result = 31 * result + data.contentHashCode()
        return result
    }

    override fun toString(): String {
        return "CANFrame(id=0x${canId.toString(16)}, dlc=$dlc, data=${data.joinToString(" ") { "0x${it.toString(16)}" }})"
    }
}

/**
 * OBD-II Data types
 */
sealed class OBDData {
    data class EngineRPM(val rpm: Int) : OBDData()
    data class VehicleSpeed(val speed: Float) : OBDData()
    data class ThrottlePosition(val position: Float) : OBDData()
    data class FuelLevel(val level: Float) : OBDData()
    data class CoolantTemp(val temp: Float) : OBDData()
    data class EngineLoad(val load: Float) : OBDData()
    data class IntakeAirTemp(val temp: Float) : OBDData()
    data class MAFRate(val rate: Float) : OBDData()
    data class BatteryVoltage(val voltage: Float) : OBDData()
}

/**
 * J1939 Data types (commercial vehicles)
 *
 * SAE J1939 standard for heavy-duty vehicles
 * Ported from production: dtg_can_bus_system.py
 *
 * Comprehensive commercial vehicle data including:
 * - Engine controllers (torque, load, emissions)
 * - Transmission (gear selection, shaft speed)
 * - Fuel system (level, rate, economy)
 * - Braking system (air pressure)
 * - Tire pressure monitoring (TPMS)
 * - Weight & load (cargo compliance)
 * - Ambient conditions
 */
sealed class J1939Data {
    /**
     * Electronic Engine Controller 1 (PGN 61444)
     * Most critical engine data: speed, torque, driver demand
     */
    data class EngineController1(
        val engineSpeed: Float,          // RPM
        val driverDemandTorque: Float,   // % (-125 to +125)
        val actualTorque: Float          // % (-125 to +125)
    ) : J1939Data()

    /**
     * Electronic Engine Controller 2 (PGN 61443)
     * Accelerator position and engine load
     */
    data class EngineController2(
        val acceleratorPedal: Float,     // % (0-100)
        val engineLoad: Float            // % (0-100)
    ) : J1939Data()

    /**
     * Electronic Engine Controller 3 (PGN 61442)
     * Emissions and diagnostics (NOx, DPF status)
     */
    data class EngineController3(
        val noxLevel: Float,             // ppm
        val dpfStatus: Int,              // Diesel particulate filter status
        val coolantTemp: Float           // °C
    ) : J1939Data()

    /**
     * Cruise Control/Vehicle Speed (PGN 65265)
     * Primary vehicle speed source for commercial vehicles
     */
    data class CruiseControlSpeed(
        val wheelBasedSpeed: Float,      // km/h
        val cruiseControlSpeed: Float,   // km/h
        val brakeSwitch: Int,            // Brake pedal switch state
        val parkingBrake: Int            // Parking brake state
    ) : J1939Data()

    /**
     * Fuel Data (PGN 65262)
     * Fuel level and consumption rate
     */
    data class FuelData(
        val fuelLevel: Float,            // % (0-100)
        val fuelRate: Float              // L/h
    ) : J1939Data()

    /**
     * Fuel Economy (PGN 65266)
     * Instantaneous and average fuel efficiency
     */
    data class FuelEconomy(
        val instantaneous: Float,        // km/L
        val average: Float               // km/L
    ) : J1939Data()

    /**
     * Transmission Controller (PGN 61445)
     * Gear selection and output shaft speed
     */
    data class TransmissionController(
        val currentGear: Int,            // Current gear (-125 to +125, 0=neutral)
        val selectedGear: Int,           // Selected gear
        val outputShaftSpeed: Float      // RPM
    ) : J1939Data()

    /**
     * Brake Controller (PGN 65215)
     * Air brake system pressure monitoring
     */
    data class BrakeController(
        val serviceBrakePressure: Float, // kPa (service brake)
        val parkingBrakePressure: Float  // kPa (parking brake)
    ) : J1939Data()

    /**
     * Tire Condition (PGN 65268)
     * TPMS: Tire pressure for all wheels
     * Critical for cargo weight monitoring
     */
    data class TireCondition(
        val frontLeftPressure: Float,    // bar
        val frontRightPressure: Float,   // bar
        val rearLeftPressure: Float,     // bar
        val rearRightPressure: Float     // bar
    ) : J1939Data()

    /**
     * Vehicle Weight (PGN 65257)
     * Axle weights for cargo compliance
     */
    data class VehicleWeight(
        val frontAxle: Float,            // kg
        val rearAxle: Float,             // kg
        val totalWeight: Float           // kg
    ) : J1939Data()

    /**
     * Ambient Conditions (PGN 65269)
     * Environmental data: temperature, pressure
     */
    data class AmbientConditions(
        val airTemperature: Float,       // °C
        val barometricPressure: Float    // kPa
    ) : J1939Data()
}

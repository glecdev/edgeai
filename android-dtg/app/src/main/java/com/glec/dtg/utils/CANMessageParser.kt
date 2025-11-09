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
     */
    fun parseJ1939PGN(frame: CANFrame): J1939Data? {
        // J1939 uses 29-bit extended CAN ID
        // PGN is extracted from bits 8-25
        val pgn = (frame.canId shr 8) and 0xFFFF

        return when (pgn) {
            0xF004 -> {
                // Electronic Engine Controller 1 (EEC1)
                // Engine speed, torque
                val speedLow = frame.data[3].toInt() and 0xFF
                val speedHigh = frame.data[4].toInt() and 0xFF
                val engineSpeed = ((speedHigh shl 8) or speedLow) * 0.125f

                J1939Data.EngineSpeed(engineSpeed)
            }

            0xFEF1 -> {
                // Cruise Control/Vehicle Speed (CCVS)
                val speedLow = frame.data[1].toInt() and 0xFF
                val speedHigh = frame.data[2].toInt() and 0xFF
                val vehicleSpeed = ((speedHigh shl 8) or speedLow) / 256.0f

                J1939Data.VehicleSpeed(vehicleSpeed)
            }

            0xFEEE -> {
                // Fuel Economy (LFE)
                val fuelLevelLow = frame.data[0].toInt() and 0xFF
                val fuelLevelHigh = frame.data[1].toInt() and 0xFF
                val fuelLevel = ((fuelLevelHigh shl 8) or fuelLevelLow) * 0.4f

                J1939Data.FuelLevel(fuelLevel)
            }

            else -> {
                Log.d(TAG, "Unknown J1939 PGN: 0x${pgn.toString(16)}")
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
 */
sealed class J1939Data {
    data class EngineSpeed(val speed: Float) : J1939Data()
    data class VehicleSpeed(val speed: Float) : J1939Data()
    data class FuelLevel(val level: Float) : J1939Data()
}

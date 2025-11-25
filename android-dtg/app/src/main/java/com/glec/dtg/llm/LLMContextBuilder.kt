package com.glec.dtg.llm

import com.glec.dtg.can.J1939Service
import com.glec.dtg.models.VehicleData
import timber.log.Timber

/**
 * LLM Context Builder - Converts J1939 CAN data to VehicleData context
 *
 * This class bridges the gap between low-level CAN bus data (J1939 protocol)
 * and high-level vehicle context needed for LLM prompts.
 *
 * **Data Flow**:
 * ```
 * J1939 CAN Bus → J1939Service → LLMContextBuilder → VehicleData → LLM Prompt
 * ```
 *
 * **J1939 PGNs Used** (per PHASE3K_LLM_INTEGRATION.md):
 * - PGN 65257: Vehicle Weight (cargo weight)
 * - PGN 65267: Tire Pressure (TPMS)
 * - PGN 65262: Engine Temperature
 * - PGN 65266: Fuel Economy
 *
 * **References**:
 * - LLM_IMPLEMENTATION_GUIDE.md - Context integration
 * - docs/J1939_PROTOCOL.md - CAN message formats
 * - android-dtg/can/J1939Service.kt - CAN data source
 *
 * @param j1939Service Service providing real-time J1939 CAN data
 */
class LLMContextBuilder(private val j1939Service: J1939Service) {

    companion object {
        private const val FUEL_EFFICIENCY_MIN = 0.1  // Ignore zero/invalid readings
        private const val TIRE_PRESSURE_DEFAULT = 220.0  // kPa (normal for truck)
        private const val ENGINE_TEMP_DEFAULT = 85.0  // °C (normal operating temp)
    }

    /**
     * Build VehicleData context from latest J1939 CAN data
     *
     * **Context Included**:
     * - Cargo weight (kg) - for load management queries
     * - Tire pressure (kPa) - for safety warnings
     * - Engine temperature (°C) - for overheating detection
     * - Fuel efficiency (km/L) - for economy advice
     * - Timestamp (ms) - for data freshness
     *
     * **Error Handling**:
     * - Falls back to safe defaults if CAN data unavailable
     * - Logs warnings for missing data
     *
     * **Performance**: <10ms (data already cached by J1939Service)
     *
     * @return VehicleData context for LLM prompt
     */
    fun buildContext(): VehicleData {
        return try {
            val canData = j1939Service.getLatestData()

            VehicleData(
                cargoWeight = extractCargoWeight(canData),
                tirePressure = extractTirePressure(canData),
                engineTemp = extractEngineTemp(canData),
                fuelEfficiency = calculateFuelEfficiency(canData),
                timestamp = System.currentTimeMillis()
            )
        } catch (e: Exception) {
            Timber.w(e, "Failed to build context from CAN data, using defaults")

            // Return safe defaults if CAN data unavailable
            VehicleData(
                cargoWeight = 0.0,
                tirePressure = TIRE_PRESSURE_DEFAULT,
                engineTemp = ENGINE_TEMP_DEFAULT,
                fuelEfficiency = 0.0,
                timestamp = System.currentTimeMillis()
            )
        }
    }

    /**
     * Extract cargo weight from J1939 PGN 65257
     *
     * **PGN 65257**: Vehicle Weight
     * - Byte 1-2: Front axle weight (kg)
     * - Byte 3-4: Rear axle weight (kg)
     * - Byte 5-6: Cargo weight (kg)
     *
     * **Formula**: cargoWeight = (Byte5 << 8) | Byte6
     *
     * @param canData J1939 data packet
     * @return Cargo weight in kg
     */
    private fun extractCargoWeight(canData: J1939Data): Double {
        return try {
            canData.pgn65257.cargoWeight.toDouble()
        } catch (e: Exception) {
            Timber.w("Failed to extract cargo weight: ${e.message}")
            0.0  // Unknown weight
        }
    }

    /**
     * Extract tire pressure from J1939 PGN 65267
     *
     * **PGN 65267**: Tire Pressure (TPMS)
     * - Supports up to 4 tires
     * - Returns front-left tire as representative
     *
     * **Normal Range**: 200-250 kPa for commercial trucks
     * **Warning**: <200 kPa (low pressure)
     *
     * @param canData J1939 data packet
     * @return Tire pressure in kPa
     */
    private fun extractTirePressure(canData: J1939Data): Double {
        return try {
            canData.pgn65267.frontLeftTirePressure.toDouble()
        } catch (e: Exception) {
            Timber.w("Failed to extract tire pressure: ${e.message}")
            TIRE_PRESSURE_DEFAULT  // Normal pressure
        }
    }

    /**
     * Extract engine temperature from J1939 PGN 65262
     *
     * **PGN 65262**: Engine Temperature
     * - Engine coolant temperature (°C)
     *
     * **Normal Range**: 80-95°C
     * **Warning**: >95°C (overheating risk)
     *
     * @param canData J1939 data packet
     * @return Engine temperature in °C
     */
    private fun extractEngineTemp(canData: J1939Data): Double {
        return try {
            canData.pgn65262.engineCoolantTemp.toDouble()
        } catch (e: Exception) {
            Timber.w("Failed to extract engine temp: ${e.message}")
            ENGINE_TEMP_DEFAULT  // Normal operating temp
        }
    }

    /**
     * Calculate fuel efficiency from J1939 data
     *
     * **Formula**: fuelEfficiency = totalDistance / totalFuel
     *
     * **Data Sources**:
     * - PGN 65248: Total distance (km)
     * - PGN 65257: Total fuel used (L)
     *
     * **Edge Cases**:
     * - Zero fuel: Return 0.0 (avoid division by zero)
     * - Zero distance: Return 0.0 (vehicle not moving)
     *
     * **Expected Range**: 5-8 km/L for loaded commercial trucks
     *
     * @param canData J1939 data packet
     * @return Fuel efficiency in km/L
     */
    private fun calculateFuelEfficiency(canData: J1939Data): Double {
        return try {
            val totalDistance = canData.totalDistance  // km
            val totalFuel = canData.totalFuel  // L

            if (totalFuel < FUEL_EFFICIENCY_MIN || totalDistance < 0.1) {
                0.0  // Insufficient data
            } else {
                totalDistance / totalFuel
            }
        } catch (e: Exception) {
            Timber.w("Failed to calculate fuel efficiency: ${e.message}")
            0.0
        }
    }
}

/**
 * J1939 CAN Data container
 *
 * TODO: Replace with actual J1939Data from can/ package
 * This is a temporary stub for development
 */
data class J1939Data(
    val pgn65257: PGN65257 = PGN65257(),  // Vehicle weight
    val pgn65262: PGN65262 = PGN65262(),  // Engine temp
    val pgn65267: PGN65267 = PGN65267(),  // Tire pressure
    val totalDistance: Double = 0.0,      // km
    val totalFuel: Double = 0.0           // L
)

/**
 * PGN 65257: Vehicle Weight
 */
data class PGN65257(
    val cargoWeight: Int = 0,           // kg
    val frontAxleWeight: Int = 0,       // kg
    val rearAxleWeight: Int = 0         // kg
)

/**
 * PGN 65262: Engine Temperature
 */
data class PGN65262(
    val engineCoolantTemp: Int = 85     // °C
)

/**
 * PGN 65267: Tire Pressure (TPMS)
 */
data class PGN65267(
    val frontLeftTirePressure: Int = 220,   // kPa
    val frontRightTirePressure: Int = 220,  // kPa
    val rearLeftTirePressure: Int = 220,    // kPa
    val rearRightTirePressure: Int = 220    // kPa
)

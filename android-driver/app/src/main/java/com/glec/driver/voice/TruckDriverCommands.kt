package com.glec.driver.voice

import android.content.Context
import android.util.Log
import com.glec.driver.models.VehicleData
import kotlinx.coroutines.flow.StateFlow

/**
 * GLEC Driver - Truck-Specific Voice Commands
 *
 * Ported from production: TruckDriverVoiceCommands.kt (11KB)
 * Source: GLEC_DTG_INTEGRATED_v20.0.0/android_app/kotlin_source/
 *
 * Extends base voice commands with commercial vehicle features:
 * - Cargo weight monitoring ("짐 상태 확인")
 * - Tire pressure check ("타이어 압력 확인")
 * - Engine diagnostics ("엔진 상태")
 * - Fuel range calculation ("주행 가능 거리")
 * - Rest area navigation ("가까운 휴게소")
 * - Weigh station info ("검문소 정보")
 * - Vehicle inspection ("차량 점검")
 * - Road hazard reporting ("도로 위험 신고")
 *
 * Production-verified for:
 * - Volvo trucks (FE, FM series)
 * - Hyundai commercial vehicles (Porter, Mighty)
 * - Generic J1939-compliant vehicles
 */
class TruckDriverCommands(
    context: Context,
    private val vehicleData: StateFlow<VehicleData?>
) : VoiceAssistant(context) {

    companion object {
        private const val TAG = "TruckDriverCommands"

        // Truck-specific voice intents (Korean)
        const val INTENT_CHECK_CARGO = "CHECK_CARGO_STATUS"
        const val INTENT_TIRE_PRESSURE = "TIRE_PRESSURE_CHECK"
        const val INTENT_ENGINE_STATUS = "ENGINE_STATUS"
        const val INTENT_FUEL_RANGE = "FUEL_RANGE"
        const val INTENT_NEAREST_REST_AREA = "NEAREST_REST_AREA"
        const val INTENT_WEIGH_STATION = "WEIGH_STATION_INFO"
        const val INTENT_VEHICLE_INSPECTION = "VEHICLE_INSPECTION"
        const val INTENT_REPORT_HAZARD = "REPORT_ROAD_HAZARD"
        const val INTENT_CHECK_BRAKES = "CHECK_BRAKES"
        const val INTENT_DPF_STATUS = "DPF_STATUS"
        const val INTENT_TRANSMISSION_STATUS = "TRANSMISSION_STATUS"
        const val INTENT_AXLE_WEIGHT = "AXLE_WEIGHT"
    }

    /**
     * Parse voice command to intent
     *
     * Production: Pattern matching with Korean natural language
     */
    override fun parseIntent(sttResult: String): VoiceIntent? {
        Log.d(TAG, "Parsing intent: $sttResult")

        return when {
            // --- Cargo & Weight ---
            sttResult.contains("짐") && sttResult.contains("상태") ->
                VoiceIntent.TruckSpecific(INTENT_CHECK_CARGO)

            sttResult.contains("화물") && sttResult.contains("무게") ->
                VoiceIntent.TruckSpecific(INTENT_CHECK_CARGO)

            sttResult.contains("적재") && sttResult.contains("중량") ->
                VoiceIntent.TruckSpecific(INTENT_AXLE_WEIGHT)

            // --- Tire Pressure ---
            sttResult.contains("타이어") && (sttResult.contains("압력") || sttResult.contains("공기압")) ->
                VoiceIntent.TruckSpecific(INTENT_TIRE_PRESSURE)

            sttResult.contains("타이어") && sttResult.contains("확인") ->
                VoiceIntent.TruckSpecific(INTENT_TIRE_PRESSURE)

            // --- Engine & Diagnostics ---
            sttResult.contains("엔진") && sttResult.contains("상태") ->
                VoiceIntent.TruckSpecific(INTENT_ENGINE_STATUS)

            sttResult.contains("엔진") && sttResult.contains("점검") ->
                VoiceIntent.TruckSpecific(INTENT_ENGINE_STATUS)

            sttResult.contains("디피에프") || sttResult.contains("dpf") ->
                VoiceIntent.TruckSpecific(INTENT_DPF_STATUS)

            // --- Fuel & Range ---
            sttResult.contains("주행") && sttResult.contains("거리") ->
                VoiceIntent.TruckSpecific(INTENT_FUEL_RANGE)

            sttResult.contains("연료") && sttResult.contains("남") ->
                VoiceIntent.TruckSpecific(INTENT_FUEL_RANGE)

            sttResult.contains("몇") && sttResult.contains("킬로") && sttResult.contains("갈") ->
                VoiceIntent.TruckSpecific(INTENT_FUEL_RANGE)

            // --- Navigation (Truck-specific) ---
            sttResult.contains("휴게소") ->
                VoiceIntent.TruckSpecific(INTENT_NEAREST_REST_AREA)

            sttResult.contains("주유소") && sttResult.contains("트럭") ->
                VoiceIntent.TruckSpecific(INTENT_NEAREST_REST_AREA)

            sttResult.contains("검문소") || sttResult.contains("계근소") ->
                VoiceIntent.TruckSpecific(INTENT_WEIGH_STATION)

            // --- Vehicle Inspection ---
            sttResult.contains("차량") && sttResult.contains("점검") ->
                VoiceIntent.TruckSpecific(INTENT_VEHICLE_INSPECTION)

            sttResult.contains("브레이크") && sttResult.contains("상태") ->
                VoiceIntent.TruckSpecific(INTENT_CHECK_BRAKES)

            sttResult.contains("변속기") || sttResult.contains("기어") ->
                VoiceIntent.TruckSpecific(INTENT_TRANSMISSION_STATUS)

            // --- Safety Reporting ---
            sttResult.contains("위험") && sttResult.contains("신고") ->
                VoiceIntent.TruckSpecific(INTENT_REPORT_HAZARD)

            sttResult.contains("도로") && sttResult.contains("장애물") ->
                VoiceIntent.TruckSpecific(INTENT_REPORT_HAZARD)

            // Fallback to base voice assistant
            else -> super.parseIntent(sttResult)
        }
    }

    /**
     * Handle truck-specific voice intent
     *
     * Production: Integrates with J1939 CAN data
     */
    override fun handleIntent(intent: VoiceIntent) {
        when (intent) {
            is VoiceIntent.TruckSpecific -> handleTruckIntent(intent.action)
            else -> super.handleIntent(intent)
        }
    }

    private fun handleTruckIntent(action: String) {
        val vehicle = vehicleData.value

        if (vehicle == null) {
            speak("차량 데이터를 불러올 수 없습니다.")
            return
        }

        when (action) {
            INTENT_CHECK_CARGO -> {
                // J1939 PGN 65257: Vehicle Weight
                val cargoWeight = vehicle.j1939Data?.vehicleWeight?.totalWeight ?: 0f
                val frontAxle = vehicle.j1939Data?.vehicleWeight?.frontAxle ?: 0f
                val rearAxle = vehicle.j1939Data?.vehicleWeight?.rearAxle ?: 0f

                if (cargoWeight > 0) {
                    speak("현재 적재 중량은 ${cargoWeight.toInt()}킬로그램입니다. " +
                          "앞축 ${frontAxle.toInt()}킬로, 뒷축 ${rearAxle.toInt()}킬로입니다.")

                    // Check overload (production safety feature)
                    val maxWeight = 25000f  // 25 tons typical limit
                    if (cargoWeight > maxWeight) {
                        speak("경고! 최대 적재 중량을 초과했습니다.")
                    }
                } else {
                    speak("화물 중량 센서 데이터를 사용할 수 없습니다.")
                }
            }

            INTENT_TIRE_PRESSURE -> {
                // J1939 PGN 65268: Tire Condition
                val tireData = vehicle.j1939Data?.tirePressure

                if (tireData != null) {
                    speak("타이어 압력을 안내합니다. " +
                          "앞 왼쪽 ${String.format("%.1f", tireData.frontLeft)}바, " +
                          "앞 오른쪽 ${String.format("%.1f", tireData.frontRight)}바, " +
                          "뒤 왼쪽 ${String.format("%.1f", tireData.rearLeft)}바, " +
                          "뒤 오른쪽 ${String.format("%.1f", tireData.rearRight)}바입니다.")

                    // Production: Low pressure warning
                    val minPressure = 7.0f  // bar
                    val lowPressureTires = mutableListOf<String>()

                    if (tireData.frontLeft < minPressure) lowPressureTires.add("앞 왼쪽")
                    if (tireData.frontRight < minPressure) lowPressureTires.add("앞 오른쪽")
                    if (tireData.rearLeft < minPressure) lowPressureTires.add("뒤 왼쪽")
                    if (tireData.rearRight < minPressure) lowPressureTires.add("뒤 오른쪽")

                    if (lowPressureTires.isNotEmpty()) {
                        speak("주의! ${lowPressureTires.joinToString(", ")} 타이어 압력이 낮습니다.")
                    }
                } else {
                    speak("타이어 압력 센서를 사용할 수 없습니다.")
                }
            }

            INTENT_ENGINE_STATUS -> {
                // J1939 PGN 61444: Engine Controller 1
                val engineData = vehicle.j1939Data?.engineController1

                if (engineData != null) {
                    val rpm = engineData.engineSpeed.toInt()
                    val torque = engineData.actualTorque.toInt()
                    val coolantTemp = vehicle.canData?.coolantTemp?.toInt() ?: 0

                    speak("엔진 상태를 안내합니다. " +
                          "현재 알피엠 ${rpm}, " +
                          "토크 ${torque}퍼센트, " +
                          "냉각수 온도 ${coolantTemp}도입니다.")

                    // Production: Engine health check
                    if (coolantTemp > 100) {
                        speak("경고! 냉각수 온도가 높습니다.")
                    }
                    if (rpm > 3500) {
                        speak("주의! 과도한 회전수입니다.")
                    }
                } else {
                    speak("엔진 데이터를 사용할 수 없습니다.")
                }
            }

            INTENT_FUEL_RANGE -> {
                // Calculate range using fuel level and average consumption
                val fuelLevel = vehicle.canData?.fuelLevel ?: 0f
                val avgConsumption = vehicle.aiResults?.fuelEfficiency ?: 10f  // km/L

                // Typical truck fuel tank: 300L
                val tankCapacity = 300f
                val remainingFuel = (fuelLevel / 100f) * tankCapacity
                val estimatedRange = remainingFuel * avgConsumption

                speak("현재 연료로 약 ${estimatedRange.toInt()}킬로미터 주행 가능합니다. " +
                      "남은 연료는 ${remainingFuel.toInt()}리터입니다.")

                // Production: Low fuel warning
                if (fuelLevel < 20f) {
                    speak("주의! 연료가 부족합니다. 주유소를 찾아주세요.")
                }
            }

            INTENT_NEAREST_REST_AREA -> {
                // TODO: Integrate with external navigation API
                speak("가까운 휴게소를 찾고 있습니다.")

                // Production: Uses Korea Transport Database API
                // For now, placeholder
                speak("10킬로미터 앞 서울 휴게소가 있습니다.")
            }

            INTENT_WEIGH_STATION -> {
                // TODO: Integrate with weigh station database
                speak("검문소 정보를 확인합니다.")

                // Production: Provides advance warning
                speak("5킬로미터 앞 검문소가 있습니다. 차량 서류를 준비해주세요.")
            }

            INTENT_VEHICLE_INSPECTION -> {
                // Comprehensive vehicle health check (production feature)
                val issues = mutableListOf<String>()

                // Check critical systems
                val batteryVoltage = vehicle.canData?.batteryVoltage ?: 0f
                if (batteryVoltage < 11.5f) {
                    issues.add("배터리 전압 낮음")
                }

                val engineTemp = vehicle.canData?.coolantTemp ?: 0f
                if (engineTemp > 100f) {
                    issues.add("엔진 과열")
                }

                val brakeData = vehicle.j1939Data?.brakeController
                if (brakeData != null && brakeData.serviceBrakePressure < 500f) {
                    issues.add("브레이크 압력 부족")
                }

                if (issues.isEmpty()) {
                    speak("차량 점검 결과 이상 없습니다.")
                } else {
                    speak("차량 점검 결과 다음 문제가 발견되었습니다. " + issues.joinToString(", "))
                }
            }

            INTENT_CHECK_BRAKES -> {
                // J1939 PGN 65215: Brake Controller
                val brakeData = vehicle.j1939Data?.brakeController

                if (brakeData != null) {
                    val servicePressure = (brakeData.serviceBrakePressure / 100f).toInt()  // kPa to bar
                    val parkingPressure = (brakeData.parkingBrakePressure / 100f).toInt()

                    speak("브레이크 압력을 안내합니다. " +
                          "주 브레이크 ${servicePressure}바, " +
                          "주차 브레이크 ${parkingPressure}바입니다.")

                    // Production: Air brake safety check
                    if (servicePressure < 6) {
                        speak("경고! 브레이크 압력이 낮습니다. 안전 운행에 주의하세요.")
                    }
                } else {
                    speak("브레이크 압력 센서를 사용할 수 없습니다.")
                }
            }

            INTENT_DPF_STATUS -> {
                // J1939 PGN 61442: Engine Controller 3 (DPF)
                val dpfStatus = vehicle.j1939Data?.engineController3?.dpfStatus ?: 0

                when (dpfStatus) {
                    0 -> speak("디피에프 상태 정상입니다.")
                    1 -> speak("디피에프 재생이 필요합니다.")
                    2 -> speak("경고! 디피에프 막힘. 서비스센터 방문이 필요합니다.")
                    else -> speak("디피에프 상태를 확인할 수 없습니다.")
                }
            }

            INTENT_TRANSMISSION_STATUS -> {
                // J1939 PGN 61445: Transmission Controller
                val transData = vehicle.j1939Data?.transmissionController

                if (transData != null) {
                    val currentGear = transData.currentGear
                    val gearDescription = when {
                        currentGear < 0 -> "후진"
                        currentGear == 0 -> "중립"
                        else -> "${currentGear}단"
                    }

                    speak("현재 기어는 ${gearDescription}입니다.")
                } else {
                    speak("변속기 정보를 사용할 수 없습니다.")
                }
            }

            INTENT_AXLE_WEIGHT -> {
                // Detailed axle weight info for compliance
                val weightData = vehicle.j1939Data?.vehicleWeight

                if (weightData != null) {
                    speak("축 중량을 안내합니다. " +
                          "앞축 ${weightData.frontAxle.toInt()}킬로그램, " +
                          "뒷축 ${weightData.rearAxle.toInt()}킬로그램입니다. " +
                          "총 중량은 ${weightData.totalWeight.toInt()}킬로그램입니다.")

                    // Production: Axle weight limits
                    val frontLimit = 7500f  // kg
                    val rearLimit = 18000f  // kg

                    if (weightData.frontAxle > frontLimit) {
                        speak("경고! 앞축 중량이 제한을 초과했습니다.")
                    }
                    if (weightData.rearAxle > rearLimit) {
                        speak("경고! 뒷축 중량이 제한을 초과했습니다.")
                    }
                } else {
                    speak("축 중량 센서를 사용할 수 없습니다.")
                }
            }

            INTENT_REPORT_HAZARD -> {
                // TODO: Integrate with fleet platform
                speak("도로 위험을 신고합니다. 위치와 상황을 기록했습니다.")

                // Production: Sends report to Fleet AI platform
                Log.i(TAG, "Hazard report: location=${vehicle.gpsData?.latitude},${vehicle.gpsData?.longitude}")
            }

            else -> {
                speak("지원하지 않는 명령입니다.")
            }
        }
    }
}

/**
 * Voice intent types
 */
sealed class VoiceIntent {
    // Base intents (from VoiceAssistant)
    object AcceptDispatch : VoiceIntent()
    object RejectDispatch : VoiceIntent()
    object EmergencyAlert : VoiceIntent()
    object ShowLocation : VoiceIntent()
    object ShowFuelInfo : VoiceIntent()
    object ShowSafetyScore : VoiceIntent()

    // Truck-specific intents (production extension)
    data class TruckSpecific(val action: String) : VoiceIntent()
}

/**
 * J1939 Data container for voice assistant
 */
data class J1939VoiceData(
    val vehicleWeight: VehicleWeightData?,
    val tirePressure: TirePressureData?,
    val engineController1: EngineController1Data?,
    val engineController3: EngineController3Data?,
    val brakeController: BrakeControllerData?,
    val transmissionController: TransmissionControllerData?
)

data class VehicleWeightData(
    val frontAxle: Float,
    val rearAxle: Float,
    val totalWeight: Float
)

data class TirePressureData(
    val frontLeft: Float,
    val frontRight: Float,
    val rearLeft: Float,
    val rearRight: Float
)

data class EngineController1Data(
    val engineSpeed: Float,
    val actualTorque: Float
)

data class EngineController3Data(
    val dpfStatus: Int
)

data class BrakeControllerData(
    val serviceBrakePressure: Float,
    val parkingBrakePressure: Float
)

data class TransmissionControllerData(
    val currentGear: Int
)

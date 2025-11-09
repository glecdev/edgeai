package com.glec.driver.api

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * GLEC Driver - External Data Service
 * Fetches real-time weather and traffic data for enhanced driving insights
 *
 * Data sources:
 * - Weather: Korea Meteorological Administration API
 * - Traffic: Korea Transport Database API
 */
class ExternalDataService {

    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .build()

    /**
     * Fetch current weather data
     * API: Korea Meteorological Administration (기상청)
     * https://www.data.go.kr/data/15084084/openapi.do
     */
    suspend fun fetchWeather(latitude: Double, longitude: Double): WeatherData? {
        return withContext(Dispatchers.IO) {
            try {
                // Convert GPS coordinates to grid coordinates (nx, ny)
                val grid = convertToGrid(latitude, longitude)

                // Build API request
                val url = buildString {
                    append("http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst")
                    append("?serviceKey=$WEATHER_API_KEY")
                    append("&numOfRows=60")
                    append("&pageNo=1")
                    append("&dataType=JSON")
                    append("&base_date=${getCurrentDate()}")
                    append("&base_time=${getCurrentHour()}")
                    append("&nx=${grid.first}")
                    append("&ny=${grid.second}")
                }

                val request = Request.Builder()
                    .url(url)
                    .get()
                    .build()

                val response = httpClient.newCall(request).execute()
                if (!response.isSuccessful) {
                    Log.e(TAG, "Weather API request failed: ${response.code}")
                    return@withContext null
                }

                val jsonResponse = response.body?.string()
                if (jsonResponse == null) {
                    Log.e(TAG, "Empty weather API response")
                    return@withContext null
                }

                parseWeatherResponse(jsonResponse)
            } catch (e: IOException) {
                Log.e(TAG, "Error fetching weather data", e)
                null
            } catch (e: Exception) {
                Log.e(TAG, "Unexpected error fetching weather", e)
                null
            }
        }
    }

    /**
     * Parse weather API response
     */
    private fun parseWeatherResponse(jsonString: String): WeatherData? {
        try {
            val json = JSONObject(jsonString)
            val response = json.getJSONObject("response")
            val body = response.getJSONObject("body")
            val items = body.getJSONObject("items").getJSONArray("item")

            var temperature: Float? = null
            var humidity: Int? = null
            var precipitation: Float? = null
            var windSpeed: Float? = null
            var skyCondition: String? = null

            // Parse weather elements
            for (i in 0 until items.length()) {
                val item = items.getJSONObject(i)
                val category = item.getString("category")
                val value = item.getString("fcstValue")

                when (category) {
                    "T1H" -> temperature = value.toFloatOrNull()  // Temperature (°C)
                    "REH" -> humidity = value.toIntOrNull()  // Humidity (%)
                    "RN1" -> precipitation = value.toFloatOrNull()  // Precipitation (mm/h)
                    "WSD" -> windSpeed = value.toFloatOrNull()  // Wind speed (m/s)
                    "SKY" -> skyCondition = when (value) {
                        "1" -> "맑음"
                        "3" -> "구름많음"
                        "4" -> "흐림"
                        else -> "알 수 없음"
                    }
                }
            }

            if (temperature != null && humidity != null) {
                return WeatherData(
                    temperature = temperature,
                    humidity = humidity,
                    precipitation = precipitation ?: 0.0f,
                    windSpeed = windSpeed ?: 0.0f,
                    skyCondition = skyCondition ?: "알 수 없음"
                )
            }

            return null
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing weather response", e)
            return null
        }
    }

    /**
     * Fetch traffic information
     * API: Korea Transport Database (교통정보 API)
     */
    suspend fun fetchTraffic(latitude: Double, longitude: Double, radius: Int = 5000): TrafficData? {
        return withContext(Dispatchers.IO) {
            try {
                // Build API request
                val url = buildString {
                    append("http://openapi.its.go.kr:8081/api/NTrafficApi")
                    append("?key=$TRAFFIC_API_KEY")
                    append("&type=all")
                    append("&minX=${longitude - 0.05}")
                    append("&maxX=${longitude + 0.05}")
                    append("&minY=${latitude - 0.05}")
                    append("&maxY=${latitude + 0.05}")
                    append("&getType=json")
                }

                val request = Request.Builder()
                    .url(url)
                    .get()
                    .build()

                val response = httpClient.newCall(request).execute()
                if (!response.isSuccessful) {
                    Log.e(TAG, "Traffic API request failed: ${response.code}")
                    return@withContext null
                }

                val jsonResponse = response.body?.string()
                if (jsonResponse == null) {
                    Log.e(TAG, "Empty traffic API response")
                    return@withContext null
                }

                parseTrafficResponse(jsonResponse)
            } catch (e: IOException) {
                Log.e(TAG, "Error fetching traffic data", e)
                null
            } catch (e: Exception) {
                Log.e(TAG, "Unexpected error fetching traffic", e)
                null
            }
        }
    }

    /**
     * Parse traffic API response
     */
    private fun parseTrafficResponse(jsonString: String): TrafficData? {
        try {
            val json = JSONObject(jsonString)
            val body = json.getJSONObject("body")
            val items = body.optJSONArray("items")

            if (items == null || items.length() == 0) {
                return TrafficData(
                    congestionLevel = "원활",
                    averageSpeed = 60.0f,
                    travelTime = 0,
                    incidents = emptyList()
                )
            }

            var totalSpeed = 0.0f
            var speedCount = 0
            val incidents = mutableListOf<TrafficIncident>()

            for (i in 0 until items.length()) {
                val item = items.getJSONObject(i)

                // Average speed
                val speed = item.optDouble("speed", 0.0).toFloat()
                if (speed > 0) {
                    totalSpeed += speed
                    speedCount++
                }

                // Traffic incidents
                val incidentType = item.optString("type", "")
                if (incidentType.isNotEmpty()) {
                    incidents.add(
                        TrafficIncident(
                            type = incidentType,
                            location = item.optString("location", "알 수 없음"),
                            severity = item.optString("severity", "보통")
                        )
                    )
                }
            }

            val averageSpeed = if (speedCount > 0) totalSpeed / speedCount else 60.0f

            val congestionLevel = when {
                averageSpeed >= 60 -> "원활"
                averageSpeed >= 40 -> "서행"
                averageSpeed >= 20 -> "정체"
                else -> "심각한 정체"
            }

            return TrafficData(
                congestionLevel = congestionLevel,
                averageSpeed = averageSpeed,
                travelTime = 0,  // Calculate based on route
                incidents = incidents
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing traffic response", e)
            return null
        }
    }

    /**
     * Convert GPS coordinates to grid coordinates (for weather API)
     */
    private fun convertToGrid(latitude: Double, longitude: Double): Pair<Int, Int> {
        // Simplified grid conversion (KMA grid system)
        // Full implementation requires Lambert Conformal Conic projection

        // Approximate conversion for Seoul area
        val nx = ((longitude - 124.0) * 100).toInt()
        val ny = ((latitude - 33.0) * 100).toInt()

        return Pair(nx, ny)
    }

    /**
     * Get current date (YYYYMMDD)
     */
    private fun getCurrentDate(): String {
        val calendar = java.util.Calendar.getInstance()
        return String.format(
            "%04d%02d%02d",
            calendar.get(java.util.Calendar.YEAR),
            calendar.get(java.util.Calendar.MONTH) + 1,
            calendar.get(java.util.Calendar.DAY_OF_MONTH)
        )
    }

    /**
     * Get current hour (HH00)
     */
    private fun getCurrentHour(): String {
        val calendar = java.util.Calendar.getInstance()
        return String.format("%02d00", calendar.get(java.util.Calendar.HOUR_OF_DAY))
    }

    companion object {
        private const val TAG = "ExternalDataService"

        // TODO: Add your API keys from data.go.kr
        private const val WEATHER_API_KEY = "YOUR_WEATHER_API_KEY"
        private const val TRAFFIC_API_KEY = "YOUR_TRAFFIC_API_KEY"
    }
}

/**
 * Weather data model
 */
data class WeatherData(
    val temperature: Float,        // °C
    val humidity: Int,             // %
    val precipitation: Float,      // mm/h
    val windSpeed: Float,          // m/s
    val skyCondition: String       // 맑음, 구름많음, 흐림
) {
    /**
     * Calculate fuel efficiency impact factor
     * Rain, wind, and temperature affect fuel consumption
     */
    fun getFuelImpactFactor(): Float {
        var factor = 1.0f

        // Rain impact (up to +15%)
        if (precipitation > 5.0f) {
            factor += 0.15f
        } else if (precipitation > 1.0f) {
            factor += 0.08f
        }

        // Wind impact (up to +10%)
        if (windSpeed > 10.0f) {
            factor += 0.10f
        } else if (windSpeed > 5.0f) {
            factor += 0.05f
        }

        // Temperature impact (optimal: 15-25°C)
        if (temperature < 0 || temperature > 35) {
            factor += 0.12f
        } else if (temperature < 5 || temperature > 30) {
            factor += 0.06f
        }

        return factor
    }
}

/**
 * Traffic data model
 */
data class TrafficData(
    val congestionLevel: String,   // 원활, 서행, 정체, 심각한 정체
    val averageSpeed: Float,       // km/h
    val travelTime: Int,           // minutes
    val incidents: List<TrafficIncident>
)

/**
 * Traffic incident model
 */
data class TrafficIncident(
    val type: String,              // 사고, 공사, 행사 등
    val location: String,
    val severity: String           // 경미, 보통, 심각
)

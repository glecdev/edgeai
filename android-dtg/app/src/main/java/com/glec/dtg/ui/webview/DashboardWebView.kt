package com.glec.dtg.ui.webview

import android.content.Context
import android.webkit.JavascriptInterface
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.util.Log
import com.glec.dtg.models.CANData
import com.google.gson.Gson
import org.json.JSONObject

/**
 * GLEC DTG - 3D Dashboard WebView
 *
 * Integrates production-verified 3D dashboard from glec-dtg-ai-production
 * Source: dtg_dashboard_volvo_fixed.html (33KB, Three.js + CAN visualization)
 *
 * Features:
 * - 3D truck model rendering (8 Volvo/Hyundai models available)
 * - Real-time telemetry display (speed, RPM, fuel, brake, steering)
 * - AI safety analysis panel
 * - Color-coded risk indicators (green/orange/red)
 * - Voice AI assistant interface
 *
 * Dashboard layout (1280x480):
 * ┌─────────────┬─────────────┬─────────────┐
 * │ Vehicle     │ 3D Model    │ AI Safety   │
 * │ Telemetry   │ + Controls  │ Analysis    │
 * │ (427x464)   │ (427x464)   │ (427x464)   │
 * └─────────────┴─────────────┴─────────────┘
 */
class DashboardWebView(context: Context) : WebView(context) {

    private val gson = Gson()
    private var onDashboardReady: (() -> Unit)? = null

    companion object {
        private const val TAG = "DashboardWebView"

        // Dashboard HTML file (from production)
        private const val DASHBOARD_HTML = "dtg_dashboard_volvo_fixed.html"

        // 3D viewer HTML file (from production)
        private const val VIEWER_3D_HTML = "dtg-3d-viewer.html"
    }

    init {
        setupWebView()
        loadDashboard()
    }

    /**
     * Setup WebView configuration for 3D rendering
     */
    private fun setupWebView() {
        settings.apply {
            // Enable JavaScript (required for Three.js)
            javaScriptEnabled = true

            // Enable DOM storage (for dashboard state)
            domStorageEnabled = true

            // Allow file access (for 3D models)
            allowFileAccess = true
            allowContentAccess = true

            // Enable WebGL (for 3D rendering)
            @Suppress("DEPRECATION")
            setRenderPriority(WebSettings.RenderPriority.HIGH)

            // Enable caching for performance
            cacheMode = WebSettings.LOAD_DEFAULT
            setAppCacheEnabled(true)

            // Disable zoom controls (fixed layout)
            setSupportZoom(false)
            builtInZoomControls = false

            // Enable hardware acceleration
            setLayerType(LAYER_TYPE_HARDWARE, null)

            // Production optimization: Force GPU rendering
            mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
        }

        // JavaScript interface for bidirectional communication
        addJavascriptInterface(DashboardBridge(), "AndroidBridge")

        // WebView client for page lifecycle
        webViewClient = object : WebViewClient() {
            override fun onPageFinished(view: WebView?, url: String?) {
                super.onPageFinished(view, url)
                Log.d(TAG, "Dashboard loaded: $url")

                // Notify dashboard is ready
                onDashboardReady?.invoke()

                // Initialize dashboard with current theme
                evaluateJavascript("initializeDashboard();", null)
            }

            override fun onReceivedError(
                view: WebView?,
                errorCode: Int,
                description: String?,
                failingUrl: String?
            ) {
                super.onReceivedError(view, errorCode, description, failingUrl)
                Log.e(TAG, "WebView error: $description (code: $errorCode)")
            }
        }

        // Chrome client for console logging (debug)
        webChromeClient = object : WebChromeClient() {
            override fun onConsoleMessage(message: android.webkit.ConsoleMessage?): Boolean {
                message?.let {
                    Log.d(TAG, "JS Console: ${it.message()} (${it.sourceId()}:${it.lineNumber()})")
                }
                return true
            }
        }
    }

    /**
     * Load production-verified dashboard
     */
    private fun loadDashboard() {
        // Load from assets
        loadUrl("file:///android_asset/$DASHBOARD_HTML")
    }

    /**
     * Load alternative 3D viewer
     */
    fun load3DViewer() {
        loadUrl("file:///android_asset/$VIEWER_3D_HTML")
    }

    /**
     * Set callback for dashboard ready event
     */
    fun setOnDashboardReady(callback: () -> Unit) {
        onDashboardReady = callback
    }

    /**
     * Update dashboard with latest CAN data
     *
     * Production format matching dtg_dashboard_volvo_fixed.html
     */
    fun updateVehicleData(canData: CANData) {
        val jsonData = gson.toJson(canData)

        val jsCode = """
            updateDashboard({
                // Vehicle telemetry (left panel)
                speed: ${canData.vehicleSpeed},
                rpm: ${canData.engineRPM},
                fuel: ${canData.fuelLevel},
                brakeForce: ${canData.brakePosition},
                steeringAngle: ${canData.steeringAngle ?: 0.0f},
                throttle: ${canData.throttlePosition},

                // IMU data (for 3D model animation)
                acceleration: {
                    x: ${canData.accelerationX},
                    y: ${canData.accelerationY},
                    z: ${canData.accelerationZ}
                },
                gyro: {
                    x: ${canData.gyroX},
                    y: ${canData.gyroY},
                    z: ${canData.gyroZ}
                },

                // Additional metrics
                coolantTemp: ${canData.coolantTemp},
                batteryVoltage: ${canData.batteryVoltage},
                timestamp: ${canData.timestamp}
            });
        """.trimIndent()

        post {
            evaluateJavascript(jsCode) { result ->
                Log.v(TAG, "Dashboard updated: $result")
            }
        }
    }

    /**
     * Update AI analysis results
     *
     * Production format for AI safety panel (right panel)
     */
    fun updateAIResults(results: AIAnalysisResult) {
        val jsCode = """
            updateAIAnalysis({
                // Safety metrics
                safetyScore: ${results.safetyScore},
                riskLevel: '${results.riskLevel}',  // 'safe', 'caution', 'danger'

                // Fuel efficiency
                fuelEfficiency: ${results.fuelEfficiency},
                predictedRange: ${results.predictedRange},

                // Behavior classification
                drivingBehavior: '${results.drivingBehavior}',  // 'normal', 'eco', 'aggressive', 'dangerous'

                // Anomaly detection
                anomalies: ${gson.toJson(results.anomalies)},

                // Processing time
                processingTime: ${results.processingTimeMs},

                // Recommendations
                recommendations: ${gson.toJson(results.recommendations)}
            });
        """.trimIndent()

        post {
            evaluateJavascript(jsCode) { result ->
                Log.v(TAG, "AI results updated: $result")
            }
        }
    }

    /**
     * Update J1939 commercial vehicle data
     *
     * Production extension for commercial vehicle dashboard
     */
    fun updateJ1939Data(
        engineTorque: Float? = null,
        cargoWeight: Float? = null,
        tirePressure: TirePressureData? = null,
        gearInfo: GearInfo? = null
    ) {
        val jsCode = """
            updateCommercialData({
                engineTorque: ${engineTorque ?: "null"},
                cargoWeight: ${cargoWeight ?: "null"},
                tirePressure: ${tirePressure?.let { gson.toJson(it) } ?: "null"},
                gearInfo: ${gearInfo?.let { gson.toJson(it) } ?: "null"}
            });
        """.trimIndent()

        post {
            evaluateJavascript(jsCode) { result ->
                Log.v(TAG, "J1939 data updated: $result")
            }
        }
    }

    /**
     * Select 3D truck model
     *
     * Production models available:
     * - volvo_truck_1.glb (228KB)
     * - volvo_truck_2.glb (1.0MB)
     * - hyundai_porter.glb (5.7MB)
     * - [5 additional variants]
     */
    fun selectTruckModel(modelName: String) {
        val jsCode = "selectTruckModel('$modelName');"

        post {
            evaluateJavascript(jsCode) { result ->
                Log.d(TAG, "Truck model selected: $modelName")
            }
        }
    }

    /**
     * Set dashboard theme
     */
    fun setTheme(isDark: Boolean) {
        val theme = if (isDark) "dark" else "light"
        val jsCode = "setTheme('$theme');"

        post {
            evaluateJavascript(jsCode, null)
        }
    }

    /**
     * JavaScript bridge for dashboard callbacks
     *
     * Production: Bidirectional communication between WebView and Android
     */
    inner class DashboardBridge {

        @JavascriptInterface
        fun onUserAction(action: String, data: String?) {
            Log.d(TAG, "User action: $action, data: $data")

            // Handle user interactions from dashboard
            // (e.g., button clicks, model selection, settings)
            when (action) {
                "MODEL_SELECTED" -> {
                    // User selected different truck model
                    Log.i(TAG, "User selected model: $data")
                }

                "RESET_TRIP" -> {
                    // User reset trip data
                    Log.i(TAG, "Trip data reset")
                }

                "TOGGLE_VIEW" -> {
                    // User toggled between dashboard views
                    Log.i(TAG, "View toggled: $data")
                }

                else -> {
                    Log.w(TAG, "Unknown action: $action")
                }
            }
        }

        @JavascriptInterface
        fun log(level: String, message: String) {
            when (level) {
                "debug" -> Log.d(TAG, "JS: $message")
                "info" -> Log.i(TAG, "JS: $message")
                "warn" -> Log.w(TAG, "JS: $message")
                "error" -> Log.e(TAG, "JS: $message")
                else -> Log.v(TAG, "JS: $message")
            }
        }

        @JavascriptInterface
        fun getDeviceInfo(): String {
            val deviceInfo = mapOf(
                "model" to android.os.Build.MODEL,
                "manufacturer" to android.os.Build.MANUFACTURER,
                "androidVersion" to android.os.Build.VERSION.RELEASE,
                "sdkVersion" to android.os.Build.VERSION.SDK_INT
            )
            return gson.toJson(deviceInfo)
        }
    }

    /**
     * AI Analysis Result data structure
     */
    data class AIAnalysisResult(
        val safetyScore: Int,              // 0-100
        val riskLevel: RiskLevel,          // safe, caution, danger
        val fuelEfficiency: Float,         // km/L
        val predictedRange: Float,         // km
        val drivingBehavior: DrivingBehavior,  // normal, eco, aggressive, dangerous
        val anomalies: List<String>,       // Detected anomalies
        val processingTimeMs: Long,        // AI inference latency
        val recommendations: List<String>  // Driving recommendations
    )

    enum class RiskLevel {
        safe, caution, danger
    }

    enum class DrivingBehavior {
        normal, eco, aggressive, dangerous
    }

    /**
     * Tire pressure data (TPMS)
     */
    data class TirePressureData(
        val frontLeft: Float,   // bar
        val frontRight: Float,  // bar
        val rearLeft: Float,    // bar
        val rearRight: Float    // bar
    )

    /**
     * Gear information (transmission)
     */
    data class GearInfo(
        val current: Int,       // Current gear (-1=reverse, 0=neutral, 1+=forward)
        val selected: Int,      // Selected gear
        val mode: String        // "manual", "automatic", "eco"
    )
}

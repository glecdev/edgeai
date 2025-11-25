package com.glec.dtg.ui.truck3d

import android.annotation.SuppressLint
import android.util.Log
import android.webkit.JavascriptInterface
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import com.glec.dtg.models.CANData
import com.google.gson.Gson

/**
 * GLEC DTG 3D Truck Viewer Composable
 *
 * Features:
 * - Three.js 3D truck visualization in WebView
 * - Real-time vehicle data updates via JavaScript bridge
 * - Diagnostic code-based fault highlighting
 * - Camera control integration
 *
 * Usage:
 * ```kotlin
 * val viewerController = remember { Truck3DViewerController() }
 *
 * Truck3DViewer(
 *     controller = viewerController,
 *     modifier = Modifier.fillMaxSize()
 * )
 *
 * // Update vehicle data
 * LaunchedEffect(canData) {
 *     viewerController.updateVehicleData(canData)
 * }
 *
 * // Highlight fault
 * viewerController.highlightFault("P0118", "엔진 냉각수 온도 센서 이상")
 * ```
 */
@SuppressLint("SetJavaScriptEnabled")
@Composable
fun Truck3DViewer(
    controller: Truck3DViewerController,
    modifier: Modifier = Modifier
) {
    var webView: WebView? by remember { mutableStateOf(null) }

    // Link webView to controller
    LaunchedEffect(webView) {
        webView?.let { controller.attachWebView(it) }
    }

    AndroidView(
        modifier = modifier,
        factory = { context ->
            WebView(context).apply {
                webViewClient = WebViewClient()
                settings.apply {
                    javaScriptEnabled = true
                    domStorageEnabled = true
                    databaseEnabled = true
                    allowFileAccess = true
                    allowContentAccess = true
                    mediaPlaybackRequiresUserGesture = false
                }

                // Add JavaScript interface for Kotlin -> JS communication
                addJavascriptInterface(
                    Truck3DJavaScriptInterface(controller),
                    "AndroidBridge"
                )

                // Load Three.js viewer from assets
                loadUrl("file:///android_asset/truck-viewer/index.html")

                webView = this

                Log.d(TAG, "Truck3DViewer: WebView initialized")
            }
        },
        update = { view ->
            // No-op, updates handled via controller
        }
    )
}

/**
 * Controller for Truck3DViewer
 *
 * Provides methods to update vehicle data, highlight faults, and control camera.
 */
class Truck3DViewerController {
    private var webView: WebView? = null
    private val gson = Gson()

    internal fun attachWebView(view: WebView) {
        webView = view
        Log.d(TAG, "Truck3DViewerController: WebView attached")
    }

    /**
     * Update vehicle data in 3D viewer
     *
     * @param data CAN data containing vehicle metrics
     */
    fun updateVehicleData(data: CANData) {
        val vehicleData = VehicleDataJS(
            speed = data.vehicleSpeed.toInt(),
            rpm = data.engineRPM,
            fuel = data.fuelLevel.toInt(),
            temp = data.coolantTemp.toInt(),
            load = data.engineLoad.toInt()
        )

        executeJavaScript("updateVehicleData('${gson.toJson(vehicleData)}')")
        Log.d(TAG, "updateVehicleData: $vehicleData")
    }

    /**
     * Highlight a faulty part in the 3D model (red alert)
     *
     * @param diagnosticCode OBD-II/J1939 diagnostic code (e.g., "P0118")
     * @param description Human-readable fault description
     */
    fun highlightFault(diagnosticCode: String, description: String) {
        executeJavaScript("highlightFault('$diagnosticCode', '$description')")
        Log.d(TAG, "highlightFault: $diagnosticCode - $description")
    }

    /**
     * Clear all fault highlights
     */
    fun clearFaults() {
        executeJavaScript("clearFaults()")
        Log.d(TAG, "clearFaults: All faults cleared")
    }

    /**
     * Set camera view preset
     *
     * @param viewType One of: "perspective", "front", "side", "top"
     */
    fun setCameraView(viewType: CameraViewType) {
        executeJavaScript("setCameraView('${viewType.value}')")
        Log.d(TAG, "setCameraView: ${viewType.value}")
    }

    /**
     * Execute JavaScript code in WebView
     */
    private fun executeJavaScript(script: String) {
        webView?.post {
            webView?.evaluateJavascript(script) { result ->
                if (result != null && result != "null") {
                    Log.d(TAG, "JavaScript result: $result")
                }
            }
        }
    }
}

/**
 * JavaScript interface for bidirectional communication
 *
 * Currently used for JS -> Kotlin callbacks (if needed in future)
 */
private class Truck3DJavaScriptInterface(
    private val controller: Truck3DViewerController
) {
    @JavascriptInterface
    fun onViewerReady() {
        Log.d(TAG, "JavaScript: Viewer ready")
    }

    @JavascriptInterface
    fun onPartClicked(partName: String) {
        Log.d(TAG, "JavaScript: Part clicked - $partName")
        // TODO: Handle part click (e.g., show detailed info)
    }

    @JavascriptInterface
    fun logError(message: String) {
        Log.e(TAG, "JavaScript Error: $message")
    }
}

/**
 * Vehicle data model for JavaScript bridge
 */
private data class VehicleDataJS(
    val speed: Int,      // km/h
    val rpm: Int,        // RPM
    val fuel: Int,       // percentage
    val temp: Int,       // °C
    val load: Int        // kg or percentage
)

/**
 * Camera view types
 */
enum class CameraViewType(val value: String) {
    PERSPECTIVE("perspective"),
    FRONT("front"),
    SIDE("side"),
    TOP("top")
}

private const val TAG = "Truck3DViewer"

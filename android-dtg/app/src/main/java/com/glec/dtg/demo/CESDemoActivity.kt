package com.glec.dtg.demo

import android.os.Bundle
import android.util.Log
import android.view.ViewGroup
import android.webkit.JavascriptInterface
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.FrameLayout
import androidx.appcompat.app.AppCompatActivity
import com.glec.dtg.R

/**
 * CES Demo Activity
 *
 * Main activity for CES exhibition demo:
 * - Displays 3D truck viewer with WebView
 * - Plays 150-second ultimate demo loop
 * - Shows auto-dismiss alerts (5 seconds)
 * - Handles voice AI interactions
 * - Continuous loop for booth display
 *
 * Setup:
 * 1. 1280x480 horizontal screen (truck dashboard size)
 * 2. Fullscreen immersive mode
 * 3. Auto-start on boot (CES booth setup)
 *
 * @see CES_DEMO_VIEWER_OPTIMIZED.md
 */
class CESDemoActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "CESDemoActivity"
        private const val DEMO_ASSET_PATH = "demo-ultimate-150s.json"
        private const val DASHBOARD_HTML = "file:///android_asset/dtg_dashboard_ces.html"
    }

    // Core components
    private lateinit var webView: WebView
    private lateinit var alertContainer: FrameLayout
    private lateinit var alertManager: AlertManager
    private lateinit var demoEngine: DemoPlaybackEngine
    private lateinit var voiceBridge: DemoVoiceBridge

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_ces_demo)

        // Setup fullscreen immersive mode
        setupFullscreen()

        // Initialize components
        initializeWebView()
        initializeAlertManager()
        initializeVoiceBridge()
        initializeDemoEngine()

        // Load and start demo
        loadDemo()
    }

    /**
     * Setup fullscreen immersive mode for CES booth
     */
    private fun setupFullscreen() {
        window.decorView.systemUiVisibility = (
            android.view.View.SYSTEM_UI_FLAG_FULLSCREEN
                or android.view.View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                or android.view.View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
            )

        supportActionBar?.hide()
    }

    /**
     * Initialize WebView for 3D dashboard
     */
    private fun initializeWebView() {
        webView = findViewById(R.id.webView)
        alertContainer = findViewById(R.id.alertContainer)

        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            allowFileAccess = true
            allowContentAccess = true
        }

        // Add JavaScript interface for bidirectional communication
        webView.addJavascriptInterface(WebAppInterface(), "Android")

        // Load dashboard HTML
        webView.webViewClient = object : WebViewClient() {
            override fun onPageFinished(view: WebView?, url: String?) {
                super.onPageFinished(view, url)
                Log.d(TAG, "Dashboard loaded, starting demo in 2 seconds...")

                // Start demo after dashboard loads
                webView.postDelayed({
                    startDemo()
                }, 2000)
            }
        }

        webView.loadUrl(DASHBOARD_HTML)
    }

    /**
     * Initialize alert manager
     */
    private fun initializeAlertManager() {
        alertManager = AlertManager(this)
        alertManager.setAlertContainer(alertContainer)
    }

    /**
     * Initialize voice bridge for TTS
     */
    private fun initializeVoiceBridge() {
        voiceBridge = DemoVoiceBridge(this)
        voiceBridge.initialize {
            Log.d(TAG, "Voice bridge ready")
        }
    }

    /**
     * Initialize demo playback engine
     */
    private fun initializeDemoEngine() {
        demoEngine = DemoPlaybackEngine(
            context = this,
            alertManager = alertManager,
            webView = webView
        )
    }

    /**
     * Load demo timeline
     */
    private fun loadDemo() {
        val success = demoEngine.loadDemo(DEMO_ASSET_PATH)

        if (!success) {
            Log.e(TAG, "Failed to load demo!")
            // Show error alert
            showErrorAlert("Demo 파일을 로드하지 못했습니다")
        }
    }

    /**
     * Start demo playback
     */
    private fun startDemo() {
        Log.d(TAG, "Starting CES demo playback...")
        demoEngine.start()
    }

    /**
     * Pause demo (for debugging)
     */
    private fun pauseDemo() {
        demoEngine.pause()
    }

    /**
     * Resume demo
     */
    private fun resumeDemo() {
        demoEngine.resume()
    }

    /**
     * Show error alert
     */
    private fun showErrorAlert(message: String) {
        val alert = AlertManager.Alert(
            id = "error_${System.currentTimeMillis()}",
            type = AlertManager.AlertType.WARNING,
            title = "오류",
            message = message,
            severity = AlertManager.Severity.HIGH,
            autoDismissSeconds = 10
        )

        alertManager.showAlert(alert)
    }

    /**
     * JavaScript interface for WebView communication
     */
    inner class WebAppInterface {

        /**
         * Called from JavaScript when user interacts with dashboard
         */
        @JavascriptInterface
        fun onDashboardClick(action: String) {
            Log.d(TAG, "Dashboard action: $action")

            when (action) {
                "pause" -> pauseDemo()
                "resume" -> resumeDemo()
                "restart" -> {
                    demoEngine.stop()
                    startDemo()
                }
            }
        }

        /**
         * Called from JavaScript to log messages
         */
        @JavascriptInterface
        fun log(message: String) {
            Log.d(TAG, "WebView: $message")
        }

        /**
         * Get current demo progress (0-100)
         */
        @JavascriptInterface
        fun getDemoProgress(): Float {
            return demoEngine.getProgress() * 100
        }

        /**
         * Speak text via TTS (called from JavaScript)
         */
        @JavascriptInterface
        fun speakText(text: String) {
            voiceBridge.speak(text, DemoVoiceBridge.SpeechPriority.NORMAL)
        }
    }

    override fun onResume() {
        super.onResume()
        // Keep screen on during CES demo
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    override fun onPause() {
        super.onPause()
        // Pause demo when activity paused
        demoEngine.pause()
    }

    override fun onDestroy() {
        super.onDestroy()
        // Cleanup resources
        demoEngine.cleanup()
        alertManager.cleanup()
        voiceBridge.shutdown()
    }

    override fun onBackPressed() {
        // Disable back button for CES booth (kiosk mode)
        // Comment this line for development/debugging
        // super.onBackPressed()
        Log.d(TAG, "Back button disabled for CES kiosk mode")
    }
}

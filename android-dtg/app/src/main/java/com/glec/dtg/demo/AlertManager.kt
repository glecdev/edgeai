package com.glec.dtg.demo

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Color
import android.os.CountDownTimer
import android.os.Handler
import android.os.Looper
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.cardview.widget.CardView
import com.glec.dtg.R
import java.util.concurrent.ConcurrentHashMap

/**
 * CES Demo Alert Manager with 5-Second Auto-Dismiss
 *
 * Features:
 * - Auto-dismiss after 5 seconds
 * - Countdown UI with circular numbers (⑤ → ④ → ③ → ② → ①)
 * - Color-coded countdown (green → yellow → red)
 * - Fade-in/fade-out animations (0.3s)
 * - No user interaction required
 *
 * @see CES_DEMO_VIEWER_OPTIMIZED.md
 */
class AlertManager(private val context: Context) {

    companion object {
        private const val AUTO_DISMISS_SECONDS = 5
        private const val FADE_DURATION_MS = 300L

        private val CIRCLED_NUMBERS = mapOf(
            5 to "⑤",
            4 to "④",
            3 to "③",
            2 to "②",
            1 to "①"
        )

        private val COLOR_GREEN = Color.parseColor("#4CAF50")
        private val COLOR_YELLOW = Color.parseColor("#FFD700")
        private val COLOR_RED = Color.parseColor("#FF5252")
        private val COLOR_GRAY = Color.GRAY
    }

    data class Alert(
        val id: String,
        val type: AlertType,
        val title: String,
        val message: String,
        val severity: Severity,
        val timestamp: Long = System.currentTimeMillis(),
        val autoDismissSeconds: Int = AUTO_DISMISS_SECONDS,
        val voiceAlert: String? = null,
        val icon: AlertIcon = AlertIcon.WARNING
    )

    enum class AlertType {
        HARSH_ACCELERATION,
        HARSH_BRAKING,
        SPEEDING,
        DROWSY_DRIVING,
        FUEL_EFFICIENCY,
        TIRE_PRESSURE,
        MANDATORY_REST,
        ROUTE_OPTIMIZATION,
        OIL_CHANGE,
        FATIGUE_DETECTION
    }

    enum class Severity {
        LOW,      // Green (#4CAF50)
        MEDIUM,   // Yellow (#FFD700)
        HIGH,     // Red (#FF5252)
        CRITICAL  // Deep Red (#D32F2F)
    }

    enum class AlertIcon {
        WARNING,
        DANGER,
        INFO,
        SUCCESS,
        MAINTENANCE
    }

    // Active alerts and their timers
    private val activeAlerts = ConcurrentHashMap<String, CountDownTimer>()
    private val handler = Handler(Looper.getMainLooper())

    // Alert view container (injected from MainActivity)
    private var alertContainer: ViewGroup? = null

    /**
     * Set the alert container where alerts will be displayed
     */
    fun setAlertContainer(container: ViewGroup) {
        this.alertContainer = container
    }

    /**
     * Show alert with auto-dismiss countdown
     */
    fun showAlert(alert: Alert) {
        handler.post {
            val container = alertContainer ?: run {
                android.util.Log.e("AlertManager", "Alert container not set!")
                return@post
            }

            // Create alert view
            val alertView = createAlertView(alert)

            // Add to container
            container.addView(alertView)

            // Fade in animation
            alertView.alpha = 0f
            alertView.animate()
                .alpha(1f)
                .setDuration(FADE_DURATION_MS)
                .start()

            // Start countdown
            startCountdown(alert, alertView)
        }
    }

    /**
     * Create alert card view
     */
    private fun createAlertView(alert: Alert): View {
        val inflater = LayoutInflater.from(context)
        val view = inflater.inflate(R.layout.alert_card, null, false)

        // Set alert content
        view.findViewById<TextView>(R.id.alertTitle)?.text = alert.title
        view.findViewById<TextView>(R.id.alertMessage)?.text = alert.message

        // Set alert icon
        val iconView = view.findViewById<ImageView>(R.id.alertIcon)
        iconView?.setImageResource(getIconResource(alert.icon))

        // Set severity color
        val cardView = view.findViewById<CardView>(R.id.alertCard)
        cardView?.setCardBackgroundColor(getSeverityColor(alert.severity))

        // Store alert ID
        view.tag = alert.id

        return view
    }

    /**
     * Start 5-second countdown with visual feedback
     */
    private fun startCountdown(alert: Alert, alertView: View) {
        val countdownView = alertView.findViewById<TextView>(R.id.countdown)

        val timer = object : CountDownTimer(
            (alert.autoDismissSeconds * 1000).toLong(),
            1000L
        ) {
            override fun onTick(millisUntilFinished: Long) {
                val secondsRemaining = (millisUntilFinished / 1000).toInt() + 1

                // Update countdown number
                val circledNumber = CIRCLED_NUMBERS[secondsRemaining] ?: ""
                countdownView?.text = circledNumber

                // Update countdown color
                countdownView?.setTextColor(getCountdownColor(secondsRemaining))

                // Optional: pulse animation
                animateCountdown(countdownView)
            }

            override fun onFinish() {
                // Auto-dismiss
                dismissAlert(alert.id, alertView)
            }
        }

        activeAlerts[alert.id] = timer
        timer.start()
    }

    /**
     * Get color based on countdown (green → yellow → red)
     */
    private fun getCountdownColor(secondsRemaining: Int): Int {
        return when (secondsRemaining) {
            5, 4 -> COLOR_GREEN   // Green
            3 -> COLOR_YELLOW     // Yellow
            2, 1 -> COLOR_RED     // Red
            else -> COLOR_GRAY
        }
    }

    /**
     * Animate countdown number (subtle pulse)
     */
    private fun animateCountdown(view: TextView?) {
        view ?: return

        val animator = ValueAnimator.ofFloat(1f, 1.2f, 1f)
        animator.duration = 300
        animator.addUpdateListener { animation ->
            val scale = animation.animatedValue as Float
            view.scaleX = scale
            view.scaleY = scale
        }
        animator.start()
    }

    /**
     * Dismiss alert with fade-out animation
     */
    fun dismissAlert(alertId: String, alertView: View? = null) {
        handler.post {
            // Cancel timer
            activeAlerts[alertId]?.cancel()
            activeAlerts.remove(alertId)

            // Find view if not provided
            val view = alertView ?: alertContainer?.findViewWithTag<View>(alertId)
            view ?: return@post

            // Fade out animation
            view.animate()
                .alpha(0f)
                .setDuration(FADE_DURATION_MS)
                .withEndAction {
                    // Remove from container
                    (view.parent as? ViewGroup)?.removeView(view)
                }
                .start()
        }
    }

    /**
     * Dismiss all active alerts
     */
    fun dismissAllAlerts() {
        val alertIds = activeAlerts.keys.toList()
        alertIds.forEach { alertId ->
            dismissAlert(alertId)
        }
    }

    /**
     * Get icon resource based on alert type
     */
    private fun getIconResource(icon: AlertIcon): Int {
        return when (icon) {
            AlertIcon.WARNING -> android.R.drawable.ic_dialog_alert
            AlertIcon.DANGER -> android.R.drawable.ic_delete
            AlertIcon.INFO -> android.R.drawable.ic_dialog_info
            AlertIcon.SUCCESS -> android.R.drawable.checkbox_on_background
            AlertIcon.MAINTENANCE -> android.R.drawable.ic_menu_preferences
        }
    }

    /**
     * Get severity color
     */
    private fun getSeverityColor(severity: Severity): Int {
        return when (severity) {
            Severity.LOW -> Color.parseColor("#E8F5E9")      // Light green
            Severity.MEDIUM -> Color.parseColor("#FFF9C4")   // Light yellow
            Severity.HIGH -> Color.parseColor("#FFEBEE")     // Light red
            Severity.CRITICAL -> Color.parseColor("#FFCDD2") // Deeper red
        }
    }

    /**
     * Cleanup resources
     */
    fun cleanup() {
        dismissAllAlerts()
        alertContainer = null
    }
}

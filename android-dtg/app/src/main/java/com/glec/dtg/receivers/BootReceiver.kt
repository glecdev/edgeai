package com.glec.dtg.receivers

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.Build
import android.util.Log
import com.glec.dtg.services.DTGForegroundService

/**
 * GLEC DTG - Boot Receiver
 * Automatically starts DTG service when device boots
 *
 * Required permissions in AndroidManifest.xml:
 * - RECEIVE_BOOT_COMPLETED
 * - FOREGROUND_SERVICE
 */
class BootReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == Intent.ACTION_BOOT_COMPLETED) {
            Log.i(TAG, "Device boot completed, starting DTG service...")

            try {
                val serviceIntent = Intent(context, DTGForegroundService::class.java).apply {
                    action = DTGForegroundService.ACTION_START_SERVICE
                }

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    context.startForegroundService(serviceIntent)
                } else {
                    context.startService(serviceIntent)
                }

                Log.i(TAG, "DTG service start requested")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start DTG service on boot", e)
            }
        }
    }

    companion object {
        private const val TAG = "BootReceiver"
    }
}

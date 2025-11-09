package com.glec.dtg

import android.app.Application
import timber.log.Timber

/**
 * GLEC DTG Application
 * Main application class
 */
class DTGApplication : Application() {

    override fun onCreate() {
        super.onCreate()

        // Initialize Timber logging
        if (BuildConfig.DEBUG) {
            Timber.plant(Timber.DebugTree())
        }

        Timber.i("DTG Application started")

        // TODO: Initialize DVC, MLflow clients if needed
        // TODO: Initialize SNPE runtime
    }
}

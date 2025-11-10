package com.glec.dtg.mqtt

import android.util.Base64
import android.util.Log
import java.security.MessageDigest
import java.security.cert.X509Certificate
import javax.net.ssl.X509TrustManager

/**
 * Certificate Pinning for MQTT TLS
 *
 * Validates server certificates against pre-configured SHA-256 pins.
 * Provides additional security layer beyond standard CA validation.
 *
 * Usage:
 * ```kotlin
 * val pinner = CertificatePinner(
 *     hostname = "mqtt.fleet.glec.co.kr",
 *     pins = listOf(
 *         "sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
 *         "sha256/BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB="  // Backup pin
 *     )
 * )
 *
 * val trustManager = PinningTrustManager(baseTrustManager, pinner)
 * ```
 *
 * @property hostname The hostname to validate pins against
 * @property pins List of SHA-256 certificate pins (format: "sha256/base64")
 *
 * @author GLEC DTG Team
 * @since Phase 3D
 */
data class CertificatePinner(
    val hostname: String,
    val pins: List<String>
) {
    companion object {
        private const val TAG = "CertificatePinner"

        /**
         * Calculate SHA-256 pin for a certificate
         *
         * @param cert X.509 certificate
         * @return SHA-256 pin in format "sha256/base64"
         */
        fun calculatePin(cert: X509Certificate): String {
            val publicKey = cert.publicKey.encoded
            val digest = MessageDigest.getInstance("SHA-256")
            val hash = digest.digest(publicKey)
            val base64 = Base64.encodeToString(hash, Base64.NO_WRAP)
            return "sha256/$base64"
        }
    }

    /**
     * Validate certificate chain against pins
     *
     * @param chain Certificate chain from server
     * @return true if any certificate matches a pin, false otherwise
     */
    fun validate(chain: Array<X509Certificate>): Boolean {
        if (pins.isEmpty()) {
            // No pins configured, skip pinning
            return true
        }

        // Calculate pins for all certificates in chain
        val certificatePins = chain.map { calculatePin(it) }

        Log.d(TAG, "Validating certificate chain for $hostname")
        Log.d(TAG, "Expected pins: $pins")
        Log.d(TAG, "Certificate pins: $certificatePins")

        // Check if any certificate matches any configured pin
        for (certPin in certificatePins) {
            if (pins.contains(certPin)) {
                Log.i(TAG, "✅ Certificate pin matched: $certPin")
                return true
            }
        }

        Log.e(TAG, "❌ Certificate pinning failed for $hostname")
        Log.e(TAG, "None of the pins matched. Expected: $pins, Got: $certificatePins")
        return false
    }

    /**
     * Validate configuration
     */
    fun isValid(): Boolean {
        if (hostname.isBlank()) {
            return false
        }

        // Check pin format
        for (pin in pins) {
            if (!pin.startsWith("sha256/") || pin.length < 15) {
                return false
            }
        }

        return true
    }
}

/**
 * X509TrustManager with Certificate Pinning
 *
 * Wraps a base TrustManager and adds certificate pinning validation.
 *
 * @property baseTrustManager The base X509TrustManager for standard validation
 * @property pinner The CertificatePinner for pin validation
 */
class PinningTrustManager(
    private val baseTrustManager: X509TrustManager,
    private val pinner: CertificatePinner
) : X509TrustManager {

    companion object {
        private const val TAG = "PinningTrustManager"
    }

    /**
     * Check server certificate chain
     *
     * Performs two-step validation:
     * 1. Standard CA validation (via baseTrustManager)
     * 2. Certificate pinning validation
     */
    override fun checkServerTrusted(chain: Array<X509Certificate>, authType: String) {
        // Step 1: Standard CA validation
        try {
            baseTrustManager.checkServerTrusted(chain, authType)
            Log.d(TAG, "✅ CA validation passed")
        } catch (e: Exception) {
            Log.e(TAG, "❌ CA validation failed", e)
            throw e
        }

        // Step 2: Certificate pinning validation
        if (!pinner.validate(chain)) {
            throw javax.net.ssl.SSLPeerUnverifiedException(
                "Certificate pinning failed for ${pinner.hostname}. " +
                "Expected pins: ${pinner.pins}, " +
                "but got: ${chain.map { CertificatePinner.calculatePin(it) }}"
            )
        }

        Log.d(TAG, "✅ Certificate pinning passed for ${pinner.hostname}")
    }

    override fun checkClientTrusted(chain: Array<X509Certificate>, authType: String) {
        baseTrustManager.checkClientTrusted(chain, authType)
    }

    override fun getAcceptedIssuers(): Array<X509Certificate> {
        return baseTrustManager.acceptedIssuers
    }
}

/**
 * Certificate Pinner Builder
 *
 * Fluent API for building certificate pinners with multiple hosts.
 *
 * Usage:
 * ```kotlin
 * val pinners = CertificatePinnerBuilder()
 *     .add("mqtt.fleet.glec.co.kr", "sha256/AAAA...=", "sha256/BBBB...=")
 *     .add("backup.fleet.glec.co.kr", "sha256/CCCC...=")
 *     .build()
 * ```
 */
class CertificatePinnerBuilder {
    private val pinners = mutableMapOf<String, CertificatePinner>()

    /**
     * Add certificate pins for a hostname
     *
     * @param hostname The hostname
     * @param pins Variable number of SHA-256 pins
     * @return this builder for chaining
     */
    fun add(hostname: String, vararg pins: String): CertificatePinnerBuilder {
        pinners[hostname] = CertificatePinner(hostname, pins.toList())
        return this
    }

    /**
     * Build map of hostname to CertificatePinner
     *
     * @return Map of hostname to pinner
     */
    fun build(): Map<String, CertificatePinner> {
        // Validate all pinners
        for ((hostname, pinner) in pinners) {
            if (!pinner.isValid()) {
                throw IllegalArgumentException("Invalid pinner for hostname: $hostname")
            }
        }
        return pinners.toMap()
    }
}

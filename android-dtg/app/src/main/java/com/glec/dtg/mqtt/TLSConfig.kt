package com.glec.dtg.mqtt

import java.io.InputStream

/**
 * TLS/SSL Configuration for MQTT
 *
 * Secure MQTT connection with TLS 1.2+ encryption and certificate pinning.
 *
 * Usage:
 * ```kotlin
 * val tlsConfig = TLSConfig(
 *     caCertInputStream = assets.open("mqtt_ca.crt"),
 *     tlsVersion = "TLSv1.2",
 *     certificatePins = listOf("sha256/AAAA...=")
 * )
 * ```
 *
 * @property caCertInputStream CA certificate input stream (required)
 * @property clientCertInputStream Client certificate input stream (optional, for mTLS)
 * @property clientKeyInputStream Client private key input stream (optional, for mTLS)
 * @property tlsVersion TLS version ("TLSv1.2" or "TLSv1.3")
 * @property cipherSuites Allowed cipher suites (null = default secure suites)
 * @property certificatePins SHA-256 certificate pins for certificate pinning
 * @property hostnameVerificationEnabled Enable hostname verification (default: true)
 *
 * @author GLEC DTG Team
 * @since Phase 3D
 */
data class TLSConfig(
    val caCertInputStream: InputStream,
    val clientCertInputStream: InputStream? = null,
    val clientKeyInputStream: InputStream? = null,
    val tlsVersion: String = "TLSv1.2",
    val cipherSuites: List<String>? = null,
    val certificatePins: List<String> = emptyList(),
    val hostnameVerificationEnabled: Boolean = true
) {
    companion object {
        /**
         * Recommended secure cipher suites for TLS 1.2+
         *
         * Prioritizes:
         * - ECDHE (Perfect Forward Secrecy)
         * - AES-GCM (Authenticated encryption)
         * - SHA256/384 (Strong hash)
         */
        val RECOMMENDED_CIPHER_SUITES = listOf(
            // TLS 1.3 (if available)
            "TLS_AES_256_GCM_SHA384",
            "TLS_AES_128_GCM_SHA256",
            "TLS_CHACHA20_POLY1305_SHA256",

            // TLS 1.2
            "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
            "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
            "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384",
            "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256",
            "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256",
            "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256"
        )

        /**
         * Minimum supported TLS version
         */
        const val MIN_TLS_VERSION = "TLSv1.2"

        /**
         * Create TLS config with CA certificate only (server authentication)
         *
         * @param caCertInputStream CA certificate input stream
         * @param tlsVersion TLS version (default: TLSv1.2)
         * @param certificatePins Certificate pins for pinning (optional)
         */
        fun createServerAuth(
            caCertInputStream: InputStream,
            tlsVersion: String = MIN_TLS_VERSION,
            certificatePins: List<String> = emptyList()
        ): TLSConfig {
            return TLSConfig(
                caCertInputStream = caCertInputStream,
                tlsVersion = tlsVersion,
                certificatePins = certificatePins,
                cipherSuites = RECOMMENDED_CIPHER_SUITES
            )
        }

        /**
         * Create TLS config with mutual TLS (mTLS)
         *
         * @param caCertInputStream CA certificate input stream
         * @param clientCertInputStream Client certificate input stream
         * @param clientKeyInputStream Client private key input stream
         * @param tlsVersion TLS version (default: TLSv1.2)
         * @param certificatePins Certificate pins for pinning (optional)
         */
        fun createMutualTLS(
            caCertInputStream: InputStream,
            clientCertInputStream: InputStream,
            clientKeyInputStream: InputStream,
            tlsVersion: String = MIN_TLS_VERSION,
            certificatePins: List<String> = emptyList()
        ): TLSConfig {
            return TLSConfig(
                caCertInputStream = caCertInputStream,
                clientCertInputStream = clientCertInputStream,
                clientKeyInputStream = clientKeyInputStream,
                tlsVersion = tlsVersion,
                certificatePins = certificatePins,
                cipherSuites = RECOMMENDED_CIPHER_SUITES
            )
        }
    }

    /**
     * Validate TLS configuration
     *
     * @return true if valid, false otherwise
     */
    fun validate(): Boolean {
        // Check TLS version
        if (tlsVersion != "TLSv1.2" && tlsVersion != "TLSv1.3") {
            return false
        }

        // Mutual TLS: both cert and key required
        if (clientCertInputStream != null && clientKeyInputStream == null) {
            return false
        }

        if (clientKeyInputStream != null && clientCertInputStream == null) {
            return false
        }

        // Certificate pins format: sha256/base64
        for (pin in certificatePins) {
            if (!pin.startsWith("sha256/") || pin.length < 15) {
                return false
            }
        }

        return true
    }

    /**
     * Check if mutual TLS is enabled
     */
    fun isMutualTLS(): Boolean {
        return clientCertInputStream != null && clientKeyInputStream != null
    }

    /**
     * Check if certificate pinning is enabled
     */
    fun isCertificatePinningEnabled(): Boolean {
        return certificatePins.isNotEmpty()
    }
}

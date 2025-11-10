package com.glec.dtg.mqtt

/**
 * GLEC DTG - MQTT Configuration
 *
 * Configuration for MQTT broker connection and behavior.
 *
 * Usage:
 * ```kotlin
 * val config = MQTTConfig(
 *     brokerUrl = "ssl://mqtt.fleet.glec.co.kr:8883",
 *     clientId = "DTG-SN-12345",
 *     username = "dtg-device-12345",
 *     password = "secure-password",
 *     tlsConfig = TLSConfig.createServerAuth(caCertInputStream)
 * )
 * ```
 *
 * @property brokerUrl MQTT broker URL (tcp:// or ssl://)
 * @property clientId Unique client identifier (device serial number)
 * @property username MQTT authentication username (optional)
 * @property password MQTT authentication password (optional)
 * @property cleanSession Clean session flag (false for persistent session)
 * @property connectionTimeout Connection timeout in seconds
 * @property keepAliveInterval Keep-alive interval in seconds
 * @property autoReconnect Enable automatic reconnection
 * @property maxReconnectDelay Maximum reconnection delay in milliseconds
 * @property queueMaxSize Maximum offline queue size
 * @property queueTTLHours Message TTL in hours
 * @property tlsConfig TLS/SSL configuration (required if brokerUrl uses ssl://)
 */
data class MQTTConfig(
    val brokerUrl: String,
    val clientId: String,
    val username: String? = null,
    val password: String? = null,
    val cleanSession: Boolean = false,
    val connectionTimeout: Int = 30,
    val keepAliveInterval: Int = 60,
    val autoReconnect: Boolean = true,
    val maxReconnectDelay: Long = 60_000L,
    val queueMaxSize: Int = 10_000,
    val queueTTLHours: Long = 24,
    val tlsConfig: TLSConfig? = null
) {
    companion object {
        /**
         * Create default configuration for testing
         */
        fun createDefault(deviceId: String): MQTTConfig {
            return MQTTConfig(
                brokerUrl = "ssl://mqtt.fleet.glec.co.kr:8883",
                clientId = deviceId,
                username = "dtg-device-$deviceId",
                password = null  // Should be loaded from secure storage
            )
        }

        /**
         * Create configuration for local testing (no TLS)
         */
        fun createLocalTest(deviceId: String, port: Int = 1883): MQTTConfig {
            return MQTTConfig(
                brokerUrl = "tcp://localhost:$port",
                clientId = deviceId,
                username = null,
                password = null,
                cleanSession = true  // Clean session for testing
            )
        }
    }

    /**
     * Validate configuration
     */
    fun validate(): Boolean {
        if (brokerUrl.isBlank()) return false
        if (clientId.isBlank()) return false
        if (!brokerUrl.startsWith("tcp://") && !brokerUrl.startsWith("ssl://")) return false
        if (connectionTimeout <= 0) return false
        if (keepAliveInterval <= 0) return false
        if (queueMaxSize <= 0) return false
        if (queueTTLHours <= 0) return false

        // TLS validation: if ssl://, tlsConfig must be provided and valid
        if (isTLSEnabled()) {
            if (tlsConfig == null) {
                return false
            }
            if (!tlsConfig.validate()) {
                return false
            }
        }

        return true
    }

    /**
     * Get broker host (without protocol and port)
     */
    fun getBrokerHost(): String {
        return brokerUrl
            .removePrefix("tcp://")
            .removePrefix("ssl://")
            .substringBefore(":")
    }

    /**
     * Get broker port
     */
    fun getBrokerPort(): Int {
        val portString = brokerUrl.substringAfterLast(":", "")
        return portString.toIntOrNull() ?: if (brokerUrl.startsWith("ssl://")) 8883 else 1883
    }

    /**
     * Check if TLS/SSL is enabled
     */
    fun isTLSEnabled(): Boolean {
        return brokerUrl.startsWith("ssl://")
    }
}

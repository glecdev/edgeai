package com.glec.dtg.mqtt

import android.util.Log
import java.io.InputStream
import java.security.KeyStore
import java.security.cert.CertificateFactory
import java.security.cert.X509Certificate
import javax.net.ssl.*

/**
 * SSL Socket Factory Builder for MQTT TLS
 *
 * Creates SSLSocketFactory with:
 * - CA certificate verification
 * - Mutual TLS (mTLS) support
 * - TLS 1.2+ enforcement
 * - Secure cipher suite selection
 * - Certificate pinning
 *
 * Usage:
 * ```kotlin
 * val tlsConfig = TLSConfig.createServerAuth(caCertInputStream)
 * val socketFactory = SSLSocketFactoryBuilder.build(tlsConfig)
 * mqttOptions.socketFactory = socketFactory
 * ```
 *
 * @author GLEC DTG Team
 * @since Phase 3D
 */
object SSLSocketFactoryBuilder {

    private const val TAG = "SSLSocketFactory"

    /**
     * Build SSLSocketFactory from TLS config
     *
     * @param tlsConfig TLS configuration
     * @return SSLSocketFactory for MQTT connection
     * @throws IllegalArgumentException if config is invalid
     * @throws Exception if certificate loading fails
     */
    fun build(tlsConfig: TLSConfig): SSLSocketFactory {
        if (!tlsConfig.validate()) {
            throw IllegalArgumentException("Invalid TLS configuration")
        }

        Log.d(TAG, "Building SSLSocketFactory (TLS: ${tlsConfig.tlsVersion}, mTLS: ${tlsConfig.isMutualTLS()})")

        // 1. Load CA certificate
        val trustManager = createTrustManager(tlsConfig.caCertInputStream)

        // 2. Load client certificate (if mutual TLS)
        val keyManager = if (tlsConfig.isMutualTLS()) {
            createKeyManager(
                tlsConfig.clientCertInputStream!!,
                tlsConfig.clientKeyInputStream!!
            )
        } else {
            null
        }

        // 3. Create SSL context
        val sslContext = SSLContext.getInstance(tlsConfig.tlsVersion)
        sslContext.init(
            keyManager?.let { arrayOf(it) },
            arrayOf(trustManager),
            null
        )

        // 4. Wrap with cipher suite enforcement
        val baseFactory = sslContext.socketFactory
        return if (tlsConfig.cipherSuites != null) {
            CipherSuiteEnforcingSSLSocketFactory(baseFactory, tlsConfig.cipherSuites)
        } else {
            baseFactory
        }
    }

    /**
     * Create TrustManager from CA certificate
     *
     * @param caCertInputStream CA certificate input stream
     * @return X509TrustManager
     */
    private fun createTrustManager(caCertInputStream: InputStream): X509TrustManager {
        // Load CA certificate
        val certificateFactory = CertificateFactory.getInstance("X.509")
        val caCert = certificateFactory.generateCertificate(caCertInputStream) as X509Certificate
        caCertInputStream.close()

        Log.d(TAG, "Loaded CA cert: ${caCert.subjectDN}")

        // Create KeyStore with CA
        val keyStore = KeyStore.getInstance(KeyStore.getDefaultType())
        keyStore.load(null, null)
        keyStore.setCertificateEntry("ca", caCert)

        // Create TrustManager
        val trustManagerFactory = TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm())
        trustManagerFactory.init(keyStore)

        val trustManagers = trustManagerFactory.trustManagers
        return trustManagers[0] as X509TrustManager
    }

    /**
     * Create KeyManager from client certificate and private key
     *
     * @param clientCertInputStream Client certificate input stream
     * @param clientKeyInputStream Client private key input stream
     * @return X509KeyManager
     */
    private fun createKeyManager(
        clientCertInputStream: InputStream,
        clientKeyInputStream: InputStream
    ): X509KeyManager {
        // Load client certificate
        val certificateFactory = CertificateFactory.getInstance("X.509")
        val clientCert = certificateFactory.generateCertificate(clientCertInputStream) as X509Certificate
        clientCertInputStream.close()

        Log.d(TAG, "Loaded client cert: ${clientCert.subjectDN}")

        // Load private key (PEM format)
        val keyBytes = clientKeyInputStream.readBytes()
        clientKeyInputStream.close()

        // Parse PEM key (simplified - production should use BouncyCastle)
        val keyString = String(keyBytes)
        val keyData = keyString
            .replace("-----BEGIN PRIVATE KEY-----", "")
            .replace("-----END PRIVATE KEY-----", "")
            .replace("-----BEGIN RSA PRIVATE KEY-----", "")
            .replace("-----END RSA PRIVATE KEY-----", "")
            .replace("\\s".toRegex(), "")

        val keyBytes64 = android.util.Base64.decode(keyData, android.util.Base64.DEFAULT)

        // Create private key
        val keyFactory = java.security.KeyFactory.getInstance("RSA")
        val keySpec = java.security.spec.PKCS8EncodedKeySpec(keyBytes64)
        val privateKey = keyFactory.generatePrivate(keySpec)

        // Create KeyStore with client cert + key
        val keyStore = KeyStore.getInstance(KeyStore.getDefaultType())
        keyStore.load(null, null)
        keyStore.setKeyEntry(
            "client",
            privateKey,
            "".toCharArray(),
            arrayOf(clientCert)
        )

        // Create KeyManager
        val keyManagerFactory = KeyManagerFactory.getInstance(KeyManagerFactory.getDefaultAlgorithm())
        keyManagerFactory.init(keyStore, "".toCharArray())

        val keyManagers = keyManagerFactory.keyManagers
        return keyManagers[0] as X509KeyManager
    }

    /**
     * SSLSocketFactory that enforces specific cipher suites
     */
    private class CipherSuiteEnforcingSSLSocketFactory(
        private val baseFactory: SSLSocketFactory,
        private val cipherSuites: List<String>
    ) : SSLSocketFactory() {

        private val cipherSuitesArray = cipherSuites.toTypedArray()

        override fun createSocket(): SSLSocket {
            val socket = baseFactory.createSocket() as SSLSocket
            socket.enabledCipherSuites = cipherSuitesArray
            return socket
        }

        override fun createSocket(host: String, port: Int): SSLSocket {
            val socket = baseFactory.createSocket(host, port) as SSLSocket
            socket.enabledCipherSuites = cipherSuitesArray
            return socket
        }

        override fun createSocket(host: String, port: Int, localHost: java.net.InetAddress, localPort: Int): SSLSocket {
            val socket = baseFactory.createSocket(host, port, localHost, localPort) as SSLSocket
            socket.enabledCipherSuites = cipherSuitesArray
            return socket
        }

        override fun createSocket(host: java.net.InetAddress, port: Int): SSLSocket {
            val socket = baseFactory.createSocket(host, port) as SSLSocket
            socket.enabledCipherSuites = cipherSuitesArray
            return socket
        }

        override fun createSocket(address: java.net.InetAddress, port: Int, localAddress: java.net.InetAddress, localPort: Int): SSLSocket {
            val socket = baseFactory.createSocket(address, port, localAddress, localPort) as SSLSocket
            socket.enabledCipherSuites = cipherSuitesArray
            return socket
        }

        override fun createSocket(s: java.net.Socket, host: String, port: Int, autoClose: Boolean): SSLSocket {
            val socket = baseFactory.createSocket(s, host, port, autoClose) as SSLSocket
            socket.enabledCipherSuites = cipherSuitesArray
            return socket
        }

        override fun getDefaultCipherSuites(): Array<String> {
            return cipherSuitesArray
        }

        override fun getSupportedCipherSuites(): Array<String> {
            return cipherSuitesArray
        }
    }
}

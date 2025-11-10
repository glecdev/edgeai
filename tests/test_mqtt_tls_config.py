#!/usr/bin/env python3
"""
GLEC DTG - MQTT TLS Configuration Tests

Tests for TLS/SSL configuration validation and certificate pinning logic.
Validates configuration rules, certificate pin formats, and security settings.

Run: pytest tests/test_mqtt_tls_config.py -v
"""

import hashlib
import base64
import re
import unittest
from typing import List, Optional


class TLSConfig:
    """
    Python implementation of TLSConfig for testing.
    Mirrors Kotlin TLSConfig behavior.
    """

    MIN_TLS_VERSION = "TLSv1.2"

    RECOMMENDED_CIPHER_SUITES = [
        # TLS 1.3
        "TLS_AES_256_GCM_SHA384",
        "TLS_AES_128_GCM_SHA256",
        "TLS_CHACHA20_POLY1305_SHA256",
        # TLS 1.2
        "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
        "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
        "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384",
        "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256",
    ]

    def __init__(
        self,
        ca_cert_path: str,
        client_cert_path: Optional[str] = None,
        client_key_path: Optional[str] = None,
        tls_version: str = "TLSv1.2",
        cipher_suites: Optional[List[str]] = None,
        certificate_pins: Optional[List[str]] = None,
        hostname_verification_enabled: bool = True
    ):
        self.ca_cert_path = ca_cert_path
        self.client_cert_path = client_cert_path
        self.client_key_path = client_key_path
        self.tls_version = tls_version
        self.cipher_suites = cipher_suites or self.RECOMMENDED_CIPHER_SUITES
        self.certificate_pins = certificate_pins or []
        self.hostname_verification_enabled = hostname_verification_enabled

    def validate(self) -> bool:
        """Validate TLS configuration"""
        # Check TLS version
        if self.tls_version not in ["TLSv1.2", "TLSv1.3"]:
            return False

        # Mutual TLS: both cert and key required
        if self.client_cert_path is not None and self.client_key_path is None:
            return False

        if self.client_key_path is not None and self.client_cert_path is None:
            return False

        # Certificate pins format: sha256/base64
        for pin in self.certificate_pins:
            if not pin.startswith("sha256/") or len(pin) < 15:
                return False

        return True

    def is_mutual_tls(self) -> bool:
        """Check if mutual TLS is enabled"""
        return self.client_cert_path is not None and self.client_key_path is not None

    def is_certificate_pinning_enabled(self) -> bool:
        """Check if certificate pinning is enabled"""
        return len(self.certificate_pins) > 0


class CertificatePinner:
    """
    Python implementation of CertificatePinner for testing.
    Validates certificate pins and hostname matching.
    """

    def __init__(self, hostname: str, pins: List[str]):
        self.hostname = hostname
        self.pins = pins

    @staticmethod
    def calculate_pin(public_key_der: bytes) -> str:
        """
        Calculate SHA-256 pin for a public key

        Args:
            public_key_der: DER-encoded public key bytes

        Returns:
            SHA-256 pin in format "sha256/base64"
        """
        digest = hashlib.sha256(public_key_der).digest()
        base64_hash = base64.b64encode(digest).decode('ascii')
        return f"sha256/{base64_hash}"

    def validate(self, certificate_pins: List[str]) -> bool:
        """
        Validate certificate chain against pins

        Args:
            certificate_pins: List of pins from certificate chain

        Returns:
            True if any certificate matches a pin
        """
        if not self.pins:
            # No pins configured, skip pinning
            return True

        # Check if any certificate matches any configured pin
        for cert_pin in certificate_pins:
            if cert_pin in self.pins:
                return True

        return False

    def is_valid(self) -> bool:
        """Validate pinner configuration"""
        if not self.hostname or self.hostname.strip() == "":
            return False

        # Check pin format
        for pin in self.pins:
            if not pin.startswith("sha256/") or len(pin) < 15:
                return False

        return True


class MQTTConfig:
    """
    Python implementation of MQTTConfig for testing.
    Validates MQTT broker configuration with TLS support.
    """

    def __init__(
        self,
        broker_url: str,
        client_id: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        clean_session: bool = False,
        connection_timeout: int = 30,
        keep_alive_interval: int = 60,
        tls_config: Optional[TLSConfig] = None
    ):
        self.broker_url = broker_url
        self.client_id = client_id
        self.username = username
        self.password = password
        self.clean_session = clean_session
        self.connection_timeout = connection_timeout
        self.keep_alive_interval = keep_alive_interval
        self.tls_config = tls_config

    def validate(self) -> bool:
        """Validate MQTT configuration"""
        if not self.broker_url or self.broker_url.strip() == "":
            return False

        if not self.client_id or self.client_id.strip() == "":
            return False

        if not self.broker_url.startswith("tcp://") and not self.broker_url.startswith("ssl://"):
            return False

        if self.connection_timeout <= 0:
            return False

        if self.keep_alive_interval <= 0:
            return False

        # TLS validation: if ssl://, tls_config must be provided and valid
        if self.is_tls_enabled():
            if self.tls_config is None:
                return False
            if not self.tls_config.validate():
                return False

        return True

    def is_tls_enabled(self) -> bool:
        """Check if TLS/SSL is enabled"""
        return self.broker_url.startswith("ssl://")


class TestTLSConfigValidation(unittest.TestCase):
    """Test TLS configuration validation"""

    def test_valid_server_auth_tls_config(self):
        """Test valid server authentication TLS config"""
        config = TLSConfig(
            ca_cert_path="/path/to/ca.crt",
            tls_version="TLSv1.2"
        )

        self.assertTrue(config.validate())
        self.assertFalse(config.is_mutual_tls())

    def test_valid_mutual_tls_config(self):
        """Test valid mutual TLS config"""
        config = TLSConfig(
            ca_cert_path="/path/to/ca.crt",
            client_cert_path="/path/to/client.crt",
            client_key_path="/path/to/client.key",
            tls_version="TLSv1.2"
        )

        self.assertTrue(config.validate())
        self.assertTrue(config.is_mutual_tls())

    def test_invalid_tls_version(self):
        """Test invalid TLS version"""
        config = TLSConfig(
            ca_cert_path="/path/to/ca.crt",
            tls_version="TLSv1.0"  # Too old
        )

        self.assertFalse(config.validate())

    def test_invalid_mutual_tls_missing_key(self):
        """Test mutual TLS with missing private key"""
        config = TLSConfig(
            ca_cert_path="/path/to/ca.crt",
            client_cert_path="/path/to/client.crt",
            client_key_path=None  # Missing
        )

        self.assertFalse(config.validate())

    def test_invalid_mutual_tls_missing_cert(self):
        """Test mutual TLS with missing certificate"""
        config = TLSConfig(
            ca_cert_path="/path/to/ca.crt",
            client_cert_path=None,  # Missing
            client_key_path="/path/to/client.key"
        )

        self.assertFalse(config.validate())

    def test_invalid_certificate_pin_format(self):
        """Test invalid certificate pin format"""
        config = TLSConfig(
            ca_cert_path="/path/to/ca.crt",
            certificate_pins=[
                "sha256/VALID_BASE64_STRING_HERE==",
                "invalid_pin"  # Missing sha256/ prefix
            ]
        )

        self.assertFalse(config.validate())

    def test_certificate_pinning_enabled(self):
        """Test certificate pinning detection"""
        config = TLSConfig(
            ca_cert_path="/path/to/ca.crt",
            certificate_pins=["sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="]
        )

        self.assertTrue(config.is_certificate_pinning_enabled())

    def test_certificate_pinning_disabled(self):
        """Test certificate pinning disabled"""
        config = TLSConfig(
            ca_cert_path="/path/to/ca.crt",
            certificate_pins=[]
        )

        self.assertFalse(config.is_certificate_pinning_enabled())


class TestCertificatePinner(unittest.TestCase):
    """Test certificate pinning validation"""

    def test_valid_pinner_configuration(self):
        """Test valid pinner configuration"""
        pinner = CertificatePinner(
            hostname="mqtt.fleet.glec.co.kr",
            pins=["sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="]
        )

        self.assertTrue(pinner.is_valid())

    def test_invalid_pinner_empty_hostname(self):
        """Test invalid pinner with empty hostname"""
        pinner = CertificatePinner(
            hostname="",
            pins=["sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="]
        )

        self.assertFalse(pinner.is_valid())

    def test_invalid_pinner_malformed_pin(self):
        """Test invalid pinner with malformed pin"""
        pinner = CertificatePinner(
            hostname="mqtt.fleet.glec.co.kr",
            pins=["invalid_pin"]
        )

        self.assertFalse(pinner.is_valid())

    def test_pin_validation_success(self):
        """Test successful pin validation"""
        pinner = CertificatePinner(
            hostname="mqtt.fleet.glec.co.kr",
            pins=[
                "sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
                "sha256/BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB="
            ]
        )

        # Certificate chain has matching pin
        certificate_pins = [
            "sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",  # Match!
            "sha256/CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC="
        ]

        self.assertTrue(pinner.validate(certificate_pins))

    def test_pin_validation_failure(self):
        """Test failed pin validation"""
        pinner = CertificatePinner(
            hostname="mqtt.fleet.glec.co.kr",
            pins=[
                "sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
                "sha256/BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB="
            ]
        )

        # Certificate chain has NO matching pins
        certificate_pins = [
            "sha256/CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC=",
            "sha256/DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD="
        ]

        self.assertFalse(pinner.validate(certificate_pins))

    def test_pin_calculation(self):
        """Test SHA-256 pin calculation"""
        # Example public key bytes (DER-encoded)
        public_key_der = b"example_public_key_data_here"

        pin = CertificatePinner.calculate_pin(public_key_der)

        # Check format
        self.assertTrue(pin.startswith("sha256/"))
        self.assertGreater(len(pin), 15)

        # Verify calculation is deterministic
        pin2 = CertificatePinner.calculate_pin(public_key_der)
        self.assertEqual(pin, pin2)


class TestMQTTConfigWithTLS(unittest.TestCase):
    """Test MQTT configuration with TLS integration"""

    def test_valid_mqtt_config_with_tls(self):
        """Test valid MQTT config with TLS"""
        tls_config = TLSConfig(
            ca_cert_path="/path/to/ca.crt",
            tls_version="TLSv1.2"
        )

        mqtt_config = MQTTConfig(
            broker_url="ssl://mqtt.fleet.glec.co.kr:8883",
            client_id="DTG-SN-12345",
            tls_config=tls_config
        )

        self.assertTrue(mqtt_config.validate())
        self.assertTrue(mqtt_config.is_tls_enabled())

    def test_invalid_mqtt_config_ssl_without_tls_config(self):
        """Test invalid MQTT config: ssl:// without TLS config"""
        mqtt_config = MQTTConfig(
            broker_url="ssl://mqtt.fleet.glec.co.kr:8883",
            client_id="DTG-SN-12345",
            tls_config=None  # Missing TLS config
        )

        self.assertFalse(mqtt_config.validate())

    def test_valid_mqtt_config_tcp_without_tls(self):
        """Test valid MQTT config: tcp:// without TLS (for testing)"""
        mqtt_config = MQTTConfig(
            broker_url="tcp://localhost:1883",
            client_id="DTG-SN-12345",
            tls_config=None
        )

        self.assertTrue(mqtt_config.validate())
        self.assertFalse(mqtt_config.is_tls_enabled())

    def test_mqtt_config_with_mutual_tls(self):
        """Test MQTT config with mutual TLS"""
        tls_config = TLSConfig(
            ca_cert_path="/path/to/ca.crt",
            client_cert_path="/path/to/client.crt",
            client_key_path="/path/to/client.key",
            tls_version="TLSv1.2"
        )

        mqtt_config = MQTTConfig(
            broker_url="ssl://mqtt.fleet.glec.co.kr:8883",
            client_id="DTG-SN-12345",
            username="dtg-device-12345",
            password="secure-password",
            tls_config=tls_config
        )

        self.assertTrue(mqtt_config.validate())
        self.assertTrue(mqtt_config.tls_config.is_mutual_tls())

    def test_mqtt_config_with_certificate_pinning(self):
        """Test MQTT config with certificate pinning"""
        tls_config = TLSConfig(
            ca_cert_path="/path/to/ca.crt",
            certificate_pins=[
                "sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
                "sha256/BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB="  # Backup pin
            ],
            tls_version="TLSv1.2"
        )

        mqtt_config = MQTTConfig(
            broker_url="ssl://mqtt.fleet.glec.co.kr:8883",
            client_id="DTG-SN-12345",
            tls_config=tls_config
        )

        self.assertTrue(mqtt_config.validate())
        self.assertTrue(mqtt_config.tls_config.is_certificate_pinning_enabled())


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestTLSConfigValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestCertificatePinner))
    suite.addTests(loader.loadTestsFromTestCase(TestMQTTConfigWithTLS))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)

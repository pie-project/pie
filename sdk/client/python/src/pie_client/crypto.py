"""Cryptographic utilities for key parsing and signing operations."""

import base64
from pathlib import Path
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ec, ed25519, padding
from cryptography.hazmat.backends import default_backend


class ParsedPrivateKey:
    """
    Represents a parsed private key that can be used for signing.

    Supports:
    - OpenSSH format for RSA, ED25519, and ECDSA keys
    - PKCS#8 PEM format for RSA, ED25519, and ECDSA keys
    - PKCS#1 PEM format for RSA keys
    - ECDSA curves supported: P-256 (nistp256), P-384 (nistp384)
    - RSA keys must be at least 2048 bits (minimum enforced for security)
    """

    def __init__(self, private_key):
        self._private_key = private_key

    @classmethod
    def parse(cls, key_content: str) -> "ParsedPrivateKey":
        """
        Parse a private key from PEM or OpenSSH format string.

        :param key_content: The key content as a string (PEM or OpenSSH format)
        :return: ParsedPrivateKey instance
        :raises ValueError: If the key cannot be parsed or is invalid
        """
        key_bytes = key_content.encode("utf-8")

        # Try parsing as OpenSSH format first (most common for SSH keys)
        try:
            private_key = serialization.load_ssh_private_key(
                key_bytes, password=None, backend=default_backend()
            )
            return cls._validate_and_wrap(private_key)
        except (ValueError, TypeError):
            pass

        # Try parsing as PEM format (PKCS#8 or PKCS#1)
        try:
            private_key = serialization.load_pem_private_key(
                key_bytes, password=None, backend=default_backend()
            )
            return cls._validate_and_wrap(private_key)
        except (ValueError, TypeError):
            pass

        raise ValueError(
            "Could not parse private key. Supported formats: "
            "OpenSSH (RSA/ED25519/ECDSA), PKCS#8 PEM, PKCS#1 PEM"
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "ParsedPrivateKey":
        """
        Load and parse a private key from a file.

        :param path: Path to the private key file
        :return: ParsedPrivateKey instance
        """
        key_content = Path(path).read_text()
        return cls.parse(key_content)

    @classmethod
    def _validate_and_wrap(cls, private_key) -> "ParsedPrivateKey":
        """Validate the key type and wrap it."""
        if isinstance(private_key, rsa.RSAPrivateKey):
            # Check RSA key size (minimum 2048 bits)
            key_size = private_key.key_size
            if key_size < 2048:
                raise ValueError(
                    f"RSA key is too weak: {key_size} bits (minimum required: 2048 bits)"
                )
            return cls(private_key)

        if isinstance(private_key, ed25519.Ed25519PrivateKey):
            return cls(private_key)

        if isinstance(private_key, ec.EllipticCurvePrivateKey):
            # Check for supported curves
            curve = private_key.curve
            if isinstance(curve, (ec.SECP256R1, ec.SECP384R1)):
                return cls(private_key)
            raise ValueError(
                f"Unsupported ECDSA curve: {curve.name}. "
                "Supported: P-256 (secp256r1), P-384 (secp384r1)"
            )

        raise ValueError(
            f"Unsupported key type: {type(private_key).__name__}. "
            "Supported: RSA, ED25519, ECDSA (P-256, P-384)"
        )

    def sign(self, data: bytes) -> bytes:
        """
        Sign data with the private key.

        For RSA keys: Uses PKCS#1 v1.5 signature scheme with SHA-256.
        For ED25519 keys: Uses Ed25519 signature scheme.
        For ECDSA keys: Uses ECDSA with SHA-256 (P-256) or SHA-384 (P-384).

        :param data: The data to sign
        :return: The signature as bytes
        """
        if isinstance(self._private_key, rsa.RSAPrivateKey):
            return self._private_key.sign(data, padding.PKCS1v15(), hashes.SHA256())

        if isinstance(self._private_key, ed25519.Ed25519PrivateKey):
            return self._private_key.sign(data)

        if isinstance(self._private_key, ec.EllipticCurvePrivateKey):
            # Use SHA-256 for P-256, SHA-384 for P-384
            curve = self._private_key.curve
            if isinstance(curve, ec.SECP256R1):
                hash_algo = hashes.SHA256()
            else:  # P-384
                hash_algo = hashes.SHA384()

            return self._private_key.sign(data, ec.ECDSA(hash_algo))

        raise ValueError(
            f"Cannot sign with key type: {type(self._private_key).__name__}"
        )

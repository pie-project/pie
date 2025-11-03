//! Cryptographic utilities for key parsing and signing operations.

use anyhow::{Context, Result, bail};
use pem;
use ring::signature::{
    ECDSA_P256_SHA256_ASN1_SIGNING, ECDSA_P384_SHA384_ASN1_SIGNING, EcdsaKeyPair, Ed25519KeyPair,
    RSA_PKCS1_SHA256, RsaKeyPair,
};
use rsa::RsaPrivateKey;
use rsa::pkcs8::{DecodePrivateKey, EncodePrivateKey};
use rsa::traits::PublicKeyParts;
use ssh_key::{Algorithm, EcdsaCurve, PrivateKey as SshPrivateKey};

/// Enum representing the supported key types
enum KeyPair {
    Rsa(RsaKeyPair),
    Ed25519(Ed25519KeyPair),
    EcdsaP256(EcdsaKeyPair),
    EcdsaP384(EcdsaKeyPair),
}

/// Represents a parsed private key that can be used for signing.
pub struct ParsedPrivateKey {
    key_pair: KeyPair,
}

impl ParsedPrivateKey {
    /// Parse a private key from PEM or SSH format string.
    ///
    /// Supports:
    /// - OpenSSH format for RSA, ED25519, and ECDSA keys
    /// - PKCS#8 PEM format for RSA, ED25519, and ECDSA keys
    /// - PKCS#1 PEM format for RSA keys
    /// - ECDSA curves supported: P-256 (nistp256), P-384 (nistp384)
    /// - RSA keys must be at least 2048 bits (minimum enforced for security)
    pub fn parse(key_content: &str) -> Result<Self> {
        // Try parsing as OpenSSH format first (most common)
        if let Ok(ssh_key) = SshPrivateKey::from_openssh(key_content) {
            return Self::from_ssh_key(ssh_key);
        }

        // Try parsing as PEM format (PKCS#8 or PKCS#1)
        if key_content.contains("-----BEGIN") {
            return Self::from_pem(key_content);
        }

        bail!(
            "Could not parse private key. Supported formats: OpenSSH (RSA/ED25519/ECDSA), PKCS#8 PEM, PKCS#1 PEM"
        )
    }

    /// Parse an SSH private key into a KeyPair (RSA, ED25519, or ECDSA)
    fn from_ssh_key(ssh_key: SshPrivateKey) -> Result<Self> {
        match ssh_key.algorithm() {
            Algorithm::Rsa { .. } => Self::rsa_from_ssh_key(ssh_key),
            Algorithm::Ed25519 => Self::ed25519_from_ssh_key(ssh_key),
            Algorithm::Ecdsa { curve } => Self::ecdsa_from_ssh_key(ssh_key, &curve),
            algo => bail!(
                "Unsupported key algorithm: {:?}. Supported: RSA, ED25519, ECDSA (P-256, P-384).",
                algo
            ),
        }
    }

    /// Parse PEM format private key
    fn from_pem(key_content: &str) -> Result<Self> {
        // Try parsing as RSA PKCS#8/PKCS#1 first
        if let Ok(rsa_key) = RsaPrivateKey::from_pkcs8_pem(key_content) {
            // Check that the key is at least 2048 bits
            Self::check_rsa_key_size(&rsa_key)?;

            // Convert to PKCS#8 DER format expected by the `ring` crate
            let pkcs8_der = rsa_key
                .to_pkcs8_der()
                .context("Failed to encode RSA key as PKCS#8 DER")?;

            let key_pair = RsaKeyPair::from_pkcs8(pkcs8_der.as_bytes())
                .map_err(|e| anyhow::anyhow!("Failed to create RSA key pair: {:?}", e))?;

            return Ok(Self {
                key_pair: KeyPair::Rsa(key_pair),
            });
        }

        // Try parsing as PKCS#8 for ED25519 or ECDSA
        if let Ok(pkcs8_der) = pem::parse(key_content) {
            // Try ED25519
            if let Ok(key_pair) = Ed25519KeyPair::from_pkcs8(pkcs8_der.contents()) {
                return Ok(Self {
                    key_pair: KeyPair::Ed25519(key_pair),
                });
            }

            // For ECDSA, we need a random number generator
            let rng = ring::rand::SystemRandom::new();

            // Try ECDSA P-256
            if let Ok(key_pair) = EcdsaKeyPair::from_pkcs8(
                &ECDSA_P256_SHA256_ASN1_SIGNING,
                pkcs8_der.contents(),
                &rng,
            ) {
                return Ok(Self {
                    key_pair: KeyPair::EcdsaP256(key_pair),
                });
            }

            // Try ECDSA P-384
            if let Ok(key_pair) = EcdsaKeyPair::from_pkcs8(
                &ECDSA_P384_SHA384_ASN1_SIGNING,
                pkcs8_der.contents(),
                &rng,
            ) {
                return Ok(Self {
                    key_pair: KeyPair::EcdsaP384(key_pair),
                });
            }
        }

        bail!("Failed to parse PEM as RSA (PKCS#8/PKCS#1), ED25519 (PKCS#8), or ECDSA (PKCS#8)")
    }

    /// Parse RSA key from SSH format
    fn rsa_from_ssh_key(ssh_key: SshPrivateKey) -> Result<Self> {
        // Extract the RSA components from SSH key
        let key_data = ssh_key.key_data();
        let rsa_keypair = key_data.rsa().context("Failed to extract RSA key data")?;

        // Convert to rsa crate's types
        let n = rsa::BigUint::from_bytes_be(rsa_keypair.public.n.as_bytes());
        let e = rsa::BigUint::from_bytes_be(rsa_keypair.public.e.as_bytes());
        let d = rsa::BigUint::from_bytes_be(rsa_keypair.private.d.as_bytes());
        let p = rsa::BigUint::from_bytes_be(rsa_keypair.private.p.as_bytes());
        let q = rsa::BigUint::from_bytes_be(rsa_keypair.private.q.as_bytes());

        // Construct the key from the RSA components
        let rsa_key = RsaPrivateKey::from_components(n, e, d, vec![p, q])
            .context("Failed to construct RSA private key from SSH components")?;

        // Check that the key is at least 2048 bits
        Self::check_rsa_key_size(&rsa_key)?;

        // Convert to PKCS#8 DER format expected by the `ring` crate
        let pkcs8_der = rsa_key
            .to_pkcs8_der()
            .context("Failed to encode RSA key as PKCS#8 DER")?;

        // Create `ring`'s `RsaKeyPair` from PKCS#8 DER
        let key_pair = RsaKeyPair::from_pkcs8(pkcs8_der.as_bytes())
            .map_err(|e| anyhow::anyhow!("Failed to create RSA key pair: {:?}", e))?;

        Ok(Self {
            key_pair: KeyPair::Rsa(key_pair),
        })
    }

    /// Parse ED25519 key from SSH format
    fn ed25519_from_ssh_key(ssh_key: SshPrivateKey) -> Result<Self> {
        let key_data = ssh_key.key_data();
        let ed25519_keypair = key_data
            .ed25519()
            .context("Failed to extract ED25519 key data")?;

        // ED25519 keys in SSH format: 32 bytes private key + 32 bytes public key
        let private_bytes = ed25519_keypair.private.as_ref();
        let public_bytes = ed25519_keypair.public.as_ref();

        // ring expects a 32-byte seed for Ed25519
        if private_bytes.len() != 32 {
            bail!(
                "Invalid ED25519 private key length: expected 32 bytes, got {}",
                private_bytes.len()
            );
        }

        let key_pair = Ed25519KeyPair::from_seed_and_public_key(private_bytes, public_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to create ED25519 key pair: {:?}", e))?;

        Ok(Self {
            key_pair: KeyPair::Ed25519(key_pair),
        })
    }

    /// Parse ECDSA key from SSH format
    fn ecdsa_from_ssh_key(ssh_key: SshPrivateKey, curve: &EcdsaCurve) -> Result<Self> {
        let key_data = ssh_key.key_data();
        let ecdsa_keypair = key_data
            .ecdsa()
            .context("Failed to extract ECDSA key data")?;

        // Get the private scalar from the SSH key
        let private_bytes = match ecdsa_keypair {
            ssh_key::private::EcdsaKeypair::NistP256 { private, .. } => private.as_slice(),
            ssh_key::private::EcdsaKeypair::NistP384 { private, .. } => private.as_slice(),
            ssh_key::private::EcdsaKeypair::NistP521 { private, .. } => private.as_slice(),
        };

        // Convert to PKCS#8 DER format (required by ring)
        let pkcs8_der = Self::ecdsa_to_pkcs8_der(private_bytes, curve)?;
        let rng = ring::rand::SystemRandom::new();

        // Create the appropriate key pair based on the curve
        let key_pair = match curve {
            EcdsaCurve::NistP256 => KeyPair::EcdsaP256(
                EcdsaKeyPair::from_pkcs8(&ECDSA_P256_SHA256_ASN1_SIGNING, &pkcs8_der, &rng)
                    .map_err(|e| {
                        anyhow::anyhow!("Failed to create ECDSA P-256 key pair: {:?}", e)
                    })?,
            ),
            EcdsaCurve::NistP384 => KeyPair::EcdsaP384(
                EcdsaKeyPair::from_pkcs8(&ECDSA_P384_SHA384_ASN1_SIGNING, &pkcs8_der, &rng)
                    .map_err(|e| {
                        anyhow::anyhow!("Failed to create ECDSA P-384 key pair: {:?}", e)
                    })?,
            ),
            EcdsaCurve::NistP521 => {
                bail!("ECDSA P-521 curve is not supported by the ring crate")
            }
        };

        Ok(Self { key_pair })
    }

    /// Convert ECDSA private key bytes to PKCS#8 DER format
    fn ecdsa_to_pkcs8_der(private_bytes: &[u8], curve: &ssh_key::EcdsaCurve) -> Result<Vec<u8>> {
        use ssh_key::EcdsaCurve;

        match curve {
            EcdsaCurve::NistP256 => {
                use p256::pkcs8::EncodePrivateKey;
                let secret_key = p256::SecretKey::from_bytes(private_bytes.into())
                    .context("Failed to parse P-256 private key")?;
                let pkcs8_der = secret_key
                    .to_pkcs8_der()
                    .context("Failed to encode P-256 key as PKCS#8")?;
                Ok(pkcs8_der.as_bytes().to_vec())
            }
            EcdsaCurve::NistP384 => {
                use p384::pkcs8::EncodePrivateKey;
                let secret_key = p384::SecretKey::from_bytes(private_bytes.into())
                    .context("Failed to parse P-384 private key")?;
                let pkcs8_der = secret_key
                    .to_pkcs8_der()
                    .context("Failed to encode P-384 key as PKCS#8")?;
                Ok(pkcs8_der.as_bytes().to_vec())
            }
            EcdsaCurve::NistP521 => {
                bail!("P-521 curve is not supported")
            }
        }
    }

    /// Check that an RSA private key is at least 2048 bits.
    fn check_rsa_key_size(rsa_key: &RsaPrivateKey) -> Result<()> {
        let key_size_bits = rsa_key.size() * 8;
        if key_size_bits < 2048 {
            bail!(
                "RSA key is too weak: {} bits (minimum required: 2048 bits)",
                key_size_bits
            );
        }
        Ok(())
    }

    /// Sign data with the private key.
    ///
    /// For RSA keys: Uses PKCS#1 v1.5 signature scheme with SHA-256.
    /// For ED25519 keys: Uses Ed25519 signature scheme.
    /// For ECDSA keys: Uses ECDSA with SHA-256 (P-256) or SHA-384 (P-384).
    pub fn sign(&self, data: &[u8]) -> Result<Vec<u8>> {
        match &self.key_pair {
            KeyPair::Rsa(rsa_key_pair) => {
                // The random number generator is required by the function signature.
                // It is not used for our deterministic signing.
                let rng = ring::rand::SystemRandom::new();

                let mut signature = vec![0u8; rsa_key_pair.public().modulus_len()];
                rsa_key_pair
                    .sign(&RSA_PKCS1_SHA256, &rng, data, &mut signature)
                    .map_err(|e| anyhow::anyhow!("Failed to sign data with RSA: {:?}", e))?;
                Ok(signature)
            }
            KeyPair::Ed25519(ed25519_key_pair) => {
                let signature = ed25519_key_pair.sign(data);
                Ok(signature.as_ref().to_vec())
            }
            KeyPair::EcdsaP256(ecdsa_key_pair) => {
                let rng = ring::rand::SystemRandom::new();
                let signature = ecdsa_key_pair.sign(&rng, data).map_err(|e| {
                    anyhow::anyhow!("Failed to sign data with ECDSA P-256: {:?}", e)
                })?;
                Ok(signature.as_ref().to_vec())
            }
            KeyPair::EcdsaP384(ecdsa_key_pair) => {
                let rng = ring::rand::SystemRandom::new();
                let signature = ecdsa_key_pair.sign(&rng, data).map_err(|e| {
                    anyhow::anyhow!("Failed to sign data with ECDSA P-384: {:?}", e)
                })?;
                Ok(signature.as_ref().to_vec())
            }
        }
    }
}

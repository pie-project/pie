//! Cryptographic utilities for key parsing and signing operations.

use anyhow::{Context, Result, bail};
use pem;
use ring::signature::{Ed25519KeyPair, RSA_PKCS1_SHA256, RsaKeyPair};
use rsa::RsaPrivateKey;
use rsa::pkcs8::{DecodePrivateKey, EncodePrivateKey};
use ssh_key::{Algorithm, PrivateKey as SshPrivateKey};

/// Enum representing the supported key types
enum KeyPair {
    Rsa(RsaKeyPair),
    Ed25519(Ed25519KeyPair),
}

/// Represents a parsed private key that can be used for signing.
pub struct ParsedPrivateKey {
    key_pair: KeyPair,
}

impl ParsedPrivateKey {
    /// Parse a private key from PEM or SSH format string.
    ///
    /// Supports:
    /// - OpenSSH format for RSA and ED25519 keys (e.g., ~/.ssh/id_rsa, ~/.ssh/id_ed25519)
    /// - PKCS#8 PEM format for RSA and ED25519 keys
    /// - PKCS#1 PEM format for RSA keys
    pub fn parse(key_content: &str) -> Result<Self> {
        // Try parsing as OpenSSH format first (most common)
        if let Ok(ssh_key) = SshPrivateKey::from_openssh(key_content) {
            return Self::from_ssh_key(ssh_key);
        }

        // Try parsing as PEM format (PKCS#8 or PKCS#1)
        if key_content.contains("-----BEGIN") {
            return Self::from_pem_format(key_content);
        }

        bail!(
            "Could not parse private key. Supported formats: OpenSSH (RSA/ED25519), PKCS#8 PEM, PKCS#1 PEM"
        )
    }

    /// Parse an SSH private key into a KeyPair (RSA or ED25519)
    fn from_ssh_key(ssh_key: SshPrivateKey) -> Result<Self> {
        match ssh_key.algorithm() {
            Algorithm::Rsa { .. } => {
                let rsa_key = Self::rsa_key_from_ssh_format(ssh_key)?;

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
            Algorithm::Ed25519 => {
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

                // Combine private and public key for ring's Ed25519KeyPair
                let mut keypair_bytes = Vec::with_capacity(64);
                keypair_bytes.extend_from_slice(private_bytes);
                keypair_bytes.extend_from_slice(public_bytes);

                let key_pair =
                    Ed25519KeyPair::from_seed_and_public_key(private_bytes, public_bytes).map_err(
                        |e| anyhow::anyhow!("Failed to create ED25519 key pair: {:?}", e),
                    )?;

                Ok(Self {
                    key_pair: KeyPair::Ed25519(key_pair),
                })
            }
            algo => bail!(
                "Unsupported key algorithm: {:?}. Only RSA and ED25519 are supported.",
                algo
            ),
        }
    }

    /// Parse PEM format private key
    fn from_pem_format(key_content: &str) -> Result<Self> {
        // Try parsing as RSA PKCS#8/PKCS#1 first
        if let Ok(rsa_key) = RsaPrivateKey::from_pkcs8_pem(key_content) {
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

        // Try parsing as ED25519 PKCS#8
        // The ring crate can parse PKCS#8 DER directly
        if let Ok(pkcs8_der) = pem::parse(key_content) {
            if let Ok(key_pair) = Ed25519KeyPair::from_pkcs8(pkcs8_der.contents()) {
                return Ok(Self {
                    key_pair: KeyPair::Ed25519(key_pair),
                });
            }
        }

        bail!("Failed to parse PEM as RSA (PKCS#8/PKCS#1) or ED25519 (PKCS#8)")
    }

    /// Convert an SSH private key to rsa::RsaPrivateKey
    fn rsa_key_from_ssh_format(ssh_key: SshPrivateKey) -> Result<RsaPrivateKey> {
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
        RsaPrivateKey::from_components(n, e, d, vec![p, q])
            .context("Failed to construct RSA private key from SSH components")
    }

    /// Sign data with the private key.
    ///
    /// For RSA keys: Uses PKCS#1 v1.5 signature scheme with SHA-256.
    /// For ED25519 keys: Uses Ed25519 signature scheme.
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
        }
    }
}

//! Cryptographic utilities for key parsing and signing operations.

use anyhow::{Context, Result, bail};
use rsa::RsaPrivateKey;
use rsa::pkcs1::DecodeRsaPrivateKey;
use rsa::pkcs1v15::SigningKey;
use rsa::pkcs8::DecodePrivateKey;
use rsa::sha2::Sha256;
use rsa::signature::{SignatureEncoding, Signer};
use ssh_key::{Algorithm, PrivateKey as SshPrivateKey};

/// Represents a parsed private key that can be used for signing.
pub struct ParsedPrivateKey {
    /// The signing key configured for SHA-256
    signing_key: SigningKey<Sha256>,
}

impl ParsedPrivateKey {
    /// Parse a private key from PEM or SSH format string.
    ///
    /// Supports:
    /// - OpenSSH format (e.g., ~/.ssh/id_rsa)
    /// - PKCS#8 PEM format
    /// - PKCS#1 PEM format
    pub fn parse(key_content: &str) -> Result<Self> {
        let key = Self::parse_key_string(key_content)?;
        key.validate()
            .context("Failed to validate RSA private key")?;
        let signing_key = SigningKey::<Sha256>::new(key);
        Ok(Self { signing_key })
    }

    /// Parse a private key from a string.
    fn parse_key_string(key_content: &str) -> Result<RsaPrivateKey> {
        // Try parsing as OpenSSH format first (most common for ~/.ssh/id_rsa)
        if let Ok(ssh_key) = SshPrivateKey::from_openssh(key_content) {
            return Self::from_ssh_key(ssh_key);
        }

        // Try parsing as PKCS#8 PEM format
        if let Ok(rsa_key) = RsaPrivateKey::from_pkcs8_pem(key_content) {
            return Ok(rsa_key);
        }

        // Try parsing as PKCS#1 PEM format
        if let Ok(rsa_key) = RsaPrivateKey::from_pkcs1_pem(key_content) {
            return Ok(rsa_key);
        }

        bail!("Could not parse private key. Supported formats: OpenSSH, PKCS#8 PEM, PKCS#1 PEM")
    }

    /// Create from an SSH private key
    fn from_ssh_key(ssh_key: SshPrivateKey) -> Result<RsaPrivateKey> {
        // Ensure it's an RSA key
        if !matches!(ssh_key.algorithm(), Algorithm::Rsa { .. }) {
            bail!(
                "Only RSA keys are supported, found: {:?}",
                ssh_key.algorithm()
            );
        }

        // Extract the RSA components
        let key_data = ssh_key.key_data();
        let rsa_keypair = key_data.rsa().context("Failed to extract RSA key data")?;

        // Convert to rsa crate's RsaPrivateKey
        let n = rsa::BigUint::from_bytes_be(rsa_keypair.public.n.as_bytes());
        let e = rsa::BigUint::from_bytes_be(rsa_keypair.public.e.as_bytes());
        let d = rsa::BigUint::from_bytes_be(rsa_keypair.private.d.as_bytes());
        let p = rsa::BigUint::from_bytes_be(rsa_keypair.private.p.as_bytes());
        let q = rsa::BigUint::from_bytes_be(rsa_keypair.private.q.as_bytes());

        let rsa_key = RsaPrivateKey::from_components(n, e, d, vec![p, q])
            .context("Failed to construct RSA private key from SSH key components")?;

        Ok(rsa_key)
    }

    /// Sign data with the private key using PKCS#1 v1.5 signature scheme with SHA-256.
    ///
    /// The signature is computed over the SHA-256 hash of the input data.
    pub fn sign(&self, data: &[u8]) -> Result<Vec<u8>> {
        let signature = self.signing_key.sign(data);
        Ok(signature.to_bytes().as_ref().to_vec())
    }
}

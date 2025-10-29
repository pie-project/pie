//! Cryptographic utilities for key parsing and signing operations.

use anyhow::{Context, Result, bail};
use ring::signature::{RSA_PKCS1_SHA256, RsaKeyPair};
use rsa::RsaPrivateKey;
use rsa::pkcs8::{DecodePrivateKey, EncodePrivateKey};
use ssh_key::{Algorithm, PrivateKey as SshPrivateKey};

/// Represents a parsed private key that can be used for signing.
pub struct ParsedPrivateKey {
    key_pair: RsaKeyPair,
}

impl ParsedPrivateKey {
    /// Parse a private key from PEM or SSH format string.
    ///
    /// Supports:
    /// - OpenSSH format (e.g., ~/.ssh/id_rsa)
    /// - PKCS#8 PEM format
    /// - PKCS#1 PEM format
    pub fn parse(key_content: &str) -> Result<Self> {
        let rsa_key = Self::parse_rsa_key(key_content)?;

        // Convert to PKCS#8 DER format expected by the `ring` crate
        let pkcs8_der = rsa_key
            .to_pkcs8_der()
            .context("Failed to encode key as PKCS#8 DER")?;

        // Create `ring`'s `RsaKeyPair` from PKCS#8 DER
        let key_pair = RsaKeyPair::from_pkcs8(pkcs8_der.as_bytes())
            .map_err(|e| anyhow::anyhow!("Failed to create RSA key pair: {:?}", e))?;

        Ok(Self { key_pair })
    }

    /// Parse a private key from various formats into `rsa::RsaPrivateKey`
    fn parse_rsa_key(key_content: &str) -> Result<RsaPrivateKey> {
        // Try parsing as OpenSSH format first (most common for ~/.ssh/id_rsa)
        if let Ok(ssh_key) = SshPrivateKey::from_openssh(key_content) {
            return Self::rsa_key_from_ssh_format(ssh_key);
        }

        // Try parsing as PEM format (PKCS#8 or PKCS#1)
        if key_content.contains("-----BEGIN") {
            return Self::rsa_key_from_pem_format(key_content);
        }

        bail!("Could not parse private key. Supported formats: OpenSSH, PKCS#8 PEM, PKCS#1 PEM")
    }

    /// Parse PEM format private key using the rsa crate
    fn rsa_key_from_pem_format(key_content: &str) -> Result<RsaPrivateKey> {
        // Parse the PEM format - the rsa crate handles both PKCS#8 and PKCS#1

        RsaPrivateKey::from_pkcs8_pem(key_content)
            .context("Failed to parse PEM as PKCS#8 or PKCS#1 format")
    }

    /// Convert an SSH private key to rsa::RsaPrivateKey
    fn rsa_key_from_ssh_format(ssh_key: SshPrivateKey) -> Result<RsaPrivateKey> {
        // Ensure it's an RSA key
        if !matches!(ssh_key.algorithm(), Algorithm::Rsa { .. }) {
            bail!(
                "Only RSA keys are supported, found: {:?}",
                ssh_key.algorithm()
            );
        }

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

    /// Sign data with the private key using PKCS#1 v1.5 signature scheme with SHA-256.
    ///
    /// The signature is computed over the SHA-256 hash of the input data.
    pub fn sign(&self, data: &[u8]) -> Result<Vec<u8>> {
        // The random number generator is required by the function signature.
        // It is not used for our deterministic signing.
        let rng = ring::rand::SystemRandom::new();

        let mut signature = vec![0u8; self.key_pair.public().modulus_len()];
        self.key_pair
            .sign(&RSA_PKCS1_SHA256, &rng, data, &mut signature)
            .map_err(|e| anyhow::anyhow!("Failed to sign data: {:?}", e))?;
        Ok(signature)
    }
}

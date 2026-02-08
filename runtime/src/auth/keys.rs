//! Public key types for authentication.
//!
//! Supports parsing from OpenSSH, PKCS#8, and PKCS#1 formats, and verification
//! using typed verifying keys from the same crates that parse (no intermediate
//! raw-byte representations).

use anyhow::{Context, Result, bail};
use ed25519_dalek::Verifier as _;
use rsa::RsaPublicKey;
use rsa::pkcs8::DecodePublicKey;
use rsa::signature::Verifier as RsaVerifier;
use rsa::traits::PublicKeyParts;
use ssh_key::public::EcdsaPublicKey;
use ssh_key::{Algorithm, EcdsaCurve, PublicKey as SshPublicKey};

/// A public key that can be used for signature verification.
///
/// Each variant holds a typed verifying key from the crate that originally
/// parsed it, eliminating raw-byte intermediate representations.
#[derive(Debug, Clone)]
pub enum PublicKey {
    /// RSA public key (2048–8192 bits, PKCS#1 v1.5 SHA-256 signatures)
    Rsa(RsaPublicKey),
    /// ED25519 public key
    Ed25519(ed25519_dalek::VerifyingKey),
    /// ECDSA P-256 public key (SHA-256 signatures)
    EcdsaP256(p256::ecdsa::VerifyingKey),
    /// ECDSA P-384 public key (SHA-384 signatures)
    EcdsaP384(p384::ecdsa::VerifyingKey),
}

impl PartialEq for PublicKey {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Rsa(a), Self::Rsa(b)) => a == b,
            (Self::Ed25519(a), Self::Ed25519(b)) => a == b,
            (Self::EcdsaP256(a), Self::EcdsaP256(b)) => {
                a.to_encoded_point(false) == b.to_encoded_point(false)
            }
            (Self::EcdsaP384(a), Self::EcdsaP384(b)) => {
                a.to_encoded_point(false) == b.to_encoded_point(false)
            }
            _ => false,
        }
    }
}

impl Eq for PublicKey {}

impl PublicKey {
    /// Attempts to parse a public key from a string in various formats.
    ///
    /// Supported formats:
    /// - OpenSSH (RSA, ED25519, ECDSA)
    /// - PKCS#8 PEM (RSA, ED25519, ECDSA)
    /// - PKCS#1 PEM (RSA)
    ///
    /// Supported key types:
    /// - RSA (2048–8192 bits, minimum 2048 enforced)
    /// - ED25519
    /// - ECDSA P-256, P-384
    pub fn parse(key_content: &str) -> Result<Self> {
        // Try OpenSSH format first (most common)
        if let Ok(ssh_key) = SshPublicKey::from_openssh(key_content) {
            return Self::from_ssh_public_key(ssh_key);
        }

        // Try PKCS#8 PEM (RSA)
        if let Ok(rsa_key) = RsaPublicKey::from_public_key_pem(key_content) {
            return Self::from_rsa_key(rsa_key);
        }

        // Try PKCS#1 PEM (RSA)
        if let Ok(rsa_key) = rsa::pkcs1::DecodeRsaPublicKey::from_pkcs1_pem(key_content) {
            return Self::from_rsa_key(rsa_key);
        }

        // Try PKCS#8 PEM (ED25519)
        if let Ok(ed_key) = ed25519_dalek::VerifyingKey::from_public_key_pem(key_content) {
            return Ok(Self::Ed25519(ed_key));
        }

        // Try PKCS#8 PEM (ECDSA P-256)
        if let Ok(p256_key) = p256::PublicKey::from_public_key_pem(key_content) {
            return Ok(Self::EcdsaP256(p256::ecdsa::VerifyingKey::from(&p256_key)));
        }

        // Try PKCS#8 PEM (ECDSA P-384)
        if let Ok(p384_key) = p384::PublicKey::from_public_key_pem(key_content) {
            return Ok(Self::EcdsaP384(p384::ecdsa::VerifyingKey::from(&p384_key)));
        }

        bail!("Could not parse public key in any supported format")
    }

    /// Converts from an SSH public key.
    fn from_ssh_public_key(ssh_key: SshPublicKey) -> Result<Self> {
        match ssh_key.algorithm() {
            Algorithm::Rsa { .. } => {
                let key_data = ssh_key.key_data();
                let rsa_public = key_data.rsa().context("Failed to extract RSA key data")?;

                let n = rsa::BigUint::from_bytes_be(rsa_public.n.as_bytes());
                let e = rsa::BigUint::from_bytes_be(rsa_public.e.as_bytes());

                let rsa_key = RsaPublicKey::new(n, e)
                    .context("Failed to construct RSA public key from SSH components")?;

                Self::from_rsa_key(rsa_key)
            }
            Algorithm::Ed25519 => {
                let key_data = ssh_key.key_data();
                let ed25519_public = key_data
                    .ed25519()
                    .context("Failed to extract ED25519 key data")?;

                let verifying_key =
                    ed25519_dalek::VerifyingKey::from_bytes(ed25519_public.as_ref())
                        .context("Invalid ED25519 public key bytes")?;

                Ok(Self::Ed25519(verifying_key))
            }
            Algorithm::Ecdsa { curve } => {
                let key_data = ssh_key.key_data();
                let ecdsa_public = key_data
                    .ecdsa()
                    .context("Failed to extract ECDSA key data")?;

                Self::from_ecdsa_key(ecdsa_public, &curve)
            }
            algo => bail!(
                "Unsupported key algorithm: {:?}. Supported: RSA, ED25519, ECDSA (P-256, P-384).",
                algo
            ),
        }
    }

    /// Converts from an RSA public key, enforcing minimum key size.
    fn from_rsa_key(rsa_key: RsaPublicKey) -> Result<Self> {
        let key_size_bits = rsa_key.size() * 8;
        if key_size_bits < 2048 {
            bail!(
                "RSA key is too weak: {} bits (minimum required: 2048 bits)",
                key_size_bits
            );
        }
        Ok(Self::Rsa(rsa_key))
    }

    /// Converts from an ECDSA public key in SSH format.
    fn from_ecdsa_key(ecdsa_public: &EcdsaPublicKey, curve: &EcdsaCurve) -> Result<Self> {
        match curve {
            EcdsaCurve::NistP256 => {
                let point_bytes = ecdsa_public.as_ref();
                let public_key = p256::PublicKey::from_sec1_bytes(point_bytes)
                    .map_err(|e| anyhow::anyhow!("Invalid P-256 public key: {e}"))?;
                Ok(Self::EcdsaP256(p256::ecdsa::VerifyingKey::from(&public_key)))
            }
            EcdsaCurve::NistP384 => {
                let point_bytes = ecdsa_public.as_ref();
                let public_key = p384::PublicKey::from_sec1_bytes(point_bytes)
                    .map_err(|e| anyhow::anyhow!("Invalid P-384 public key: {e}"))?;
                Ok(Self::EcdsaP384(p384::ecdsa::VerifyingKey::from(&public_key)))
            }
            EcdsaCurve::NistP521 => {
                bail!("ECDSA P-521 curve is not supported")
            }
        }
    }

    /// Converts to OpenSSH public key format string for serialization.
    pub(super) fn to_ssh_public_key_string(&self) -> Result<String> {
        match self {
            Self::Rsa(rsa_key) => {
                let n_bytes = rsa_key.n().to_bytes_be();
                let e_bytes = rsa_key.e().to_bytes_be();

                let ssh_rsa = ssh_key::public::RsaPublicKey {
                    e: ssh_key::Mpint::from_positive_bytes(&e_bytes)
                        .context("Failed to create Mpint for e")?,
                    n: ssh_key::Mpint::from_positive_bytes(&n_bytes)
                        .context("Failed to create Mpint for n")?,
                };
                let public_key = SshPublicKey::from(ssh_rsa);
                public_key
                    .to_openssh()
                    .map_err(|e| anyhow::anyhow!("Failed to encode RSA key to OpenSSH: {e}"))
            }
            Self::Ed25519(verifying_key) => {
                let ssh_ed25519 =
                    ssh_key::public::Ed25519PublicKey::try_from(verifying_key.as_bytes().as_ref())
                        .context("Failed to create ED25519 SSH key")?;
                let public_key = SshPublicKey::from(ssh_ed25519);
                public_key
                    .to_openssh()
                    .map_err(|e| anyhow::anyhow!("Failed to encode ED25519 key to OpenSSH: {e}"))
            }
            Self::EcdsaP256(verifying_key) => {
                let encoded_point = verifying_key.to_encoded_point(false);
                let ssh_ecdsa = ssh_key::public::EcdsaPublicKey::NistP256(encoded_point);
                let public_key = SshPublicKey::from(ssh_ecdsa);
                public_key
                    .to_openssh()
                    .context("Failed to encode ECDSA P-256 key to OpenSSH")
            }
            Self::EcdsaP384(verifying_key) => {
                let encoded_point = verifying_key.to_encoded_point(false);
                let ssh_ecdsa = ssh_key::public::EcdsaPublicKey::NistP384(encoded_point);
                let public_key = SshPublicKey::from(ssh_ecdsa);
                public_key
                    .to_openssh()
                    .context("Failed to encode ECDSA P-384 key to OpenSSH")
            }
        }
    }

    /// Verify a signature using the appropriate algorithm for the key type.
    pub fn verify(&self, message: &[u8], signature: &[u8]) -> Result<()> {
        match self {
            Self::Rsa(rsa_key) => {
                use rsa::pkcs1v15::Signature;
                let verifying_key =
                    rsa::pkcs1v15::VerifyingKey::<sha2::Sha256>::new(rsa_key.clone());
                let sig = Signature::try_from(signature)
                    .context("Invalid RSA signature format")?;
                RsaVerifier::verify(&verifying_key, message, &sig)
                    .map_err(|_| anyhow::anyhow!("RSA signature verification failed"))
            }
            Self::Ed25519(verifying_key) => {
                let sig = ed25519_dalek::Signature::from_slice(signature)
                    .context("Invalid ED25519 signature format")?;
                verifying_key
                    .verify(message, &sig)
                    .map_err(|_| anyhow::anyhow!("ED25519 signature verification failed"))
            }
            Self::EcdsaP256(verifying_key) => {
                use p256::ecdsa::signature::Verifier;
                let sig = p256::ecdsa::DerSignature::from_bytes(signature.into())
                    .context("Invalid ECDSA P-256 signature format")?;
                verifying_key
                    .verify(message, &sig)
                    .map_err(|_| anyhow::anyhow!("ECDSA P-256 signature verification failed"))
            }
            Self::EcdsaP384(verifying_key) => {
                use p384::ecdsa::signature::Verifier;
                let sig = p384::ecdsa::DerSignature::from_bytes(signature.into())
                    .context("Invalid ECDSA P-384 signature format")?;
                verifying_key
                    .verify(message, &sig)
                    .map_err(|_| anyhow::anyhow!("ECDSA P-384 signature verification failed"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Test key material (generated at test time)
    // =========================================================================

    fn gen_ed25519_openssh() -> (ed25519_dalek::SigningKey, String) {
        use ring::rand::SecureRandom;
        let rng = ring::rand::SystemRandom::new();
        let mut secret = [0u8; 32];
        rng.fill(&mut secret).unwrap();
        let signing_key = ed25519_dalek::SigningKey::from_bytes(&secret);
        let verifying_key = signing_key.verifying_key();
        let ssh_ed25519 =
            ssh_key::public::Ed25519PublicKey::try_from(verifying_key.as_bytes().as_ref())
                .unwrap();
        let ssh_pub = SshPublicKey::from(ssh_ed25519);
        (signing_key, ssh_pub.to_openssh().unwrap())
    }

    fn gen_p256_openssh() -> (p256::ecdsa::SigningKey, String) {
        use ring::rand::SecureRandom;
        let rng = ring::rand::SystemRandom::new();
        let mut scalar = [0u8; 32];
        // Keep generating until we get a valid scalar (non-zero, < order)
        let signing_key = loop {
            rng.fill(&mut scalar).unwrap();
            if let Ok(sk) = p256::ecdsa::SigningKey::from_slice(&scalar) {
                break sk;
            }
        };
        let verifying_key = signing_key.verifying_key();
        let encoded_point = verifying_key.to_encoded_point(false);
        let ssh_ecdsa = ssh_key::public::EcdsaPublicKey::NistP256(encoded_point);
        let ssh_pub = SshPublicKey::from(ssh_ecdsa);
        (signing_key, ssh_pub.to_openssh().unwrap())
    }

    fn gen_p384_openssh() -> (p384::ecdsa::SigningKey, String) {
        use ring::rand::SecureRandom;
        let rng = ring::rand::SystemRandom::new();
        let mut scalar = [0u8; 48];
        let signing_key = loop {
            rng.fill(&mut scalar).unwrap();
            if let Ok(sk) = p384::ecdsa::SigningKey::from_slice(&scalar) {
                break sk;
            }
        };
        let verifying_key = signing_key.verifying_key();
        let encoded_point = verifying_key.to_encoded_point(false);
        let ssh_ecdsa = ssh_key::public::EcdsaPublicKey::NistP384(encoded_point);
        let ssh_pub = SshPublicKey::from(ssh_ecdsa);
        (signing_key, ssh_pub.to_openssh().unwrap())
    }

    // =========================================================================
    // Parsing tests
    // =========================================================================

    #[test]
    fn parse_ed25519_openssh() {
        let (_sk, openssh) = gen_ed25519_openssh();
        let key = PublicKey::parse(&openssh).unwrap();
        assert!(matches!(key, PublicKey::Ed25519(_)));
    }

    #[test]
    fn parse_ecdsa_p256_openssh() {
        let (_sk, openssh) = gen_p256_openssh();
        let key = PublicKey::parse(&openssh).unwrap();
        assert!(matches!(key, PublicKey::EcdsaP256(_)));
    }

    #[test]
    fn parse_ecdsa_p384_openssh() {
        let (_sk, openssh) = gen_p384_openssh();
        let key = PublicKey::parse(&openssh).unwrap();
        assert!(matches!(key, PublicKey::EcdsaP384(_)));
    }

    #[test]
    fn parse_ed25519_pkcs8_pem() {
        use ed25519_dalek::pkcs8::spki::EncodePublicKey;
        use ring::rand::SecureRandom;
        let rng = ring::rand::SystemRandom::new();
        let mut secret = [0u8; 32];
        rng.fill(&mut secret).unwrap();
        let signing_key = ed25519_dalek::SigningKey::from_bytes(&secret);
        let pem = signing_key
            .verifying_key()
            .to_public_key_pem(Default::default())
            .unwrap();
        let key = PublicKey::parse(&pem).unwrap();
        assert!(matches!(key, PublicKey::Ed25519(_)));
    }

    #[test]
    fn parse_garbage_fails() {
        assert!(PublicKey::parse("not a valid key").is_err());
    }

    // =========================================================================
    // Roundtrip: parse → to_ssh_public_key_string → parse again
    // =========================================================================

    #[test]
    fn roundtrip_ed25519() {
        let (_sk, openssh) = gen_ed25519_openssh();
        let key = PublicKey::parse(&openssh).unwrap();
        let ssh_str = key.to_ssh_public_key_string().unwrap();
        let key2 = PublicKey::parse(&ssh_str).unwrap();
        assert_eq!(key, key2);
    }

    #[test]
    fn roundtrip_p256() {
        let (_sk, openssh) = gen_p256_openssh();
        let key = PublicKey::parse(&openssh).unwrap();
        let ssh_str = key.to_ssh_public_key_string().unwrap();
        let key2 = PublicKey::parse(&ssh_str).unwrap();
        assert_eq!(key, key2);
    }

    #[test]
    fn roundtrip_p384() {
        let (_sk, openssh) = gen_p384_openssh();
        let key = PublicKey::parse(&openssh).unwrap();
        let ssh_str = key.to_ssh_public_key_string().unwrap();
        let key2 = PublicKey::parse(&ssh_str).unwrap();
        assert_eq!(key, key2);
    }

    // =========================================================================
    // Signature verification
    // =========================================================================

    #[test]
    fn verify_ed25519_signature() {
        use ed25519_dalek::Signer;
        let (sk, openssh) = gen_ed25519_openssh();
        let key = PublicKey::parse(&openssh).unwrap();
        let message = b"hello world";
        let sig = sk.sign(message);
        assert!(key.verify(message, &sig.to_bytes()).is_ok());
    }

    #[test]
    fn verify_ed25519_wrong_signature() {
        let (_sk, openssh) = gen_ed25519_openssh();
        let key = PublicKey::parse(&openssh).unwrap();
        let bad_sig = [0u8; 64];
        assert!(key.verify(b"hello", &bad_sig).is_err());
    }

    #[test]
    fn verify_p256_signature() {
        use p256::ecdsa::{SigningKey, signature::Signer};
        let (sk, openssh) = gen_p256_openssh();
        let key = PublicKey::parse(&openssh).unwrap();
        let message = b"test message";
        let sig: p256::ecdsa::DerSignature = sk.sign(message);
        assert!(key.verify(message, sig.as_bytes()).is_ok());
    }

    #[test]
    fn verify_p384_signature() {
        use p384::ecdsa::{SigningKey, signature::Signer};
        let (sk, openssh) = gen_p384_openssh();
        let key = PublicKey::parse(&openssh).unwrap();
        let message = b"test message";
        let sig: p384::ecdsa::DerSignature = sk.sign(message);
        assert!(key.verify(message, sig.as_bytes()).is_ok());
    }

    // =========================================================================
    // PartialEq
    // =========================================================================

    #[test]
    fn equality_same_key() {
        let (_sk, openssh) = gen_ed25519_openssh();
        let k1 = PublicKey::parse(&openssh).unwrap();
        let k2 = PublicKey::parse(&openssh).unwrap();
        assert_eq!(k1, k2);
    }

    #[test]
    fn inequality_different_keys() {
        let (_sk1, openssh1) = gen_ed25519_openssh();
        let (_sk2, openssh2) = gen_ed25519_openssh();
        let k1 = PublicKey::parse(&openssh1).unwrap();
        let k2 = PublicKey::parse(&openssh2).unwrap();
        assert_ne!(k1, k2);
    }

    #[test]
    fn inequality_different_types() {
        let (_sk1, openssh1) = gen_ed25519_openssh();
        let (_sk2, openssh2) = gen_p256_openssh();
        let k1 = PublicKey::parse(&openssh1).unwrap();
        let k2 = PublicKey::parse(&openssh2).unwrap();
        assert_ne!(k1, k2);
    }
}

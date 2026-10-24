use crate::error::{Error, Result};
use pem;
use ring::signature::{
    ECDSA_P256_SHA256_ASN1_SIGNING, ECDSA_P384_SHA384_ASN1_SIGNING, EcdsaKeyPair, Ed25519KeyPair,
    RSA_PKCS1_SHA256, RsaKeyPair,
};
use rsa::RsaPrivateKey;
use rsa::pkcs8::{DecodePrivateKey, EncodePrivateKey};
use rsa::traits::PublicKeyParts;
use ssh_key::{Algorithm, EcdsaCurve, PrivateKey as SshPrivateKey};

fn key_error(message: impl Into<String>) -> Error {
    Error::PrivateKey(message.into())
}

enum KeyPair {
    Rsa(RsaKeyPair),
    Ed25519(Ed25519KeyPair),
    EcdsaP256(EcdsaKeyPair),
    EcdsaP384(EcdsaKeyPair),
}

pub struct ParsedPrivateKey {
    key_pair: KeyPair,
}

impl ParsedPrivateKey {
    pub fn parse(key_content: &str) -> Result<Self> {
        if let Ok(ssh_key) = SshPrivateKey::from_openssh(key_content) {
            return Self::from_ssh_key(ssh_key);
        }

        if key_content.contains("-----BEGIN") {
            return Self::from_pem(key_content);
        }

        Err(key_error(
            "Could not parse private key. Supported formats: OpenSSH (RSA/ED25519/ECDSA), PKCS#8 PEM, PKCS#1 PEM",
        ))
    }

    fn from_ssh_key(ssh_key: SshPrivateKey) -> Result<Self> {
        match ssh_key.algorithm() {
            Algorithm::Rsa { .. } => Self::rsa_from_ssh_key(ssh_key),
            Algorithm::Ed25519 => Self::ed25519_from_ssh_key(ssh_key),
            Algorithm::Ecdsa { curve } => Self::ecdsa_from_ssh_key(ssh_key, &curve),
            algo => Err(key_error(format!(
                "Unsupported key algorithm: {algo:?}. Supported: RSA, ED25519, ECDSA (P-256, P-384)."
            ))),
        }
    }

    fn from_pem(key_content: &str) -> Result<Self> {
        if let Ok(rsa_key) = RsaPrivateKey::from_pkcs8_pem(key_content) {
            Self::check_rsa_key_size(&rsa_key)?;

            let pkcs8_der = rsa_key
                .to_pkcs8_der()
                .map_err(|e| key_error(format!("Failed to encode RSA key as PKCS#8 DER: {e}")))?;

            let key_pair = RsaKeyPair::from_pkcs8(pkcs8_der.as_bytes())
                .map_err(|e| key_error(format!("Failed to create RSA key pair: {e:?}")))?;

            return Ok(Self {
                key_pair: KeyPair::Rsa(key_pair),
            });
        }

        if let Ok(pkcs8_der) = pem::parse(key_content) {
            if let Ok(key_pair) = Ed25519KeyPair::from_pkcs8(pkcs8_der.contents()) {
                return Ok(Self {
                    key_pair: KeyPair::Ed25519(key_pair),
                });
            }

            let rng = ring::rand::SystemRandom::new();

            if let Ok(key_pair) = EcdsaKeyPair::from_pkcs8(
                &ECDSA_P256_SHA256_ASN1_SIGNING,
                pkcs8_der.contents(),
                &rng,
            ) {
                return Ok(Self {
                    key_pair: KeyPair::EcdsaP256(key_pair),
                });
            }

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

        Err(key_error(
            "Failed to parse PEM as RSA (PKCS#8/PKCS#1), ED25519 (PKCS#8), or ECDSA (PKCS#8)",
        ))
    }

    fn rsa_from_ssh_key(ssh_key: SshPrivateKey) -> Result<Self> {
        let key_data = ssh_key.key_data();
        let rsa_keypair = key_data
            .rsa()
            .ok_or_else(|| key_error("Failed to extract RSA key data"))?;

        let n = rsa::BigUint::from_bytes_be(rsa_keypair.public.n.as_bytes());
        let e = rsa::BigUint::from_bytes_be(rsa_keypair.public.e.as_bytes());
        let d = rsa::BigUint::from_bytes_be(rsa_keypair.private.d.as_bytes());
        let p = rsa::BigUint::from_bytes_be(rsa_keypair.private.p.as_bytes());
        let q = rsa::BigUint::from_bytes_be(rsa_keypair.private.q.as_bytes());

        let rsa_key = RsaPrivateKey::from_components(n, e, d, vec![p, q]).map_err(|e| {
            key_error(format!(
                "Failed to construct RSA private key from SSH components: {e}"
            ))
        })?;

        Self::check_rsa_key_size(&rsa_key)?;

        let pkcs8_der = rsa_key
            .to_pkcs8_der()
            .map_err(|e| key_error(format!("Failed to encode RSA key as PKCS#8 DER: {e}")))?;

        let key_pair = RsaKeyPair::from_pkcs8(pkcs8_der.as_bytes())
            .map_err(|e| key_error(format!("Failed to create RSA key pair: {e:?}")))?;

        Ok(Self {
            key_pair: KeyPair::Rsa(key_pair),
        })
    }

    fn ed25519_from_ssh_key(ssh_key: SshPrivateKey) -> Result<Self> {
        let key_data = ssh_key.key_data();
        let ed25519_keypair = key_data
            .ed25519()
            .ok_or_else(|| key_error("Failed to extract ED25519 key data"))?;

        let private_bytes = ed25519_keypair.private.as_ref();
        let public_bytes = ed25519_keypair.public.as_ref();

        if private_bytes.len() != 32 {
            return Err(key_error(format!(
                "Invalid ED25519 private key length: expected 32 bytes, got {}",
                private_bytes.len()
            )));
        }

        let key_pair = Ed25519KeyPair::from_seed_and_public_key(private_bytes, public_bytes)
            .map_err(|e| key_error(format!("Failed to create ED25519 key pair: {e:?}")))?;

        Ok(Self {
            key_pair: KeyPair::Ed25519(key_pair),
        })
    }

    fn ecdsa_from_ssh_key(ssh_key: SshPrivateKey, curve: &EcdsaCurve) -> Result<Self> {
        let key_data = ssh_key.key_data();
        let ecdsa_keypair = key_data
            .ecdsa()
            .ok_or_else(|| key_error("Failed to extract ECDSA key data"))?;

        let private_bytes = match ecdsa_keypair {
            ssh_key::private::EcdsaKeypair::NistP256 { private, .. } => private.as_slice(),
            ssh_key::private::EcdsaKeypair::NistP384 { private, .. } => private.as_slice(),
            ssh_key::private::EcdsaKeypair::NistP521 { private, .. } => private.as_slice(),
        };

        let pkcs8_der = Self::ecdsa_to_pkcs8_der(private_bytes, curve)?;
        let rng = ring::rand::SystemRandom::new();

        let key_pair = match curve {
            EcdsaCurve::NistP256 => KeyPair::EcdsaP256(
                EcdsaKeyPair::from_pkcs8(&ECDSA_P256_SHA256_ASN1_SIGNING, &pkcs8_der, &rng)
                    .map_err(|e| {
                        key_error(format!("Failed to create ECDSA P-256 key pair: {e:?}"))
                    })?,
            ),
            EcdsaCurve::NistP384 => KeyPair::EcdsaP384(
                EcdsaKeyPair::from_pkcs8(&ECDSA_P384_SHA384_ASN1_SIGNING, &pkcs8_der, &rng)
                    .map_err(|e| {
                        key_error(format!("Failed to create ECDSA P-384 key pair: {e:?}"))
                    })?,
            ),
            EcdsaCurve::NistP521 => {
                return Err(key_error(
                    "ECDSA P-521 curve is not supported by the ring crate",
                ));
            }
        };

        Ok(Self { key_pair })
    }

    fn ecdsa_to_pkcs8_der(private_bytes: &[u8], curve: &ssh_key::EcdsaCurve) -> Result<Vec<u8>> {
        use ssh_key::EcdsaCurve;

        match curve {
            EcdsaCurve::NistP256 => {
                use p256::pkcs8::EncodePrivateKey;
                let secret_key = p256::SecretKey::from_slice(private_bytes)
                    .map_err(|e| key_error(format!("Failed to parse P-256 private key: {e}")))?;
                let pkcs8_der = secret_key
                    .to_pkcs8_der()
                    .map_err(|e| key_error(format!("Failed to encode P-256 key as PKCS#8: {e}")))?;
                Ok(pkcs8_der.as_bytes().to_vec())
            }
            EcdsaCurve::NistP384 => {
                use p384::pkcs8::EncodePrivateKey;
                let secret_key = p384::SecretKey::from_slice(private_bytes)
                    .map_err(|e| key_error(format!("Failed to parse P-384 private key: {e}")))?;
                let pkcs8_der = secret_key
                    .to_pkcs8_der()
                    .map_err(|e| key_error(format!("Failed to encode P-384 key as PKCS#8: {e}")))?;
                Ok(pkcs8_der.as_bytes().to_vec())
            }
            EcdsaCurve::NistP521 => Err(key_error("P-521 curve is not supported")),
        }
    }

    fn check_rsa_key_size(rsa_key: &RsaPrivateKey) -> Result<()> {
        let key_size_bits = rsa_key.size() * 8;
        if key_size_bits < 2048 {
            return Err(key_error(format!(
                "RSA key is too weak: {key_size_bits} bits (minimum required: 2048 bits)"
            )));
        }
        Ok(())
    }

    pub fn sign(&self, data: &[u8]) -> Result<Vec<u8>> {
        match &self.key_pair {
            KeyPair::Rsa(rsa_key_pair) => {
                let rng = ring::rand::SystemRandom::new();

                let mut signature = vec![0u8; rsa_key_pair.public().modulus_len()];
                rsa_key_pair
                    .sign(&RSA_PKCS1_SHA256, &rng, data, &mut signature)
                    .map_err(|e| key_error(format!("Failed to sign data with RSA: {e:?}")))?;
                Ok(signature)
            }
            KeyPair::Ed25519(ed25519_key_pair) => {
                let signature = ed25519_key_pair.sign(data);
                Ok(signature.as_ref().to_vec())
            }
            KeyPair::EcdsaP256(ecdsa_key_pair) => {
                let rng = ring::rand::SystemRandom::new();
                let signature = ecdsa_key_pair.sign(&rng, data).map_err(|e| {
                    key_error(format!("Failed to sign data with ECDSA P-256: {e:?}"))
                })?;
                Ok(signature.as_ref().to_vec())
            }
            KeyPair::EcdsaP384(ecdsa_key_pair) => {
                let rng = ring::rand::SystemRandom::new();
                let signature = ecdsa_key_pair.sign(&rng, data).map_err(|e| {
                    key_error(format!("Failed to sign data with ECDSA P-384: {e:?}"))
                })?;
                Ok(signature.as_ref().to_vec())
            }
        }
    }
}

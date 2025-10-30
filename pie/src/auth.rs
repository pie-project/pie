use anyhow::{Context, Result, bail};
use pem;
use ring::signature::{ED25519, RSA_PKCS1_2048_8192_SHA256, UnparsedPublicKey};
use rsa::RsaPublicKey;
use rsa::pkcs1::{DecodeRsaPublicKey, EncodeRsaPublicKey};
use rsa::pkcs8::DecodePublicKey;
use rsa::traits::PublicKeyParts;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use ssh_key::{Algorithm, PublicKey as SshPublicKey};
use std::{collections::HashMap, fs, path::Path};

/// Structure representing the authorized_clients.toml file format.
#[derive(Deserialize, Serialize, Debug, Default)]
pub struct AuthorizedClients {
    /// Map of username to their list of authorized public keys
    #[serde(default)]
    clients: HashMap<String, ClientKeys>,
}

impl AuthorizedClients {
    /// Loads the authorized clients from the given TOML file.
    pub fn load(auth_path: &Path) -> Result<Self> {
        let content = fs::read_to_string(auth_path).context(format!(
            "Failed to read authorized clients file at {:?}",
            auth_path
        ))?;
        toml::from_str(&content).context(format!(
            "Failed to parse authorized clients file at {:?}",
            auth_path
        ))
    }

    /// Saves the authorized clients to the given TOML file.
    pub fn save(&self, auth_path: &Path) -> Result<()> {
        if let Some(parent) = auth_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create parent dir for {:?}", auth_path))?;
        }
        let content = toml::to_string_pretty(self)
            .context(format!("Failed to serialize authorized clients to TOML"))?;
        fs::write(auth_path, content).context(format!(
            "Failed to write authorized clients file at {:?}",
            auth_path
        ))
    }

    /// Checks if the authorized clients are empty.
    pub fn is_empty(&self) -> bool {
        self.clients.is_empty()
    }

    /// Returns the number of authorized clients.
    pub fn len(&self) -> usize {
        self.clients.len()
    }

    /// Returns an iterator over the authorized clients.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &ClientKeys)> {
        self.clients.iter()
    }

    /// Returns the client keys for the given username.
    pub fn get(&self, username: &str) -> Option<&ClientKeys> {
        self.clients.get(username)
    }

    /// Inserts a new authorized client and its public key into the authorized clients.
    pub fn insert(&mut self, username: &str, public_key: PublicKey) {
        self.clients
            .entry(username.to_owned())
            .and_modify(|client_keys| {
                // Check if key already exists
                if !client_keys.keys.contains(&public_key) {
                    client_keys.keys.push(public_key.clone());
                    println!("Added new key to existing user '{}'", username);
                } else {
                    println!("Key already exists for user '{}'", username);
                }
            })
            .or_insert_with(|| {
                println!("Created new user '{}'", username);
                ClientKeys::new(public_key)
            });
    }

    /// Removes an authorized client and its public keys from the authorized clients.
    pub fn remove(&mut self, username: &str) -> Option<ClientKeys> {
        self.clients.remove(username)
    }
}

/// A public key that can be used for signature verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PublicKey {
    /// RSA public key stored in PKCS#1 DER format
    Rsa(Vec<u8>),
    /// ED25519 public key stored as raw 32-byte key
    Ed25519(Box<[u8; 32]>),
}

impl PublicKey {
    /// Attempts to parse a public key from a string in various formats.
    ///
    /// Supported formats:
    /// - OpenSSH (RSA, ED25519)
    /// - PKCS#8 PEM (RSA, ED25519)
    /// - PKCS#1 PEM (RSA)
    pub fn parse(key_content: &str) -> Result<Self> {
        // Try parsing as OpenSSH format first (most common)
        if let Ok(ssh_key) = SshPublicKey::from_openssh(key_content) {
            return Self::from_ssh_public_key(ssh_key);
        }

        // Try parsing as PKCS#8 PEM format (RSA)
        if let Ok(rsa_key) = RsaPublicKey::from_public_key_pem(key_content) {
            return Self::from_rsa_key(rsa_key);
        }

        // Try parsing as PKCS#1 PEM format (RSA)
        if let Ok(rsa_key) = RsaPublicKey::from_pkcs1_pem(key_content) {
            return Self::from_rsa_key(rsa_key);
        }

        // Try parsing as PKCS#8 PEM format (ED25519)
        if let Ok(pkcs8_der) = pem::parse(key_content) {
            // Try to parse as ED25519 public key from PKCS#8
            if let Ok(ed25519_key) =
                ssh_key::public::Ed25519PublicKey::try_from(pkcs8_der.contents())
            {
                return Self::from_ed25519_key(ed25519_key.as_ref());
            }
        }

        bail!("Could not parse public key in any supported format")
    }

    /// Converts from an SSH public key.
    fn from_ssh_public_key(ssh_key: SshPublicKey) -> Result<Self> {
        match ssh_key.algorithm() {
            Algorithm::Rsa { .. } => {
                // Extract the RSA components
                let key_data = ssh_key.key_data();
                let rsa_public = key_data.rsa().context("Failed to extract RSA key data")?;

                // Convert to rsa crate's RsaPublicKey
                let n = rsa::BigUint::from_bytes_be(rsa_public.n.as_bytes());
                let e = rsa::BigUint::from_bytes_be(rsa_public.e.as_bytes());

                let rsa_key = RsaPublicKey::new(n, e)
                    .context("Failed to construct RSA public key from SSH key components")?;

                Self::from_rsa_key(rsa_key)
            }
            Algorithm::Ed25519 => {
                // Extract the ED25519 public key bytes
                let key_data = ssh_key.key_data();
                let ed25519_public = key_data
                    .ed25519()
                    .context("Failed to extract ED25519 key data")?;

                Self::from_ed25519_key(ed25519_public.as_ref())
            }
            algo => bail!(
                "Unsupported key algorithm: {:?}. Supported: RSA, ED25519.",
                algo
            ),
        }
    }

    /// Converts from an RSA public key.
    fn from_rsa_key(rsa_key: RsaPublicKey) -> Result<Self> {
        // Convert to PKCS#1 DER format for ring verification
        let pkcs1_der = rsa_key
            .to_pkcs1_der()
            .context("Failed to encode public key as PKCS#1 DER")?
            .as_bytes()
            .to_vec();

        Ok(Self::Rsa(pkcs1_der))
    }

    /// Converts from an ED25519 public key.
    fn from_ed25519_key(public_key_bytes: &[u8; 32]) -> Result<Self> {
        Ok(Self::Ed25519(Box::new(*public_key_bytes)))
    }

    /// Converts to SSH public key format for serialization
    fn to_ssh_public_key_string(&self) -> Result<String> {
        match self {
            Self::Rsa(pkcs1_der) => {
                let rsa_key = RsaPublicKey::from_pkcs1_der(pkcs1_der)
                    .context("Failed to parse PKCS#1 DER")?;

                // Extract RSA components (n and e) as big-endian bytes
                let n_bytes = rsa_key.n().to_bytes_be();
                let e_bytes = rsa_key.e().to_bytes_be();

                // Create SSH RSA public key from components
                let ssh_rsa = ssh_key::public::RsaPublicKey {
                    e: ssh_key::Mpint::from_positive_bytes(&e_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to create Mpint for e: {}", e))?,
                    n: ssh_key::Mpint::from_positive_bytes(&n_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to create Mpint for n: {}", e))?,
                };
                let public_key = SshPublicKey::from(ssh_rsa);
                Ok(public_key.to_openssh().map_err(|e| {
                    anyhow::anyhow!("Failed to encode RSA key to OpenSSH format: {}", e)
                })?)
            }
            Self::Ed25519(bytes) => {
                let ssh_ed25519 =
                    ssh_key::public::Ed25519PublicKey::try_from(bytes.as_ref().as_ref())
                        .map_err(|e| anyhow::anyhow!("Failed to create ED25519 SSH key: {}", e))?;
                let public_key = SshPublicKey::from(ssh_ed25519);
                Ok(public_key.to_openssh().map_err(|e| {
                    anyhow::anyhow!("Failed to encode ED25519 key to OpenSSH format: {}", e)
                })?)
            }
        }
    }

    /// Verify a signature using the appropriate algorithm for the key type.
    pub fn verify(&self, message: &[u8], signature: &[u8]) -> Result<()> {
        match self {
            Self::Rsa(pkcs1_der) => {
                let public_key = UnparsedPublicKey::new(&RSA_PKCS1_2048_8192_SHA256, pkcs1_der);
                public_key
                    .verify(message, signature)
                    .map_err(|_| anyhow::anyhow!("RSA signature verification failed"))
            }
            Self::Ed25519(bytes) => {
                let public_key = UnparsedPublicKey::new(&ED25519, bytes.as_ref());
                public_key
                    .verify(message, signature)
                    .map_err(|_| anyhow::anyhow!("ED25519 signature verification failed"))
            }
        }
    }
}

/// Structure representing keys for a single client/user.
#[derive(Debug)]
pub struct ClientKeys {
    /// List of authorized public keys for this user
    keys: Vec<PublicKey>,
}

impl ClientKeys {
    fn new(public_key: PublicKey) -> Self {
        Self {
            keys: vec![public_key],
        }
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn keys(&self) -> &[PublicKey] {
        &self.keys
    }
}

impl Serialize for ClientKeys {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;

        // Serialize keys to SSH public key format (OpenSSH)
        let keys_pem: Result<Vec<String>, _> = self
            .keys
            .iter()
            .map(|key| {
                key.to_ssh_public_key_string()
                    .map_err(serde::ser::Error::custom)
            })
            .collect();

        let keys_pem = keys_pem?;
        let mut state = serializer.serialize_struct("ClientKeys", 1)?;
        state.serialize_field("keys", &keys_pem)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for ClientKeys {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ClientKeysHelper {
            keys: Vec<String>,
        }

        let helper = ClientKeysHelper::deserialize(deserializer)?;
        let keys: Result<Vec<PublicKey>, _> = helper
            .keys
            .iter()
            .map(|key_str| {
                PublicKey::parse(key_str).map_err(|e| {
                    serde::de::Error::custom(format!("Failed to parse public key: {}", e))
                })
            })
            .collect();

        Ok(ClientKeys { keys: keys? })
    }
}

use anyhow::{Context, Result, bail};
use rsa::RsaPublicKey;
use rsa::pkcs1::DecodeRsaPublicKey;
use rsa::pkcs8::{DecodePublicKey, EncodePublicKey, LineEnding};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use ssh_key::{Algorithm, PublicKey};
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
    pub fn insert(&mut self, username: &str, public_key: RsaPublicKey) {
        self.clients
            .entry(username.to_owned())
            .and_modify(|client_keys| {
                if !client_keys.keys.contains(&public_key) {
                    client_keys.keys.push(public_key.clone());
                    println!("Added new key to existing user '{}'", username);
                } else {
                    println!("Key already exists for user '{}'", username);
                }
            })
            .or_insert_with(|| {
                println!("Created new user '{}'", username);
                ClientKeys {
                    keys: vec![public_key],
                }
            });
    }

    /// Removes an authorized client and its public keys from the authorized clients.
    pub fn remove(&mut self, username: &str) -> Option<ClientKeys> {
        self.clients.remove(username)
    }
}

/// Structure representing keys for a single client/user.
#[derive(Debug)]
pub struct ClientKeys {
    /// List of authorized public keys for this user
    keys: Vec<RsaPublicKey>,
}

impl ClientKeys {
    pub fn new(public_key: RsaPublicKey) -> Self {
        Self {
            keys: vec![public_key],
        }
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }
}

impl Serialize for ClientKeys {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;

        // Serialize keys to canonical PEM format
        let keys_pem: Result<Vec<String>, _> = self
            .keys
            .iter()
            .map(|key| {
                key.to_public_key_pem(LineEnding::LF)
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
        let keys: Result<Vec<RsaPublicKey>, _> = helper
            .keys
            .iter()
            .map(|key_str| {
                parse_rsa_public_key(key_str).map_err(|e| {
                    serde::de::Error::custom(format!("Failed to parse public key: {}", e))
                })
            })
            .collect();

        Ok(ClientKeys { keys: keys? })
    }
}

/// Attempts to parse an RSA public key from a string in various formats.
///
/// Supported formats:
/// - OpenSSH
/// - PKCS#8 PEM
/// - PKCS#1 PEM
pub fn parse_rsa_public_key(key_content: &str) -> Result<RsaPublicKey> {
    // Try parsing as OpenSSH format first (most common for ~/.ssh/id_rsa.pub)
    if let Ok(ssh_key) = PublicKey::from_openssh(key_content) {
        return from_ssh_public_key(ssh_key);
    }

    // Try parsing as PKCS#8 PEM format
    if let Ok(rsa_key) = RsaPublicKey::from_public_key_pem(key_content) {
        return Ok(rsa_key);
    }

    // Try parsing as PKCS#1 PEM format
    if let Ok(rsa_key) = RsaPublicKey::from_pkcs1_pem(key_content) {
        return Ok(rsa_key);
    }

    bail!("Could not parse public key in any supported format")
}

/// Converts an SSH public key to an RSA public key.
fn from_ssh_public_key(ssh_key: PublicKey) -> Result<RsaPublicKey> {
    // Ensure it's an RSA key
    if !matches!(ssh_key.algorithm(), Algorithm::Rsa { .. }) {
        bail!(
            "Only RSA keys are supported, found: {:?}",
            ssh_key.algorithm()
        );
    }

    // Extract the RSA components
    let key_data = ssh_key.key_data();
    let rsa_public = key_data.rsa().context("Failed to extract RSA key data")?;

    // Convert to rsa crate's RsaPublicKey
    let n = rsa::BigUint::from_bytes_be(rsa_public.n.as_bytes());
    let e = rsa::BigUint::from_bytes_be(rsa_public.e.as_bytes());

    let rsa_key = RsaPublicKey::new(n, e)
        .context("Failed to construct RSA public key from SSH key components")?;

    Ok(rsa_key)
}

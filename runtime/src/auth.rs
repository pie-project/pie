//! Auth Service - Authentication and authorization management
//!
//! This module provides a singleton actor for managing user authentication,
//! public key verification, and internal token validation.

use anyhow::{Context, Result, bail};
use pem;
use ring::rand::{SecureRandom, SystemRandom};
use ring::signature::{
    ECDSA_P256_SHA256_ASN1, ECDSA_P384_SHA384_ASN1, ED25519, RSA_PKCS1_2048_8192_SHA256,
    UnparsedPublicKey,
};
use rsa::RsaPublicKey;
use rsa::pkcs1::{DecodeRsaPublicKey, EncodeRsaPublicKey};
use rsa::pkcs8::DecodePublicKey;
use rsa::traits::PublicKeyParts;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use ssh_key::public::EcdsaPublicKey;
use ssh_key::{Algorithm, EcdsaCurve, PublicKey as SshPublicKey};
use std::path::{Path, PathBuf};
use std::{collections::HashMap, fs};
use std::sync::LazyLock;
use tokio::sync::oneshot;

#[cfg(unix)]
use std::fs::OpenOptions;

#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

use crate::actor::{Actor, Handle, SendError};

// =============================================================================
// Actor Setup (Singleton)
// =============================================================================

/// Global singleton Auth actor.
static ACTOR: LazyLock<Actor<Message>> = LazyLock::new(Actor::new);

/// Spawns the Auth actor with configuration.
pub fn spawn(config: AuthConfig) {
    ACTOR.spawn_with::<AuthActor, _>(|| AuthActor::with_config(config));
}

/// Check if the auth actor is spawned.
pub fn is_spawned() -> bool {
    ACTOR.is_spawned()
}

// =============================================================================
// Configuration
// =============================================================================

/// Auth configuration.
#[derive(Debug)]
pub struct AuthConfig {
    pub enable_auth: bool,
    pub authorized_users: AuthorizedUsers,
    pub internal_auth_token: String,
}

// =============================================================================
// Messages
// =============================================================================

/// Messages for the Auth actor.
#[derive(Debug)]
pub enum Message {
    /// Load authorized users from path
    LoadUsers {
        path: PathBuf,
        response: oneshot::Sender<Result<()>>,
    },

    /// Save authorized users to path
    SaveUsers {
        path: PathBuf,
        response: oneshot::Sender<Result<()>>,
    },

    /// Check if a username is authorized
    IsUserAuthorized {
        username: String,
        response: oneshot::Sender<bool>,
    },

    /// Get public keys for a user (for challenge-response auth)
    GetUserKeys {
        username: String,
        response: oneshot::Sender<Option<Vec<PublicKey>>>,
    },

    /// Verify a signature against all user keys
    VerifySignature {
        username: String,
        challenge: Vec<u8>,
        signature: Vec<u8>,
        response: oneshot::Sender<bool>,
    },

    /// Verify internal auth token
    VerifyInternalToken {
        token: String,
        response: oneshot::Sender<bool>,
    },

    /// Insert a new user
    InsertUser {
        username: String,
        response: oneshot::Sender<InsertUserResult>,
    },

    /// Remove a user
    RemoveUser {
        username: String,
        response: oneshot::Sender<RemoveUserResult>,
    },

    /// Insert a key for a user
    InsertKey {
        username: String,
        key_name: String,
        public_key: PublicKey,
        response: oneshot::Sender<InsertKeyResult>,
    },

    /// Remove a key from a user
    RemoveKey {
        username: String,
        key_name: String,
        response: oneshot::Sender<RemoveKeyResult>,
    },

    /// Generate a new challenge for authentication
    GenerateChallenge {
        response: oneshot::Sender<Result<Vec<u8>>>,
    },

    /// List all users
    ListUsers {
        response: oneshot::Sender<Vec<String>>,
    },

    /// Check if auth is enabled
    IsAuthEnabled {
        response: oneshot::Sender<bool>,
    },
}

impl Message {
    /// Sends this message to the Auth actor.
    pub fn send(self) -> Result<(), SendError> {
        ACTOR.send(self)
    }
}

// =============================================================================
// Convenience Wrappers
// =============================================================================

/// Check if authentication is enabled.
pub async fn is_auth_enabled() -> Result<bool> {
    let (tx, rx) = oneshot::channel();
    Message::IsAuthEnabled { response: tx }.send()?;
    Ok(rx.await?)
}

/// Get public keys for a user (for challenge-response auth).
pub async fn get_user_keys(username: String) -> Result<Option<Vec<PublicKey>>> {
    let (tx, rx) = oneshot::channel();
    Message::GetUserKeys { username, response: tx }.send()?;
    Ok(rx.await?)
}

/// Generate a new challenge for authentication.
pub async fn generate_challenge() -> Result<Vec<u8>> {
    let (tx, rx) = oneshot::channel();
    Message::GenerateChallenge { response: tx }.send()?;
    rx.await?
}

/// Verify a signature against all user keys.
pub async fn verify_signature(username: String, challenge: Vec<u8>, signature: Vec<u8>) -> Result<bool> {
    let (tx, rx) = oneshot::channel();
    Message::VerifySignature { username, challenge, signature, response: tx }.send()?;
    Ok(rx.await?)
}

/// Verify internal auth token.
pub async fn verify_internal_token(token: String) -> Result<bool> {
    let (tx, rx) = oneshot::channel();
    Message::VerifyInternalToken { token, response: tx }.send()?;
    Ok(rx.await?)
}

// =============================================================================
// Result Types
// =============================================================================

/// Result of inserting a user
#[derive(Debug, PartialEq)]
pub enum InsertUserResult {
    /// User was created
    CreatedUser,
    /// User already exists
    UserExists,
}

/// Result of inserting a key for a user
#[derive(Debug, PartialEq)]
pub enum InsertKeyResult {
    /// Key was added successfully
    AddedKey,
    /// A key with this name already exists for this user
    KeyNameExists,
    /// User not found
    UserNotFound,
}

/// Result of removing a key
#[derive(Debug, PartialEq)]
pub enum RemoveKeyResult {
    /// Key was removed successfully
    RemovedKey,
    /// Key name not found for this user
    KeyNotFound,
    /// User not found
    UserNotFound,
}

/// Result of removing a user
#[derive(Debug, PartialEq)]
pub enum RemoveUserResult {
    /// User was removed
    RemovedUser,
    /// User not found
    UserNotFound,
}

// =============================================================================
// AuthService (Business Logic)
// =============================================================================

/// The auth service handles all authentication operations.
/// This is the core business logic, separate from the actor message handling.
#[derive(Debug)]
pub struct AuthService {
    enable_auth: bool,
    authorized_users: AuthorizedUsers,
    internal_auth_token: String,
    rng: SystemRandom,
}

impl AuthService {
    pub fn new(config: AuthConfig) -> Self {
        AuthService {
            enable_auth: config.enable_auth,
            authorized_users: config.authorized_users,
            internal_auth_token: config.internal_auth_token,
            rng: SystemRandom::new(),
        }
    }

    // ==================== Auth State ====================

    pub fn is_auth_enabled(&self) -> bool {
        self.enable_auth
    }

    // ==================== User Operations ====================

    pub fn load_users(&mut self, path: &Path) -> Result<()> {
        self.authorized_users = AuthorizedUsers::load(path)?;
        Ok(())
    }

    pub fn save_users(&self, path: &Path) -> Result<()> {
        self.authorized_users.save(path)
    }

    pub fn is_user_authorized(&self, username: &str) -> bool {
        self.authorized_users.get(username).is_some()
    }

    pub fn get_user_keys(&self, username: &str) -> Option<Vec<PublicKey>> {
        self.authorized_users
            .get(username)
            .map(|keys| keys.public_keys().cloned().collect())
    }

    pub fn list_users(&self) -> Vec<String> {
        self.authorized_users.iter().map(|(k, _)| k.clone()).collect()
    }

    pub fn insert_user(&mut self, username: &str) -> InsertUserResult {
        self.authorized_users.insert_user(username)
    }

    pub fn remove_user(&mut self, username: &str) -> RemoveUserResult {
        self.authorized_users.remove_user(username)
    }

    pub fn insert_key(
        &mut self,
        username: &str,
        key_name: String,
        public_key: PublicKey,
    ) -> InsertKeyResult {
        self.authorized_users.insert_key_for_user(username, key_name, public_key)
    }

    pub fn remove_key(&mut self, username: &str, key_name: &str) -> RemoveKeyResult {
        self.authorized_users.remove_key(username, key_name)
    }

    // ==================== Authentication ====================

    pub fn generate_challenge(&self) -> Result<Vec<u8>> {
        let mut challenge = [0u8; 48];
        self.rng.fill(&mut challenge).map_err(|e| {
            anyhow::anyhow!("Failed to generate random challenge: {}", e)
        })?;
        Ok(challenge.to_vec())
    }

    pub fn verify_signature(&self, username: &str, challenge: &[u8], signature: &[u8]) -> bool {
        if let Some(user_keys) = self.authorized_users.get(username) {
            user_keys
                .public_keys()
                .any(|key| key.verify(challenge, signature).is_ok())
        } else {
            false
        }
    }

    pub fn verify_internal_token(&self, token: &str) -> bool {
        token == self.internal_auth_token
    }
}

// =============================================================================
// AuthActor
// =============================================================================

struct AuthActor {
    service: AuthService,
}

impl AuthActor {
    fn with_config(config: AuthConfig) -> Self {
        AuthActor {
            service: AuthService::new(config),
        }
    }
}

impl Handle for AuthActor {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::LoadUsers { path, response } => {
                let result = self.service.load_users(&path);
                let _ = response.send(result);
            }
            Message::SaveUsers { path, response } => {
                let result = self.service.save_users(&path);
                let _ = response.send(result);
            }
            Message::IsUserAuthorized { username, response } => {
                let result = self.service.is_user_authorized(&username);
                let _ = response.send(result);
            }
            Message::GetUserKeys { username, response } => {
                let result = self.service.get_user_keys(&username);
                let _ = response.send(result);
            }
            Message::VerifySignature {
                username,
                challenge,
                signature,
                response,
            } => {
                let result = self.service.verify_signature(&username, &challenge, &signature);
                let _ = response.send(result);
            }
            Message::VerifyInternalToken { token, response } => {
                let result = self.service.verify_internal_token(&token);
                let _ = response.send(result);
            }
            Message::InsertUser { username, response } => {
                let result = self.service.insert_user(&username);
                let _ = response.send(result);
            }
            Message::RemoveUser { username, response } => {
                let result = self.service.remove_user(&username);
                let _ = response.send(result);
            }
            Message::InsertKey {
                username,
                key_name,
                public_key,
                response,
            } => {
                let result = self.service.insert_key(&username, key_name, public_key);
                let _ = response.send(result);
            }
            Message::RemoveKey {
                username,
                key_name,
                response,
            } => {
                let result = self.service.remove_key(&username, &key_name);
                let _ = response.send(result);
            }
            Message::GenerateChallenge { response } => {
                let result = self.service.generate_challenge();
                let _ = response.send(result);
            }
            Message::ListUsers { response } => {
                let result = self.service.list_users();
                let _ = response.send(result);
            }
            Message::IsAuthEnabled { response } => {
                let result = self.service.is_auth_enabled();
                let _ = response.send(result);
            }
        }
    }
}

// =============================================================================
// AuthorizedUsers
// =============================================================================

/// Structure representing the authorized_users.toml file format.
#[derive(Deserialize, Serialize, Debug, Default)]
pub struct AuthorizedUsers {
    /// Map of username to their list of authorized public keys
    #[serde(default)]
    users: HashMap<String, UserKeys>,
}

impl AuthorizedUsers {
    /// Loads the authorized users from the given TOML file.
    pub fn load(auth_path: &Path) -> Result<Self> {
        // Check file permissions (Unix only)
        #[cfg(unix)]
        check_file_permissions(auth_path)?;

        let content = fs::read_to_string(auth_path).context(format!(
            "Failed to read authorized users file at '{}'",
            auth_path.display()
        ))?;
        toml::from_str(&content).context(format!(
            "Failed to parse authorized users file at '{}'",
            auth_path.display()
        ))
    }

    /// Saves the authorized users to the given TOML file.
    pub fn save(&self, auth_path: &Path) -> Result<()> {
        if let Some(parent) = auth_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create parent dir for '{}'", auth_path.display())
            })?;
        }

        // Check if file exists and handle permissions (Unix only)
        #[cfg(unix)]
        {
            // File exists, verify its permissions
            if auth_path.exists() {
                check_file_permissions(auth_path)?;
            // File doesn't exist, create it with correct permissions
            } else {
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .mode(0o600)
                    .open(auth_path)
                    .context(format!(
                        "Failed to create authorized clients file at '{}'",
                        auth_path.display()
                    ))?;
            }
        }

        let content = toml::to_string_pretty(self)
            .context(format!("Failed to serialize authorized users to TOML"))?;
        fs::write(auth_path, content).context(format!(
            "Failed to write authorized users file at {}",
            auth_path.display()
        ))
    }

    /// Checks if the authorized users are empty.
    pub fn is_empty(&self) -> bool {
        self.users.is_empty()
    }

    /// Returns the number of authorized users.
    pub fn len(&self) -> usize {
        self.users.len()
    }

    /// Returns an iterator over the authorized users.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &UserKeys)> {
        self.users.iter()
    }

    /// Returns the user keys for the given username.
    pub fn get(&self, username: &str) -> Option<&UserKeys> {
        self.users.get(username)
    }

    /// Inserts a new authorized user without any keys.
    pub fn insert_user(&mut self, username: &str) -> InsertUserResult {
        if self.users.contains_key(username) {
            InsertUserResult::UserExists
        } else {
            self.users.insert(username.to_owned(), UserKeys::new());
            InsertUserResult::CreatedUser
        }
    }

    /// Adds a key to an existing authorized user.
    /// Key names must be unique per user.
    pub fn insert_key_for_user(
        &mut self,
        username: &str,
        key_name: String,
        public_key: PublicKey,
    ) -> InsertKeyResult {
        if let Some(user_keys) = self.users.get_mut(username) {
            if user_keys.has_key_name(&key_name) {
                InsertKeyResult::KeyNameExists
            } else {
                user_keys.insert_key(key_name, public_key);
                InsertKeyResult::AddedKey
            }
        } else {
            InsertKeyResult::UserNotFound
        }
    }

    /// Removes a specific key from a user by key name.
    pub fn remove_key(&mut self, username: &str, key_name: &str) -> RemoveKeyResult {
        if let Some(user_keys) = self.users.get_mut(username) {
            let removed = user_keys.remove_key(key_name);
            if removed {
                RemoveKeyResult::RemovedKey
            } else {
                RemoveKeyResult::KeyNotFound
            }
        } else {
            RemoveKeyResult::UserNotFound
        }
    }

    /// Removes an authorized user and all their public keys from the authorized users.
    pub fn remove_user(&mut self, username: &str) -> RemoveUserResult {
        if self.users.remove(username).is_some() {
            RemoveUserResult::RemovedUser
        } else {
            RemoveUserResult::UserNotFound
        }
    }
}

// =============================================================================
// PublicKey
// =============================================================================

/// A public key that can be used for signature verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PublicKey {
    /// RSA public key stored in PKCS#1 DER format
    Rsa(Vec<u8>),
    /// ED25519 public key stored as raw 32-byte key
    Ed25519(Box<[u8; 32]>),
    /// ECDSA P-256 public key stored as uncompressed point (65 bytes: 0x04 + 32-byte x + 32-byte y)
    EcdsaP256(Box<[u8; 65]>),
    /// ECDSA P-384 public key stored as uncompressed point (97 bytes: 0x04 + 48-byte x + 48-byte y)
    EcdsaP384(Box<[u8; 97]>),
}

impl PublicKey {
    /// Attempts to parse a public key from a string in various formats.
    ///
    /// Supported formats:
    /// - OpenSSH (RSA, ED25519, ECDSA)
    /// - PKCS#8 PEM (RSA, ED25519, ECDSA)
    /// - PKCS#1 PEM (RSA)
    ///
    /// Supported key lengths:
    /// - RSA (2048-8192 bits, minimum 2048 bits enforced)
    /// - ED25519 (256 bits)
    /// - ECDSA (256, 384 bits)
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

        // Try parsing as PKCS#8 PEM format (ED25519 or ECDSA)
        if let Ok(pkcs8_der) = pem::parse(key_content) {
            // Try to parse as ED25519 public key from PKCS#8
            if let Ok(ed25519_key) =
                ssh_key::public::Ed25519PublicKey::try_from(pkcs8_der.contents())
            {
                return Self::from_ed25519_key(ed25519_key.as_ref());
            }

            // Try to parse as ECDSA public key from PKCS#8
            if let Ok(ecdsa_key) = Self::try_parse_ecdsa_from_pkcs8(pkcs8_der.contents()) {
                return Ok(ecdsa_key);
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
            Algorithm::Ecdsa { curve } => {
                // Extract the ECDSA public key
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

    /// Converts from an RSA public key.
    fn from_rsa_key(rsa_key: RsaPublicKey) -> Result<Self> {
        // Check that the key is at least 2048 bits
        let key_size_bits = rsa_key.size() * 8;
        if key_size_bits < 2048 {
            bail!(
                "RSA key is too weak: {} bits (minimum required: 2048 bits)",
                key_size_bits
            );
        }

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

    /// Converts from an ECDSA public key from SSH format.
    fn from_ecdsa_key(ecdsa_public: &EcdsaPublicKey, curve: &EcdsaCurve) -> Result<Self> {
        match curve {
            EcdsaCurve::NistP256 => {
                let point_bytes = ecdsa_public.as_ref();
                if point_bytes.len() != 65 {
                    bail!(
                        "Invalid P-256 public key point length: expected 65 bytes, got {}",
                        point_bytes.len()
                    );
                }
                let mut bytes = [0u8; 65];
                bytes.copy_from_slice(point_bytes);
                Ok(Self::EcdsaP256(Box::new(bytes)))
            }
            EcdsaCurve::NistP384 => {
                let point_bytes = ecdsa_public.as_ref();
                if point_bytes.len() != 97 {
                    bail!(
                        "Invalid P-384 public key point length: expected 97 bytes, got {}",
                        point_bytes.len()
                    );
                }
                let mut bytes = [0u8; 97];
                bytes.copy_from_slice(point_bytes);
                Ok(Self::EcdsaP384(Box::new(bytes)))
            }
            EcdsaCurve::NistP521 => {
                bail!("ECDSA P-521 curve is not supported by the ring crate")
            }
        }
    }

    /// Try to parse ECDSA public key from PKCS#8 DER bytes.
    fn try_parse_ecdsa_from_pkcs8(pkcs8_der: &[u8]) -> Result<Self> {
        use p256::pkcs8::DecodePublicKey;

        // Try P-256
        if let Ok(p256_key) = p256::PublicKey::from_public_key_der(pkcs8_der) {
            use p256::elliptic_curve::sec1::ToEncodedPoint;
            let encoded_point = p256_key.to_encoded_point(false);
            let point_bytes = encoded_point.as_bytes();
            if point_bytes.len() != 65 {
                bail!(
                    "Invalid P-256 public key point length: expected 65 bytes, got {}",
                    point_bytes.len()
                );
            }
            let mut bytes = [0u8; 65];
            bytes.copy_from_slice(point_bytes);
            return Ok(Self::EcdsaP256(Box::new(bytes)));
        }

        // Try P-384
        if let Ok(p384_key) = p384::PublicKey::from_public_key_der(pkcs8_der) {
            use p384::elliptic_curve::sec1::ToEncodedPoint;
            let encoded_point = p384_key.to_encoded_point(false);
            let point_bytes = encoded_point.as_bytes();
            if point_bytes.len() != 97 {
                bail!(
                    "Invalid P-384 public key point length: expected 97 bytes, got {}",
                    point_bytes.len()
                );
            }
            let mut bytes = [0u8; 97];
            bytes.copy_from_slice(point_bytes);
            return Ok(Self::EcdsaP384(Box::new(bytes)));
        }

        bail!("Failed to parse ECDSA public key from PKCS#8")
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
                        .context("Failed to create Mpint for e")?,
                    n: ssh_key::Mpint::from_positive_bytes(&n_bytes)
                        .context("Failed to create Mpint for n")?,
                };
                let public_key = SshPublicKey::from(ssh_rsa);
                Ok(public_key.to_openssh().map_err(|e| {
                    anyhow::anyhow!("Failed to encode RSA key to OpenSSH format: {}", e)
                })?)
            }
            Self::Ed25519(bytes) => {
                let ssh_ed25519 =
                    ssh_key::public::Ed25519PublicKey::try_from(bytes.as_ref().as_ref())
                        .context("Failed to create ED25519 SSH key")?;
                let public_key = SshPublicKey::from(ssh_ed25519);
                Ok(public_key.to_openssh().map_err(|e| {
                    anyhow::anyhow!("Failed to encode ED25519 key to OpenSSH format: {}", e)
                })?)
            }
            Self::EcdsaP256(point_bytes) => {
                use p256::EncodedPoint;
                let encoded_point = EncodedPoint::from_bytes(point_bytes.as_ref())
                    .context("Failed to create EncodedPoint from P-256 bytes")?;
                let ssh_ecdsa = ssh_key::public::EcdsaPublicKey::NistP256(encoded_point);
                let public_key = SshPublicKey::from(ssh_ecdsa);
                Ok(public_key
                    .to_openssh()
                    .context("Failed to encode ECDSA P-256 key to OpenSSH format")?)
            }
            Self::EcdsaP384(point_bytes) => {
                use p384::EncodedPoint;
                let encoded_point = EncodedPoint::from_bytes(point_bytes.as_ref())
                    .context("Failed to create EncodedPoint from P-384 bytes")?;
                let ssh_ecdsa = ssh_key::public::EcdsaPublicKey::NistP384(encoded_point);
                let public_key = SshPublicKey::from(ssh_ecdsa);
                Ok(public_key
                    .to_openssh()
                    .context("Failed to encode ECDSA P-384 key to OpenSSH format")?)
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
            Self::EcdsaP256(point_bytes) => {
                let public_key =
                    UnparsedPublicKey::new(&ECDSA_P256_SHA256_ASN1, point_bytes.as_ref());
                public_key
                    .verify(message, signature)
                    .map_err(|_| anyhow::anyhow!("ECDSA P-256 signature verification failed"))
            }
            Self::EcdsaP384(point_bytes) => {
                let public_key =
                    UnparsedPublicKey::new(&ECDSA_P384_SHA384_ASN1, point_bytes.as_ref());
                public_key
                    .verify(message, signature)
                    .map_err(|_| anyhow::anyhow!("ECDSA P-384 signature verification failed"))
            }
        }
    }
}

// =============================================================================
// UserKeys
// =============================================================================

/// Structure representing keys for a single user.
#[derive(Debug)]
pub struct UserKeys {
    /// Map of key names to their public keys
    /// Key names must be unique per user, but the same public key can have multiple names
    keys: HashMap<String, PublicKey>,
}

impl UserKeys {
    fn new() -> Self {
        Self {
            keys: HashMap::new(),
        }
    }

    /// Returns the number of keys in the client.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Returns an iterator over (name, key) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &PublicKey)> {
        self.keys.iter()
    }

    /// Returns all public keys (values).
    pub fn public_keys(&self) -> impl Iterator<Item = &PublicKey> {
        self.keys.values()
    }

    /// Check if a key with the given name exists.
    pub fn has_key_name(&self, name: &str) -> bool {
        self.keys.contains_key(name)
    }

    /// Remove a key by name. Returns true if a key was removed.
    pub fn remove_key(&mut self, name: &str) -> bool {
        self.keys.remove(name).is_some()
    }

    /// Insert a new key with the given name. Returns true if this is a new name,
    /// false if the name already existed (and was replaced).
    pub fn insert_key(&mut self, name: String, public_key: PublicKey) -> bool {
        self.keys.insert(name, public_key).is_none()
    }
}

impl Serialize for UserKeys {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;

        // Serialize as a map of name -> SSH public key string
        let keys_map: Result<HashMap<&str, String>, _> = self
            .keys
            .iter()
            .map(|(name, key)| {
                key.to_ssh_public_key_string()
                    .map(|key_str| (name.as_str(), key_str))
                    .map_err(serde::ser::Error::custom)
            })
            .collect();

        let keys_map = keys_map?;
        let mut state = serializer.serialize_struct("UserKeys", 1)?;
        state.serialize_field("keys", &keys_map)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for UserKeys {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct UserKeysHelper {
            keys: HashMap<String, String>,
        }

        let helper = UserKeysHelper::deserialize(deserializer)?;
        let keys: Result<HashMap<String, PublicKey>, _> = helper
            .keys
            .into_iter()
            .map(|(name, key_str)| {
                PublicKey::parse(&key_str)
                    .map(|public_key| (name.clone(), public_key))
                    .map_err(|e| {
                        serde::de::Error::custom(format!(
                            "Failed to parse public key '{}': {}",
                            name, e
                        ))
                    })
            })
            .collect();

        Ok(UserKeys { keys: keys? })
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Check file permissions and bail if they're not 0o600 (Unix only).
#[cfg(unix)]
fn check_file_permissions(path: &Path) -> Result<()> {
    let metadata = fs::metadata(path).context(format!(
        "Failed to read metadata for file at '{}'",
        path.display()
    ))?;
    let permissions = metadata.permissions();
    let mode = permissions.mode() & 0o777;

    // Check if permissions are too permissive (should be 0o600)
    if mode != 0o600 {
        bail!(
            "File at '{}' has insecure permissions: {:o}. \
            Run: `chmod 600 '{}'`",
            path.display(),
            mode,
            path.display()
        );
    }
    Ok(())
}

/// Generates a random 64-character string for internal client connection authentication.
///
/// Uses `ring::rand::SystemRandom` for cryptographic randomness with rejection sampling
/// to avoid modulo bias.
pub fn generate_internal_auth_token() -> Result<String> {
    const CHARSET: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let rng = SystemRandom::new();
    let mut internal_auth_token = String::with_capacity(64);

    // Rejection sampling threshold: 256 - (256 % 62) = 248
    // Accept only bytes < 248 to ensure uniform distribution
    let threshold = 256 - (256 % CHARSET.len());

    while internal_auth_token.len() < 64 {
        let mut random_bytes = [0u8; 128];
        rng.fill(&mut random_bytes).map_err(|e| {
            anyhow::anyhow!(
                "Failed to generate random bytes for internal auth token: {}",
                e
            )
        })?;

        for &byte in &random_bytes {
            if internal_auth_token.len() >= 64 {
                break;
            }

            // Reject bytes >= threshold to avoid modulo bias
            if (byte as usize) < threshold {
                let idx = (byte as usize) % CHARSET.len();
                internal_auth_token.push(CHARSET[idx] as char);
            }
        }
    }

    Ok(internal_auth_token)
}

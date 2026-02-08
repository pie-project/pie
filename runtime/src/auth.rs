//! Auth Service - Authentication and authorization management
//!
//! This module provides a singleton actor for managing user authentication,
//! public key verification, and internal token validation.

mod keys;
mod users;

pub use keys::PublicKey;
pub use users::{AuthorizedUsers, UserKeys};

use anyhow::Result;
use base64::Engine;
use ring::rand::{SecureRandom, SystemRandom};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use tokio::sync::oneshot;

use crate::service::{Service, ServiceHandler};

// =============================================================================
// Actor Setup (Singleton)
// =============================================================================

/// Global singleton Auth actor.
static SERVICE: LazyLock<Service<Message>> = LazyLock::new(Service::new);

/// Spawns the Auth actor.
pub fn spawn(enable_auth: bool, authorized_users_path: &Path) {
    SERVICE
        .spawn(|| AuthService::new(enable_auth, authorized_users_path))
        .expect("Auth already spawned");
}

// =============================================================================
// Convenience Wrappers
// =============================================================================

/// Check if a user exists in the authorized users list.
pub async fn user_exists(username: String) -> Result<bool> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::UserExists { username, response: tx })?;
    Ok(rx.await?)
}

/// Check if authentication is enabled.
pub async fn is_auth_enabled() -> Result<bool> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::IsAuthEnabled { response: tx })?;
    Ok(rx.await?)
}

/// Generate a new challenge for authentication.
pub async fn generate_challenge() -> Result<Vec<u8>> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::GenerateChallenge { response: tx })?;
    rx.await?
}

/// Verify a signature against all user keys.
pub async fn verify_signature(
    username: String,
    challenge: Vec<u8>,
    signature: Vec<u8>,
) -> Result<bool> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::VerifySignature { username, challenge, signature, response: tx })?;
    Ok(rx.await?)
}

/// Verify internal auth token.
pub async fn verify_internal_token(token: String) -> Result<bool> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::VerifyInternalToken { token, response: tx })?;
    Ok(rx.await?)
}

/// Get the internal auth token.
pub async fn get_internal_auth_token() -> Result<String> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::GetInternalAuthToken { response: tx })?;
    Ok(rx.await?)
}

// =============================================================================
// AuthService (Business Logic)
// =============================================================================

/// The auth service handles all authentication operations.
#[derive(Debug)]
pub struct AuthService {
    enable_auth: bool,
    authorized_users: AuthorizedUsers,
    authorized_users_path: PathBuf,
    internal_auth_token: String,
    rng: SystemRandom,
}

impl AuthService {
    pub fn new(enable_auth: bool, authorized_users_path: &Path) -> Self {
        let authorized_users = if enable_auth {
            AuthorizedUsers::load(authorized_users_path)
                .expect("Failed to load authorized users")
        } else {
            AuthorizedUsers::default()
        };

        let internal_auth_token =
            generate_internal_auth_token().expect("Failed to generate internal auth token");

        AuthService {
            enable_auth,
            authorized_users,
            authorized_users_path: authorized_users_path.to_path_buf(),
            internal_auth_token,
            rng: SystemRandom::new(),
        }
    }

    pub fn is_auth_enabled(&self) -> bool {
        self.enable_auth
    }

    fn save(&self) {
        if let Err(e) = self.authorized_users.save(&self.authorized_users_path) {
            tracing::error!("Failed to save authorized users: {e}");
        }
    }

    pub fn user_exists(&self, username: &str) -> bool {
        self.authorized_users.get(username).is_some()
    }

    pub fn list_users(&self) -> Vec<String> {
        self.authorized_users.iter().map(|(k, _)| k.clone()).collect()
    }

    pub fn insert_user(&mut self, username: &str) -> Result<()> {
        self.authorized_users.insert_user(username)?;
        self.save();
        Ok(())
    }

    pub fn remove_user(&mut self, username: &str) -> Result<()> {
        self.authorized_users.remove_user(username)?;
        self.save();
        Ok(())
    }

    pub fn insert_key(
        &mut self,
        username: &str,
        key_name: String,
        public_key: PublicKey,
    ) -> Result<()> {
        self.authorized_users.insert_key_for_user(username, key_name, public_key)?;
        self.save();
        Ok(())
    }

    pub fn remove_key(&mut self, username: &str, key_name: &str) -> Result<()> {
        self.authorized_users.remove_key(username, key_name)?;
        self.save();
        Ok(())
    }

    pub fn generate_challenge(&self) -> Result<Vec<u8>> {
        let mut challenge = [0u8; 48];
        self.rng
            .fill(&mut challenge)
            .map_err(|e| anyhow::anyhow!("Failed to generate random challenge: {e}"))?;
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
// Messages
// =============================================================================

/// Messages for the Auth actor.
#[derive(Debug)]
pub enum Message {
    UserExists {
        username: String,
        response: oneshot::Sender<bool>,
    },
    VerifySignature {
        username: String,
        challenge: Vec<u8>,
        signature: Vec<u8>,
        response: oneshot::Sender<bool>,
    },
    VerifyInternalToken {
        token: String,
        response: oneshot::Sender<bool>,
    },
    InsertUser {
        username: String,
        response: oneshot::Sender<Result<()>>,
    },
    RemoveUser {
        username: String,
        response: oneshot::Sender<Result<()>>,
    },
    InsertKey {
        username: String,
        key_name: String,
        public_key: PublicKey,
        response: oneshot::Sender<Result<()>>,
    },
    RemoveKey {
        username: String,
        key_name: String,
        response: oneshot::Sender<Result<()>>,
    },
    GenerateChallenge {
        response: oneshot::Sender<Result<Vec<u8>>>,
    },
    ListUsers {
        response: oneshot::Sender<Vec<String>>,
    },
    IsAuthEnabled {
        response: oneshot::Sender<bool>,
    },
    GetInternalAuthToken {
        response: oneshot::Sender<String>,
    },
}

impl ServiceHandler for AuthService {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::UserExists { username, response } => {
                let _ = response.send(self.user_exists(&username));
            }
            Message::VerifySignature { username, challenge, signature, response } => {
                let _ = response.send(self.verify_signature(&username, &challenge, &signature));
            }
            Message::VerifyInternalToken { token, response } => {
                let _ = response.send(self.verify_internal_token(&token));
            }
            Message::InsertUser { username, response } => {
                let _ = response.send(self.insert_user(&username));
            }
            Message::RemoveUser { username, response } => {
                let _ = response.send(self.remove_user(&username));
            }
            Message::InsertKey { username, key_name, public_key, response } => {
                let _ = response.send(self.insert_key(&username, key_name, public_key));
            }
            Message::RemoveKey { username, key_name, response } => {
                let _ = response.send(self.remove_key(&username, &key_name));
            }
            Message::GenerateChallenge { response } => {
                let _ = response.send(self.generate_challenge());
            }
            Message::ListUsers { response } => {
                let _ = response.send(self.list_users());
            }
            Message::IsAuthEnabled { response } => {
                let _ = response.send(self.is_auth_enabled());
            }
            Message::GetInternalAuthToken { response } => {
                let _ = response.send(self.internal_auth_token.clone());
            }
        }
    }
}

// =============================================================================
// Token Generation
// =============================================================================

/// Generates a URL-safe base64 token (64 characters) for internal authentication.
///
/// Uses `ring::rand::SystemRandom` for cryptographic randomness.
/// 48 random bytes → 64 characters in base64url (no padding).
fn generate_internal_auth_token() -> Result<String> {
    let mut bytes = [0u8; 48];
    SystemRandom::new()
        .fill(&mut bytes)
        .map_err(|e| anyhow::anyhow!("Failed to generate random bytes: {e}"))?;
    Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes))
}

//! Authorized users persistence and user key management.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::{fs, result};

use super::keys::PublicKey;

// =============================================================================
// AuthorizedUsers
// =============================================================================

/// Structure representing the authorized_users.toml file format.
#[derive(Deserialize, Serialize, Debug, Default)]
pub struct AuthorizedUsers {
    /// Map of username to their authorized keys.
    #[serde(default)]
    users: HashMap<String, UserKeys>,
}

impl AuthorizedUsers {
    /// Loads the authorized users from the given TOML file.
    pub fn load(auth_path: &Path) -> Result<Self> {
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

    /// Saves the authorized users to the given TOML file atomically.
    pub fn save(&self, auth_path: &Path) -> Result<()> {
        if let Some(parent) = auth_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create parent dir for '{}'", auth_path.display())
            })?;
        }

        #[cfg(unix)]
        if auth_path.exists() {
            check_file_permissions(auth_path)?;
        }

        let content = toml::to_string_pretty(self)
            .context("Failed to serialize authorized users to TOML")?;

        // Atomic save: write to temp file, then rename
        let tmp_path = auth_path.with_extension("tmp");
        fs::write(&tmp_path, &content).with_context(|| {
            format!("Failed to write temp file at '{}'", tmp_path.display())
        })?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&tmp_path, std::fs::Permissions::from_mode(0o600))
                .with_context(|| {
                    format!("Failed to set permissions on '{}'", tmp_path.display())
                })?;
        }

        fs::rename(&tmp_path, auth_path).with_context(|| {
            format!(
                "Failed to rename '{}' to '{}'",
                tmp_path.display(),
                auth_path.display()
            )
        })
    }

    pub fn is_empty(&self) -> bool {
        self.users.is_empty()
    }

    pub fn len(&self) -> usize {
        self.users.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &UserKeys)> {
        self.users.iter()
    }

    pub fn get(&self, username: &str) -> Option<&UserKeys> {
        self.users.get(username)
    }

    pub fn insert_user(&mut self, username: &str) -> Result<()> {
        if self.users.contains_key(username) {
            bail!("User '{}' already exists", username)
        } else {
            self.users.insert(username.to_owned(), UserKeys::new());
            Ok(())
        }
    }

    pub fn insert_key_for_user(
        &mut self,
        username: &str,
        key_name: String,
        public_key: PublicKey,
    ) -> Result<()> {
        if let Some(user_keys) = self.users.get_mut(username) {
            if user_keys.has_key_name(&key_name) {
                bail!("Key '{}' already exists for user '{}'", key_name, username)
            } else {
                user_keys.insert_key(key_name, public_key);
                Ok(())
            }
        } else {
            bail!("User '{}' not found", username)
        }
    }

    pub fn remove_key(&mut self, username: &str, key_name: &str) -> Result<()> {
        if let Some(user_keys) = self.users.get_mut(username) {
            if user_keys.remove_key(key_name) {
                Ok(())
            } else {
                bail!("Key '{}' not found for user '{}'", key_name, username)
            }
        } else {
            bail!("User '{}' not found", username)
        }
    }

    pub fn remove_user(&mut self, username: &str) -> Result<()> {
        if self.users.remove(username).is_some() {
            Ok(())
        } else {
            bail!("User '{}' not found", username)
        }
    }
}

// =============================================================================
// UserKeys
// =============================================================================

/// Keys for a single user. Serialized as `{ keys: { name: "ssh-... key" } }`.
///
/// Uses a derive-based raw intermediary for serde: the TOML stores OpenSSH
/// strings which are parsed into typed `PublicKey` values on deserialization.
#[derive(Debug)]
pub struct UserKeys {
    keys: HashMap<String, PublicKey>,
}

/// Raw TOML representation — just string keys.
#[derive(Serialize, Deserialize)]
struct UserKeysRaw {
    keys: HashMap<String, String>,
}

impl UserKeys {
    pub(super) fn new() -> Self {
        Self {
            keys: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &PublicKey)> {
        self.keys.iter()
    }

    pub fn public_keys(&self) -> impl Iterator<Item = &PublicKey> {
        self.keys.values()
    }

    pub fn has_key_name(&self, name: &str) -> bool {
        self.keys.contains_key(name)
    }

    pub fn remove_key(&mut self, name: &str) -> bool {
        self.keys.remove(name).is_some()
    }

    pub fn insert_key(&mut self, name: String, public_key: PublicKey) -> bool {
        self.keys.insert(name, public_key).is_none()
    }
}

impl TryFrom<UserKeysRaw> for UserKeys {
    type Error = anyhow::Error;

    fn try_from(raw: UserKeysRaw) -> Result<Self> {
        let keys: Result<HashMap<String, PublicKey>> = raw
            .keys
            .into_iter()
            .map(|(name, key_str)| {
                PublicKey::parse(&key_str)
                    .with_context(|| format!("Failed to parse public key '{name}'"))
                    .map(|pk| (name, pk))
            })
            .collect();
        Ok(Self { keys: keys? })
    }
}

impl TryFrom<&UserKeys> for UserKeysRaw {
    type Error = anyhow::Error;

    fn try_from(user_keys: &UserKeys) -> Result<Self> {
        let keys: Result<HashMap<String, String>> = user_keys
            .keys
            .iter()
            .map(|(name, key)| {
                key.to_ssh_public_key_string()
                    .map(|s| (name.clone(), s))
            })
            .collect();
        Ok(Self { keys: keys? })
    }
}

impl Serialize for UserKeys {
    fn serialize<S>(&self, serializer: S) -> result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let raw = UserKeysRaw::try_from(self).map_err(serde::ser::Error::custom)?;
        raw.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for UserKeys {
    fn deserialize<D>(deserializer: D) -> result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = UserKeysRaw::deserialize(deserializer)?;
        UserKeys::try_from(raw).map_err(serde::de::Error::custom)
    }
}

// =============================================================================
// Utility
// =============================================================================

/// Check file permissions and bail if they're not 0o600 (Unix only).
#[cfg(unix)]
fn check_file_permissions(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let metadata = fs::metadata(path).context(format!(
        "Failed to read metadata for file at '{}'",
        path.display()
    ))?;
    let mode = metadata.permissions().mode() & 0o777;

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Helper: generate an Ed25519 OpenSSH key string for test fixtures.
    fn gen_ed25519_key() -> (PublicKey, String) {
        use ring::rand::SecureRandom;
        let rng = ring::rand::SystemRandom::new();
        let mut secret = [0u8; 32];
        rng.fill(&mut secret).unwrap();
        let signing_key = ed25519_dalek::SigningKey::from_bytes(&secret);
        let verifying_key = signing_key.verifying_key();
        let ssh_ed25519 =
            ssh_key::public::Ed25519PublicKey::try_from(verifying_key.as_bytes().as_ref())
                .unwrap();
        let ssh_pub = ssh_key::PublicKey::from(ssh_ed25519);
        let openssh = ssh_pub.to_openssh().unwrap();
        let key = PublicKey::parse(&openssh).unwrap();
        (key, openssh)
    }

    // =========================================================================
    // UserKeys CRUD
    // =========================================================================

    #[test]
    fn user_keys_insert_and_has() {
        let mut uk = UserKeys::new();
        let (key, _) = gen_ed25519_key();
        assert!(uk.insert_key("laptop".into(), key));
        assert!(uk.has_key_name("laptop"));
        assert!(!uk.has_key_name("desktop"));
    }

    #[test]
    fn user_keys_insert_duplicate_returns_false() {
        let mut uk = UserKeys::new();
        let (key1, _) = gen_ed25519_key();
        let (key2, _) = gen_ed25519_key();
        assert!(uk.insert_key("laptop".into(), key1));
        assert!(!uk.insert_key("laptop".into(), key2)); // overwrite → returns false
    }

    #[test]
    fn user_keys_remove() {
        let mut uk = UserKeys::new();
        let (key, _) = gen_ed25519_key();
        uk.insert_key("laptop".into(), key);
        assert!(uk.remove_key("laptop"));
        assert!(!uk.remove_key("laptop"));
        assert_eq!(uk.len(), 0);
    }

    #[test]
    fn user_keys_public_keys_iterator() {
        let mut uk = UserKeys::new();
        let (k1, _) = gen_ed25519_key();
        let (k2, _) = gen_ed25519_key();
        uk.insert_key("a".into(), k1);
        uk.insert_key("b".into(), k2);
        assert_eq!(uk.public_keys().count(), 2);
    }

    // =========================================================================
    // AuthorizedUsers CRUD
    // =========================================================================

    #[test]
    fn insert_and_get_user() {
        let mut au = AuthorizedUsers::default();
        au.insert_user("alice").unwrap();
        assert!(au.get("alice").is_some());
        assert!(au.get("bob").is_none());
    }

    #[test]
    fn insert_duplicate_user_fails() {
        let mut au = AuthorizedUsers::default();
        au.insert_user("alice").unwrap();
        assert!(au.insert_user("alice").is_err());
    }

    #[test]
    fn remove_user() {
        let mut au = AuthorizedUsers::default();
        au.insert_user("alice").unwrap();
        au.remove_user("alice").unwrap();
        assert!(au.is_empty());
    }

    #[test]
    fn remove_nonexistent_user_fails() {
        let mut au = AuthorizedUsers::default();
        assert!(au.remove_user("ghost").is_err());
    }

    #[test]
    fn insert_key_for_user() {
        let mut au = AuthorizedUsers::default();
        au.insert_user("alice").unwrap();
        let (key, _) = gen_ed25519_key();
        au.insert_key_for_user("alice", "laptop".into(), key).unwrap();
        let user_keys = au.get("alice").unwrap();
        assert!(user_keys.has_key_name("laptop"));
    }

    #[test]
    fn insert_key_for_nonexistent_user_fails() {
        let mut au = AuthorizedUsers::default();
        let (key, _) = gen_ed25519_key();
        assert!(au.insert_key_for_user("ghost", "laptop".into(), key).is_err());
    }

    #[test]
    fn insert_duplicate_key_name_fails() {
        let mut au = AuthorizedUsers::default();
        au.insert_user("alice").unwrap();
        let (k1, _) = gen_ed25519_key();
        let (k2, _) = gen_ed25519_key();
        au.insert_key_for_user("alice", "laptop".into(), k1).unwrap();
        assert!(au.insert_key_for_user("alice", "laptop".into(), k2).is_err());
    }

    #[test]
    fn remove_key() {
        let mut au = AuthorizedUsers::default();
        au.insert_user("alice").unwrap();
        let (key, _) = gen_ed25519_key();
        au.insert_key_for_user("alice", "laptop".into(), key).unwrap();
        au.remove_key("alice", "laptop").unwrap();
        assert!(!au.get("alice").unwrap().has_key_name("laptop"));
    }

    #[test]
    fn remove_nonexistent_key_fails() {
        let mut au = AuthorizedUsers::default();
        au.insert_user("alice").unwrap();
        assert!(au.remove_key("alice", "ghost").is_err());
    }

    #[test]
    fn list_users() {
        let mut au = AuthorizedUsers::default();
        au.insert_user("alice").unwrap();
        au.insert_user("bob").unwrap();
        assert_eq!(au.len(), 2);
        let names: Vec<_> = au.iter().map(|(k, _)| k.clone()).collect();
        assert!(names.contains(&"alice".to_string()));
        assert!(names.contains(&"bob".to_string()));
    }

    // =========================================================================
    // TOML serde roundtrip
    // =========================================================================

    #[test]
    fn toml_serde_roundtrip() {
        let mut au = AuthorizedUsers::default();
        au.insert_user("alice").unwrap();
        let (key, _) = gen_ed25519_key();
        au.insert_key_for_user("alice", "laptop".into(), key.clone())
            .unwrap();

        // Serialize → deserialize
        let toml_str = toml::to_string_pretty(&au).unwrap();
        let au2: AuthorizedUsers = toml::from_str(&toml_str).unwrap();

        assert!(au2.get("alice").is_some());
        let alice_keys = au2.get("alice").unwrap();
        assert!(alice_keys.has_key_name("laptop"));
        // Verify the key survived the roundtrip
        let roundtripped_key = alice_keys.iter().next().unwrap().1;
        assert_eq!(&key, roundtripped_key);
    }

    #[test]
    fn toml_output_is_valid_toml() {
        let mut au = AuthorizedUsers::default();
        au.insert_user("alice").unwrap();
        let (key, _) = gen_ed25519_key();
        au.insert_key_for_user("alice", "laptop".into(), key).unwrap();

        let toml_str = toml::to_string_pretty(&au).unwrap();
        // Must be valid TOML
        assert!(toml_str.parse::<toml::Table>().is_ok());
        // Check structural aspects
        assert!(toml_str.contains("[users.alice.keys]"));
    }

    // =========================================================================
    // File save/load roundtrip
    // =========================================================================

    #[test]
    fn save_and_load_roundtrip() {
        let mut au = AuthorizedUsers::default();
        au.insert_user("bob").unwrap();
        let (key, _) = gen_ed25519_key();
        au.insert_key_for_user("bob", "workstation".into(), key.clone())
            .unwrap();

        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        au.save(&path).unwrap();

        let au2 = AuthorizedUsers::load(&path).unwrap();
        assert!(au2.get("bob").is_some());
        let bob_keys = au2.get("bob").unwrap();
        assert!(bob_keys.has_key_name("workstation"));
        assert_eq!(bob_keys.iter().next().unwrap().1, &key);
    }

    #[cfg(unix)]
    #[test]
    fn save_sets_permissions_to_600() {
        use std::os::unix::fs::PermissionsExt;

        let au = AuthorizedUsers::default();
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        au.save(&path).unwrap();

        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }
}

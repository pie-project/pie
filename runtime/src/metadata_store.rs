//! Engine-lifetime metadata key-value store for inferlets.
//!
//! The store is intentionally narrow: it is an in-memory, namespaced byte
//! hashmap for runtime metadata that should survive across inferlet request
//! instances but may be lost when the engine process exits.

use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

use anyhow::{Result, bail};

/// Maximum caller-supplied namespace length in bytes.
pub const MAX_NAMESPACE_BYTES: usize = 256;
/// Maximum caller-supplied key length in bytes.
pub const MAX_KEY_BYTES: usize = 256;
/// Maximum stored value length in bytes.
pub const MAX_VALUE_BYTES: usize = 1024 * 1024;
/// Maximum aggregate stored value bytes for this process.
pub const MAX_TOTAL_BYTES: usize = 16 * 1024 * 1024;

const MAX_OWNER_PART_BYTES: usize = 256;

static STORE: LazyLock<RwLock<Store>> = LazyLock::new(|| RwLock::new(Store::default()));

#[derive(Default)]
struct Store {
    entries: HashMap<MetadataKey, Vec<u8>>,
    total_value_bytes: usize,
}

/// Host-derived metadata owner identity.
///
/// Guests control `namespace` and `key`, but not this owner. The runtime builds
/// it from the current `InstanceState`, so separate users/programs cannot
/// collide by choosing the same namespace/key strings.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MetadataOwner {
    username: String,
    program: String,
}

impl MetadataOwner {
    pub fn new(username: impl Into<String>, program: impl Into<String>) -> Result<Self> {
        let username = username.into();
        let program = program.into();
        validate_part("owner username", &username, MAX_OWNER_PART_BYTES)?;
        validate_part("owner program", &program, MAX_OWNER_PART_BYTES)?;
        Ok(Self { username, program })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MetadataKey {
    owner: MetadataOwner,
    namespace: String,
    key: String,
}

impl MetadataKey {
    fn new(owner: &MetadataOwner, namespace: &str, key: &str) -> Result<Self> {
        validate_part("namespace", namespace, MAX_NAMESPACE_BYTES)?;
        validate_part("key", key, MAX_KEY_BYTES)?;

        Ok(Self {
            owner: owner.clone(),
            namespace: namespace.to_string(),
            key: key.to_string(),
        })
    }
}

fn validate_part(label: &str, value: &str, max_bytes: usize) -> Result<()> {
    if value.is_empty() {
        bail!("metadata {label} must not be empty");
    }
    if value.len() > max_bytes {
        bail!(
            "metadata {label} must be at most {max_bytes} bytes (got {})",
            value.len()
        );
    }
    Ok(())
}

/// Store or overwrite a metadata value.
pub fn put(owner: &MetadataOwner, namespace: &str, key: &str, value: Vec<u8>) -> Result<()> {
    let metadata_key = MetadataKey::new(owner, namespace, key)?;
    if value.len() > MAX_VALUE_BYTES {
        bail!(
            "metadata value must be at most {MAX_VALUE_BYTES} bytes (got {})",
            value.len()
        );
    }

    let mut store = STORE
        .write()
        .map_err(|_| anyhow::anyhow!("metadata store lock poisoned"))?;

    let replaced_len = store
        .entries
        .get(&metadata_key)
        .map_or(0, std::vec::Vec::len);
    let new_total = store
        .total_value_bytes
        .saturating_sub(replaced_len)
        .checked_add(value.len())
        .ok_or_else(|| anyhow::anyhow!("metadata store total byte count overflow"))?;

    if new_total > MAX_TOTAL_BYTES {
        bail!("metadata store total byte cap exceeded: {new_total} > {MAX_TOTAL_BYTES}");
    }

    store.total_value_bytes = new_total;
    store.entries.insert(metadata_key, value);
    Ok(())
}

/// Retrieve a metadata value, if present.
pub fn get(owner: &MetadataOwner, namespace: &str, key: &str) -> Result<Option<Vec<u8>>> {
    let metadata_key = MetadataKey::new(owner, namespace, key)?;
    let store = STORE
        .read()
        .map_err(|_| anyhow::anyhow!("metadata store lock poisoned"))?;
    Ok(store.entries.get(&metadata_key).cloned())
}

/// Delete a metadata value. Returns whether an entry existed.
pub fn delete(owner: &MetadataOwner, namespace: &str, key: &str) -> Result<bool> {
    let metadata_key = MetadataKey::new(owner, namespace, key)?;
    let mut store = STORE
        .write()
        .map_err(|_| anyhow::anyhow!("metadata store lock poisoned"))?;
    if let Some(value) = store.entries.remove(&metadata_key) {
        store.total_value_bytes = store.total_value_bytes.saturating_sub(value.len());
        Ok(true)
    } else {
        Ok(false)
    }
}

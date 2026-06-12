use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

use anyhow::{Result, bail};

pub const MAX_NAMESPACE_BYTES: usize = 256;
pub const MAX_KEY_BYTES: usize = 256;
pub const MAX_VALUE_BYTES: usize = 1024 * 1024;
pub const MAX_ENTRIES: usize = 4096;
pub const MAX_TOTAL_BYTES: usize = 16 * 1024 * 1024;

static STORE: LazyLock<RwLock<Store>> = LazyLock::new(|| RwLock::new(Store::default()));

#[derive(Default)]
struct Store {
    entries: HashMap<MetadataKey, Vec<u8>>,
    total_metadata_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MetadataOwner {
    username: String,
    program: String,
}

impl MetadataOwner {
    pub fn new(username: impl Into<String>, program: impl Into<String>) -> Self {
        Self {
            username: username.into(),
            program: program.into(),
        }
    }

    fn stored_bytes(&self) -> usize {
        self.username.len() + self.program.len()
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

    fn stored_bytes(&self, value_len: usize) -> Result<usize> {
        self.owner
            .stored_bytes()
            .checked_add(self.namespace.len())
            .and_then(|bytes| bytes.checked_add(self.key.len()))
            .and_then(|bytes| bytes.checked_add(value_len))
            .ok_or_else(|| anyhow::anyhow!("metadata store byte count overflow"))
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

    let is_new_entry = !store.entries.contains_key(&metadata_key);
    if is_new_entry && store.entries.len() >= MAX_ENTRIES {
        bail!(
            "metadata store entry cap exceeded: {} >= {MAX_ENTRIES}",
            store.entries.len()
        );
    }

    let replaced_bytes = store
        .entries
        .get(&metadata_key)
        .map_or(Ok(0), |value| metadata_key.stored_bytes(value.len()))?;
    let stored_bytes = metadata_key.stored_bytes(value.len())?;
    let new_total = store
        .total_metadata_bytes
        .saturating_sub(replaced_bytes)
        .checked_add(stored_bytes)
        .ok_or_else(|| anyhow::anyhow!("metadata store total byte count overflow"))?;

    if new_total > MAX_TOTAL_BYTES {
        bail!("metadata store total byte cap exceeded: {new_total} > {MAX_TOTAL_BYTES}");
    }

    store.total_metadata_bytes = new_total;
    store.entries.insert(metadata_key, value);
    Ok(())
}

pub fn get(owner: &MetadataOwner, namespace: &str, key: &str) -> Result<Option<Vec<u8>>> {
    let metadata_key = MetadataKey::new(owner, namespace, key)?;
    let store = STORE
        .read()
        .map_err(|_| anyhow::anyhow!("metadata store lock poisoned"))?;
    Ok(store.entries.get(&metadata_key).cloned())
}

pub fn delete(owner: &MetadataOwner, namespace: &str, key: &str) -> Result<bool> {
    let metadata_key = MetadataKey::new(owner, namespace, key)?;
    let mut store = STORE
        .write()
        .map_err(|_| anyhow::anyhow!("metadata store lock poisoned"))?;
    if let Some((removed_key, value)) = store.entries.remove_entry(&metadata_key) {
        store.total_metadata_bytes = store
            .total_metadata_bytes
            .saturating_sub(removed_key.stored_bytes(value.len())?);
        Ok(true)
    } else {
        Ok(false)
    }
}

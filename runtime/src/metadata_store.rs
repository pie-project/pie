//! Engine-lifetime metadata key-value store for inferlets.
//!
//! The store is intentionally narrow: it is an in-memory, namespaced byte
//! hashmap for runtime metadata that should survive across inferlet request
//! instances but may be lost when the engine process exits.

use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

use anyhow::{Result, bail};

static STORE: LazyLock<RwLock<HashMap<MetadataKey, Vec<u8>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MetadataKey {
    namespace: String,
    key: String,
}

impl MetadataKey {
    fn new(namespace: &str, key: &str) -> Result<Self> {
        validate_part("namespace", namespace)?;
        validate_part("key", key)?;

        Ok(Self {
            namespace: namespace.to_string(),
            key: key.to_string(),
        })
    }
}

fn validate_part(label: &str, value: &str) -> Result<()> {
    if value.is_empty() {
        bail!("metadata {label} must not be empty");
    }
    Ok(())
}

/// Store or overwrite a metadata value.
pub fn put(namespace: &str, key: &str, value: Vec<u8>) -> Result<()> {
    let metadata_key = MetadataKey::new(namespace, key)?;
    let mut store = STORE
        .write()
        .map_err(|_| anyhow::anyhow!("metadata store lock poisoned"))?;
    store.insert(metadata_key, value);
    Ok(())
}

/// Retrieve a metadata value, if present.
pub fn get(namespace: &str, key: &str) -> Result<Option<Vec<u8>>> {
    let metadata_key = MetadataKey::new(namespace, key)?;
    let store = STORE
        .read()
        .map_err(|_| anyhow::anyhow!("metadata store lock poisoned"))?;
    Ok(store.get(&metadata_key).cloned())
}

/// Delete a metadata value. Returns whether an entry existed.
pub fn delete(namespace: &str, key: &str) -> Result<bool> {
    let metadata_key = MetadataKey::new(namespace, key)?;
    let mut store = STORE
        .write()
        .map_err(|_| anyhow::anyhow!("metadata store lock poisoned"))?;
    Ok(store.remove(&metadata_key).is_some())
}

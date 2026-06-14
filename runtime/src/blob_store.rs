//! Small daemon-lifetime named byte blobs for inferlet state.
//!
//! This is intentionally in-memory and process-lifetime only. It gives guests a
//! bounded place to persist small compatibility-scoped state across per-request
//! WASM instances without relying on per-instance `/scratch` or a persistent
//! Wasmtime store.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, ensure, Result};

const DEFAULT_MAX_BLOB_BYTES: usize = 1024 * 1024;
const DEFAULT_MAX_BLOBS_PER_USER: usize = 256;
const DEFAULT_MAX_TOTAL_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_MAX_TTL: Duration = Duration::from_secs(30 * 60);

static STORE: LazyLock<Mutex<BlobStore>> = LazyLock::new(|| {
    Mutex::new(BlobStore::new(
        DEFAULT_MAX_BLOB_BYTES,
        DEFAULT_MAX_BLOBS_PER_USER,
        DEFAULT_MAX_TOTAL_BYTES,
        DEFAULT_MAX_TTL,
    ))
});

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct BlobKey {
    username: String,
    name: String,
}

#[derive(Clone, Debug)]
struct BlobEntry {
    bytes: Vec<u8>,
    expires_at_ms: u64,
    last_used_ms: u64,
}

#[derive(Debug)]
pub(crate) struct BlobStore {
    max_blob_bytes: usize,
    max_blobs_per_user: usize,
    max_total_bytes: usize,
    max_ttl_ms: u64,
    total_bytes: usize,
    entries: HashMap<BlobKey, BlobEntry>,
}

impl BlobStore {
    pub(crate) fn new(
        max_blob_bytes: usize,
        max_blobs_per_user: usize,
        max_total_bytes: usize,
        max_ttl: Duration,
    ) -> Self {
        Self {
            max_blob_bytes: max_blob_bytes.max(1),
            max_blobs_per_user: max_blobs_per_user.max(1),
            max_total_bytes: max_total_bytes.max(1),
            max_ttl_ms: duration_ms(max_ttl).max(1),
            total_bytes: 0,
            entries: HashMap::new(),
        }
    }

    pub(crate) fn save(
        &mut self,
        now_ms: u64,
        username: &str,
        name: &str,
        bytes: Vec<u8>,
        ttl: Duration,
    ) -> Result<()> {
        ensure!(!username.is_empty(), "blob username must not be empty");
        ensure!(!name.is_empty(), "blob name must not be empty");
        ensure!(
            bytes.len() <= self.max_blob_bytes,
            "blob is {} bytes, over per-blob limit {}",
            bytes.len(),
            self.max_blob_bytes,
        );
        ensure!(
            bytes.len() <= self.max_total_bytes,
            "blob is {} bytes, over total blob-store limit {}",
            bytes.len(),
            self.max_total_bytes,
        );

        self.prune_expired(now_ms);

        let key = BlobKey {
            username: username.to_string(),
            name: name.to_string(),
        };
        if let Some(old) = self.entries.remove(&key) {
            self.total_bytes = self.total_bytes.saturating_sub(old.bytes.len());
        }

        let ttl_ms = duration_ms(ttl).min(self.max_ttl_ms).max(1);
        self.total_bytes += bytes.len();
        self.entries.insert(
            key.clone(),
            BlobEntry {
                bytes,
                expires_at_ms: now_ms.saturating_add(ttl_ms),
                last_used_ms: now_ms,
            },
        );

        self.evict_until_user_within_cap(now_ms, username, &key);
        self.evict_until_total_within_cap(now_ms, &key);

        if !self.entries.contains_key(&key) {
            return Err(anyhow!("blob store capacity cannot retain new blob"));
        }
        Ok(())
    }

    pub(crate) fn open(&mut self, now_ms: u64, username: &str, name: &str) -> Result<Option<Vec<u8>>> {
        ensure!(!username.is_empty(), "blob username must not be empty");
        ensure!(!name.is_empty(), "blob name must not be empty");
        self.prune_expired(now_ms);
        let key = BlobKey {
            username: username.to_string(),
            name: name.to_string(),
        };
        let Some(entry) = self.entries.get_mut(&key) else {
            return Ok(None);
        };
        entry.last_used_ms = now_ms;
        Ok(Some(entry.bytes.clone()))
    }

    pub(crate) fn delete(&mut self, now_ms: u64, username: &str, name: &str) -> Result<()> {
        ensure!(!username.is_empty(), "blob username must not be empty");
        ensure!(!name.is_empty(), "blob name must not be empty");
        self.prune_expired(now_ms);
        let key = BlobKey {
            username: username.to_string(),
            name: name.to_string(),
        };
        if let Some(old) = self.entries.remove(&key) {
            self.total_bytes = self.total_bytes.saturating_sub(old.bytes.len());
        }
        Ok(())
    }

    fn prune_expired(&mut self, now_ms: u64) {
        let mut removed = 0usize;
        self.entries.retain(|_, entry| {
            let keep = entry.expires_at_ms > now_ms;
            if !keep {
                removed += entry.bytes.len();
            }
            keep
        });
        self.total_bytes = self.total_bytes.saturating_sub(removed);
    }

    fn evict_until_user_within_cap(&mut self, now_ms: u64, username: &str, protected: &BlobKey) {
        self.prune_expired(now_ms);
        loop {
            let count = self.entries.keys().filter(|key| key.username == username).count();
            if count <= self.max_blobs_per_user {
                break;
            }
            let Some(victim) = self.oldest_matching(|key, _| key.username == username && key != protected) else {
                break;
            };
            self.remove_key(&victim);
        }
    }

    fn evict_until_total_within_cap(&mut self, now_ms: u64, protected: &BlobKey) {
        self.prune_expired(now_ms);
        while self.total_bytes > self.max_total_bytes {
            let Some(victim) = self.oldest_matching(|key, _| key != protected) else {
                break;
            };
            self.remove_key(&victim);
        }
    }

    fn oldest_matching(&self, mut predicate: impl FnMut(&BlobKey, &BlobEntry) -> bool) -> Option<BlobKey> {
        self.entries
            .iter()
            .filter(|(key, entry)| predicate(key, entry))
            .min_by_key(|(_, entry)| entry.last_used_ms)
            .map(|(key, _)| key.clone())
    }

    fn remove_key(&mut self, key: &BlobKey) {
        if let Some(old) = self.entries.remove(key) {
            self.total_bytes = self.total_bytes.saturating_sub(old.bytes.len());
        }
    }
}

pub fn save_blob(username: &str, name: &str, bytes: Vec<u8>, ttl_ms: u64) -> Result<()> {
    STORE.lock().unwrap().save(
        now_ms(),
        username,
        name,
        bytes,
        Duration::from_millis(ttl_ms),
    )
}

pub fn open_blob(username: &str, name: &str) -> Result<Option<Vec<u8>>> {
    STORE.lock().unwrap().open(now_ms(), username, name)
}

pub fn delete_blob(username: &str, name: &str) -> Result<()> {
    STORE.lock().unwrap().delete(now_ms(), username, name)
}

fn duration_ms(duration: Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn blob_survives_independent_callers_until_ttl() {
        let mut store = BlobStore::new(16, 4, 64, Duration::from_millis(100));
        store.save(0, "alice", "thread-a", b"cache".to_vec(), Duration::from_millis(50)).unwrap();

        assert_eq!(store.open(10, "alice", "thread-a").unwrap(), Some(b"cache".to_vec()));
        assert_eq!(store.open(51, "alice", "thread-a").unwrap(), None);
    }

    #[test]
    fn users_are_isolated_even_when_names_match() {
        let mut store = BlobStore::new(16, 4, 64, Duration::from_secs(60));
        store.save(0, "alice", "same", b"alice".to_vec(), Duration::from_secs(60)).unwrap();
        store.save(0, "bob", "same", b"bob".to_vec(), Duration::from_secs(60)).unwrap();

        assert_eq!(store.open(1, "alice", "same").unwrap(), Some(b"alice".to_vec()));
        assert_eq!(store.open(1, "bob", "same").unwrap(), Some(b"bob".to_vec()));
    }

    #[test]
    fn rejects_blobs_over_per_blob_cap() {
        let mut store = BlobStore::new(3, 4, 64, Duration::from_secs(60));

        assert!(store.save(0, "alice", "too-large", vec![1, 2, 3, 4], Duration::from_secs(60)).is_err());
        assert_eq!(store.open(0, "alice", "too-large").unwrap(), None);
    }

    #[test]
    fn evicts_lru_entries_to_honor_total_bytes_cap() {
        let mut store = BlobStore::new(16, 8, 6, Duration::from_secs(60));
        store.save(0, "alice", "a", b"aa".to_vec(), Duration::from_secs(60)).unwrap();
        store.save(1, "alice", "b", b"bb".to_vec(), Duration::from_secs(60)).unwrap();
        store.save(2, "alice", "c", b"cc".to_vec(), Duration::from_secs(60)).unwrap();
        assert_eq!(store.open(3, "alice", "a").unwrap(), Some(b"aa".to_vec()));

        store.save(4, "alice", "d", b"dd".to_vec(), Duration::from_secs(60)).unwrap();

        assert_eq!(store.open(5, "alice", "b").unwrap(), None);
        assert_eq!(store.open(5, "alice", "a").unwrap(), Some(b"aa".to_vec()));
        assert_eq!(store.open(5, "alice", "c").unwrap(), Some(b"cc".to_vec()));
        assert_eq!(store.open(5, "alice", "d").unwrap(), Some(b"dd".to_vec()));
    }

    #[test]
    fn evicts_lru_entries_to_honor_per_user_count_cap() {
        let mut store = BlobStore::new(16, 2, 64, Duration::from_secs(60));
        store.save(0, "alice", "a", b"a".to_vec(), Duration::from_secs(60)).unwrap();
        store.save(1, "alice", "b", b"b".to_vec(), Duration::from_secs(60)).unwrap();
        assert_eq!(store.open(2, "alice", "a").unwrap(), Some(b"a".to_vec()));

        store.save(3, "alice", "c", b"c".to_vec(), Duration::from_secs(60)).unwrap();

        assert_eq!(store.open(4, "alice", "b").unwrap(), None);
        assert_eq!(store.open(4, "alice", "a").unwrap(), Some(b"a".to_vec()));
        assert_eq!(store.open(4, "alice", "c").unwrap(), Some(b"c".to_vec()));
    }
}

//! Where other workers' state is: the gateway's directory, and the links to
//! fetch it by key. Both are the link layer's, set when it connects.

use std::collections::HashSet;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use futures::future::BoxFuture;

use crate::store::kv::hash::Hash256;
use crate::store::registry::with_kv_lock;

/// A worker that holds a run from token 0: how many pages of it, and where
/// it serves its peers.
#[derive(Debug, Clone)]
pub struct Holder {
    pub addr: String,
    pub pages: u32,
}

pub trait Directory: Send + Sync {
    /// The workers, not this one, that hold a run of `keys`, longest first.
    fn locate(&self, keys: Vec<Hash256>) -> BoxFuture<'static, Result<Vec<Holder>>>;
}

pub trait Remote: Send + Sync {
    /// The bytes of pages `keys[from..]` at the worker serving `addr`.
    fn fetch(
        &self,
        addr: String,
        keys: Vec<Hash256>,
        from: u32,
    ) -> BoxFuture<'static, Result<super::prefill::PageStream>>;
}

static DIRECTORY: RwLock<Option<Arc<dyn Directory>>> = RwLock::new(None);
static REMOTE: RwLock<Option<Arc<dyn Remote>>> = RwLock::new(None);

pub fn set_directory(directory: Option<Arc<dyn Directory>>) {
    *DIRECTORY.write().unwrap() = directory;
}

pub fn set_remote(remote: Option<Arc<dyn Remote>>) {
    *REMOTE.write().unwrap() = remote;
}

pub(crate) fn directory() -> Option<Arc<dyn Directory>> {
    DIRECTORY.read().unwrap().clone()
}

pub(crate) fn remote() -> Option<Arc<dyn Remote>> {
    REMOTE.read().unwrap().clone()
}

/// Every sealed page this worker holds, by key: what it reports to the
/// directory.
pub fn sealed_keys() -> HashSet<Hash256> {
    let mut keys = HashSet::new();
    for stores in crate::store::registry::all_for_model(0) {
        keys.extend(with_kv_lock(&stores.kv, "offload-report", |kv| {
            kv.sealed_keys()
        }));
    }
    keys
}

/// Every cache entry this worker holds, as its pages' keys from token 0.
pub fn cached_runs() -> Vec<Vec<Hash256>> {
    let mut runs = Vec::new();
    for stores in crate::store::registry::all_for_model(0) {
        runs.extend(with_kv_lock(&stores.kv, "offload-report", |kv| {
            kv.cached_runs()
        }));
    }
    runs
}

/// Every page this worker's persistent disk tier keeps a copy of, by key.
pub fn persisted_keys() -> HashSet<Hash256> {
    let mut keys = HashSet::new();
    for stores in crate::store::registry::all_for_model(0) {
        keys.extend(with_kv_lock(&stores.kv, "offload-report", |kv| {
            kv.persisted_keys()
        }));
    }
    keys
}

/// The runs whose owner asked a prefix of them kept on disk: each run's last
/// key, and the tokens asked.
pub fn persist_hints() -> Vec<(Hash256, u64)> {
    let mut hints = Vec::new();
    for stores in crate::store::registry::all_for_model(0) {
        hints.extend(with_kv_lock(&stores.kv, "offload-report", |kv| {
            kv.persist_hints()
        }));
    }
    hints.sort_unstable();
    hints
}

/// The tokens one kv page holds, and whether the disk tier is kept across
/// runs; none without a model.
pub fn disk_tier() -> Option<(u32, bool)> {
    let stores = crate::store::registry::all_for_model(0);
    let page_size = stores.first()?.kv_page_size;
    let persistent = stores
        .iter()
        .any(|stores| with_kv_lock(&stores.kv, "offload-report", |kv| kv.disk_persistent()));
    Some((page_size, persistent))
}

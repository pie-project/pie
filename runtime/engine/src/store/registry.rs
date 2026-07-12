//! Per-(model, driver) store registry (kv_refact.md, `store/registry.rs`).
//!
//! Maps a model/driver pair to its owning `KvStore` and `RsStore` so
//! `pipeline::fire` and the WIT host resources resolve handles without each
//! component holding a direct store reference. Mirrors the retired
//! `arena::registry` shape: an append-only static keyed by `model_idx`
//! (lock-step with bootstrap model registration), each entry a `Vec` indexed
//! by driver ordinal.
//!
//! ## Locking discipline (required)
//! Lock a store **synchronously** for prepare/commit/abort and release it
//! **before** awaiting the driver — never across an `await`:
//! ```text
//! prepare: let mut kv = registry.kv.lock();   // sync
//!          kv.prepare_write(..)               // sync
//!          drop(kv);                          // unlock
//!          driver copies + launch, await      // no lock held
//! commit:  let mut kv = registry.kv.lock();   // re-lock
//!          kv.commit(prepared, ..)            // sync
//! ```

use std::sync::{Arc, LazyLock, Mutex};

use super::kv::KvStore;
use super::rs::RsStore;

/// The typed stores for one (model, driver).
#[derive(Clone)]
pub struct Stores {
    pub kv: Arc<Mutex<KvStore>>,
    pub rs: Arc<Mutex<RsStore>>,
    /// Tokens per KV page for this model/driver.
    pub kv_page_size: u32,
}

static REGISTRY: LazyLock<boxcar::Vec<Vec<Stores>>> = LazyLock::new(boxcar::Vec::new);

/// Register a model's per-driver stores at bootstrap. Capacities come from
/// the driver-preallocated static pools. Returns the assigned model index.
pub fn register_model(kv_page_size: u32, num_kv_pages: &[usize], num_rs_slots: &[usize]) -> usize {
    register_model_with_swap(
        kv_page_size,
        num_kv_pages,
        &vec![0; num_kv_pages.len()],
        num_rs_slots,
    )
}

pub fn register_model_with_swap(
    kv_page_size: u32,
    num_kv_pages: &[usize],
    num_host_pages: &[usize],
    num_rs_slots: &[usize],
) -> usize {
    let stores: Vec<Stores> = (0..num_kv_pages.len())
        .map(|d| Stores {
            kv: Arc::new(Mutex::new(KvStore::new_with_swap(
                num_kv_pages[d] as u32,
                num_host_pages.get(d).copied().unwrap_or(0) as u32,
                rand::random::<[u8; 32]>(),
            ))),
            rs: Arc::new(Mutex::new(RsStore::new(
                num_rs_slots.get(d).copied().unwrap_or(0) as u32,
            ))),
            kv_page_size,
        })
        .collect();
    REGISTRY.push(stores)
}

/// The stores for `(model_idx, driver_idx)`; cheap `Arc` clones. Panics if
/// never registered — a bootstrap wiring bug, not a runtime condition.
pub fn get(model_idx: usize, driver_idx: usize) -> Stores {
    try_get(model_idx, driver_idx).unwrap_or_else(|| {
        panic!("store registry: no stores for model {model_idx} driver {driver_idx}")
    })
}

pub fn try_get(model_idx: usize, driver_idx: usize) -> Option<Stores> {
    REGISTRY.get(model_idx)?.get(driver_idx).cloned()
}

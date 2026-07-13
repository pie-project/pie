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

use std::sync::{Arc, LazyLock, Mutex, RwLock};

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

static REGISTRY: LazyLock<boxcar::Vec<RwLock<Vec<Option<Stores>>>>> =
    LazyLock::new(boxcar::Vec::new);

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
    let stores: Vec<Option<Stores>> = (0..num_kv_pages.len())
        .map(|d| {
            Some(Stores {
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
        })
        .collect();
    REGISTRY.push(RwLock::new(stores))
}

pub fn register_driver_with_swap(
    model_idx: usize,
    driver_idx: usize,
    kv_page_size: u32,
    base_page: u32,
    num_kv_pages: usize,
    num_host_pages: usize,
    num_rs_slots: usize,
) -> anyhow::Result<()> {
    let model = REGISTRY
        .get(model_idx)
        .ok_or_else(|| anyhow::anyhow!("store registry: unknown model {model_idx}"))?;
    let mut stores = model.write().unwrap();
    anyhow::ensure!(
        stores.len() == driver_idx,
        "store registry: dynamic driver id {driver_idx} is not the next slot {} for model {model_idx}",
        stores.len()
    );
    if let Some(existing) = stores.iter().flatten().next() {
        anyhow::ensure!(
            existing.kv_page_size == kv_page_size,
            "store registry: KV page size {kv_page_size} does not match model {model_idx} page size {}",
            existing.kv_page_size
        );
    }
    stores.push(Some(Stores {
        kv: Arc::new(Mutex::new(KvStore::new_with_swap_range(
            base_page,
            num_kv_pages as u32,
            num_host_pages as u32,
            rand::random::<[u8; 32]>(),
        ))),
        rs: Arc::new(Mutex::new(RsStore::new(num_rs_slots as u32))),
        kv_page_size,
    }));
    Ok(())
}

pub fn unregister_driver(model_idx: usize, driver_idx: usize) -> anyhow::Result<()> {
    let model = REGISTRY
        .get(model_idx)
        .ok_or_else(|| anyhow::anyhow!("store registry: unknown model {model_idx}"))?;
    let mut stores = model.write().unwrap();
    let slot = stores.get_mut(driver_idx).ok_or_else(|| {
        anyhow::anyhow!("store registry: unknown driver {driver_idx} for model {model_idx}")
    })?;
    anyhow::ensure!(
        slot.take().is_some(),
        "store registry: driver {driver_idx} for model {model_idx} is already unregistered"
    );
    Ok(())
}

/// The stores for `(model_idx, driver_idx)`; cheap `Arc` clones. Panics if
/// never registered — a bootstrap wiring bug, not a runtime condition.
pub fn get(model_idx: usize, driver_idx: usize) -> Stores {
    try_get(model_idx, driver_idx).unwrap_or_else(|| {
        panic!("store registry: no stores for model {model_idx} driver {driver_idx}")
    })
}

pub fn try_get(model_idx: usize, driver_idx: usize) -> Option<Stores> {
    REGISTRY
        .get(model_idx)?
        .read()
        .unwrap()
        .get(driver_idx)
        .cloned()
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_store_slots_unregister_without_reusing_driver_ids() {
        let model = register_model(16, &[8], &[0]);
        register_driver_with_swap(model, 1, 16, 10, 4, 0, 0).unwrap();
        assert!(try_get(model, 1).is_some());
        unregister_driver(model, 1).unwrap();
        assert!(try_get(model, 1).is_none());
        register_driver_with_swap(model, 2, 16, 20, 4, 0, 0).unwrap();
        assert!(try_get(model, 2).is_some());
    }
}

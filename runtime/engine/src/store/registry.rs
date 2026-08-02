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
//! Lock a store **synchronously** for prepare/publication/settlement and release it
//! **before** awaiting the driver — never across an `await`:
//! ```text
//! prepare: let mut kv = registry.kv.lock();   // sync
//!          kv.prepare_write(..)               // sync
//!          drop(kv);                          // unlock
//!          driver copies + launch, await      // no lock held
//! settle: let mut kv = registry.kv.lock();    // re-lock
//!         kv.settle(..)                       // sync
//! ```

use std::sync::atomic::Ordering;
use std::sync::{Arc, LazyLock, Mutex, RwLock};

use super::kv::KvStore;
use super::rs::RsStore;

/// parking_lot mutexes do not poison: a panic unwinding mid-mutation would
/// leave a half-updated `KvStore` silently live (std's poisoning made the
/// next `lock().unwrap()` crash loud). This taint flag restores fail-loud —
/// set on unwind inside [`with_kv_lock`], asserted on every entry.
///
/// KNOWN BLAST RADIUS: the flag is process-global, not per-store — one
/// panic taints every `(model, driver)` store in the process, and in the
/// test binary a single panicking `with_kv_lock` cascades into every later
/// KV test in the same process. Acceptable while deployment is one store;
/// a multi-store engine should move the flag beside the mutex it guards
/// (a `KvStoreLock { mutex, tainted }` newtype in `Stores`).
static KV_TAINTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

struct KvTaintOnPanic;

impl Drop for KvTaintOnPanic {
    fn drop(&mut self) {
        if std::thread::panicking() {
            KV_TAINTED.store(true, Ordering::SeqCst);
        }
    }
}

#[inline(always)]
pub fn with_kv_lock<T>(
    store: &parking_lot::Mutex<KvStore>,
    tag: &'static str,
    operation: impl FnOnce(&mut KvStore) -> T,
) -> T {
    assert!(
        !KV_TAINTED.load(Ordering::Relaxed),
        "KV store tainted by an earlier panic mid-mutation ({tag})"
    );
    let taint = KvTaintOnPanic;
    let mut guard = store.lock();
    let result = operation(&mut guard);
    drop(guard);
    drop(taint);
    result
}

/// The typed stores for one (model, driver).
#[derive(Clone)]
pub struct Stores {
    // parking_lot: adaptive spinning beats the futex round trip under the
    // contended herd (every guest's finalize/prepare takes this lock; the
    // measured wake-herd lock storm cost ~2 ms per wave — §15).
    pub kv: Arc<parking_lot::Mutex<KvStore>>,
    pub rs: Arc<Mutex<RsStore>>,
    /// Tokens per KV page for this model/driver.
    pub kv_page_size: u32,
}

static REGISTRY: LazyLock<boxcar::Vec<RwLock<Vec<Option<Stores>>>>> =
    LazyLock::new(boxcar::Vec::new);

/// Test convenience: register a model with no host swap pages. Production
/// bootstrap always sizes swap explicitly via [`register_model_with_swap`].
#[cfg(test)]
pub fn register_model(kv_page_size: u32, num_kv_pages: &[usize], num_rs_slots: &[usize]) -> usize {
    register_model_with_swap(
        kv_page_size,
        num_kv_pages,
        &vec![0; num_kv_pages.len()],
        num_rs_slots,
    )
}

/// Register a model's per-driver stores at bootstrap. Capacities come from
/// the driver-preallocated static pools. Returns the assigned model index.
pub fn register_model_with_swap(
    kv_page_size: u32,
    num_kv_pages: &[usize],
    num_host_pages: &[usize],
    num_rs_slots: &[usize],
) -> usize {
    let stores: Vec<Option<Stores>> = (0..num_kv_pages.len())
        .map(|d| {
            let kv = Arc::new(parking_lot::Mutex::new(KvStore::new_with_swap(
                num_kv_pages[d] as u32,
                num_host_pages.get(d).copied().unwrap_or(0) as u32,
                rand::random::<[u8; 32]>(),
            )));
            Some(Stores {
                kv,
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
    if stores.len() <= driver_idx {
        stores.resize_with(driver_idx + 1, || None);
    }
    anyhow::ensure!(
        stores[driver_idx].is_none(),
        "store registry: driver {driver_idx} is already registered for model {model_idx}"
    );
    if let Some(existing) = stores.iter().flatten().next() {
        anyhow::ensure!(
            existing.kv_page_size == kv_page_size,
            "store registry: KV page size {kv_page_size} does not match model {model_idx} page size {}",
            existing.kv_page_size
        );
    }
    let kv = Arc::new(parking_lot::Mutex::new(KvStore::new_with_swap_range(
        base_page,
        num_kv_pages as u32,
        num_host_pages as u32,
        rand::random::<[u8; 32]>(),
    )));
    stores[driver_idx] = Some(Stores {
        kv,
        rs: Arc::new(Mutex::new(RsStore::new(num_rs_slots as u32))),
        kv_page_size,
    });
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

pub fn all_for_model(model_idx: usize) -> Vec<Stores> {
    REGISTRY
        .get(model_idx)
        .map(|stores| stores.read().unwrap().iter().flatten().cloned().collect())
        .unwrap_or_default()
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

    #[test]
    fn dynamic_store_slots_allow_global_driver_id_gaps() {
        let model = register_model(16, &[8], &[0]);
        register_driver_with_swap(model, 4, 16, 40, 4, 0, 0).unwrap();
        assert!(try_get(model, 1).is_none());
        assert!(try_get(model, 3).is_none());
        assert!(try_get(model, 4).is_some());
        assert_eq!(all_for_model(model).len(), 2);
    }
}

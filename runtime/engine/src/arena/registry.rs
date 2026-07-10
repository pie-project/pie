//! Per-(model, driver) arena registry.
//!
//! The unified [`Arena`] is one per driver, shared by the KV and RS working-set
//! host resources and the forward-pass `execute()` prepare. Those run on the
//! **instance thread** (inside WASM host methods, which hold `InstanceState`
//! and the resource table) — *not* inside the `context` message-passing actor,
//! which has no access to the WASM resource table. So the arena cannot live as
//! a plain field on the `ContextManager` actor; it must be reachable
//! synchronously from `InstanceState`.
//!
//! This registry is that home: a standalone, append-only static keyed by
//! `model_idx` (lock-step with `context::SERVICES` / `PAGE_SIZES`), each model
//! owning a `Vec<Arc<Mutex<Arena>>>` indexed by driver ordinal. Being
//! independent of `ContextManager`, it survives the Phase-5 context teardown
//! with zero extraction.
//!
//! ## Locking discipline (required)
//! Lock an arena **synchronously** for the transaction ops and release it
//! **before** awaiting the driver — never hold the lock across an `await`:
//! ```text
//! prepare: let mut a = registry::get(model, drv).lock();  // sync
//!          a.txn_alloc / txn_cow / txn_pin / resolve  ...    // sync
//!          drop(a);                                          // unlock
//!          driver d2d + submit().await                       // no lock held
//! commit:  let mut a = registry::get(model, drv).lock();  // re-lock
//!          a.txn_commit(txn)  ...                            // sync
//! ```
//! This keeps the critical section short and await-free, so there is no
//! cross-instance deadlock or contention across the driver round-trip.

use std::sync::{Arc, LazyLock, Mutex};

use crate::driver::DriverId;

use super::{Arena, ArenaConfig};

/// Append-only registry, indexed by `model_idx`. Each entry is a model's
/// per-driver arenas, indexed by driver ordinal.
static REGISTRY: LazyLock<boxcar::Vec<Vec<Arc<Mutex<Arena>>>>> = LazyLock::new(boxcar::Vec::new);

/// Register a model's per-driver arenas at bootstrap (called from
/// `context::spawn`, in lock-step with `SERVICES`/`PAGE_SIZES`).
///
/// Capacities mirror the legacy per-driver pools so the arena can take over
/// physical allocation in step: one KV-page-sized block per KV page, one block
/// per RS slot (v1 single-slot slabs), CPU stash sized to the offload tier. No
/// scratch consumers in v1. The arena's `device` is the driver ordinal, the
/// same id the driver copy ops are addressed by. Returns the assigned model
/// index (equals `context::SERVICES`'s `model_idx`).
pub fn register_model(
    page_size: usize,
    num_gpu_pages: &[usize],
    num_cpu_pages: &[usize],
    num_rs_slots: &[usize],
) -> usize {
    let arenas: Vec<Arc<Mutex<Arena>>> = (0..num_gpu_pages.len())
        .map(|d| {
            Arc::new(Mutex::new(Arena::new(ArenaConfig {
                device: d as DriverId,
                block_size: page_size as u32,
                kv_pages: num_gpu_pages[d] as u32,
                rs_blocks: num_rs_slots.get(d).copied().unwrap_or(0) as u32,
                scratch_blocks: 0,
                cpu_blocks: num_cpu_pages.get(d).copied().unwrap_or(0) as u32,
            })))
        })
        .collect();
    REGISTRY.push(arenas)
}

/// The arena for `(model_idx, driver_idx)`. The returned `Arc` is cheap to
/// clone; lock it synchronously and release before any `await` (see the module
/// docs). Panics if the model/driver was never registered — a bootstrap wiring
/// bug, not a runtime condition.
pub fn get(model_idx: usize, driver_idx: usize) -> Arc<Mutex<Arena>> {
    try_get(model_idx, driver_idx).unwrap_or_else(|| {
        panic!("arena registry: no arena for model {model_idx} driver {driver_idx}")
    })
}

/// Fallible lookup — `None` if the model or driver is not registered.
pub fn try_get(model_idx: usize, driver_idx: usize) -> Option<Arc<Mutex<Arena>>> {
    REGISTRY
        .get(model_idx)
        .and_then(|drivers| drivers.get(driver_idx))
        .map(Arc::clone)
}

/// Number of registered models.
pub fn model_count() -> usize {
    REGISTRY.count()
}

/// Number of drivers registered for `model_idx` (0 if the model is absent).
pub fn driver_count(model_idx: usize) -> usize {
    REGISTRY.get(model_idx).map(Vec::len).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{ArenaKind, Residency};

    #[test]
    fn register_lookup_and_lock_sync() {
        // Two drivers: driver 0 KV=8/RS=2, driver 1 KV=4/RS=0 (RS-unsupported).
        let m = register_model(16, &[8, 4], &[8, 4], &[2, 0]);
        assert_eq!(driver_count(m), 2);

        {
            let a = get(m, 0);
            let mut g = a.lock().unwrap();
            assert_eq!(g.block_size(), 16);
            assert_eq!(g.capacity(ArenaKind::KvPage), 8);
            assert_eq!(g.capacity(ArenaKind::RsSlab), 2);
            let h = g.alloc(ArenaKind::KvPage, 1).unwrap();
            assert_eq!(g.residency(h.object_id).unwrap(), Residency::Gpu);
        } // lock released before any further work

        {
            let a = get(m, 1);
            let g = a.lock().unwrap();
            assert_eq!(g.capacity(ArenaKind::RsSlab), 0); // RS-unsupported driver
        }

        assert!(try_get(m, 5).is_none());
        assert!(try_get(99_999, 0).is_none());
    }

    #[test]
    fn distinct_models_get_distinct_arenas() {
        let a = register_model(8, &[2], &[2], &[0]);
        let b = register_model(8, &[2], &[2], &[0]);
        assert_ne!(a, b);
        // Allocating in model a must not affect model b's pool.
        let _h = get(a, 0)
            .lock()
            .unwrap()
            .alloc(ArenaKind::KvPage, 1)
            .unwrap();
        assert_eq!(get(a, 0).lock().unwrap().used(ArenaKind::KvPage), 1);
        assert_eq!(get(b, 0).lock().unwrap().used(ArenaKind::KvPage), 0);
    }
}

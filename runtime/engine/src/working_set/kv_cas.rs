//! Per-(model, driver) KV CAS-index registry (Lane C).
//!
//! Sibling to bravo's `arena` registry (`arena::get(model_idx, driver_idx)`),
//! deliberately kept in the KV lane so bravo's `Arc<Mutex<Arena>>` registry stays
//! Lane-B-only and survives echo's Phase-5 teardown (no inverted layering).
//!
//! Holds one [`KvCas`] (the content-hash sharing index, W6/W7) per
//! `(model_idx, driver_idx)`, behind `Arc<Mutex<_>>`. It is registered at
//! bootstrap in the same per-model loop as the arena (right after
//! `arena::register_model`, before `context::spawn`), so `model_idx` stays
//! lock-step with the arena registry and the registration survives Phase 5.
//!
//! ## Locking discipline
//! The only operation needing both an arena lock and a CAS lock is `seal`. The
//! lock order is **always `arena` → `kv_cas`**, and the CAS lock is acquired only
//! in the forward txn's commit/seal step (after the arena lock), never before it
//! and never across an `await`. `resolve_read` / `resolve_write` /
//! `cow_write_slot` take the arena lock only. No cycle ⇒ deadlock-free without
//! bundling the two registries.

use std::sync::{Arc, LazyLock, Mutex};

use crate::working_set::kv::KvCas;

/// `model_idx -> (driver_idx -> KvCas)`. Append-only (`boxcar`) so registration
/// is lock-free and indices are stable, mirroring `context::PAGE_SIZES`.
static REGISTRY: LazyLock<boxcar::Vec<Vec<Arc<Mutex<KvCas>>>>> = LazyLock::new(boxcar::Vec::new);

/// Register a model's KV CAS indices — one fresh [`KvCas`] per driver — and
/// return the assigned `model_idx`. Call exactly once per model at bootstrap,
/// immediately after the arena registration for the same model, so the returned
/// index matches the arena registry's.
pub fn register_cas(num_drivers: usize) -> usize {
    let drivers: Vec<Arc<Mutex<KvCas>>> = (0..num_drivers)
        .map(|_| Arc::new(Mutex::new(KvCas::new())))
        .collect();
    REGISTRY.push(drivers)
}

/// The KV CAS index for `(model_idx, driver_idx)`. Panics if the model/driver was
/// not registered at bootstrap (a wiring bug, not a runtime input error).
pub fn get(model_idx: usize, driver_idx: usize) -> Arc<Mutex<KvCas>> {
    try_get(model_idx, driver_idx)
        .unwrap_or_else(|| panic!("kv_cas: no CAS index for (model {model_idx}, driver {driver_idx})"))
}

/// Fallible accessor: `None` if `(model_idx, driver_idx)` is not registered.
pub fn try_get(model_idx: usize, driver_idx: usize) -> Option<Arc<Mutex<KvCas>>> {
    REGISTRY
        .get(model_idx)
        .and_then(|drivers| drivers.get(driver_idx))
        .cloned()
}

/// Number of models currently registered (diagnostics/tests).
pub fn model_count() -> usize {
    REGISTRY.count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_get_roundtrip() {
        // Indices are stable and lock-step with registration order.
        let m = register_cas(2);
        assert!(get(m, 0).lock().unwrap().is_empty());
        assert!(get(m, 1).lock().unwrap().is_empty());
        // Same handle is shared (Arc) across gets — CAS state is per (model,driver).
        let a = get(m, 0);
        let b = get(m, 0);
        assert!(Arc::ptr_eq(&a, &b));
        // Unregistered driver/model → None.
        assert!(try_get(m, 9).is_none());
        assert!(try_get(m + 10_000, 0).is_none());
    }
}

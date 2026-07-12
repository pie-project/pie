//! Typed resource stores (kv_refact.md): device memory as pages, slots, and
//! mappings. Replaces the generic KV-page-sized `Arena` with resource-specific
//! stores over typed static backing pools.
//!
//! - [`kv`]: `KvStore` — the mapping trie, hash lifecycle, and prepare/commit/
//!   abort protocol over the physical KV page pool.
//! - `rs`: `RsStore` — the recurrent-state slot store (GDN/Mamba2 folded
//!   state) with the same prepare/commit/abort protocol.
//! - `pool`/`genmap`: the physical-id free list and generational key map the
//!   typed stores are built on.
//! - `registry`: per-(model, driver) lookup of the owning `KvStore`/`RsStore`.
//! - [`reclaim`]: the pressure ladder (idle-lease drop, preempt-youngest,
//!   wait queue, restore-on-free) — the only submodule external tests reach
//!   directly (`store::reclaim::contention()`), since it is this crate's
//!   sole KV-contention diagnostic surface.

pub(crate) mod genmap;
pub(crate) mod kv;
pub(crate) mod pool;
pub mod reclaim;
pub(crate) mod registry;
pub(crate) mod rs;

/// Coarse worker-routing signal derived from real KV residency and contention.
pub fn kv_pressure_bucket() -> u8 {
    if let Some(orchestrator) = reclaim::contention() {
        return orchestrator.kv_pressure_bucket();
    }
    let Some(stores) = registry::try_get(0, 0) else {
        return 0;
    };
    let kv = stores.kv.lock().unwrap();
    let total = kv.capacity_pages();
    if total == 0 {
        return 0;
    }
    let used = total.saturating_sub(kv.available_pages() as u32);
    (f64::from(used) / f64::from(total) * 255.0).round() as u8
}

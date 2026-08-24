//! What [`crate::layout`] planned, allocated.
//!
//! The KV pages, recurrent slabs, and swap regions — the long-lived device
//! memory a fire binds against. Shapes are decided in `layout/`; this is the
//! half that needs a card.

#![forbid(unsafe_code)]

pub mod kv_cache;
pub mod kv_cache_live;
pub mod recurrent_state_cache;
pub mod swap_pool;

// `mla_cache` and `compressed_plane_cache` STOOD HERE, with their geometry in
// `layout/{mla,compressed_plane}_geometry.rs`. Nothing in the driver ever
// allocated either: `Deployment::of` refuses every MLA/latent SKU by name, and
// the `model_costs` arms that sized them died with the legacy walk. What kept
// them compiling was their own parity tests, which is a module held up by the
// test that would have reported it dead.

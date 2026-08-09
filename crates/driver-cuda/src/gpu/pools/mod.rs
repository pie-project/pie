//! What [`crate::layout`] planned, allocated.
//!
//! The KV pages, the recurrent slabs, the swap regions — the long-lived
//! device memory a fire binds against. Every shape here was decided in
//! `layout/`; this is the half that needs a card, which is why the two
//! are separate directories rather than one `store/`.

pub mod dsv4_compress_cache;
pub mod kv_cache;
pub mod kv_cache_live;
pub mod mla_cache;
pub mod recurrent_state_cache;
pub mod swap_pool;

//! What [`crate::layout`] planned, allocated.
//!
//! The KV pages, recurrent slabs, and swap regions — the long-lived device
//! memory a fire binds against. Shapes are decided in `layout/`; this is the
//! half that needs a card.

#![forbid(unsafe_code)]

pub mod compressed_plane_cache;
pub mod kv_cache;
pub mod kv_cache_live;
pub mod mla_cache;
pub mod recurrent_state_cache;
pub mod swap_pool;

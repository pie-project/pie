//! What [`crate::layout`] planned, allocated.
//!
//! The KV pages, the recurrent slabs, the swap regions — the long-lived
//! device memory a fire binds against. Every shape here was decided in
//! `layout/`; this is the half that needs a card, which is why the two
//! are separate directories rather than one `store/`.

// THE CRATE'S THESIS, held by the compiler on the second half where it
// became reachable (§7). `pools/` allocates device memory and needed
// exactly TWO `unsafe` blocks to do it, and neither was about
// allocating: `kv_cache_live` launched the envelope seed and
// synchronised a stream by naming CUDA symbols directly, which is a
// pool reaching past `device` and `bind` for vocabulary they already
// have. `StreamRef::synchronize` and `bind::abi::seed_envelopes_empty`
// are those two, said once where the pointers are understood.
//
// That is §7's rule read from the other side: a module that says WHAT
// to launch should not have to spell HOW. And it is a good sign for the
// `layout`/`pools` split that the half which ALLOCATES needed none —
// the `unsafe` was never in the allocation.
#![forbid(unsafe_code)]

pub mod compressed_plane_cache;
pub mod kv_cache;
pub mod kv_cache_live;
pub mod mla_cache;
pub mod recurrent_state_cache;
pub mod swap_pool;

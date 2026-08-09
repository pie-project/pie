//! `layout/`: how big, where, how many — and none of it needs a card.
//!
//! Geometry, memory budgets, swap plans, the load plan and the profile
//! cache. Every answer here is arithmetic over shapes and budgets, which
//! is why this directory is OUTSIDE `gpu/` and builds with no CUDA
//! feature selected at all (`tests/portable_half.rs`).
//!
//! The line between this and [`crate::gpu::pools`] is the one this
//! crate's `Cargo.toml` already drew: **`kv_geometry` says what shape
//! the pages are and `kv_cache` allocates them, and only the second one
//! needs a card.** It used to run between 22 files inside one `store/`;
//! raised to a directory, the `#[cfg]` collapses from per-file to one.

// THE CRATE'S THESIS, held by the compiler on the half where it is
// reachable today (§8 row 11). `lib.rs` cannot carry this: `gpu/` has
// 342 `unsafe` blocks and needs them, because that is where CUDA is.
// `layout/` needed exactly one — a `libc::flock` on the profile
// cache — and it did not need to be `unsafe` at all; `File::lock` is
// the same advisory lock with the descriptor's lifetime in the type.
//
// The rule this makes checkable: an answer that does not need a card
// does not need `unsafe` either. A future module that reaches for one
// here fails the build rather than the review.
#![forbid(unsafe_code)]

pub mod budget;
pub mod calibrate;
pub mod dsv4_geometry;
pub mod dtoa;
pub mod json;
pub mod kv_format;
pub mod kv_geometry;
pub mod memory_planner;
pub mod mla_geometry;
pub mod model_costs;
pub mod planner_policy;
pub mod profile_cache;
pub mod profile_key;
pub mod recurrent_layout;
pub mod rendezvous;
pub mod swap_plan;
/// How big the fire's scratch is — arithmetic, so it lives here rather
/// than beside the buffer it sizes.
pub mod workspace;

pub use kv_format::{KvCacheFormat, KvCacheScaleLayout, KvCacheScheme};

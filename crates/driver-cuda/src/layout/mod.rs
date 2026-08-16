//! `layout/`: how big, where, how many.
//!
//! Geometry, memory budgets, swap plans, the load plan and the profile cache
//! — all arithmetic over shapes and budgets, so this directory builds with no
//! CUDA feature selected (`tests/portable_half.rs`). `kv_geometry` says what
//! shape the pages are; [`crate::pools`]'s `kv_cache` allocates them.

// Arithmetic only, so unsafe can be forbidden here.
#![forbid(unsafe_code)]

pub mod budget;
pub mod calibrate;
pub mod compressed_plane_geometry;
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
/// How big the fire's scratch is.
pub mod workspace;

pub use kv_format::{KvCacheFormat, KvCacheScaleLayout, KvCacheScheme};

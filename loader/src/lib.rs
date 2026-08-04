//! Runtime-owned planner for Pie model loading.
//!
//! `plan = compile(source_facts, program, target)`. None of the three inputs is
//! a model's name: the driver states what it needs as a contract over the
//! checkpoint's byte space, the loader reads the checkpoint's own metadata, and
//! the target carries the numbers a device measured. `tests/standalone.rs` pins
//! that as four properties rather than as prose.
//!
//! CUDA and `WeightStore` ownership stay on the C++ side. Reading a checkpoint
//! is `crate::checkpoint`'s alone, and the compiler below it opens nothing —
//! `crate::testkit::host_executor` does, which is exactly why it is not the
//! compiler.

pub mod cache_key;
pub mod checkpoint;
pub mod contract;
pub mod dump;
pub mod error;
pub mod extent;
pub mod ffi;
pub mod plan;
#[cfg(feature = "testkit")]
pub mod testkit;
pub mod types;
pub mod verify;
pub mod weight_store;

/// Single source for the planner's debug-logging gate (`PIE_LOAD_PLANNER_DEBUG`).
pub(crate) fn planner_debug_enabled() -> bool {
    std::env::var_os("PIE_LOAD_PLANNER_DEBUG").is_some()
}

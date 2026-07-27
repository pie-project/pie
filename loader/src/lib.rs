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
//! `crate::host_executor` does, which is exactly why it is not the compiler.

pub mod artifact;
pub mod backend;
pub mod checkpoint;
pub mod contract;
pub mod contract_writer;
pub mod dump;
pub mod error;
pub mod ffi;
pub mod frontend;
pub mod host_executor;
pub mod ir;
pub mod load_plan;
pub mod optimizer;
pub mod planner;
pub mod reference;
pub mod typecheck;
pub mod types;
pub mod verify;

/// Single source for the planner's debug-logging gate (`PIE_LOAD_PLANNER_DEBUG`).
pub(crate) fn planner_debug_enabled() -> bool {
    std::env::var_os("PIE_LOAD_PLANNER_DEBUG").is_some()
}

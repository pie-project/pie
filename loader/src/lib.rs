//! Runtime-owned planner for Pie model loading.
//!
//! The crate intentionally keeps CUDA, file IO, and `WeightStore` ownership on
//! the C++ side. Rust receives metadata/config/ABI data and returns a flat
//! executable LoadPlan.

pub mod artifact;
pub mod backend;
pub mod checkpoint;
pub mod contract;
pub mod contract_writer;
pub mod dump;
pub mod error;
pub mod ffi;
pub mod frontend;
pub mod host;
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

//! Pure Rust compiler boundary for Pie weight loading.
//!
//! The crate intentionally keeps CUDA, file IO, and `WeightStore` ownership on
//! the C++ side. Rust receives metadata/config/ABI data and returns a flat
//! executable storage program view.

pub mod abi;
pub mod checkpoint_header;
pub mod config;
pub mod dump;
pub mod error;
pub mod frontend;
pub mod gguf;
pub mod host_executor;
pub mod inproc;
pub mod ir;
pub mod optimizer;
pub mod reference;
pub mod schema;
pub mod schemas;
pub mod semantic;
pub mod source;
pub mod storage;
pub mod storage_compiler;
pub mod typecheck;
pub mod types;

/// Single source for the loader's debug-logging gate (`PIE_WEIGHT_LOADER_DEBUG`).
pub(crate) fn wl_debug_enabled() -> bool {
    std::env::var_os("PIE_WEIGHT_LOADER_DEBUG").is_some()
}

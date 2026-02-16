//! FFI module for Python-Rust interop.
//!
//! Contains PyO3 bindings exposed to Python via the `_runtime` extension module.

pub mod pybindings;

pub use pybindings::{Config, RuntimeHandle};

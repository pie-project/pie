//! Re-export IPC bridge types from the central ffi module.
//!
//! This module exists for backwards compatibility with code that imports
//! from `legacy_model::ffi_bridge`.

pub use crate::ffi::AsyncIpcClient;

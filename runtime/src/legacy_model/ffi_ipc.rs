//! Re-export IPC types from the central ffi module.
//!
//! This module exists for backwards compatibility with code that imports
//! from `legacy_model::ffi_ipc`.

pub use crate::ffi::{FfiIpcBackend, FfiIpcQueue, IpcChannels, IpcRequest, IpcResponse};

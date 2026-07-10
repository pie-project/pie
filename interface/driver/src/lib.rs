//! pie-driver-abi — the final local runtime ↔ driver contract.
//!
//! This crate now exposes exactly three public surfaces:
//!
//! - [`local`]: plain `#[repr(C)]` direct-FFI descriptors and symbol declarations.
//! - [`capabilities`]: reduced cold-path JSON facts used at create time.
//! - [`transfer`]: Rust-only KV transfer vocabulary shared with cross-node transport.
//!
//! The committed `include/pie_driver_abi.h` header is generated from [`local`]
//! via `pie-driver-abi-cbindgen`.

pub mod capabilities;
pub mod local;
pub mod transfer;

pub use capabilities::DriverCapabilities;
pub use local::*;
pub use transfer::{KvDtype, KvHandle, KvLayout, KvLayoutKind, KvRegion, MemoryDomain};

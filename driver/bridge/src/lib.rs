//! pie-bridge — canonical wire schema (rkyv) + IPC for the pie runtime
//! ↔ driver interface.
//!
//! The wire format IS Rust source: the structs in [`schema`] carry
//! `#[schema(...)]` which derives `Archive` + `Serialize` + `Deserialize`
//! and emits every `extern "C"` reader, parse entry, descriptor type,
//! and builder mechanically from the type name. Downstream Rust
//! consumers (`pie`, `pie-driver-dummy`) import these structs directly
//! — no IDL, no codegen, no mirror types.
//!
//! C++ and Python downstream drivers go through the auto-generated C
//! ABI (gated behind the `cabi` feature) and PyO3 wrappers (gated
//! behind the `python` feature) — rkyv itself is Rust-only.

pub mod brle;
pub mod capabilities;
pub mod schema;
pub mod wire;

// Schema types are the public API. Re-export at the crate root so
// `pie_bridge::Frame` etc. work without an extra namespace hop.
pub use brle::Brle;
pub use capabilities::DriverCapabilities;
pub use schema::*;

#[cfg(feature = "ipc")]
pub mod ipc;

pub mod ffi;

#[cfg(feature = "python")]
mod python;

include!(concat!(env!("OUT_DIR"), "/schema_hash.rs"));

//! pie-ipc — in-proc IPC mechanism for the pie runtime ↔ driver
//! interface.
//!
//! The canonical wire schema vocabulary now lives in the [`pie_driver_abi`]
//! crate (the dependency floor). This crate re-exports it for source
//! compatibility and adds the transport machinery: the rkyv encode/
//! parse helpers ([`wire`]), the in-proc C-ABI vtable ([`ffi`]), and the
//! shared-memory ring ([`ipc`]).
//!
//! Downstream Rust consumers can keep importing `pie_ipc::Frame`
//! etc.; new code that needs only the schema vocabulary (without the
//! IPC mechanism) should depend on `pie-driver-abi` directly.

// Schema vocabulary lives in `pie-driver-abi`. Re-export it so existing
// `pie_ipc::Frame`, `pie_ipc::schema::*`, `pie_ipc::Brle`,
// `pie_ipc::DriverCapabilities`, `pie_ipc::SCHEMA_HASH`, etc.
// paths keep resolving. The wire/ffi/ipc modules below also resolve
// `crate::schema::*` through the `schema` module re-export.
pub use pie_driver_abi::SCHEMA_HASH;
pub use pie_driver_abi::brle::Brle;
pub use pie_driver_abi::capabilities::DriverCapabilities;
pub use pie_driver_abi::schema::*;
// Flat-POD wire types (`CopyDir`, `AdapterBinding`, …) + their `Archived*`
// forms live in `pie_driver_abi::pod`; re-export so `pie_ipc::CopyDir` etc. and
// the C-ABI round-trip tests keep resolving them.
pub use pie_driver_abi::pod::*;
pub use pie_driver_abi::{brle, capabilities, pod, schema};

pub mod wire;

#[cfg(feature = "ipc")]
pub mod ipc;

pub mod ffi;

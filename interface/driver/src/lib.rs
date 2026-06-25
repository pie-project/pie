//! pie-driver-abi — canonical runtime ↔ driver ABI vocabulary for pie.
//!
//! This crate owns the wire ring and the in-node driver surface: the rich
//! `#[schema]` rkyv types in [`schema`], the flat-POD descriptors in [`pod`]
//! (cast-readable by C++/Python), the run-length tensor codec in [`brle`], the
//! [`capabilities`] JSON handshake, and the [`kv`] layout/handle vocabulary the
//! data-plane transport moves. It imports nothing internal except its own derive
//! macro, so transport, runtime, drivers, and the in-proc IPC mechanism can all
//! share this ABI without dragging in any transport or IPC machinery.
//!
//! The wire format IS Rust source: the structs in [`schema`] carry
//! `#[schema(...)]` which derives `Archive` + `Serialize` + `Deserialize` and
//! emits every `extern "C"` reader, parse entry, descriptor type, and builder
//! mechanically from the type name. Downstream Rust consumers import these
//! structs directly — no IDL, no codegen, no mirror types.
//!
//! C++ downstream drivers go through the auto-generated C ABI (`include/`); rkyv
//! itself is Rust-only.

pub mod brle;
pub mod capabilities;
pub mod kv;
pub mod pod;
pub mod schema;

// Schema types are the public API. Re-export at the crate root so
// `pie_driver_abi::Frame` etc. work without an extra namespace hop.
pub use brle::Brle;
pub use capabilities::DriverCapabilities;
pub use kv::{KvDtype, KvHandle, KvLayout, KvLayoutKind, KvRegion, MemoryDomain};
// Flat-POD wire types (see `pod`): plain `#[repr(C)]`/`#[repr(u8)]` + rkyv,
// embedded by value in the rich types via `#[schema(pod)]`.
pub use pod::*;
pub use schema::*;

include!(concat!(env!("OUT_DIR"), "/schema_hash.rs"));

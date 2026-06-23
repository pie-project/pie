//! pie-schema — canonical wire schema (rkyv) vocabulary for the pie
//! runtime ↔ driver interface.
//!
//! This crate is the dependency floor: it imports nothing internal, so
//! transport, runtime, drivers, the in-proc IPC mechanism, and the
//! controller can all share the schema vocabulary without dragging in
//! any transport or IPC machinery.
//!
//! The wire format IS Rust source: the structs in [`schema`] carry
//! `#[schema(...)]` which derives `Archive` + `Serialize` + `Deserialize`
//! and emits every `extern "C"` reader, parse entry, descriptor type,
//! and builder mechanically from the type name. Downstream Rust
//! consumers import these structs directly — no IDL, no codegen, no
//! mirror types.
//!
//! C++ downstream drivers go through the auto-generated C ABI (gated
//! behind the `cabi` feature); rkyv itself is Rust-only.

pub mod brle;
pub mod capabilities;
pub mod cluster;
// The disaggregated-serving control contract. Reached as `pie_schema::control::*`
// (NOT flat-re-exported below): its `WorkerId`/`WorkerInfo`/`Role` deliberately
// shadow the legacy `cluster` names this redesign supersedes, so the module path
// keeps the two unambiguous until the legacy control bits are removed.
pub mod control;
pub mod edge;
pub mod kv;
pub mod message;
pub mod pod;
pub mod schema;

// Schema types are the public API. Re-export at the crate root so
// `pie_schema::Frame` etc. work without an extra namespace hop.
pub use brle::Brle;
pub use capabilities::DriverCapabilities;
pub use cluster::{
    HealthStatus, LoadState, Placement, RequestId, RequestMeta, Role, WorkerId, WorkerInfo,
};
pub use edge::{GatewayFrame, SessionId, WorkerFrame};
pub use kv::{KvDtype, KvHandle, KvLayout, KvLayoutKind, KvRegion, MemoryDomain};
pub use message::{ClientMessage, ServerMessage};
// Flat-POD wire types (see `pod`): plain `#[repr(C)]`/`#[repr(u8)]` + rkyv,
// embedded by value in the rich types via `#[schema(pod)]`.
pub use pod::*;
pub use schema::*;

include!(concat!(env!("OUT_DIR"), "/schema_hash.rs"));

//! The worker's own copies of the cross-crate RPC service contracts.
//!
//! So the worker can interop with the standalone **controller** and the
//! **gateway** over the wire without depending on `pie-controller` /
//! `pie-gateway`, it carries its own copies of the `#[tarpc::service]`
//! definitions. The shared *data* lives on the floor in `pie-schema`; the
//! transport is tarpc-native (`serde_transport::{tcp,unix}`), dialed inline at
//! the call sites — no custom transport module.
//!
//! INVARIANT: these definitions must stay **wire-compatible** with the
//! controller's `ControlApi` ([`control_api`]) and the gateway's
//! `WorkerSessionApi` ([`worker_session_api`]) — same method names, argument
//! order, and (schema-owned) argument/return types. tarpc derives the wire
//! format from these signatures, so a drift here is a silent protocol break.

pub mod control_api;
pub mod worker_session_api;

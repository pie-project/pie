//! The worker's own copy of the cross-crate gateway RPC service contract.
//!
//! So the worker can interop with the standalone **gateway** over the wire
//! without depending on `pie-gateway`, it carries its own copy of the
//! `#[tarpc::service]` definition for that edge. The control-plane contract is
//! shared through `pie-control`. The shared *data* lives on the floor in
//! `pie-schema`; the transport is tarpc-native
//! (`serde_transport::{tcp,unix}`), dialed inline at the call sites — no custom
//! transport module.
//!
//! INVARIANT: the gateway definition must stay **wire-compatible** with the
//! gateway's `WorkerSessionApi` ([`worker_session_api`]) — same method names,
//! argument order, and (schema-owned) argument/return types. tarpc derives the
//! wire format from these signatures, so a drift here is a silent protocol
//! break.

pub mod worker_session_api;

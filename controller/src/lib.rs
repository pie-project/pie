//! `pie-controller` — Pie's cluster-coordination **control plane**.
//!
//! This crate decides *what runs where*: it registers workers, routes requests,
//! pairs prefill↔decode, and tracks liveness/load. It is **control plane only**
//! — it moves small coordination metadata (worker ids, roles, addresses, load)
//! and **never touches tensor bytes**. KV blocks and activations travel on the
//! data plane (`pie-transport`), which this crate has no knowledge of.
//!
//! # One trait, two deployments
//!
//! [`Controller`] is the seam the worker wires to. Two impls back it, selected
//! by the worker's launch flag:
//!
//! - **on-device** (`--single-node`): the worker constructs an
//!   [`InProcController`] in its own address space and calls it directly — no
//!   network, no serialization.
//! - **distributed** (`--role=… --controller=addr`): the worker dials a
//!   [`RemoteController`], which frames each call over a socket to a standalone
//!   controller process ([`run_as_process`], backed by the same
//!   [`InProcController`]).
//!
//! # Minimal start (YAGNI)
//!
//! Per the ratified spec the coordinator is a worker **registry + round-robin
//! `route`** with **pushed** load (`report`); `pair` is the trivial same-node
//! decision. Real PD pairing, least-loaded routing, and placement scaling are
//! deferred — the trait keeps them swappable with worker code untouched.
//! [`pairing`] holds the (kept-but-unextended) prefill↔decode A↔B machinery for
//! that future.
//!
//! # Three orthogonal axes
//!
//! The control plane keeps three concerns from leaking into one another: the
//! **role** axis (prefill/decode/encode — [`Role`]), the **topology** axis
//! (single-node vs distributed — the two [`Controller`] impls), and the
//! **backend** axis (cuda/portable/dummy), which never appears here at all.
//!
//! # Dependency direction
//!
//! Edges point **downward only**. The controller depends on `pie-schema` for the
//! shared vocabulary ([`pie_schema::cluster`]) and nothing else — never on the
//! runtime, transport, or driver crates. The control-RPC message envelope is
//! local in [`protocol`].

mod controller;
mod error;
mod health;
mod pairing;
mod protocol;
mod role;
mod rpc;
mod serve;

pub use controller::{Controller, ControllerConfig, InProcController};
pub use error::{ControllerError, Result};
pub use health::HealthChecker;
pub use pairing::{Pair, PairId, PairingTable};
pub use protocol::{ControlRequest, ControlResponse};
pub use role::RoleTable;
pub use rpc::RemoteController;
pub use serve::{ProcessConfig, run_as_process};

// Shared coordination vocabulary lives on the dependency floor (`pie-schema`).
// Re-export it so downstream code can use `pie_controller::{WorkerId, Role, …}`
// without taking its own `pie-schema` dependency.
pub use pie_schema::{
    HealthStatus, LoadState, Placement, RequestId, RequestMeta, Role, WorkerId, WorkerInfo,
};

//! Control-plane RPC message vocabulary.
//!
//! The request/response envelope workers exchange with the controller over the
//! distributed control channel — one variant per [`crate::Controller`] trait
//! method. The embedded (in-proc) deployment skips this entirely and calls the
//! trait directly; the distributed deployment frames these over a socket (see
//! [`crate::rpc`]).
//!
//! The shared vocabulary (`WorkerId`, `WorkerInfo`, `LoadState`, `RequestMeta`,
//! `Placement`) lives on the floor in [`pie_schema::cluster`]; this module only
//! adds the message framing around it.

use serde::{Deserialize, Serialize};

use pie_schema::{LoadState, Placement, RequestId, RequestMeta, WorkerId, WorkerInfo};

/// A request a worker sends to the controller. One per trait method.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlRequest {
    /// Join the cluster; the controller replies with a freshly minted
    /// [`WorkerId`]. → [`ControlResponse::Registered`].
    Register(WorkerInfo),
    /// Push current load (heartbeat + load). → [`ControlResponse::Ack`].
    Report { worker: WorkerId, load: LoadState },
    /// Ask where to place a request. → [`ControlResponse::Routed`].
    Route(RequestMeta),
    /// Ask for the prefill/decode worker pair for a request.
    /// → [`ControlResponse::Paired`].
    Pair(RequestId),
}

/// The controller's reply to a [`ControlRequest`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlResponse {
    /// `Register` succeeded; here is your id.
    Registered(WorkerId),
    /// `Report` succeeded with no payload.
    Ack,
    /// `Route` decision.
    Routed(Placement),
    /// `Pair` decision: `(prefill, decode)` workers (equal in the minimal
    /// same-node start).
    Paired(WorkerId, WorkerId),
    /// The controller rejected the request; carries the rendered error. Relays a
    /// [`crate::ControllerError`] (e.g. no eligible worker) to a remote caller.
    Error(String),
}

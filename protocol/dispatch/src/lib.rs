//! Gateway data-plane RPC contract — the `GatewayInbound` + `WorkerControl`
//! tarpc services.
//!
//! Post-inversion (Pie 0.5.0) the worker dials INTO the gateway (1:N fan-in):
//! the gateway is the listening server, the worker the client. One
//! worker-initiated connection carries BOTH traits, split at each end with
//! `spawn_twoway` (the heavy token traffic stays on the plain client→server
//! direction; latency-sensitive commands go reverse):
//!
//! - [`GatewayInbound`] — served by the GATEWAY, called by the worker. Bulk,
//!   forward: token push + register + load report + redirect.
//! - [`WorkerControl`] — served by the WORKER, called by the gateway.
//!   Latency-sensitive, varied: dispatch + cancel + set_priority + drain.
//!
//! The macros generate `GatewayInboundClient` (held by the worker) and
//! `WorkerControlClient` (held by the gateway, one per connected worker in its
//! registry). The crate also owns the shared two-way connection glue that splits
//! the one worker→gateway socket into both service channels — see [`link`]
//! ([`spawn_twoway`] + the typed [`accept_gateway_link`] /
//! [`connect_gateway_link`] constructors). It is defined ONCE here so the muxed
//! frame's wire layout and the channel ordering can't drift between the two ends.
//!
//! Deps are `pie-schema` + `tarpc` and tarpc-transitive foundations
//! (serde/futures/tokio/thiserror/tracing) only — zero internal pie edges beyond
//! `pie-schema`, so the data-type floor `pie-schema` stays tarpc-free, same
//! purity invariant as [`pie_control`](https://docs.rs/pie-control).
//!
//! NOTE — this `WorkerControl` is the gateway↔worker DATA plane (dispatch a
//! turn). It is unrelated to the worker's CONTROL-plane embed seam, which is
//! named `ControlLink` (`pie-worker`, register/heartbeat/report/watch against
//! the controller). Two different surfaces, two different names (manager M2).
//!
//! Wire data types are defined in [`pie_schema::gateway`]; `WorkerId` /
//! `WorkerStatus` are reused from [`pie_schema::control`].

use pie_schema::control::{WorkerId, WorkerStatus};
use pie_schema::gateway::{Accepted, Control, Priority, ReqId, Request, Tokens};

mod link;
pub use link::{
    ChannelOrIoError, TwoWayMessage, accept_gateway_link, connect_gateway_link, dispatch_codec,
    spawn_twoway,
};

/// Served by the GATEWAY, called by the worker (worker → gateway). The bulk,
/// forward direction: the worker announces itself, streams tokens, reports load,
/// and bounces turns it can no longer serve.
#[tarpc::service]
pub trait GatewayInbound {
    /// Announce this connection's worker identity (the controller-minted
    /// [`WorkerId`] presented verbatim) so the gateway can bind the reverse
    /// `WorkerControlClient` into its registry and join it against the routing
    /// table. The first call on a freshly dialed-in connection.
    async fn register(worker_id: WorkerId);

    /// Push one chunk of a turn's output stream. The reply [`Control`] piggybacks
    /// ordinary cancel (continue / abort) so the common cancel path needs no
    /// extra round-trip. A token stream is modeled as repeated calls (tarpc has
    /// weak native server-streaming); concurrent turns interleave on the one
    /// connection, keyed by `req_id`.
    async fn push_tokens(req_id: ReqId, chunk: Tokens) -> Control;

    /// Push coarse load for gateway-local freshness (same [`WorkerStatus`] the
    /// worker reports to the controller). Additive: the gateway's admission can
    /// gate off the controller's `RoutingTable` alone, so this is freshness, not
    /// a hard dependency.
    async fn report(worker_id: WorkerId, status: WorkerStatus);

    /// Bounce an already-accepted turn the worker can no longer serve (post-hoc
    /// final-admission reject — it filled / started draining after `dispatch`).
    /// The gateway re-routes the turn to another worker.
    async fn redirect(req_id: ReqId);
}

/// Served by the WORKER, called by the gateway (gateway → worker). The
/// latency-sensitive, varied direction over the reverse channel.
#[tarpc::service]
pub trait WorkerControl {
    /// Dispatch one turn. The worker has final admission (design §7): it answers
    /// [`Accepted::Ok`] (will stream tokens), [`Accepted::Reject`], or
    /// [`Accepted::Redirect`]. Idempotent + ack-based: a re-sent `Request` (same
    /// `req_id`) is the same turn, so the gateway re-routes on a transport/no-ack
    /// failure without duplicating work. The `Request` carries a `BlobRef`, never
    /// raw bytes.
    async fn dispatch(req: Request) -> Accepted;

    /// Immediately abort an in-flight turn over the reverse channel — for when
    /// the worker isn't currently pushing (e.g. mid-prefill) so the piggybacked
    /// [`Control::Abort`] on `push_tokens` can't reach it promptly.
    async fn cancel(req_id: ReqId);

    /// Adjust an in-flight turn's scheduling priority. (Spec-locked surface;
    /// the runtime hook is a tracked follow-on — may start best-effort/no-op.)
    async fn set_priority(req_id: ReqId, p: Priority);

    /// Begin graceful drain: stop accepting new turns, let in-flight ones finish.
    /// (Spec-locked surface; the runtime hook is a tracked follow-on — may start
    /// best-effort/no-op.)
    async fn drain();
}

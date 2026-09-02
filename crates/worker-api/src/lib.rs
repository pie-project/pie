//! Gateway<->worker data-plane RPC — the `GatewayInbound` + `WorkerControl`
//! tarpc services and the glue that splits one socket into both. The worker
//! dials into the gateway; [`spawn_twoway`] splits that one connection into
//! forward token traffic and reverse control commands.

use controller_api::WorkerStatus;
use ids::{ReqId, WorkerId};

mod data;
mod link;

pub use data::{Accepted, BlobRef, Control, Priority, Request, Tokens};
pub use link::{
    ChannelOrIoError, TwoWayMessage, accept_gateway_link, connect_gateway_link, dispatch_codec,
    spawn_twoway,
};

/// Served by the GATEWAY, called by the worker: the bulk, forward direction.
#[tarpc::service]
pub trait GatewayInbound {
    /// Announce this connection's worker identity so the gateway can bind the
    /// reverse `WorkerControlClient` into its registry. First call on a dial-in.
    async fn register(worker_id: WorkerId);

    /// Push one chunk of a turn's output stream. The reply [`Control`]
    /// piggybacks ordinary cancel, so that path needs no extra round-trip.
    async fn push_tokens(req_id: ReqId, chunk: Tokens) -> Control;

    /// Coarse load for freshness only — admission can gate off the controller's
    /// `RoutingTable` alone, so this is not a hard dependency.
    async fn report(worker_id: WorkerId, status: WorkerStatus);

    /// Bounce an already-accepted turn the worker can no longer serve.
    async fn redirect(req_id: ReqId);
}

/// Served by the WORKER, called by the gateway over the reverse channel.
#[tarpc::service]
pub trait WorkerControl {
    /// Dispatch one turn; the worker has final admission. Idempotent and
    /// ack-based, so the gateway re-routes on a no-ack failure without
    /// duplicating work.
    async fn dispatch(req: Request) -> Accepted;

    /// Immediately abort an in-flight turn, for when the worker is not pushing
    /// and the piggybacked [`Control::Abort`] cannot reach it promptly.
    async fn cancel(req_id: ReqId);

    /// Adjust an in-flight turn's scheduling priority. Spec-locked surface.
    async fn set_priority(req_id: ReqId, p: Priority);

    /// Stop accepting new turns, let in-flight ones finish. Spec-locked surface.
    async fn drain();
}

use controller_api::WorkerStatus;
use ids::{ReqId, SessionId, WorkerId};

mod data;
mod link;

pub use data::{Accepted, BlobRef, Control, Priority, Request, Tokens};
pub use link::{
    ChannelOrIoError, TwoWayMessage, accept_gateway_link, connect_gateway_link, dispatch_codec,
    spawn_twoway,
};

#[tarpc::service]
pub trait GatewayInbound {
    async fn register(worker_id: WorkerId);

    async fn push_tokens(req_id: ReqId, chunk: Tokens) -> Control;

    async fn report(worker_id: WorkerId, status: WorkerStatus);

    async fn redirect(req_id: ReqId);
}

#[tarpc::service]
pub trait WorkerControl {
    async fn dispatch(req: Request) -> Accepted;

    async fn cancel(req_id: ReqId);

    async fn set_priority(req_id: ReqId, p: Priority);

    // The session's client is gone: tear it down so its processes are
    // terminated and their KV seats returned, rather than running on
    // unread. Sent to every connected worker; one that never held the
    // session ignores it.
    async fn end_session(session: SessionId);

    async fn drain();
}

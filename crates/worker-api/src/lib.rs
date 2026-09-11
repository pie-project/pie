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

    async fn close_session(session: SessionId);

    async fn set_priority(req_id: ReqId, p: Priority);

    async fn drain();
}

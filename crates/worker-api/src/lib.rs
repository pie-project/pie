use controller_api::WorkerStatus;
use ids::{ReqId, WorkerId};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemoryReport {
    pub working_set: u64,
    pub ceiling: u64,
    pub weights: u64,
    pub scratch: u64,
    pub floor: u64,
    pub pool: u64,
    pub minimum: u64,
}

#[tarpc::service]
pub trait WorkerControl {
    async fn memory() -> Option<MemoryReport>;

    async fn dispatch(req: Request) -> Accepted;

    async fn cancel(req_id: ReqId);

    async fn set_priority(req_id: ReqId, p: Priority);

    async fn drain();
}

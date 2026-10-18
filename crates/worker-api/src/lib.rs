use controller_api::WorkerStatus;
use ids::{ReqId, SessionId, WorkerId};

mod data;
mod link;
pub mod peer;

pub use data::{Accepted, BlobRef, Control, Priority, Request, StateReport, Tokens};
pub use link::{
    ChannelOrIoError, TwoWayMessage, accept_gateway_link, connect_gateway_link, dispatch_codec,
    spawn_twoway,
};

#[tarpc::service]
pub trait GatewayInbound {
    async fn register(worker_id: WorkerId);

    async fn push_tokens(req_id: ReqId, chunk: Tokens) -> Control;

    /// For the cluster policy, ahead of the exit event.
    async fn counted(req_id: ReqId, prompt: u64, generated: u64);

    async fn report(worker_id: WorkerId, status: WorkerStatus);

    async fn redirect(req_id: ReqId);

    /// What this worker holds, since its last report.
    async fn state(worker_id: WorkerId, report: StateReport);

    /// The workers, not `worker_id`, that hold a run of `keys` from token 0:
    /// each one's bulk address and how many pages of the run, longest first.
    async fn locate(worker_id: WorkerId, keys: Vec<[u8; 32]>) -> Vec<(WorkerId, String, u32)>;
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

    /// Take the run `keys` (from token 0) from the worker whose bulk address
    /// is `from`, and keep it: the cluster policy's replication. Answers how
    /// many pages came.
    async fn replicate(keys: Vec<[u8; 32]>, from: String) -> Result<u32, String>;

    /// `changes` is a JSON object whose `null`s remove names; none closes the
    /// session.
    async fn plex_session(session: String, changes: Option<String>);
}

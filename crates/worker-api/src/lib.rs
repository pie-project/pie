use controller_api::WorkerStatus;
use ids::{ReqId, SessionId, WorkerId};

mod data;
mod link;
pub mod peer;

pub use controller_api::StateReport;
pub use data::{Accepted, BlobRef, Control, Priority, Request, Tokens};
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

    /// What a worker that links to no controller (a pool) holds, since its
    /// last report: passed on to the controller's directory. A worker with
    /// a controller reports there itself.
    async fn state(worker_id: WorkerId, report: StateReport);
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
    /// is `from`, and keep it: the cluster policy's replication. `retention`
    /// is the group's cluster retention key (0 when none names it): a pool
    /// lets lower keys go first. `disk`: keep it on this worker's persistent
    /// disk tier, from the run it holds itself. Answers how many pages came.
    async fn replicate(
        keys: Vec<[u8; 32]>,
        from: String,
        retention: i64,
        disk: bool,
    ) -> Result<u32, String>;

    /// `changes` is a JSON object whose `null`s remove names; none closes the
    /// session.
    async fn plex_session(session: String, changes: Option<String>);
}

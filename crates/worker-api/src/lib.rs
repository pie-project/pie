use controller_api::WorkerStatus;
use ids::{ReqId, SessionId, WorkerId};

mod data;
mod link;
pub mod peer;

pub use controller_api::StateReport;
pub use data::{Accepted, BlobRef, Control, Priority, Request, Tokens};
pub use link::{
    ChannelOrIoError, LINK_MAX_FRAME_BYTES, TwoWayMessage, accept_gateway_link,
    connect_gateway_link, dispatch_codec, spawn_twoway, until_link_lost,
};

#[tarpc::service]
pub trait GatewayInbound {
    async fn register(worker_id: WorkerId);

    async fn push_tokens(req_id: ReqId, chunk: Tokens) -> Control;

    async fn report(worker_id: WorkerId, status: WorkerStatus);

    async fn redirect(req_id: ReqId);

    /// The process the request launched started on this worker: its
    /// execution was admitted.
    async fn started(req_id: ReqId);

    /// What a worker that links to no controller (a pool) holds, since its
    /// last report: passed on to the controller's directory. A worker with
    /// a controller reports there itself.
    async fn state(worker_id: WorkerId, report: StateReport);
}

#[tarpc::service]
pub trait WorkerControl {
    async fn dispatch(req: Request) -> Accepted;

    async fn cancel(req_id: ReqId);

    /// The workers the request's process may prefill on, best first, from
    /// now on (empty: the worker's own configuration decides).
    async fn set_executors(req_id: ReqId, executors: Vec<WorkerId>);

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

    /// The cluster's share grants as of `grants.version`, whole; an older
    /// version than the worker holds is ignored.
    async fn share_groups(grants: controller_api::ShareGrants);
}

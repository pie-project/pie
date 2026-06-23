//! The control-plane RPC contract — the worker's copy of the controller's
//! `ControlApi`.
//!
//! `#[tarpc::service]` generates [`ControlApiClient`], which the worker uses to
//! dial a standalone controller process in distributed mode (`--role/--
//! controller`). Only the client half is used here; the server half is
//! implemented by the controller process.
//!
//! WIRE CONTRACT: must match `pie_controller`'s `ControlApi` exactly. The
//! argument/return types are the shared floor vocabulary in `pie-schema`.

use pie_schema::{LoadState, Placement, RequestId, RequestMeta, WorkerId, WorkerInfo};

/// Coordination errors travel as a rendered string (the controller's error type
/// isn't `Serialize`).
pub type RpcResult<T> = Result<T, String>;

#[tarpc::service]
pub trait ControlApi {
    /// Register a worker on boot; the controller mints and returns its
    /// [`WorkerId`].
    async fn register(info: WorkerInfo) -> RpcResult<WorkerId>;

    /// Push current load (doubles as the liveness heartbeat).
    async fn report(worker: WorkerId, load: LoadState) -> RpcResult<()>;

    /// Decide which worker should serve a request.
    async fn route(meta: RequestMeta) -> RpcResult<Placement>;

    /// Decide the `(prefill, decode)` worker pair for a request.
    async fn pair(req: RequestId) -> RpcResult<(WorkerId, WorkerId)>;

    /// Resolve a [`WorkerId`] to its registered [`WorkerInfo`].
    async fn resolve(worker: WorkerId) -> RpcResult<WorkerInfo>;
}

//! The tarpc control-plane service definition — the unified RPC surface.
//!
//! `#[tarpc::service]` generates the [`ControlApiClient`] (dialed by workers and
//! the gateway) and the server-side `ControlApi` trait (implemented by
//! [`crate::serve`] over an [`crate::InProcController`]). One variant per
//! [`crate::Controller`] trait method.
//!
//! Coordination errors are carried as `Result<T, String>`: [`ControllerError`]
//! isn't `Serialize`, so the server renders it to a string and the client maps
//! it back to [`ControllerError::Remote`]. The shared vocabulary (`WorkerId`,
//! `WorkerInfo`, `LoadState`, `RequestMeta`, `Placement`) lives on the floor in
//! [`pie_schema::cluster`]; this module only declares the RPC envelope.
//!
//! [`ControllerError`]: crate::ControllerError
//! [`ControllerError::Remote`]: crate::ControllerError::Remote

use pie_schema::{LoadState, Placement, RequestId, RequestMeta, WorkerId, WorkerInfo};

/// Result alias for service methods — coordination errors travel as a rendered
/// string (see module docs).
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

    /// Resolve a [`WorkerId`] to its registered [`WorkerInfo`] (control address
    /// + role) so a router can dial the placed worker.
    async fn resolve(worker: WorkerId) -> RpcResult<WorkerInfo>;
}

//! The edge-session RPC contract — the worker's copy of the gateway's
//! `WorkerSessionApi`.
//!
//! `#[tarpc::service]` generates the server trait the worker implements (so a
//! gateway can proxy client sessions to it in distributed mode) plus an unused
//! client. The frame/session types are the shared floor vocabulary in
//! `pie-schema`.
//!
//! WIRE CONTRACT: must match `pie_gateway::edge_rpc::WorkerSessionApi` exactly.

use pie_schema::{GatewayFrame, SessionId, WorkerFrame};

/// RPC result type: worker-side runtime errors are returned as rendered strings.
pub type RpcResult<T> = Result<T, String>;

#[tarpc::service]
pub trait WorkerSessionApi {
    /// Open a new logical session on the worker and return its handle.
    async fn open() -> RpcResult<SessionId>;

    /// Send one client-originated message into a worker session.
    async fn send(session: SessionId, frame: GatewayFrame) -> RpcResult<()>;

    /// Receive zero or more server-originated messages from a worker session.
    async fn recv(session: SessionId, max_wait_ms: u64) -> RpcResult<Vec<WorkerFrame>>;

    /// Close a logical session and release any worker-side resources.
    async fn close(session: SessionId) -> RpcResult<()>;
}

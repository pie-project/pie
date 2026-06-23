//! Bidirectional gateway<->worker RPC contract.
//!
//! The gateway terminates the client WebSocket and speaks this service to a
//! chosen worker. The contract models a duplex session channel:
//!
//! - gateway -> worker: [`send`] pushes one client message at a time
//! - worker -> gateway: [`recv`] long-polls server messages (batched)
//!
//! This is the edge plane's own RPC surface; it lives here (not on the
//! `pie-schema` floor) so the schema crate stays a pure data vocabulary with
//! no tarpc machinery. The worker implements the server side by depending on
//! this crate; the gateway dials it via [`WorkerSessionApiClient`].
//!
//! [`send`]: WorkerSessionApi::send
//! [`recv`]: WorkerSessionApi::recv

pub use pie_schema::{GatewayFrame, SessionId, WorkerFrame};

/// RPC result type: worker-side runtime errors are returned as rendered strings.
pub type RpcResult<T> = Result<T, String>;

#[tarpc::service]
pub trait WorkerSessionApi {
    /// Open a new logical session on the worker and return its handle.
    async fn open() -> RpcResult<SessionId>;

    /// Send one client-originated message into a worker session.
    async fn send(session: SessionId, frame: GatewayFrame) -> RpcResult<()>;

    /// Receive zero or more server-originated messages from a worker session.
    ///
    /// `max_wait_ms` is a long-poll upper bound: the worker can return early as
    /// soon as any message is available, or an empty vector on timeout.
    async fn recv(session: SessionId, max_wait_ms: u64) -> RpcResult<Vec<WorkerFrame>>;

    /// Close a logical session and release any worker-side resources.
    async fn close(session: SessionId) -> RpcResult<()>;
}

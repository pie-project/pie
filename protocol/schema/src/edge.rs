//! Shared edge-session wire structs (gateway<->worker, and worker<->local
//! client). Pure data — the `#[tarpc::service]` traits that carry these live in
//! the components that own each RPC surface (gateway, worker), so the floor
//! stays free of RPC machinery.

use crate::message::{ClientMessage, ServerMessage};
use serde::{Deserialize, Serialize};

/// Opaque edge-session identifier minted by the worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub u64);

/// One message flowing from the client-facing edge (gateway, or the worker's own
/// local client server) into a worker session.
#[derive(Debug, Serialize, Deserialize)]
pub struct GatewayFrame {
    pub message: ClientMessage,
}

/// One message flowing from a worker session back out to the edge.
#[derive(Debug, Serialize, Deserialize)]
pub struct WorkerFrame {
    pub message: ServerMessage,
}

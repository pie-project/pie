//! Edge-session wire frames (gateway↔worker, and worker↔local client). Pure
//! data — the `#[tarpc::service]` traits that carry these live in the
//! components that own each RPC surface (gateway, worker), so this crate stays
//! free of RPC machinery.
//!
//! `SessionId` moved to `pie-ids`; these frames carry only the message payload,
//! so this module references no id atom.

use crate::message::{ClientMessage, ServerMessage};
use serde::{Deserialize, Serialize};

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

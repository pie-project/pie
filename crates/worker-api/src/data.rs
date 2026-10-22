use serde::{Deserialize, Serialize};

use client_api::{ClientMessage, ServerMessage};
use ids::{ReqId, SessionId, TenantId, WorkerId};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlobRef {
    pub hash: String,
    pub size: u64,
    pub kind: String,
    pub origin: String,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize,
)]
pub enum Priority {
    Low,
    #[default]
    Normal,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    pub req_id: ReqId,
    pub session: SessionId,
    pub tenant: TenantId,
    pub priority: Priority,
    pub blobs: Vec<BlobRef>,
    pub message: ClientMessage,
    #[serde(default)]
    pub cluster_props: client_api::Props,
    /// Workers the cluster policy names to run this process's passes, best
    /// first (`placement.executors`); empty leaves them to the worker plane.
    #[serde(default)]
    pub executors: Vec<WorkerId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Accepted {
    Ok { worker: WorkerId },
    Reject,
    Redirect { worker: WorkerId },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Tokens {
    Chunk(ServerMessage),
    Eos,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Control {
    Continue,
    Abort,
}

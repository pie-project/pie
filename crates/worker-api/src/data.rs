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

/// What a worker holds, as it tells its gateway's directory: the sealed
/// pages it gained and lost by key since its last report (all of them when
/// `full`), the runs it keeps whole from token 0 (gained in full, lost by
/// their last key), and where it serves its pages' bytes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StateReport {
    pub peer: Option<String>,
    pub added: Vec<[u8; 32]>,
    pub removed: Vec<[u8; 32]>,
    pub full: bool,
    pub runs_added: Vec<Vec<[u8; 32]>>,
    pub runs_removed: Vec<[u8; 32]>,
    /// A pool: it loads no model and keeps what it is sent.
    pub pool: bool,
    /// What the worker policy published for the cluster plane, as
    /// `(subject, name, JSON value)`; all of it, whenever any changes.
    pub published: Option<Vec<(String, String, String)>>,
}

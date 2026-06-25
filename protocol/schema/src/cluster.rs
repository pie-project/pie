//! Cluster-coordination vocabulary — NOT a rkyv wire type.
//!
//! The control plane (`pie-controller`) coordinates which worker serves which
//! inference stage, routes requests, and tracks liveness/load. These are the
//! shared vocabulary for that coordination. Worker, runtime, and the data-plane
//! transport read them too (a worker must know its id + assigned role; transport
//! addresses peers by [`WorkerId`]), so they live here on the dependency floor.
//!
//! Like `DriverCapabilities`, this is deliberately NOT under `#[schema]`:
//! coordination metadata travels as plain serde over a control channel, never on
//! the rkyv tensor ring, so it is not part of `SCHEMA_HASH`. The control plane is
//! cross-node Rust↔Rust and low-rate, so serde is sufficient (no zero-copy /
//! C-ABI need); keeping it off the wire hash also decouples control-plane
//! evolution from the driver handshake.

use serde::{Deserialize, Serialize};

/// Opaque cluster-unique worker handle, minted by the controller at
/// registration. The data-plane transport addresses peers by this id too.
///
/// Newtype rather than a bare `u64` so a worker id can never be confused with a
/// request id or any other counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WorkerId(pub u64);

impl std::fmt::Display for WorkerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "worker#{}", self.0)
    }
}

/// The role taxonomy — what stage of inference a worker serves.
///
/// One of three orthogonal coordination axes, independent of the *backend* axis
/// (cuda / portable / dummy) and the *topology* axis (on-device vs distributed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Role {
    /// Consumes prompt tokens and produces the initial KV state.
    Prefill,
    /// Consumes KV state and produces output tokens step by step.
    Decode,
    /// Encodes non-text modalities (image / audio) into embeddings.
    Encode,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Role::Prefill => "prefill",
            Role::Decode => "decode",
            Role::Encode => "encode",
        })
    }
}

/// What a worker tells the controller about itself when it joins the cluster.
///
/// Static identity/capability declared once at registration. Dynamic state
/// (live load, KV headroom) is pushed separately as [`LoadState`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// Where peers reach this worker's control endpoint (e.g. `"10.0.0.4:7000"`).
    pub control_addr: String,
    /// Role the worker is requesting, if it already knows. `None` means "assign
    /// me one".
    pub preferred_role: Option<Role>,
}

/// Liveness verdict for a single worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Heartbeats are arriving on time.
    Healthy,
    /// Heartbeats are late but the worker has not yet timed out.
    Degraded,
    /// No heartbeat within the timeout window.
    Unreachable,
}

/// Dynamic load a worker **pushes** to the controller (the controller never
/// polls). Soft state — the worker is the source of truth; the controller keeps
/// only this reconstructable summary for routing decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadState {
    /// In-flight requests on this worker.
    pub active_requests: u32,
    /// Free KV pages — headroom for admitting new work.
    pub kv_pages_free: u32,
}

/// Opaque handle to an inference request, used as routing/pairing input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RequestId(pub u64);

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "request#{}", self.0)
    }
}

/// The metadata the controller routes on. Deliberately small — the controller
/// never sees request bodies or tokens, only the shape it needs to place work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestMeta {
    /// Identity of the request being placed.
    pub id: RequestId,
    /// Prompt length in tokens (a size hint for admission, not the tokens).
    pub prompt_tokens: u32,
}

/// The controller's routing decision: which worker should serve a request.
///
/// Minimal-start models a monolithic worker (prefill+decode co-located). A
/// future PD-split placement would name separate prefill/decode workers; the
/// `Controller::pair` seam already exposes that shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Placement {
    /// The worker selected to serve the request.
    pub worker: WorkerId,
}

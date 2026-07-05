//! pie-ids — the deduplicated identity atoms shared across the pie interface
//! crates.
//!
//! This crate is the dependency leaf of the `interface/` tree: it imports
//! nothing internal, so the controller-, worker-, and client-facing interface
//! crates can all share one canonical set of id newtypes instead of each
//! minting its own. These are cross-node control/edge vocabulary (plain serde,
//! never `#[repr(C)]`/rkyv); the driver ABI keeps its own in-node C ids and does
//! NOT depend on this crate.

use serde::{Deserialize, Serialize};

/// Controller-minted, cluster-unique worker handle. Newtype so it can never be
/// confused with a gateway id or any other counter. The data-plane transport
/// addresses peers by this id too.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WorkerId(pub u64);

impl std::fmt::Display for WorkerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "worker#{}", self.0)
    }
}

/// Controller-minted, cluster-unique gateway handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GatewayId(pub u64);

impl std::fmt::Display for GatewayId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "gateway#{}", self.0)
    }
}

/// Either kind of cluster member, carried by the unified `heartbeat` call so the
/// controller can route liveness to the right registry. Workers and gateways are
/// minted into separate id spaces, so a single flat counter can't identify a
/// node — hence an enum, not a bare newtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeId {
    Worker(WorkerId),
    Gateway(GatewayId),
}

impl From<WorkerId> for NodeId {
    fn from(id: WorkerId) -> Self {
        NodeId::Worker(id)
    }
}

impl From<GatewayId> for NodeId {
    fn from(id: GatewayId) -> Self {
        NodeId::Gateway(id)
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeId::Worker(id) => write!(f, "{id}"),
            NodeId::Gateway(id) => write!(f, "{id}"),
        }
    }
}

/// Logical session id, gateway-minted and stable across the turns of one session
/// (one-shot = a 1-turn session; WS = many). Distinct from [`ReqId`]:
/// `SessionId` spans turns, `ReqId` is one turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SessionId(pub u64);

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sess#{}", self.0)
    }
}

/// Per-turn id, gateway-minted at dispatch. Keys a single in-flight turn across
/// `dispatch` / `push_tokens` / `cancel` / `set_priority` / `redirect`. A
/// multi-turn WS session produces one fresh `ReqId` per user prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ReqId(pub u64);

impl std::fmt::Display for ReqId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "req#{}", self.0)
    }
}

/// Opaque handle to an inference request, used as routing/pairing input by the
/// controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RequestId(pub u64);

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "request#{}", self.0)
    }
}

/// The edge-supplied principal a turn is attributed to (tenant / user id),
/// extracted by the gateway's light identity gate from the trusted edge header.
/// Used for routing, quota, and isolation — NOT authentication. An opaque string
/// so the gateway does not pin a tenant scheme.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TenantId(pub String);

impl std::fmt::Display for TenantId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

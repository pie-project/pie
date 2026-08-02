//! Error type for the transport data plane.

use thiserror::Error;

/// Result alias for transport operations.
pub type Result<T> = std::result::Result<T, TransportError>;

/// Failures surfaced by the worker↔worker data plane.
#[derive(Debug, Error)]
pub enum TransportError {
    /// The requested path does not participate / is not built. metal and vulkan
    /// are single-node — their driver shim never exports a handle, so they never
    /// reach an engine. Also raised when a transfer is routed to the `nixl`
    /// engine in a build compiled without `feature = "nixl"`. Callers treat this
    /// as "this path simply isn't on the fabric", not a bug.
    #[error("transport path not available: {0}")]
    Unsupported(&'static str),

    /// An engine could not register a driver-exported handle (out of resources,
    /// unpinnable memory, a region that isn't actually device-resident, ...).
    #[error("failed to register KV handle: {0}")]
    Registration(String),

    /// A transfer (or its completion) failed at the engine level.
    #[error("KV transfer failed: {0}")]
    Transfer(String),

    /// The source and destination KV layouts disagree, so page offsets would not
    /// line up. Pairing must reject mismatched workers before the data plane ever
    /// runs; this guards the invariant at transfer time.
    #[error("source/destination KV layout mismatch")]
    LayoutMismatch,

    /// A page index addressed memory outside a handle's registered region.
    #[error("page {page} is out of bounds for the KV region")]
    PageOutOfBounds { page: u32 },

    /// `send`/`recv` named a worker the engine has no registered handle for. For
    /// the local engine this means the peer is not co-located (not registered on
    /// this node).
    #[error("no registered KV handle for worker {worker}")]
    UnknownPeer { worker: u64 },

    /// `poll` was given a transfer id the registry never issued.
    #[error("unknown transfer id {id}")]
    UnknownTransfer { id: u64 },
}

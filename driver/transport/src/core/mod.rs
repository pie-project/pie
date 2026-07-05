//! Backend-agnostic transport core.
//!
//! Defines the uniform interface every engine implements and the
//! register → transfer → complete lifecycle vocabulary. Engines (`local`,
//! `nixl`) plug in behind the [`Engine`] trait; the [`crate::registry`] binds a
//! driver-exported handle to one and dispatches.
//!
//! The KV handle the data plane consumes lives on the schema floor
//! ([`pie_driver_abi::KvHandle`]) — transport never owns or interprets the
//! bytes, it only moves pages between workers.

use crate::error::Result;
use pie_driver_abi::KvHandle;

/// Worker identity on the data plane — re-exported from the interface leaf.
///
/// The controller assigns ids and decides pairings; transport only names a
/// transfer's `src`/`dst` and reports completion. Single-source vocab: this is
/// `pie_ids::WorkerId`, shared with the controller and worker.
pub use pie_ids::WorkerId;

/// The KV pages to move within a handle, at whole-page granularity. Page
/// indices address the paged KV arena described by the handle's layout.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PageSet {
    /// Page indices to move.
    pub pages: Vec<u32>,
}

impl PageSet {
    /// A page set from a list of indices.
    pub fn new(pages: Vec<u32>) -> Self {
        Self { pages }
    }

    /// Number of pages.
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }
}

/// Opaque token for an in-flight transfer. Poll it via the registry/engine for
/// completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransferId(pub u64);

/// Async completion state of a transfer — the signal the runtime awaits.
///
/// Transport only *reports* state; *when* to await is the scheduler's call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Completion {
    /// The transfer has not finished yet.
    Pending,
    /// The transfer completed successfully.
    Done,
    /// The transfer failed; the string is a human-readable reason.
    Failed(String),
}

/// Which engine backs a registered handle or transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineKind {
    /// Same-node device-to-device copy (co-located prefill+decode).
    Local,
    /// Cross-node RDMA/NIXL path (deferred behind `feature = "nixl"`).
    Nixl,
}

/// A driver-exported handle bound to an engine — the output of
/// [`Engine::register`]. Carries the engine tag and owning worker so the
/// registry can route subsequent `send`/`recv`/`poll` calls.
#[derive(Debug, Clone)]
pub struct RegisteredHandle {
    pub(crate) engine: EngineKind,
    pub(crate) owner: WorkerId,
    pub(crate) handle: KvHandle,
}

impl RegisteredHandle {
    /// The engine this handle is bound to.
    pub fn engine(&self) -> EngineKind {
        self.engine
    }

    /// The worker that owns the underlying KV cache.
    pub fn owner(&self) -> WorkerId {
        self.owner
    }

    /// The underlying driver-exported handle.
    pub fn handle(&self) -> &KvHandle {
        &self.handle
    }
}

/// A paired peer's connection info, handed over by the controller's pairing
/// decision. Transport *executes* against this — it never computes it.
///
/// Bundles everything a cross-node engine needs to target the peer: the peer's
/// id, its exported [`KvHandle`] (where its pages physically live, for building
/// the remote descriptor list), and the opaque connect-metadata creds (for
/// NIXL, the peer agent's `get_local_md` blob). serde because it crosses the
/// control channel as pairing metadata.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PeerConn {
    /// The peer worker.
    pub worker: WorkerId,
    /// The peer's exported KV handle (remote region addresses + layout).
    pub handle: KvHandle,
    /// Opaque connect-metadata creds (mechanism-specific; empty for `local`).
    pub metadata: Vec<u8>,
}

/// The uniform, backend-agnostic data-plane interface.
///
/// `local` and `nixl` implement it; the registry dispatches to the right one.
/// Object-safe so the registry can hold engines behind a trait object.
///
/// Lifecycle: [`register`](Engine::register) →
/// [`send`](Engine::send)/[`recv`](Engine::recv) → [`poll`](Engine::poll).
/// Transfers are async — the start calls return a [`TransferId`] and completion
/// is observed via `poll`.
pub trait Engine {
    /// The engine kind this implementation provides.
    fn kind(&self) -> EngineKind;

    /// Register a driver-exported handle owned by `owner` so this engine can
    /// move its pages.
    fn register(&self, owner: WorkerId, handle: KvHandle) -> Result<RegisteredHandle>;

    /// Start sending `pages` of `handle` to worker `dst`. Async — returns a
    /// token to poll.
    fn send(&self, handle: &RegisteredHandle, pages: &PageSet, dst: WorkerId)
    -> Result<TransferId>;

    /// Start receiving `pages` into the local `slot` from worker `src`. Async —
    /// returns a token to poll.
    fn recv(&self, slot: &RegisteredHandle, pages: &PageSet, src: WorkerId) -> Result<TransferId>;

    /// Poll an in-flight transfer's completion.
    fn poll(&self, id: TransferId) -> Result<Completion>;

    /// Register a remote peer's connection info — the [`PeerConn`] the
    /// controller's pairing handoff carries (peer id, its exported handle, and
    /// for NIXL the peer agent's `get_local_md` blob). Subsequent
    /// [`send`](Engine::send)/[`recv`](Engine::recv) to that peer target it.
    ///
    /// The local engine has no remote peers (co-located handles are known via
    /// [`register`](Engine::register)), so its implementation is a no-op.
    fn connect(&self, peer: &PeerConn) -> Result<()>;

    /// This engine's own connect metadata, to advertise to peers via the
    /// controller. Empty for the local engine; for NIXL, the agent's local
    /// metadata blob (valid once memory is registered).
    fn local_metadata(&self) -> Result<Vec<u8>>;
}

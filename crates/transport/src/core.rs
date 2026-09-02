//! Backend-agnostic transport core: register → transfer → complete lifecycle.

use crate::error::Result;
use engine::KvHandle;

/// Worker identity on the data plane (alias of `ids::WorkerId`).
pub use ids::WorkerId;

/// KV pages to move within a handle, at whole-page granularity.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PageSet {
    pub pages: Vec<u32>,
}

impl PageSet {
    pub fn new(pages: Vec<u32>) -> Self {
        Self { pages }
    }

    pub fn len(&self) -> usize {
        self.pages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }
}

/// Opaque token for an in-flight transfer; poll for completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransferId(pub u64);

/// Completion state of a transfer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Completion {
    Pending,
    Done,
    /// Failed; string is a human-readable reason.
    Failed(String),
}

/// Which backend backs a registered handle or transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Same-node device-to-device copy.
    Local,
    /// Cross-node RDMA/NIXL path (behind `feature = "nixl"`).
    Nixl,
}

/// Handle bound to a transfer backend; carries backend tag and owner for routing.
#[derive(Debug, Clone)]
pub struct RegisteredHandle {
    pub(crate) backend: BackendKind,
    pub(crate) owner: WorkerId,
    pub(crate) handle: KvHandle,
}

impl RegisteredHandle {
    pub fn backend(&self) -> BackendKind {
        self.backend
    }

    pub fn owner(&self) -> WorkerId {
        self.owner
    }

    pub fn handle(&self) -> &KvHandle {
        &self.handle
    }
}

/// Peer connection info from the controller's pairing handoff. Crosses the
/// control channel as pairing metadata (serde).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PeerConn {
    pub worker: WorkerId,
    /// Peer's exported KV handle (remote region addresses + layout).
    pub handle: KvHandle,
    /// Connect-metadata creds; mechanism-specific, empty for `local`.
    pub metadata: Vec<u8>,
}

/// Backend-agnostic data-plane interface; `local` and `nixl` implement it.
///
/// Lifecycle: register → send/recv → poll. Transfers are async — the start
/// calls return a [`TransferId`] and completion is observed via `poll`.
pub trait Backend {
    fn kind(&self) -> BackendKind;

    /// Register an engine-exported handle owned by `owner`.
    fn register(&self, owner: WorkerId, handle: KvHandle) -> Result<RegisteredHandle>;

    /// Start sending `pages` of `handle` to `dst`; async, returns a token to poll.
    fn send_mapped(
        &self,
        handle: &RegisteredHandle,
        src_pages: &PageSet,
        dst_pages: &PageSet,
        dst: WorkerId,
    ) -> Result<TransferId>;

    fn send(
        &self,
        handle: &RegisteredHandle,
        pages: &PageSet,
        dst: WorkerId,
    ) -> Result<TransferId> {
        self.send_mapped(handle, pages, pages, dst)
    }

    /// Start receiving `pages` into the local `slot` from `src`; async, returns
    /// a token to poll.
    fn recv_mapped(
        &self,
        slot: &RegisteredHandle,
        dst_pages: &PageSet,
        src_pages: &PageSet,
        src: WorkerId,
    ) -> Result<TransferId>;

    fn recv(&self, slot: &RegisteredHandle, pages: &PageSet, src: WorkerId) -> Result<TransferId> {
        self.recv_mapped(slot, pages, pages, src)
    }

    fn poll(&self, id: TransferId) -> Result<Completion>;

    /// Register a remote peer's connection info. No-op for the local backend.
    fn connect(&self, peer: &PeerConn) -> Result<()>;

    /// This backend's connect metadata to advertise via the controller; empty
    /// for the local backend.
    fn local_metadata(&self) -> Result<Vec<u8>>;
}

use crate::error::Result;
use engine::KvHandle;

pub use ids::WorkerId;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransferId(pub u64);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Completion {
    Pending,
    Done,
    Failed(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Nixl,
}

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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PeerConn {
    pub worker: WorkerId,
    pub handle: KvHandle,
    pub metadata: Vec<u8>,
}

pub trait Backend {
    fn kind(&self) -> BackendKind;

    /// Registers `handle`'s regions under `owner`; transfers then read and
    /// write them at their `base`/`len`, peers' included, and nothing
    /// deregisters them.
    ///
    /// # Safety
    ///
    /// Every region of `handle` must stay allocated for as long as this
    /// backend lives, and must not be accessed in ways that race a transfer
    /// into or out of it (a peer's read or write included) while one may be
    /// in flight.
    #[allow(unsafe_code)]
    unsafe fn register(&self, owner: WorkerId, handle: KvHandle) -> Result<RegisteredHandle>;

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

    fn connect(&self, peer: &PeerConn) -> Result<()>;

    fn local_metadata(&self) -> Result<Vec<u8>>;
}

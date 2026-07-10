//! Local engine — same-node device-to-device KV copy.
//!
//! The minimal baseline for co-located prefill+decode: a prefill worker's KV
//! pages are copied directly into a co-located decode worker's cache, with zero
//! network and zero serialization. The actual D2D memcpy is issued through the
//! [`D2dCopier`] seam — a real implementation wraps `cudaMemcpyDeviceToDevice` /
//! `hipMemcpy` (issued by the driver/runtime that owns the buffers), kept behind
//! a trait so the engine compiles and is testable without a device.
//!
//! Push semantics: [`send`](LocalEngine) drives the copy from the sender's
//! handle into the co-located destination's region; [`recv`](LocalEngine)
//! acknowledges (the bytes are placed by the paired send). This matches a
//! same-node DMA where one side issues the transfer.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::{
    Completion, Engine, EngineKind, PageSet, RegisteredHandle, TransferId, WorkerId,
};
use crate::error::{Result, TransportError};
use pie_driver_abi::KvHandle;

/// The device-copy primitive the local engine drives. A real implementation
/// issues a same-node device-to-device copy; tests use a recording fake.
pub trait D2dCopier: Send + Sync {
    /// Copy `len` bytes from `src_addr` to `dst_addr` within the same node.
    fn copy(&self, src_addr: u64, dst_addr: u64, len: u64) -> Result<()>;
}

/// Same-node D2D KV-copy engine.
pub struct LocalEngine {
    copier: Box<dyn D2dCopier>,
    /// Co-located handles by owning worker id, so a `send`/`recv` can resolve
    /// the peer's region on this node.
    peers: Mutex<HashMap<u64, KvHandle>>,
    /// Completion state per issued transfer.
    transfers: Mutex<HashMap<u64, Completion>>,
    next_id: AtomicU64,
}

impl LocalEngine {
    /// Build a local engine over the given device-copy primitive.
    pub fn new(copier: Box<dyn D2dCopier>) -> Self {
        Self {
            copier,
            peers: Mutex::new(HashMap::new()),
            transfers: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(0),
        }
    }

    fn fresh_id(&self) -> TransferId {
        TransferId(self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    fn record(&self, status: Completion) -> TransferId {
        let id = self.fresh_id();
        self.transfers.lock().unwrap().insert(id.0, status);
        id
    }

    /// Copy `pages` from `src` into `dst` at matching page offsets. Both handles
    /// are co-located (same node); the copy is a whole-page D2D move.
    ///
    /// INVARIANT: this addresses only `regions.first()` on each side — it assumes
    /// a single contiguous KV region per handle (the current producer shape; see
    /// the driver export shim). A multi-region handle (e.g. per-layer or sharded
    /// arenas) is out of current scope: only its first region would be moved.
    /// Generalizing to multi-region addressing is deferred with the RDMA path.
    fn copy_pages(&self, src: &KvHandle, dst: &KvHandle, pages: &PageSet) -> Result<()> {
        if !src.layout.compatible_with(&dst.layout) {
            return Err(TransportError::LayoutMismatch);
        }
        let page_bytes = src.page_bytes();
        let src_region = src.regions.first().ok_or(TransportError::Unsupported(
            "source handle has no KV region",
        ))?;
        let dst_region = dst.regions.first().ok_or(TransportError::Unsupported(
            "destination handle has no KV region",
        ))?;

        for &page in &pages.pages {
            let offset = page as u64 * page_bytes;
            if offset + page_bytes > src_region.len || offset + page_bytes > dst_region.len {
                return Err(TransportError::PageOutOfBounds { page });
            }
            self.copier.copy(
                src_region.base + offset,
                dst_region.base + offset,
                page_bytes,
            )?;
        }
        Ok(())
    }
}

impl Engine for LocalEngine {
    fn kind(&self) -> EngineKind {
        EngineKind::Local
    }

    fn register(&self, owner: WorkerId, handle: KvHandle) -> Result<RegisteredHandle> {
        self.peers.lock().unwrap().insert(owner.0, handle.clone());
        Ok(RegisteredHandle {
            engine: EngineKind::Local,
            owner,
            handle,
        })
    }

    fn send(
        &self,
        handle: &RegisteredHandle,
        pages: &PageSet,
        dst: WorkerId,
    ) -> Result<TransferId> {
        let dst_handle = {
            let peers = self.peers.lock().unwrap();
            peers
                .get(&dst.0)
                .cloned()
                .ok_or(TransportError::UnknownPeer { worker: dst.0 })?
        };
        self.copy_pages(&handle.handle, &dst_handle, pages)?;
        Ok(self.record(Completion::Done))
    }

    /// Receive `pages` into the local `slot` from co-located worker `src`.
    ///
    /// INVARIANT — no-op-ack semantics: in the same-node local path the bytes are
    /// moved by the *paired* [`send`](LocalEngine::send) (push: the sender drives
    /// the D2D copy into the receiver's region). `recv` therefore intentionally
    /// **ignores `slot` and `pages`** and only (a) validates that `src` is a
    /// registered co-located peer and (b) acknowledges with `Completion::Done`.
    /// It performs no copy. The arguments are kept for interface uniformity with
    /// a future cross-node (`nixl`) engine, where `recv` posts the actual pull.
    fn recv(&self, slot: &RegisteredHandle, pages: &PageSet, src: WorkerId) -> Result<TransferId> {
        if !self.peers.lock().unwrap().contains_key(&src.0) {
            return Err(TransportError::UnknownPeer { worker: src.0 });
        }
        let _ = (slot, pages);
        Ok(self.record(Completion::Done))
    }

    fn poll(&self, id: TransferId) -> Result<Completion> {
        self.transfers
            .lock()
            .unwrap()
            .get(&id.0)
            .cloned()
            .ok_or(TransportError::UnknownTransfer { id: id.0 })
    }

    /// No-op: the local engine has no remote peers — co-located handles are
    /// known via [`register`](LocalEngine::register), not connect metadata.
    fn connect(&self, _peer: &crate::core::PeerConn) -> Result<()> {
        Ok(())
    }

    /// Empty: same-node D2D needs no advertised connect metadata.
    fn local_metadata(&self) -> Result<Vec<u8>> {
        Ok(Vec::new())
    }
}

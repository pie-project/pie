//! Backend registry — binds an engine-exported handle to a transfer backend and dispatches
//! the data-plane lifecycle to it.
//!
//! This is the single entry point the runtime drives. It receives the
//! controller's pairing decision ("send A's pages to B") already made and only
//! *executes* it — no routing or scheduling lives here.
//!
//! The caller picks the backend for a handle at [`register`](Registry::register)
//! time (informed by the pairing — co-located → `local`, cross-node → `nixl`).
//! The registry mints a globally-unique [`TransferId`] per transfer and routes
//! `poll` back to the issuing backend, so ids never collide across backends.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::backends::local::{D2dCopier, LocalBackend};
use crate::core::{
    Backend, BackendKind, Completion, PageSet, RegisteredHandle, TransferId, WorkerId,
};
use crate::error::{Result, TransportError};
use engine::KvHandle;

/// Where an outward [`TransferId`] was issued: which backend, and that backend's
/// own (per-backend) transfer id.
#[derive(Clone, Copy)]
struct Route {
    kind: BackendKind,
    inner: TransferId,
}

/// Binds engine-exported handles to transfer backends and dispatches transfers.
pub struct Registry {
    local: LocalBackend,
    #[cfg(feature = "nixl")]
    nixl: Option<crate::backends::nixl::NixlBackend>,
    /// Outward transfer id → the backend + inner id that issued it. The registry
    /// owns id assignment so per-backend counters can't collide.
    routes: Mutex<HashMap<u64, Route>>,
    next_id: AtomicU64,
}

impl Registry {
    /// Build a registry with only the local backend — the minimal start.
    pub fn local_only(copier: Box<dyn D2dCopier>) -> Self {
        Self {
            local: LocalBackend::new(copier),
            #[cfg(feature = "nixl")]
            nixl: None,
            routes: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(0),
        }
    }

    /// Build a registry with both the local backend and a cross-node NIXL backend.
    #[cfg(feature = "nixl")]
    pub fn with_nixl(copier: Box<dyn D2dCopier>, nixl: crate::backends::nixl::NixlBackend) -> Self {
        Self {
            local: LocalBackend::new(copier),
            nixl: Some(nixl),
            routes: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(0),
        }
    }

    fn backend(&self, kind: BackendKind) -> Result<&dyn Backend> {
        match kind {
            BackendKind::Local => Ok(&self.local),
            BackendKind::Nixl => {
                #[cfg(feature = "nixl")]
                {
                    self.nixl
                        .as_ref()
                        .map(|n| n as &dyn Backend)
                        .ok_or(TransportError::Unsupported("nixl backend not enabled"))
                }
                #[cfg(not(feature = "nixl"))]
                {
                    Err(TransportError::Unsupported(
                        "nixl backend not built (enable feature \"nixl\")",
                    ))
                }
            }
        }
    }

    /// Mint a globally-unique outward id for a backend's inner transfer id, so
    /// per-backend counters can never collide.
    fn route(&self, kind: BackendKind, inner: TransferId) -> TransferId {
        let out = TransferId(self.next_id.fetch_add(1, Ordering::Relaxed));
        self.routes
            .lock()
            .unwrap()
            .insert(out.0, Route { kind, inner });
        out
    }

    /// Register an engine-exported handle owned by `owner` with `backend` (the
    /// caller picks it from the pairing — co-located → `Local`, cross-node →
    /// `Nixl`).
    pub fn register(
        &self,
        owner: WorkerId,
        handle: KvHandle,
        backend: BackendKind,
    ) -> Result<RegisteredHandle> {
        self.backend(backend)?.register(owner, handle)
    }

    /// Register a remote peer's connection info with `backend`.
    pub fn connect(&self, backend: BackendKind, peer: &crate::core::PeerConn) -> Result<()> {
        self.backend(backend)?.connect(peer)
    }

    /// This worker's connect metadata for `backend`, to advertise to peers.
    pub fn local_metadata(&self, backend: BackendKind) -> Result<Vec<u8>> {
        self.backend(backend)?.local_metadata()
    }

    /// Start sending `pages` of `handle` to worker `dst`.
    pub fn send(
        &self,
        handle: &RegisteredHandle,
        pages: &PageSet,
        dst: WorkerId,
    ) -> Result<TransferId> {
        let kind = handle.backend();
        let inner = self.backend(kind)?.send(handle, pages, dst)?;
        Ok(self.route(kind, inner))
    }

    pub fn send_mapped(
        &self,
        handle: &RegisteredHandle,
        src_pages: &PageSet,
        dst_pages: &PageSet,
        dst: WorkerId,
    ) -> Result<TransferId> {
        let kind = handle.backend();
        let inner = self
            .backend(kind)?
            .send_mapped(handle, src_pages, dst_pages, dst)?;
        Ok(self.route(kind, inner))
    }

    /// Start receiving `pages` into the local `slot` from worker `src`.
    pub fn recv(
        &self,
        slot: &RegisteredHandle,
        pages: &PageSet,
        src: WorkerId,
    ) -> Result<TransferId> {
        let kind = slot.backend();
        let inner = self.backend(kind)?.recv(slot, pages, src)?;
        Ok(self.route(kind, inner))
    }

    /// Poll an in-flight transfer's completion.
    pub fn poll(&self, id: TransferId) -> Result<Completion> {
        let route = *self
            .routes
            .lock()
            .unwrap()
            .get(&id.0)
            .ok_or(TransportError::UnknownTransfer { id: id.0 })?;
        self.backend(route.kind)?.poll(route.inner)
    }
}

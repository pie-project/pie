use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::backends::local::{D2dCopier, LocalBackend};
use crate::core::{
    Backend, BackendKind, Completion, PageSet, RegisteredHandle, TransferId, WorkerId,
};
use crate::error::{Result, TransportError};
use engine::KvHandle;

#[derive(Clone, Copy)]
struct Route {
    kind: BackendKind,
    inner: TransferId,
}

pub struct Registry {
    local: LocalBackend,
    #[cfg(feature = "nixl")]
    nixl: Option<crate::backends::nixl::NixlBackend>,
    routes: Mutex<HashMap<u64, Route>>,
    next_id: AtomicU64,
}

impl Registry {
    pub fn local_only(copier: Box<dyn D2dCopier>) -> Self {
        Self {
            local: LocalBackend::new(copier),
            #[cfg(feature = "nixl")]
            nixl: None,
            routes: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(0),
        }
    }

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

    fn route(&self, kind: BackendKind, inner: TransferId) -> TransferId {
        let out = TransferId(self.next_id.fetch_add(1, Ordering::Relaxed));
        self.routes
            .lock()
            .unwrap()
            .insert(out.0, Route { kind, inner });
        out
    }

    /// Registers `handle` with the `backend` backend ([`Backend::register`]).
    ///
    /// # Safety
    ///
    /// As for [`Backend::register`], for as long as this registry lives.
    #[allow(unsafe_code)]
    pub unsafe fn register(
        &self,
        owner: WorkerId,
        handle: KvHandle,
        backend: BackendKind,
    ) -> Result<RegisteredHandle> {
        // SAFETY: the backend lives as long as this registry, and the caller
        // keeps `handle`'s regions alive and unraced that long.
        unsafe { self.backend(backend)?.register(owner, handle) }
    }

    pub fn connect(&self, backend: BackendKind, peer: &crate::core::PeerConn) -> Result<()> {
        self.backend(backend)?.connect(peer)
    }

    pub fn local_metadata(&self, backend: BackendKind) -> Result<Vec<u8>> {
        self.backend(backend)?.local_metadata()
    }

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

//! Engine registry — binds a driver-exported handle to an engine and dispatches
//! the data-plane lifecycle to it.
//!
//! This is the single entry point the runtime drives. It receives the
//! controller's pairing decision ("send A's pages to B") already made and only
//! *executes* it — no routing or scheduling lives here.
//!
//! The caller picks the engine for a handle at [`register`](Registry::register)
//! time (informed by the pairing — co-located → `local`, cross-node → `nixl`).
//! The registry mints a globally-unique [`TransferId`] per transfer and routes
//! `poll` back to the issuing engine, so ids never collide across engines.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::{
    Completion, Engine, EngineKind, PageSet, RegisteredHandle, TransferId, WorkerId,
};
use crate::engines::local::{D2dCopier, LocalEngine};
use crate::error::{Result, TransportError};
use pie_schema::kv::KvHandle;

/// Where an outward [`TransferId`] was issued: which engine, and that engine's
/// own (per-engine) transfer id.
#[derive(Clone, Copy)]
struct Route {
    kind: EngineKind,
    inner: TransferId,
}

/// Binds driver-exported handles to engines and dispatches transfers.
pub struct Registry {
    local: LocalEngine,
    #[cfg(feature = "nixl")]
    nixl: Option<crate::engines::nixl::NixlEngine>,
    /// Outward transfer id → the engine + inner id that issued it. The registry
    /// owns id assignment so per-engine counters can't collide.
    routes: Mutex<HashMap<u64, Route>>,
    next_id: AtomicU64,
}

impl Registry {
    /// Build a registry with only the local engine — the minimal start.
    pub fn local_only(copier: Box<dyn D2dCopier>) -> Self {
        Self {
            local: LocalEngine::new(copier),
            #[cfg(feature = "nixl")]
            nixl: None,
            routes: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(0),
        }
    }

    /// Build a registry with both the local engine and a cross-node NIXL engine.
    #[cfg(feature = "nixl")]
    pub fn with_nixl(copier: Box<dyn D2dCopier>, nixl: crate::engines::nixl::NixlEngine) -> Self {
        Self {
            local: LocalEngine::new(copier),
            nixl: Some(nixl),
            routes: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(0),
        }
    }

    fn engine(&self, kind: EngineKind) -> Result<&dyn Engine> {
        match kind {
            EngineKind::Local => Ok(&self.local),
            EngineKind::Nixl => {
                #[cfg(feature = "nixl")]
                {
                    self.nixl
                        .as_ref()
                        .map(|n| n as &dyn Engine)
                        .ok_or(TransportError::Unsupported("nixl engine not enabled"))
                }
                #[cfg(not(feature = "nixl"))]
                {
                    Err(TransportError::Unsupported(
                        "nixl engine not built (enable feature \"nixl\")",
                    ))
                }
            }
        }
    }

    /// Mint a globally-unique outward id for an engine's inner transfer id, so
    /// per-engine counters can never collide.
    fn route(&self, kind: EngineKind, inner: TransferId) -> TransferId {
        let out = TransferId(self.next_id.fetch_add(1, Ordering::Relaxed));
        self.routes
            .lock()
            .unwrap()
            .insert(out.0, Route { kind, inner });
        out
    }

    /// Register a driver-exported handle owned by `owner` with `engine` (the
    /// caller picks it from the pairing — co-located → `Local`, cross-node →
    /// `Nixl`).
    pub fn register(
        &self,
        owner: WorkerId,
        handle: KvHandle,
        engine: EngineKind,
    ) -> Result<RegisteredHandle> {
        self.engine(engine)?.register(owner, handle)
    }

    /// Register a remote peer's connection info with `engine`.
    pub fn connect(&self, engine: EngineKind, peer: &crate::core::PeerConn) -> Result<()> {
        self.engine(engine)?.connect(peer)
    }

    /// This worker's connect metadata for `engine`, to advertise to peers.
    pub fn local_metadata(&self, engine: EngineKind) -> Result<Vec<u8>> {
        self.engine(engine)?.local_metadata()
    }

    /// Start sending `pages` of `handle` to worker `dst`.
    pub fn send(
        &self,
        handle: &RegisteredHandle,
        pages: &PageSet,
        dst: WorkerId,
    ) -> Result<TransferId> {
        let kind = handle.engine();
        let inner = self.engine(kind)?.send(handle, pages, dst)?;
        Ok(self.route(kind, inner))
    }

    /// Start receiving `pages` into the local `slot` from worker `src`.
    pub fn recv(
        &self,
        slot: &RegisteredHandle,
        pages: &PageSet,
        src: WorkerId,
    ) -> Result<TransferId> {
        let kind = slot.engine();
        let inner = self.engine(kind)?.recv(slot, pages, src)?;
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
        self.engine(route.kind)?.poll(route.inner)
    }
}

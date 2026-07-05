//! `pie-transport` — the worker↔worker P2P KV-tensor data plane.
//!
//! # What this crate is
//!
//! The data plane that moves KV-cache tensors directly between workers. A
//! controller pairs worker A with worker B and then **steps out**; from that
//! moment KV tensors flow P2P A↔B, *bypassing the controller*. This crate owns
//! that movement and nothing else — it never makes policy.
//!
//! # Shape (per the Controller & Transport design)
//!
//! ```text
//! core/       backend-agnostic interface: register → send/recv → poll
//! engines/
//!     local/  same-node device-to-device copy (co-located PD, zero network)
//!     nixl/   cross-node RDMA/TCP/NVMe via NIXL  [feature = "nixl", deferred]
//! registry/   binds a driver-exported handle to an engine, dispatches
//! ```
//!
//! **Minimal start = `core` + `local`.** Co-located prefill+decode defers all
//! RDMA (YAGNI); the `nixl` engine is a stub behind `feature = "nixl"` and is
//! the only place RDMA lives. Backends are asymmetric: cuda/rocm cross-node use
//! NIXL, co-located peers use `local`, and **metal/vulkan never participate**
//! (single-node; NIXL is Linux-only).
//!
//! # Boundaries
//!
//!   * **↔driver (handle boundary):** the driver pins its KV buffers and exports
//!     a [`pie_driver_abi::kv::KvHandle`]; transport consumes it without owning or
//!     interpreting the bytes. The per-backend registration shim lives on the
//!     driver's export surface. Transport never imports the driver — they meet
//!     only through the handle type on the schema floor.
//!   * **↔controller:** receives a pairing decision and executes it. No routing
//!     or scheduling here.
//!   * **↔runtime:** transfers are async — transport exposes the start and a
//!     completion signal ([`Completion`]); *when* to await is the scheduler's
//!     job.
//!   * **↔interface/driver:** the KV layout and handle type live in
//!     [`pie_driver_abi::kv`], shared by driver / transport / runtime / controller.

pub mod core;
pub mod engines;
pub mod error;
pub mod registry;

pub use crate::core::{
    Completion, Engine, EngineKind, PageSet, PeerConn, RegisteredHandle, TransferId, WorkerId,
};
pub use engines::local::{D2dCopier, LocalEngine};
pub use error::{Result, TransportError};
pub use registry::Registry;

pub use pie_driver_abi::kv::{KvDtype, KvHandle, KvLayout, KvLayoutKind, KvRegion, MemoryDomain};

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn layout() -> KvLayout {
        KvLayout {
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 64,
            page_size: 16,
            dtype: KvDtype::Bf16,
            kind: KvLayoutKind::KvSeparate,
        }
    }

    /// A handle with one region big enough for `n_pages`, based at `base`.
    fn handle(base: u64, n_pages: u64) -> KvHandle {
        let l = layout();
        KvHandle {
            regions: vec![KvRegion {
                base,
                len: n_pages * l.page_bytes(),
                domain: MemoryDomain::CudaDevice(0),
            }],
            layout: l,
        }
    }

    /// Records every D2D copy the local engine issues. Cloning shares the log,
    /// so a test can inspect calls after the copier is moved into the registry.
    #[derive(Clone, Default)]
    struct FakeCopier {
        calls: Arc<Mutex<Vec<(u64, u64, u64)>>>,
    }
    impl D2dCopier for FakeCopier {
        fn copy(&self, src_addr: u64, dst_addr: u64, len: u64) -> Result<()> {
            self.calls.lock().unwrap().push((src_addr, dst_addr, len));
            Ok(())
        }
    }

    #[test]
    fn local_send_copies_pages_at_matching_offsets() {
        let copier = FakeCopier::default();
        let calls = copier.calls.clone();
        let reg = Registry::local_only(Box::new(copier));

        let prefill = reg
            .register(WorkerId(1), handle(0x1000, 8), EngineKind::Local)
            .unwrap();
        let _decode = reg
            .register(WorkerId(2), handle(0x9000, 8), EngineKind::Local)
            .unwrap();

        let pages = PageSet::new(vec![0, 3]);
        let id = reg.send(&prefill, &pages, WorkerId(2)).unwrap();
        assert_eq!(reg.poll(id).unwrap(), Completion::Done);

        let page_bytes = layout().page_bytes();
        let recorded = calls.lock().unwrap().clone();
        assert_eq!(
            recorded,
            vec![
                (0x1000, 0x9000, page_bytes),
                (0x1000 + 3 * page_bytes, 0x9000 + 3 * page_bytes, page_bytes),
            ]
        );
    }

    #[test]
    fn local_recv_acknowledges_colocated_peer() {
        let reg = Registry::local_only(Box::<FakeCopier>::default());
        let decode = reg
            .register(WorkerId(2), handle(0x9000, 8), EngineKind::Local)
            .unwrap();
        reg.register(WorkerId(1), handle(0x1000, 8), EngineKind::Local)
            .unwrap();

        let id = reg
            .recv(&decode, &PageSet::new(vec![0]), WorkerId(1))
            .unwrap();
        assert_eq!(reg.poll(id).unwrap(), Completion::Done);
    }

    #[test]
    fn send_to_unregistered_peer_is_unknown_peer() {
        let reg = Registry::local_only(Box::<FakeCopier>::default());
        let prefill = reg
            .register(WorkerId(1), handle(0x1000, 8), EngineKind::Local)
            .unwrap();
        let err = reg
            .send(&prefill, &PageSet::new(vec![0]), WorkerId(99))
            .unwrap_err();
        assert!(matches!(err, TransportError::UnknownPeer { worker: 99 }));
    }

    #[test]
    fn page_out_of_bounds_is_rejected() {
        let reg = Registry::local_only(Box::<FakeCopier>::default());
        let prefill = reg
            .register(WorkerId(1), handle(0x1000, 2), EngineKind::Local)
            .unwrap();
        reg.register(WorkerId(2), handle(0x9000, 2), EngineKind::Local)
            .unwrap();
        let err = reg
            .send(&prefill, &PageSet::new(vec![5]), WorkerId(2))
            .unwrap_err();
        assert!(matches!(err, TransportError::PageOutOfBounds { page: 5 }));
    }

    #[test]
    fn poll_unknown_transfer_errors() {
        let reg = Registry::local_only(Box::<FakeCopier>::default());
        let err = reg.poll(TransferId(123)).unwrap_err();
        assert!(matches!(err, TransportError::UnknownTransfer { id: 123 }));
    }

    /// A handle tagged for an engine that isn't built (nixl off) routes to an
    /// `Unsupported` error rather than panicking.
    #[test]
    fn routing_to_unbuilt_engine_is_unsupported() {
        let reg = Registry::local_only(Box::<FakeCopier>::default());
        let nixl_handle = RegisteredHandle {
            engine: EngineKind::Nixl,
            owner: WorkerId(1),
            handle: handle(0x1000, 4),
        };
        let err = reg
            .send(&nixl_handle, &PageSet::new(vec![0]), WorkerId(2))
            .unwrap_err();
        assert!(matches!(err, TransportError::Unsupported(_)));
    }

    /// The registry mints globally-unique outward transfer ids and routes each
    /// `poll` back correctly — the namespacing that prevents per-engine id
    /// collisions once a second engine (nixl) is present.
    #[test]
    fn registry_mints_distinct_transfer_ids() {
        let reg = Registry::local_only(Box::<FakeCopier>::default());
        let a = reg
            .register(WorkerId(1), handle(0x1000, 8), EngineKind::Local)
            .unwrap();
        reg.register(WorkerId(2), handle(0x9000, 8), EngineKind::Local)
            .unwrap();

        let id1 = reg.send(&a, &PageSet::new(vec![0]), WorkerId(2)).unwrap();
        let id2 = reg.send(&a, &PageSet::new(vec![1]), WorkerId(2)).unwrap();
        assert_ne!(id1, id2, "outward ids must be unique");
        assert_eq!(reg.poll(id1).unwrap(), Completion::Done);
        assert_eq!(reg.poll(id2).unwrap(), Completion::Done);
    }

    /// The local engine has no connect-metadata: `connect` is a no-op and
    /// `local_metadata` is empty.
    #[test]
    fn local_engine_has_no_connect_metadata() {
        let reg = Registry::local_only(Box::<FakeCopier>::default());
        let peer = PeerConn {
            worker: WorkerId(5),
            handle: handle(0x1000, 4),
            metadata: b"ignored".to_vec(),
        };
        reg.connect(EngineKind::Local, &peer).unwrap();
        assert!(reg.local_metadata(EngineKind::Local).unwrap().is_empty());
    }
}

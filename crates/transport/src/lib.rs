//! `transport`: the worker-to-worker P2P KV-tensor data plane. A controller pairs worker A with worker B and steps out; from that moment KV tensors flow P2P A<->B, bypassing the controller. This crate owns that movement and nothing else — it never makes policy.
//!
//! `core/` is the backend-agnostic interface (register -> send/recv -> poll); `backends/local` does same-node device-to-device copy, `backends/nixl` (behind `feature = "nixl"`) cross-node RDMA/TCP/NVMe; `registry/` binds an engine-exported handle to a transfer backend and dispatches. Backends are asymmetric: cuda/rocm cross-node use NIXL, co-located peers use `local`, metal/vulkan never participate (single-node).
//!
//! The engine pins its KV buffers and exports a [`engine::KvHandle`]; transport consumes it without owning or interpreting the bytes, and never imports the engine. Transfers are async — transport exposes the start and a completion signal ([`Completion`]); when to await is the scheduler's job.

pub mod backends;
pub mod core;
pub mod error;
pub mod registry;

pub use crate::core::{
    Backend, BackendKind, Completion, PageSet, PeerConn, RegisteredHandle, TransferId, WorkerId,
};
pub use backends::local::{D2dCopier, LocalBackend};
#[cfg(feature = "nixl")]
pub use backends::nixl::NixlBackend;
pub use error::{Result, TransportError};
pub use registry::Registry;

// A cache row's element type is the model's: `transport::Dtype` is that type.
pub use engine::{KvHandle, KvLayout, KvLayoutKind, KvRegion, MemoryDomain};
pub use dtype::Dtype;

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
            dtype: Dtype::Bf16,
            kind: KvLayoutKind::KvSeparate,
            storage_format: "test-bf16".to_string(),
            region_page_bytes: Vec::new(),
        }
    }

    /// A handle with one region big enough for `n_pages`, based at `base`.
    fn handle(base: u64, n_pages: u64) -> KvHandle {
        let l = layout();
        KvHandle {
            regions: vec![KvRegion {
                base,
                len: n_pages * l.page_bytes(),
                page_stride: l.page_bytes(),
                domain: MemoryDomain::CudaDevice(0),
            }],
            layout: l,
        }
    }

    fn multi_handle(base: u64, n_pages: u64) -> KvHandle {
        let mut l = layout();
        l.region_page_bytes = vec![64, 32];
        KvHandle {
            regions: vec![
                KvRegion {
                    base,
                    len: n_pages * 64,
                    page_stride: 64,
                    domain: MemoryDomain::CudaDevice(0),
                },
                KvRegion {
                    base: base + 0x1000,
                    len: n_pages * 32,
                    page_stride: 32,
                    domain: MemoryDomain::CudaDevice(0),
                },
            ],
            layout: l,
        }
    }

    /// Records every D2D copy the local backend issues. Cloning shares the log,
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
    fn local_recv_acknowledges_colocated_peer() {
        let reg = Registry::local_only(Box::<FakeCopier>::default());
        let decode = reg
            .register(WorkerId(2), handle(0x9000, 8), BackendKind::Local)
            .unwrap();
        reg.register(WorkerId(1), handle(0x1000, 8), BackendKind::Local)
            .unwrap();

        let id = reg
            .recv(&decode, &PageSet::new(vec![0]), WorkerId(1))
            .unwrap();
        assert_eq!(reg.poll(id).unwrap(), Completion::Done);
    }

    #[test]
    fn local_mapped_send_copies_distinct_pages_across_all_regions() {
        let copier = FakeCopier::default();
        let calls = copier.calls.clone();
        let reg = Registry::local_only(Box::new(copier));
        let source = reg
            .register(WorkerId(1), multi_handle(0x1000, 8), BackendKind::Local)
            .unwrap();
        reg.register(WorkerId(2), multi_handle(0x9000, 8), BackendKind::Local)
            .unwrap();
        let id = reg
            .send_mapped(
                &source,
                &PageSet::new(vec![1]),
                &PageSet::new(vec![3]),
                WorkerId(2),
            )
            .unwrap();
        assert_eq!(reg.poll(id).unwrap(), Completion::Done);
        assert_eq!(
            calls.lock().unwrap().as_slice(),
            &[
                (0x1000 + 64, 0x9000 + 3 * 64, 64),
                (0x2000 + 32, 0xA000 + 3 * 32, 32),
            ]
        );
    }

    #[test]
    fn send_to_unregistered_peer_is_unknown_peer() {
        let reg = Registry::local_only(Box::<FakeCopier>::default());
        let prefill = reg
            .register(WorkerId(1), handle(0x1000, 8), BackendKind::Local)
            .unwrap();
        let err = reg
            .send(&prefill, &PageSet::new(vec![0]), WorkerId(99))
            .unwrap_err();
        assert!(matches!(err, TransportError::UnknownPeer { worker: 99 }));
    }

    /// A handle tagged for a backend that isn't built (nixl off) routes to an
    /// `Unsupported` error rather than panicking.
    #[test]
    fn routing_to_unbuilt_backend_is_unsupported() {
        let reg = Registry::local_only(Box::<FakeCopier>::default());
        let nixl_handle = RegisteredHandle {
            backend: BackendKind::Nixl,
            owner: WorkerId(1),
            handle: handle(0x1000, 4),
        };
        let err = reg
            .send(&nixl_handle, &PageSet::new(vec![0]), WorkerId(2))
            .unwrap_err();
        assert!(matches!(err, TransportError::Unsupported(_)));
    }

    /// The local backend has no connect-metadata: `connect` is a no-op and
    /// `local_metadata` is empty.
    #[test]
    fn local_backend_has_no_connect_metadata() {
        let reg = Registry::local_only(Box::<FakeCopier>::default());
        let peer = PeerConn {
            worker: WorkerId(5),
            handle: handle(0x1000, 4),
            metadata: b"ignored".to_vec(),
        };
        reg.connect(BackendKind::Local, &peer).unwrap();
        assert!(reg.local_metadata(BackendKind::Local).unwrap().is_empty());
    }
}

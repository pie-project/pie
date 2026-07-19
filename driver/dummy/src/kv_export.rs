//! Driver-side KV-export seam — the *producer* half of the runtime ↔
//! transport KV-transfer handshake.
//!
//! A backend that can expose its KV cache for remote transfer produces a
//! [`pie_driver_abi::KvHandle`]: the [`KvRegion`]s backing the cache plus the
//! paged [`pie_driver_abi::KvLayout`]. Transport's `Engine::register` consumes
//! that value by move — the two sides meet only at `KvHandle`, with no trait or
//! import spanning the driver and transport crates. Remote-access credentials
//! are mechanism-specific and opaque (e.g. a NIXL agent's metadata blob),
//! exchanged at the connect level, so the region carries only `(base, len,
//! domain)`.
//!
//! The dummy driver has no real KV cache, so it pins a small synthetic
//! host-DRAM buffer and exports it as a single [`MemoryDomain::HostPinned`]
//! region — enough for the single-node UCX `shm,tcp` transport e2e to register
//! and move bytes without a GPU or RDMA NIC. Single-node backends with nothing
//! to export (e.g. metal/vulkan) decline by returning `None`. Real per-backend
//! device-memory export + RDMA registration is deferred.

use pie_driver_abi::{KvDtype, KvExport, KvHandle, KvLayout, KvLayoutKind, KvRegion, MemoryDomain};

/// The dummy driver's host-DRAM KV-export stub.
///
/// Owns a synthetic, page-sized host buffer so the exported region's `base`/
/// `len` stay valid for the handle's lifetime (the driver-pins-its-buffers
/// contract). It is a plain heap allocation — a real driver would page-lock the
/// memory — which is sufficient for the UCX `shm,tcp` host path (no CUDA
/// pinning required there).
pub struct DummyKvExport {
    /// Synthetic host-DRAM buffer backing the exported region. Allocated once
    /// and never resized, so its data pointer is stable for `&self`'s lifetime.
    buf: Vec<u8>,
    /// Paged geometry the exported handle advertises.
    layout: KvLayout,
}

impl DummyKvExport {
    /// Allocate a synthetic export buffer matching the configured page count.
    pub fn new(num_pages: u32, page_size: u32) -> Self {
        // Minimal but self-consistent layout. page_bytes() here is
        // 1 layer · 2 planes · 1 head · 64 head_dim · 16 tokens/page · 2 B
        // = 4096 bytes/page.
        let layout = KvLayout {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 64,
            page_size,
            dtype: KvDtype::Bf16,
            kind: KvLayoutKind::KvSeparate,
            storage_format: "dummy-bf16-v1".to_string(),
            region_page_bytes: Vec::new(),
        };
        let page_stride = layout.page_bytes();
        let len = page_stride * num_pages as u64;
        Self {
            buf: vec![0u8; len as usize],
            layout,
        }
    }

    /// The exported buffer's bytes — for a transport e2e to seed a test
    /// pattern on the source side and verify it on the destination.
    pub fn region_bytes(&self) -> &[u8] {
        &self.buf
    }

    /// Mutable view of the exported buffer. The data pointer is unchanged (the
    /// `Vec` is never resized), so the previously exported `KvHandle` stays
    /// valid across writes.
    pub fn region_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.buf
    }
}

impl Default for DummyKvExport {
    fn default() -> Self {
        Self::new(8, 16)
    }
}

impl KvExport for DummyKvExport {
    /// Export the synthetic host buffer as one `HostPinned` region. The region
    /// carries no credentials — the transport engine registers it (NIXL
    /// `register_mem`) and produces the opaque metadata exchanged at connect.
    fn export_kv_handle(&self) -> Option<KvHandle> {
        Some(KvHandle {
            regions: vec![KvRegion {
                base: self.buf.as_ptr() as u64,
                len: self.buf.len() as u64,
                page_stride: self.layout.page_bytes(),
                domain: MemoryDomain::HostPinned,
            }],
            layout: self.layout.clone(),
        })
    }
}

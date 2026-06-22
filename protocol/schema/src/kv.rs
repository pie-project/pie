//! KV-layout vocabulary — how a paged KV cache maps onto bytes.
//!
//! The data plane needs to know how a KV page maps onto bytes so it can turn a
//! page index into an `(offset, len)` for a one-sided RDMA op. That vocabulary
//! lives here, in the schema floor, so transport, drivers, and the controller
//! can agree on page geometry without depending on each other.
//!
//! Keep the page → byte math behind [`KvLayout`] so callers never encode layout
//! knowledge themselves.
//!
//! These are control-plane vocabulary (handle/region/layout cross the pairing
//! side channel), so they derive `serde` like `cluster`/`DriverCapabilities` —
//! deliberately NOT `#[schema]`, and NOT part of `SCHEMA_HASH`.

use serde::{Deserialize, Serialize};

/// Element type of the KV cache. Only the byte width matters to transport; the
/// numeric semantics are the model's concern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvDtype {
    F32,
    F16,
    Bf16,
    /// 8-bit float (e4m3) KV cache.
    F8E4M3,
    /// 8-bit integer (quantized) KV cache.
    I8,
}

impl KvDtype {
    /// Size of one element in bytes.
    pub const fn size(self) -> usize {
        match self {
            KvDtype::F32 => 4,
            KvDtype::F16 | KvDtype::Bf16 => 2,
            KvDtype::F8E4M3 | KvDtype::I8 => 1,
        }
    }
}

/// How K and V are arranged within a page. Transport only needs the multiplier
/// (almost always 2: one K plane + one V plane), but the discriminant is kept
/// so a future MLA / single-latent layout can be added without a silent change
/// to the page-size math.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvLayoutKind {
    /// Separate key and value planes per page (the common paged-attention case).
    KvSeparate,
    /// A single fused latent plane per page (e.g. MLA-style caches).
    FusedLatent,
}

impl KvLayoutKind {
    /// Number of distinct planes stored per page.
    const fn planes(self) -> usize {
        match self {
            KvLayoutKind::KvSeparate => 2,
            KvLayoutKind::FusedLatent => 1,
        }
    }
}

/// Paged KV-cache geometry — the minimum a peer needs to address pages
/// identically on both ends of a link.
///
/// Field names line up with the driver capability handshake
/// (`kv_page_size`, etc.).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvLayout {
    /// Transformer layers cached.
    pub num_layers: u32,
    /// KV heads per layer (post-GQA grouping).
    pub num_kv_heads: u32,
    /// Channels per head.
    pub head_dim: u32,
    /// Tokens per page.
    pub page_size: u32,
    /// Element type.
    pub dtype: KvDtype,
    /// K/V arrangement within a page.
    pub kind: KvLayoutKind,
}

impl KvLayout {
    /// Bytes occupied by a single KV page across all layers.
    ///
    /// `layers · planes · kv_heads · head_dim · page_size · dtype_bytes`.
    /// A peer must agree on this exact value or page offsets diverge — see
    /// [`KvLayout::compatible_with`].
    pub fn page_bytes(&self) -> u64 {
        self.num_layers as u64
            * self.kind.planes() as u64
            * self.num_kv_heads as u64
            * self.head_dim as u64
            * self.page_size as u64
            * self.dtype.size() as u64
    }

    /// True when two workers can move pages between each other unmodified.
    /// Pairing must check this before handing the link to the data plane; the
    /// transport layer re-checks at transfer time and raises its layout-mismatch
    /// error otherwise.
    pub fn compatible_with(&self, other: &KvLayout) -> bool {
        self == other
    }
}

// =============================================================================
// Driver-exported KV handle
// =============================================================================
//
// The data plane never allocates or owns the KV cache. The driver pins its KV
// buffers and exports a handle describing where they live; transport consumes it
// without interpreting the bytes. This handle type is the shared contract —
// driver(export) / transport(consume) / runtime(ledger) / controller(pairing
// metadata) all speak it — so it lives on the schema floor next to the layout.
// It is mechanism-neutral: remote-access credentials (e.g. a NIXL agent's
// metadata blob) are opaque and travel at the connect level, NOT in the region.
// Single-node backends (metal/vulkan) export nothing.

/// Where a registrable KV region physically lives. This is the only place the
/// backend "how" axis surfaces in the handle — reduced to what a copier/NIC
/// needs to address the memory, not any backend-specific control logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryDomain {
    /// Page-locked host memory (a staging buffer the driver DMAs through).
    HostPinned,
    /// CUDA device memory on the given device ordinal.
    CudaDevice(u32),
    /// ROCm/HIP device memory on the given device ordinal.
    RocmDevice(u32),
}

/// One contiguous span of KV-cache memory the driver pinned and exported.
///
/// `base` is a process-virtual address within the domain's space; consumers
/// treat it as an opaque integer and only ever add page offsets to it. The
/// region carries no remote-access credentials: those are mechanism-specific
/// (a NIXL agent registers the region and produces an opaque metadata blob) and
/// are exchanged at the connect level, keeping this type backend-neutral.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvRegion {
    /// Base virtual address of the region within its domain.
    pub base: u64,
    /// Length of the region in bytes.
    pub len: u64,
    /// Physical domain of the bytes.
    pub domain: MemoryDomain,
}

/// The driver-exported, engine-agnostic KV handle the data plane consumes.
///
/// Produced by the per-backend registration shim on the driver's export
/// surface. The page → byte math lives in [`KvLayout`]; the regions say where
/// those bytes are. Transport never imports the driver — they meet only here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvHandle {
    /// The contiguous regions backing the KV cache.
    pub regions: Vec<KvRegion>,
    /// Paged geometry of the cache, for page → byte addressing.
    pub layout: KvLayout,
}

impl KvHandle {
    /// Bytes per KV page, from the layout.
    pub fn page_bytes(&self) -> u64 {
        self.layout.page_bytes()
    }
}

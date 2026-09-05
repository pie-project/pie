//! Moving state that is already on a device: KV pages, recurrent slots, pool pages.

use serde::{Deserialize, Serialize};

use model_ir::Dtype;

/// Which memory a range of bytes lives in. The device ordinal rides inside
/// the variant that has one, since "which GPU" is meaningless for
/// host-pinned memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryDomain {
    /// Page-locked host memory — reachable by every device on the node, and
    /// what an inline transfer stages through.
    HostPinned,
    /// CUDA device memory on this ordinal.
    CudaDevice(u32),
    /// ROCm device memory on this ordinal.
    RocmDevice(u32),
    /// Metal shared memory (unified: the CPU may map it).
    MetalShared,
    /// Metal private memory (device-only).
    MetalPrivate,
    /// Vulkan device-local memory on this ordinal.
    VulkanDevice(u32),
    /// wgpu device memory on this ordinal. Spelled the way the shell,
    /// the platform and the artifact word are (`wgpu`), not `WebGpu`.
    WgpuDevice(u32),
}

impl MemoryDomain {
    /// The device ordinal, for the domains that have one.
    #[must_use]
    pub const fn ordinal(self) -> Option<u32> {
        match self {
            MemoryDomain::CudaDevice(ordinal)
            | MemoryDomain::RocmDevice(ordinal)
            | MemoryDomain::VulkanDevice(ordinal)
            | MemoryDomain::WgpuDevice(ordinal) => Some(ordinal),
            MemoryDomain::HostPinned | MemoryDomain::MetalShared | MemoryDomain::MetalPrivate => {
                None
            }
        }
    }

    /// True for the domains the host may address directly.
    #[must_use]
    pub const fn host_visible(self) -> bool {
        matches!(
            self,
            MemoryDomain::HostPinned | MemoryDomain::MetalShared
        )
    }
}

/// How the two halves of a cache row are laid out in a page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvLayoutKind {
    /// Separate K and V planes — two planes per layer.
    KvSeparate,
    /// One fused latent plane, as MLA writes it — one plane per layer.
    FusedLatent,
}

impl KvLayoutKind {
    /// How many planes a layer occupies under this layout.
    #[must_use]
    pub const fn planes(self) -> u64 {
        match self {
            KvLayoutKind::KvSeparate => 2,
            KvLayoutKind::FusedLatent => 1,
        }
    }
}

/// The geometry of one KV pool, in enough detail that a peer can compute a
/// page's address without asking. Two loads may exchange pages iff their
/// layouts are exactly equal — a page is raw bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvLayout {
    /// Layers with a cache row.
    pub num_layers: u32,
    /// KV heads per layer.
    pub num_kv_heads: u32,
    /// Elements per head.
    pub head_dim: u32,
    /// Tokens per page.
    pub page_size: u32,
    /// The cache row's element type — the model's own dtype (decision 1).
    pub dtype: Dtype,
    /// Whether K and V share a plane.
    pub kind: KvLayoutKind,
    /// A backend's name for a packing the four numbers above do not determine
    /// (a swizzle, a paired-plane interleave). Empty means the plain layout.
    #[serde(default)]
    pub storage_format: String,
    /// Per-region page bytes, when the pool is cut into regions of unequal
    /// size (a quantized pool's scales region, say). Non-empty OVERRIDES the
    /// derived formula in [`KvLayout::page_bytes`].
    #[serde(default)]
    pub region_page_bytes: Vec<u64>,
}

impl KvLayout {
    /// How many bytes one page occupies. `region_page_bytes` wins when set;
    /// otherwise the geometry multiplies out, element size in bits so a
    /// quantized pool is not rounded up.
    #[must_use]
    pub fn page_bytes(&self) -> u64 {
        if !self.region_page_bytes.is_empty() {
            return self.region_page_bytes.iter().copied().sum();
        }
        let elements = u64::from(self.num_layers)
            * self.kind.planes()
            * u64::from(self.num_kv_heads)
            * u64::from(self.head_dim)
            * u64::from(self.page_size);
        (elements * self.dtype.bits()).div_ceil(8)
    }

    /// May a page of `other` be written into a pool of this layout?
    #[must_use]
    pub fn compatible_with(&self, other: &KvLayout) -> bool {
        self == other
    }
}

/// One contiguous span of a KV pool, as an address a peer may write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvRegion {
    /// Device address of the first byte.
    pub base: u64,
    /// How many bytes the region spans.
    pub len: u64,
    /// Bytes between consecutive pages in this region.
    pub page_stride: u64,
    /// Which memory it lives in.
    pub domain: MemoryDomain,
}

/// A whole KV pool, addressable from outside the engine that owns it — a
/// prefill worker exports one, a decode worker registers it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvHandle {
    /// The pool's spans, in region order.
    pub regions: Vec<KvRegion>,
    /// The geometry every span is cut to.
    pub layout: KvLayout,
}

impl KvHandle {
    /// How many bytes one page occupies.
    #[must_use]
    pub fn page_bytes(&self) -> u64 {
        self.layout.page_bytes()
    }

    /// How many pages the pool holds, or `None` if the regions disagree.
    /// Every region must hold the same page count: a page id indexes all of
    /// them at once (region 0 holds page `p`'s K plane, region 1 its scales).
    #[must_use]
    pub fn page_capacity(&self) -> Option<u64> {
        let mut capacity = None;
        for region in &self.regions {
            if region.page_stride == 0 || region.len % region.page_stride != 0 {
                return None;
            }
            let pages = region.len / region.page_stride;
            if capacity
                .replace(pages)
                .is_some_and(|current| current != pages)
            {
                return None;
            }
        }
        capacity
    }
}

/// Something that can hand out its KV pool's address. A trait rather than a
/// method on [`Engine`](crate::Engine) because transport takes it on things
/// that are not `Engine`s — a mock pool in a test, a peer's cached handle.
pub trait KvExport {
    /// This pool's address, or `None` if it is not exportable.
    fn export_kv_handle(&self) -> Option<KvHandle>;
}

/// One page-to-page token move inside (or between) KV pools. Token-granular
/// rather than page-granular: a fork copies a partial page's live tokens
/// into a fresh page and leaves the rest.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvMove {
    /// Page written.
    pub dst_page_id: u32,
    /// First token slot written in it.
    pub dst_token_offset: u32,
    /// Page read.
    pub src_page_id: u32,
    /// First token slot read from it.
    pub src_token_offset: u32,
}

/// The `copy_kv` verb's argument.
///
/// `src_page_ids`/`dst_page_ids` are the whole-page moves and `moves` are the
/// token-granular ones; a submission may carry either or both.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCopy {
    /// Where the pages are read from.
    pub src: MemoryDomain,
    /// Where they are written.
    pub dst: MemoryDomain,
    /// Pages read, parallel to `dst_page_ids`.
    pub src_page_ids: Vec<u32>,
    /// Pages written, parallel to `src_page_ids`.
    pub dst_page_ids: Vec<u32>,
    /// Token-granular moves.
    pub moves: Vec<KvMove>,
}

impl Default for KvCopy {
    fn default() -> KvCopy {
        KvCopy {
            src: MemoryDomain::HostPinned,
            dst: MemoryDomain::HostPinned,
            src_page_ids: Vec::new(),
            dst_page_ids: Vec::new(),
            moves: Vec::new(),
        }
    }
}

impl KvCopy {
    /// Is this a submission the contract describes? Checks that the two
    /// page lists are parallel.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`](crate::Error::Invalid) when they are not.
    pub fn validate(&self) -> crate::Result<()> {
        if self.src_page_ids.len() != self.dst_page_ids.len() {
            return Err(crate::Error::Invalid(format!(
                "src_page_ids has {} entries and dst_page_ids {}",
                self.src_page_ids.len(),
                self.dst_page_ids.len()
            )));
        }
        Ok(())
    }
}

/// One span of recurrent state moved between slots.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateMove {
    /// Slot read.
    pub src_slot_id: u32,
    /// Slot written.
    pub dst_slot_id: u32,
    /// First token read from it.
    pub src_token_offset: u32,
    /// First token written in it.
    pub dst_token_offset: u32,
    /// How many tokens.
    pub token_count: u32,
}

/// The `copy_state` verb's argument.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct StateCopy {
    /// The spans to move, in order.
    pub moves: Vec<StateMove>,
}

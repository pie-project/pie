//! Rust-only KV transfer vocabulary.
//!
//! Cross-node movement still needs a small shared description of exported KV
//! memory. These types stay in `pie-driver-abi` so transport, runtime, and
//! drivers can agree on page geometry without pulling the local direct-FFI
//! surface into the generated C header.

use serde::{Deserialize, Serialize};

/// Element type of the KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvDtype {
    F32,
    F16,
    Bf16,
    F8E4M3,
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

/// How K and V are arranged within a page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvLayoutKind {
    KvSeparate,
    FusedLatent,
}

impl KvLayoutKind {
    const fn planes(self) -> usize {
        match self {
            KvLayoutKind::KvSeparate => 2,
            KvLayoutKind::FusedLatent => 1,
        }
    }
}

/// Paged KV-cache geometry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvLayout {
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub page_size: u32,
    pub dtype: KvDtype,
    pub kind: KvLayoutKind,
    #[serde(default)]
    pub storage_format: String,
    #[serde(default)]
    pub region_page_bytes: Vec<u64>,
}

impl KvLayout {
    pub fn page_bytes(&self) -> u64 {
        if !self.region_page_bytes.is_empty() {
            return self.region_page_bytes.iter().copied().sum();
        }
        self.num_layers as u64
            * self.kind.planes() as u64
            * self.num_kv_heads as u64
            * self.head_dim as u64
            * self.page_size as u64
            * self.dtype.size() as u64
    }

    pub fn compatible_with(&self, other: &KvLayout) -> bool {
        self == other
    }
}

/// Physical memory domain of an exported KV region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryDomain {
    HostPinned,
    CudaDevice(u32),
    RocmDevice(u32),
}

/// One contiguous exported KV region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvRegion {
    pub base: u64,
    pub len: u64,
    pub page_stride: u64,
    pub domain: MemoryDomain,
}

/// Driver-exported KV handle consumed by transport.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvHandle {
    pub regions: Vec<KvRegion>,
    pub layout: KvLayout,
}

impl KvHandle {
    pub fn page_bytes(&self) -> u64 {
        self.layout.page_bytes()
    }

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

/// Driver-side producer seam for transport registration.
pub trait KvExport {
    fn export_kv_handle(&self) -> Option<KvHandle>;
}

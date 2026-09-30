use serde::{Deserialize, Serialize};

use model_ir::Dtype;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryDomain {
    HostPinned,
    CudaDevice(u32),
    RocmDevice(u32),
    MetalShared,
    MetalPrivate,
    VulkanDevice(u32),
    WgpuDevice(u32),
    XlaDevice(u32),
    /// The engine's own slot file, one kv page per slot.
    LocalDisk,
}

impl MemoryDomain {
    #[must_use]
    pub const fn ordinal(self) -> Option<u32> {
        match self {
            MemoryDomain::CudaDevice(ordinal)
            | MemoryDomain::RocmDevice(ordinal)
            | MemoryDomain::VulkanDevice(ordinal)
            | MemoryDomain::WgpuDevice(ordinal)
            | MemoryDomain::XlaDevice(ordinal) => Some(ordinal),
            MemoryDomain::HostPinned
            | MemoryDomain::MetalShared
            | MemoryDomain::MetalPrivate
            | MemoryDomain::LocalDisk => None,
        }
    }

    #[must_use]
    pub const fn host_visible(self) -> bool {
        matches!(self, MemoryDomain::HostPinned | MemoryDomain::MetalShared)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvLayoutKind {
    KvSeparate,
    FusedLatent,
}

impl KvLayoutKind {
    #[must_use]
    pub const fn planes(self) -> u64 {
        match self {
            KvLayoutKind::KvSeparate => 2,
            KvLayoutKind::FusedLatent => 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvLayout {
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub page_size: u32,
    pub dtype: Dtype,
    pub kind: KvLayoutKind,
    #[serde(default)]
    pub storage_format: String,
    #[serde(default)]
    pub region_page_bytes: Vec<u64>,
}

impl KvLayout {
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

    #[must_use]
    pub fn compatible_with(&self, other: &KvLayout) -> bool {
        self == other
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvRegion {
    pub base: u64,
    pub len: u64,
    pub page_stride: u64,
    pub domain: MemoryDomain,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvHandle {
    pub regions: Vec<KvRegion>,
    pub layout: KvLayout,
}

impl KvHandle {
    #[must_use]
    pub fn page_bytes(&self) -> u64 {
        self.layout.page_bytes()
    }

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

pub trait KvExport {
    fn export_kv_handle(&self) -> Option<KvHandle>;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvMove {
    pub dst_page_id: u32,
    pub dst_token_offset: u32,
    pub src_page_id: u32,
    pub src_token_offset: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCopy {
    pub src: MemoryDomain,
    pub dst: MemoryDomain,
    pub src_page_ids: Vec<u32>,
    pub dst_page_ids: Vec<u32>,
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

    /// A whole-page move between `device` and a tier below it: the tier,
    /// whether the pages leave the device, and the page ids on the device
    /// and in the tier. `None` for any other copy.
    #[must_use]
    pub fn tier_move(&self, device: MemoryDomain) -> Option<(MemoryDomain, bool, &[u32], &[u32])> {
        let tier = |domain| matches!(domain, MemoryDomain::HostPinned | MemoryDomain::LocalDisk);
        if !self.moves.is_empty() {
            return None;
        }
        if self.src == device && tier(self.dst) {
            Some((self.dst, true, &self.src_page_ids, &self.dst_page_ids))
        } else if tier(self.src) && self.dst == device {
            Some((self.src, false, &self.dst_page_ids, &self.src_page_ids))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateMove {
    pub src_slot_id: u32,
    pub dst_slot_id: u32,
    pub src_token_offset: u32,
    pub dst_token_offset: u32,
    pub token_count: u32,
}

/// Which pools a state copy's slot ids name: host slots index the pool
/// `PoolFacts::host_state_slots` advertises.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateDirection {
    #[default]
    DeviceToDevice,
    DeviceToHost,
    HostToDevice,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct StateCopy {
    pub moves: Vec<StateMove>,
    #[serde(default)]
    pub direction: StateDirection,
}

impl StateCopy {
    /// Refuses a copy to or from host slots on `engine`, which reserves none.
    pub fn device_only(&self, engine: &'static str) -> crate::Result<()> {
        if self.direction == StateDirection::DeviceToDevice {
            return Ok(());
        }
        Err(crate::Error::Unsupported {
            verb: "`copy_state` to or from host slots, which this load does not reserve",
            engine,
        })
    }
}

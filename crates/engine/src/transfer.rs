use serde::{Deserialize, Serialize};

use poem_ir::Dtype;

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

/// `tokens` kv cells copied from one page to another, the unit a device
/// copies in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvRun {
    pub src_page: u32,
    pub src_token: u32,
    pub dst_page: u32,
    pub dst_token: u32,
    pub tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCopy {
    pub src: MemoryDomain,
    pub dst: MemoryDomain,
    pub src_page_ids: Vec<u32>,
    pub dst_page_ids: Vec<u32>,
    pub moves: Vec<KvMove>,
    /// Windowed kv pages `(src, dst)` moved in the copy's direction, in the
    /// windowed planes' own page ids: a ring's pages parking on the host.
    #[serde(default)]
    pub windowed: Vec<(u32, u32)>,
}

impl Default for KvCopy {
    fn default() -> KvCopy {
        KvCopy {
            src: MemoryDomain::HostPinned,
            dst: MemoryDomain::HostPinned,
            src_page_ids: Vec::new(),
            dst_page_ids: Vec::new(),
            moves: Vec::new(),
            windowed: Vec::new(),
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

    /// The copy over pages of `page_size` tokens as runs: one per whole
    /// page pair, and the per-token moves merged where each continues the
    /// last within one page. A run whose two ends overlap on one page is
    /// refused: a device copy of overlapping regions is undefined, not a
    /// shift.
    pub fn runs(&self, page_size: u32) -> crate::Result<Vec<KvRun>> {
        self.validate()?;
        let mut runs: Vec<KvRun> = self
            .src_page_ids
            .iter()
            .zip(&self.dst_page_ids)
            .map(|(&src_page, &dst_page)| KvRun {
                src_page,
                src_token: 0,
                dst_page,
                dst_token: 0,
                tokens: page_size,
            })
            .collect();
        for (at, cell) in self.moves.iter().enumerate() {
            if cell.src_token_offset >= page_size || cell.dst_token_offset >= page_size {
                return Err(crate::Error::Invalid(format!(
                    "kv move {at} names token offsets {}/{} in pages of {page_size} tokens",
                    cell.src_token_offset, cell.dst_token_offset
                )));
            }
            if cell.src_page_id == cell.dst_page_id
                && cell.src_token_offset == cell.dst_token_offset
            {
                continue;
            }
            let run = runs.last_mut().filter(|run| {
                run.src_page == cell.src_page_id
                    && run.dst_page == cell.dst_page_id
                    && run.src_token + run.tokens == cell.src_token_offset
                    && run.dst_token + run.tokens == cell.dst_token_offset
                    && run.src_token + run.tokens < page_size
            });
            match run {
                Some(run) => run.tokens += 1,
                None => runs.push(KvRun {
                    src_page: cell.src_page_id,
                    src_token: cell.src_token_offset,
                    dst_page: cell.dst_page_id,
                    dst_token: cell.dst_token_offset,
                    tokens: 1,
                }),
            }
        }
        if let Some(run) = runs.iter().find(|run| {
            run.src_page == run.dst_page && run.src_token.abs_diff(run.dst_token) < run.tokens
        }) {
            return Err(crate::Error::Invalid(format!(
                "a kv move of {} tokens reads page {} from token {} and writes the same page \
                 at token {} — the two ends overlap, and a device copy whose ends overlap is \
                 undefined rather than a shift",
                run.tokens, run.src_page, run.src_token, run.dst_token
            )));
        }
        Ok(runs)
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

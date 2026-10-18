use eta_ir::registry::{GeometryClass, ModelProfile, PortMask};
use serde::{Deserialize, Serialize};

use crate::transfer::{KvHandle, MemoryDomain};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceFacts {
    pub backend: String,
    pub domain: MemoryDomain,
    pub sms: u32,
    pub unified_memory: bool,
    pub fp8_native: bool,
    pub native_mxfp4_moe: bool,
    pub storage_alignment: u32,
    pub storage_max_tile_bytes: u64,
    pub codegen_backend: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCopyDomains {
    pub device_to_device: bool,
    pub device_to_host: bool,
    pub host_to_device: bool,
    pub host_to_host: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FireLimits {
    pub max_lanes: u32,
    pub max_tokens: u32,
    pub max_page_refs: u32,
    pub max_context: u32,
    /// The rows one prefill fire takes best, when the engine prefers fewer
    /// than `max_tokens` (0: no preference). A guest chunks a long prompt to
    /// it: on Metal, the Neural Engine takes its share of a fire only up to
    /// this many rows.
    #[serde(default)]
    pub prefill_rows: u32,
    /// 0: as many as `max_tokens`.
    #[serde(default)]
    pub max_readouts: u32,
    /// The lanes of one fire that may capture attention scores (0: every
    /// lane).
    #[serde(default)]
    pub max_score_lanes: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolFacts {
    pub kv_pages: u32,
    pub kv_page_size: u32,
    pub state_slots: u32,
    pub state_slot_bytes: u64,
    /// Buffer pages of `kv_page_size` unfolded rows, a pool beside the state
    /// slots.
    #[serde(default)]
    pub buffer_pages: u32,
    pub adapter_banks: u32,
    pub elastic_page_bytes: u64,
    pub elastic_budget_pages: u64,
    /// The windowed kv pool, page 0 its null page; zero when the model
    /// declares no windowed space or the engine pages its sliding rows in
    /// full. `window_tokens` is the model's fact: the window its sliding
    /// rows are read through, whether or not a pool backs them.
    #[serde(default)]
    pub window_pages: u32,
    #[serde(default)]
    pub window_tokens: u32,
    /// The tiers below the device: host pages kv can be suspended into by
    /// `kv_copy`, host windowed pages for a ring's pages, the state slots and
    /// buffer pages a `StateCopy` may park in (host memory, or slot files on
    /// an engine with no host tier), and slot-file pages `kv_copy` serves to
    /// and from `MemoryDomain::LocalDisk`. Zero means the engine serves no
    /// such copies.
    #[serde(default)]
    pub host_kv_pages: u32,
    #[serde(default)]
    pub host_window_pages: u32,
    #[serde(default)]
    pub host_state_slots: u32,
    #[serde(default)]
    pub host_buffer_pages: u32,
    #[serde(default)]
    pub disk_kv_pages: u32,
    /// The engine would rather post one frame a lane than seat fewer lanes
    /// than it batches when its state slots are short: the runtime then
    /// dispatches one frame deep, so a lane holds one slot. False keeps the
    /// configured depth whatever the slots seat.
    #[serde(default)]
    pub one_frame_when_short: bool,
    /// The engine would rather a lane already decoding claim its next pages
    /// before a prompt waiting to be seated does, and a prompt wait for a
    /// lane to finish rather than suspend one. False keeps the runtime's
    /// order of arrival, a waiting prompt older than a lane preempting it.
    #[serde(default)]
    pub seated_first: bool,
}

impl PoolFacts {
    /// Keeps only the tier capacity every rank of `ranks` has: a group moves
    /// what all of them can hold.
    pub fn least_tiers(&mut self, ranks: impl Iterator<Item = PoolFacts>) {
        for rank in ranks {
            self.host_kv_pages = self.host_kv_pages.min(rank.host_kv_pages);
            self.host_window_pages = self.host_window_pages.min(rank.host_window_pages);
            self.host_state_slots = self.host_state_slots.min(rank.host_state_slots);
            self.host_buffer_pages = self.host_buffer_pages.min(rank.host_buffer_pages);
            self.disk_kv_pages = self.disk_kv_pages.min(rank.disk_kv_pages);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Capabilities {
    pub device: DeviceFacts,
    pub pools: PoolFacts,
    pub limits: FireLimits,
    pub profile: ModelProfile,
    pub ports: PortMask,
    pub geometry: GeometryClass,
    pub kv_copy: KvCopyDomains,
    pub kv_handle: Option<KvHandle>,
    pub media_encode: bool,
    #[serde(default)]
    pub device_channel_commit: bool,

    #[serde(default)]
    pub rs_verbs: bool,

    /// The rows inside one lane whose recurrent state a fire lands in slots
    /// of their own as it runs (`Lane::rs_snapshots`); 0 where it lands only
    /// the state after the lane's last row, so a pass is cut at each.
    #[serde(default)]
    pub rs_snapshots: u32,

    #[serde(default)]
    pub bidirectional_attention: bool,

    #[serde(default)]
    pub adapter_attachments: bool,

    /// The facts the loaded trace's rows branch on, which a lane's word is
    /// classified by.
    #[serde(default)]
    pub facts: poem_ir::Facts,
}

impl Capabilities {
    #[must_use]
    pub fn admits(&self, wanted: GeometryClass) -> bool {
        self.ports.covers(wanted.ports())
    }
}

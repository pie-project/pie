use eta_ir::registry::{GeometryClass, PortMask};
use serde::{Deserialize, Serialize};

use crate::chunk::ChunkFacts;
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
    /// The most lanes a fire of prompts only (no lane decoding) carries,
    /// past which the engine holds no body for it (0: as many as
    /// `max_lanes`).
    #[serde(default)]
    pub prefill_lanes: u32,
    /// 0: as many as `max_tokens`.
    #[serde(default)]
    pub max_readouts: u32,
    /// The most recurrent-state rows one lane may hold unfolded (rows
    /// written past its fold: `first − folded`), what the engine sizes its
    /// rs scratch for. A pass that would leave a lane past it is refused
    /// before anything is submitted; the runtime never folds behind the
    /// guest's back. 0: unbounded (an engine that does not state it).
    #[serde(default)]
    pub max_unfolded_rows: u32,
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
    /// The window the model's sliding rows are read through; zero when it
    /// declares no windowed row.
    #[serde(default)]
    pub window_tokens: u32,
    /// The tiers below the device: host pages kv can be suspended into by
    /// `kv_copy`, and slot-file pages `kv_copy` serves to and from
    /// `MemoryDomain::LocalDisk`. Zero means the engine serves no such
    /// copies.
    #[serde(default)]
    pub host_kv_pages: u32,
    #[serde(default)]
    pub disk_kv_pages: u32,
    /// The slot files are kept across runs (`disk_kv_persistent`).
    #[serde(default)]
    pub disk_kv_persistent: bool,
    /// And what they held when the last run left is there still: they were
    /// found with the meta this load would write.
    #[serde(default)]
    pub disk_kv_kept: bool,
    /// The engine would rather a lane already decoding claim its next pages
    /// before a prompt waiting to be seated does, and a prompt wait for a
    /// lane to finish rather than suspend one. False keeps the runtime's
    /// order of arrival, a waiting prompt older than a lane preempting it.
    #[serde(default)]
    pub seated_first: bool,
    /// The engine would rather a process wait to start until the pools hold
    /// the started ones to the pages a finished one held, than start each at
    /// once and evict lanes the pools then run short for.
    #[serde(default)]
    pub seat_to_end: bool,
    /// With `seat_to_end`, the engine would rather the starts the pools do
    /// not hold to the end at once be spread over a life, so they finish
    /// apart, than a wave start together and run to its end together.
    #[serde(default)]
    pub seat_spread: bool,
}

impl PoolFacts {
    /// Keeps only the tier capacity every rank of `ranks` has: a group moves
    /// what all of them can hold.
    pub fn least_tiers(&mut self, ranks: impl Iterator<Item = PoolFacts>) {
        for rank in ranks {
            self.host_kv_pages = self.host_kv_pages.min(rank.host_kv_pages);
            self.disk_kv_pages = self.disk_kv_pages.min(rank.disk_kv_pages);
            self.disk_kv_persistent &= rank.disk_kv_persistent;
            self.disk_kv_kept &= rank.disk_kv_kept;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Capabilities {
    pub device: DeviceFacts,
    pub pools: PoolFacts,
    /// The pools as chunks.
    #[serde(default)]
    pub chunks: ChunkFacts,
    pub limits: FireLimits,
    pub ports: PortMask,
    pub geometry: GeometryClass,
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

    #[serde(default)]
    pub has_lora: bool,

    #[serde(default)]
    pub has_attn_page_mask: bool,

    #[serde(default)]
    pub draft_block: u32,
    #[serde(default)]
    pub draft_mask_token: u32,
    #[serde(default)]
    pub draft_proposals_from: u32,

    /// The facts the loaded trace's rows branch on, which a lane's word is
    /// classified by.
    #[serde(default)]
    pub facts: poem_ir::Facts,

    /// The guest's inputs by id, with the lane words that read each.
    #[serde(default)]
    pub inputs: Vec<poem_ir::InputDecl>,

    #[serde(default)]
    pub classes: poem_ir::LaneClasses,
}

impl Capabilities {
    #[must_use]
    pub fn admits(&self, wanted: GeometryClass) -> bool {
        self.ports.covers(wanted.ports())
    }
}

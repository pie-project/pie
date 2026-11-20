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
}

/// What a load states beside its chunk pool ([`ChunkFacts`], which holds
/// every count, size and tier of its state).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolFacts {
    pub adapter_banks: u32,
    /// The window the model's sliding rows are read through, a fact of the
    /// model: an engine that pages its sliding rows in full states no window
    /// list for it. Zero when the model declares no windowed row.
    #[serde(default)]
    pub window_tokens: u32,
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

    /// What a guest stage's cost is held to on this device
    /// ([`GuestCost`]): none stated, the default, holds it to nothing.
    #[serde(default)]
    pub guest_cost: GuestCost,
    /// The frames this engine keeps in flight, its run-ahead depth (at
    /// least 1): the one number the runtime dispatches frames by and sizes
    /// a beam group's pool with (`BEAM-PAGE-LISTS.md` §3's `F`). The load
    /// request's depth is what the runtime asks for; this is what the
    /// engine runs.
    pub frames_in_flight: u32,
}

/// What bounds a guest stage's time on a device: the bound it states on one
/// submission, past which the driver faults the submission (and with it
/// every tenant's fire in it), and the rates its load measured that cost a
/// stage. The runtime refuses a registration whose stage the rates cost past
/// the bound (`eta_ir::validate::SubmitBound`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct GuestCost {
    /// The device's stated bound on one submission, ns; `None` where it
    /// states none.
    #[serde(default)]
    pub submit_bound_ns: Option<u64>,
    /// What the load measured one launch takes, ns (`[rates] launch_ns`).
    #[serde(default)]
    pub launch_ns: Option<u64>,
    /// What the load measured an elementwise pass moves, bytes a ns
    /// (`[rates] elementwise_gbps`).
    #[serde(default)]
    pub elementwise_gbps: Option<f64>,
}

impl Capabilities {
    #[must_use]
    pub fn admits(&self, wanted: GeometryClass) -> bool {
        self.ports.covers(wanted.ports())
    }
}

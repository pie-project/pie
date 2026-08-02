//! Owned driver verb plans shared by local and remote backends.
//!
//! These are process-independent values. Borrowed pointers and completion cells
//! stay in the runtime's local submission layer.

use serde::{Deserialize, Serialize};

use crate::{
    PIE_MEMORY_DOMAIN_HOST_PINNED, PieKvMoveCell, PieMemoryDomain, PiePoolRange, PieStateCopyRange,
};

pub const CHANNEL_TICKET_NONE: u64 = u64::MAX;

/// Binary run-length encoded attention-mask row.
///
/// Even run indices are false and odd run indices are true. A row beginning
/// with true therefore starts with a zero-length false run.
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EncodedMask {
    pub runs: Vec<u32>,
    pub total_size: u64,
}

impl EncodedMask {
    pub fn new(runs: Vec<u32>, total_size: u64) -> Self {
        Self { runs, total_size }
    }

    pub fn len(&self) -> usize {
        self.total_size as usize
    }

    pub fn is_empty(&self) -> bool {
        self.total_size == 0
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchPlan {
    pub token_ids: Vec<u32>,
    pub position_ids: Vec<u32>,
    pub kv_page_indices: Vec<u32>,
    pub kv_page_indptr: Vec<u32>,
    pub kv_last_page_lens: Vec<u32>,
    pub qo_indptr: Vec<u32>,
    pub rs_slot_ids: Vec<u32>,
    pub rs_slot_flags: Vec<u8>,
    pub rs_fold_lens: Vec<u32>,
    pub rs_buffer_slot_ids: Vec<u32>,
    pub rs_buffer_slot_indptr: Vec<u32>,
    /// The buffered prefix each row REPLAYS ahead of its own tokens: the slabs
    /// (`rs_buffer_read_slot_ids`), the row CSR over them
    /// (`rs_buffer_read_indptr`, `R + 1`), and how many tokens of them to
    /// replay (`rs_buffer_read_lens`, `R`). Empty when nothing is buffered.
    pub rs_buffer_read_slot_ids: Vec<u32>,
    pub rs_buffer_read_indptr: Vec<u32>,
    pub rs_buffer_read_lens: Vec<u32>,
    /// Physical offset of each row's logical buffer token 0 (`R`). A fold that
    /// lands mid-page cannot release the page it half-consumed, so the
    /// survivors keep their offsets and every buffer span is `head + logical`.
    pub rs_buffer_heads: Vec<u32>,
    /// WorkingSet-relative buffer page -> physical slot, for channel-resolved
    /// `rs-geometry`. Per REQUEST ROW, unlike [`Self::kv_translation`]: a pass
    /// binds one RS working set per request, so there is no single table for
    /// the fire. `rs_translation_indptr` is the row CSR, `R + 1` entries.
    pub rs_translation: Vec<u32>,
    pub rs_translation_indptr: Vec<u32>,
    pub masks: Vec<EncodedMask>,
    pub mask_indptr: Vec<u32>,
    pub sampling_indices: Vec<u32>,
    pub sampling_indptr: Vec<u32>,
    pub context_ids: Vec<u64>,
    pub single_token_mode: bool,
    pub device_resolved_geometry: bool,
    pub has_user_mask: bool,
    /// tart STRUCTURAL v0/S-2 (0.3 re-port): the pass's layer truncation
    /// (`set-max-layers`). Engine-internal — it crosses the driver ABI as
    /// the region table's per-region k, never as a scalar word.
    pub max_layers: Option<u32>,
    /// The program's hook stages write the `attn_page_mask` sink (Track B
    /// page substitution). Engine-internal, same precedent as
    /// `max_layers`: the admission gate reads it — a page-mask hook needs
    /// the full-R paged decode path, so it cannot ride the banded walk
    /// and its group stays depth-homogeneous.
    #[serde(default)]
    pub hook_page_mask: bool,
    /// The program binds an `AttnMask` descriptor port to a CHANNEL, so the
    /// driver resolves a dense per-cell mask pre-forward. Such a fire must be
    /// submitted SOLO: a multi-program batch cannot merge one program's dense
    /// mask with another's geometry, and the driver fails loud rather than
    /// execute a wrong one.
    #[serde(default)]
    pub dense_device_mask: bool,
    /// Exclusive physical KV page high-water required before this launch.
    #[serde(default)]
    pub required_kv_pages: u32,
    pub image_indptr: Vec<u32>,
    pub image_grids: Vec<u32>,
    pub image_anchor_positions: Vec<u32>,
    pub image_pixels: Vec<u8>,
    pub image_pixel_indptr: Vec<u32>,
    pub image_mrope_positions: Vec<u32>,
    pub image_mrope_indptr: Vec<u32>,
    pub image_patch_positions: Vec<u32>,
    pub image_anchor_rows: Vec<u32>,
    pub audio_features: Vec<u8>,
    pub audio_feature_indptr: Vec<u32>,
    pub audio_anchor_rows: Vec<u32>,
    pub audio_indptr: Vec<u32>,
    pub embed_rows: Vec<u8>,
    pub embed_indptr: Vec<u32>,
    pub embed_shapes: Vec<u32>,
    pub embed_dtypes: Vec<u8>,
    pub embed_anchor_rows: Vec<u32>,
    pub embed_block_indptr: Vec<u32>,
    pub kv_len: Vec<u32>,
    pub kv_len_device: Vec<u64>,
    pub kv_translation: Vec<u32>,
    pub kv_write_lower_bounds: Vec<u64>,
    pub kv_write_upper_bounds: Vec<u64>,
    pub kv_translation_version: u64,
    pub channel_expected_head: Vec<u64>,
    pub channel_expected_tail: Vec<u64>,
}

pub const RS_FLAG_RESET: u8 = 1;
pub const RS_FLAG_FOLD: u8 = 2;
pub const RS_FLAG_BUFFER_WRITE: u8 = 4;
/// This row's fold length is not host-known; it comes from the `rs_fold_len`
/// descriptor port and `rs_fold_lens[r]` is a placeholder.
pub const RS_FLAG_FOLD_LEN_DEVICE: u8 = 8;

#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgramRegistration {
    pub program_hash: u64,
    /// Kernels the host generated for this driver's backend, empty unless the
    /// driver advertised a
    /// [`codegen_backend`](crate::capabilities::DriverCapabilities::codegen_backend).
    #[serde(default)]
    pub emitted_kernels: Vec<EmittedKernel>,
    /// The emitter version `emitted_kernels` was built with; part of the
    /// driver's compile-cache key, so a bump must miss rather than reuse.
    #[serde(default)]
    pub emitter_version: u32,
    /// Per-region bind verdicts and intrinsic side-table analysis, joined to
    /// `emitted_kernels` on `(stage_index, region_index)`.
    #[serde(default)]
    pub region_analysis: Vec<RegionAnalysis>,
    /// The program itself, in the shape a driver executes it.
    #[serde(default)]
    pub launch: LaunchPackage,
    /// The canonical PTIR container, for the in-workspace reference driver
    /// only.
    ///
    /// `pie-driver-dummy` is a Rust crate that links the compiler's own IR and
    /// interpreter, so it cannot drift from them the way a hand-written C++
    /// mirror can. It is the one consumer that still wants PTIR.
    ///
    /// This is deliberately **not** part of the C ABI: [`PieProgramDesc`] has
    /// no counterpart field, so a native driver cannot see PTIR even by
    /// accident.
    ///
    /// [`PieProgramDesc`]: crate::local::PieProgramDesc
    #[serde(default)]
    pub reference_ptir: Vec<u8>,
}

/// **The launch package** — the owned counterpart of
/// [`PieLaunchPackage`](crate::local::PieLaunchPackage).
///
/// This is what a driver receives instead of PTIR. `stages` and `plans` are
/// parallel arrays in attachment order.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchPackage {
    pub values: Vec<LaunchValue>,
    pub channels: Vec<LaunchChannel>,
    pub ports: Vec<LaunchPort>,
    /// Program-wide name table for second-party kernels and sinks.
    pub names: Vec<String>,
    pub stages: Vec<LaunchStage>,
    pub plans: Vec<LaunchStagePlan>,
}

/// One declared SSA value. Owned counterpart of
/// [`PieLaunchValue`](crate::local::PieLaunchValue).
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchValue {
    pub id: u32,
    /// `PIE_VALUE_*`.
    pub source: u8,
    pub dtype: u8,
    /// `PTIR_INTR_*` when `source` is `PIE_VALUE_INTRINSIC`.
    pub intrinsic: u8,
    pub channel: u32,
    pub literal_bits: u32,
    pub shape: Vec<u32>,
}

/// One op in a stage DAG. Owned counterpart of
/// [`PieLaunchOp`](crate::local::PieLaunchOp).
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchOp {
    /// `PTIR_OP_*`.
    pub code: u16,
    pub result_count: u16,
    pub result_id: u32,
    /// `PTIR_INTR_*`, for `intrinsic_val`.
    pub intrinsic: u16,
    pub lit_dtype: u8,
    pub dtype: u8,
    pub pred_tag: u8,
    pub rng_kind: u8,
    pub lit_bits: u32,
    pub pred_payload: u32,
    /// Channel slot, or `u32::MAX` when the op touches no channel.
    pub channel: u32,
    pub name_index: u32,
    pub imm: u32,
    pub imm2: u32,
    pub imm3: u32,
    pub args: Vec<u32>,
    pub shape: Vec<u32>,
}

/// One channel declaration. Owned counterpart of
/// [`PieLaunchChannel`](crate::local::PieLaunchChannel).
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchChannel {
    pub id: u32,
    pub capacity: u32,
    pub dtype: u8,
    /// `PIE_CHANNEL_*` bits.
    pub flags: u8,
    /// -1 private, 0 import, 1 export.
    pub extern_dir: i8,
    /// `PIE_READINESS_*` — the direction this channel's first op requires.
    pub readiness: u8,
    pub shape: Vec<u32>,
    pub extern_name: Vec<u8>,
}

/// One descriptor-port binding. Owned counterpart of
/// [`PieLaunchPort`](crate::local::PieLaunchPort).
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchPort {
    /// `PTIR_PORT_*`.
    pub port: u8,
    pub is_const: bool,
    pub const_dtype: u8,
    pub channel: u32,
    pub const_shape: Vec<u32>,
    pub const_data: Vec<u8>,
}

/// A `(channel, value)` pair. Owned counterpart of
/// [`PieLaunchPut`](crate::local::PieLaunchPut).
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchPut {
    pub channel: u32,
    pub value: u32,
}

/// One stage program. Owned counterpart of
/// [`PieLaunchStage`](crate::local::PieLaunchStage).
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchStage {
    /// Prologue 0, OnAttnProj 1, OnAttn 2, Epilogue 3.
    pub kind: u8,
    pub ops: Vec<LaunchOp>,
    pub puts: Vec<LaunchPut>,
    pub takes: Vec<u32>,
    pub reads: Vec<u32>,
}

/// One region. Owned counterpart of
/// [`PieLaunchRegion`](crate::local::PieLaunchRegion).
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchRegion {
    /// `PIE_REGION_GENERATED` or `PIE_REGION_LIBRARY`.
    pub kind: u8,
    pub library: u8,
    pub schedule: u8,
    pub nodes: Vec<u32>,
    pub inputs: Vec<u32>,
    pub outputs: Vec<u32>,
    pub sinks: Vec<LaunchPut>,
}

/// One normalized value type. Owned counterpart of
/// [`PieLaunchPlanValue`](crate::local::PieLaunchPlanValue).
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchPlanValue {
    pub dtype: u8,
    /// Per-dimension extent kind: `PIE_EXTENT_STATIC` or a runtime extent.
    pub extents: Vec<u8>,
    /// Per-dimension literal extent, meaningful where `extents` is static.
    pub dims: Vec<u32>,
}

/// One lane-binding rule. Owned counterpart of
/// [`PieLaunchChannelRule`](crate::local::PieLaunchChannelRule).
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchChannelRule {
    pub value: u32,
    pub local: u32,
}

/// The per-program launch plan for one stage. Owned counterpart of
/// [`PieLaunchStagePlan`](crate::local::PieLaunchStagePlan).
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchStagePlan {
    pub signature_hash: u64,
    /// Graph-cache identity (`pie_plan::stage_identity`).
    pub identity: u64,
    /// `PIE_STAGE_REQUIRES_*` bits.
    pub flags: u32,
    pub mtp_rows: u32,
    pub ops: Vec<LaunchOp>,
    /// Source op positions each normalized op covers.
    pub source_ops: Vec<Vec<u32>>,
    pub value_types: Vec<LaunchPlanValue>,
    /// Local channel slot → program-global dense channel index.
    pub channel_bindings: Vec<u32>,
    /// Local name slot → canonical second-party kernel name.
    pub names: Vec<String>,
    pub singleton: Vec<LaunchRegion>,
    pub fused: Vec<LaunchRegion>,
    /// Runtime extents any value in the stage depends on, ascending.
    pub used_extents: Vec<u8>,
    pub channel_rules: Vec<LaunchChannelRule>,
    /// Why `PIE_STAGE_GROUPED_VALID` is clear. Empty when it is set.
    pub error: String,
}

/// Every per-region decision the host made. The owned counterpart of
/// [`PieRegionAnalysis`](crate::local::PieRegionAnalysis).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionAnalysis {
    pub stage_index: u32,
    pub region_index: u32,
    /// `PIE_REGION_*` bits.
    pub flags: u32,
    pub direct_argmax: Vec<DirectArgmax>,
    /// Nodes the rewrites make redundant, ascending.
    pub skipped: Vec<u32>,
}

/// One `argmax` that reads a logits intrinsic's buffer directly. The owned
/// counterpart of [`PieDirectArgmax`](crate::local::PieDirectArgmax).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectArgmax {
    pub node: u32,
    pub source_value: u32,
    pub intrinsic: u16,
    pub requires_single_row: u8,
}

/// One host-emitted kernel. The owned counterpart of
/// [`PieEmittedKernel`](crate::local::PieEmittedKernel).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmittedKernel {
    /// `PIE_KERNEL_*`.
    pub kind: u32,
    pub stage_index: u32,
    pub region_index: u32,
    /// Entry-point symbol; empty when emission failed.
    pub entry_name: String,
    /// Backend source; empty when emission failed.
    pub source: String,
    /// Why emission failed. Empty on success.
    ///
    /// A failure is not necessarily fatal: a driver may have a slower path for
    /// the same region, and recording *why* is what lets it tell a deliberate
    /// fallback from a bug.
    pub error: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelRegistrationPlan {
    pub driver_id: usize,
    pub channel_id: u64,
    pub shape: Vec<u32>,
    pub dtype: u8,
    pub host_role: u8,
    pub seeded: bool,
    pub extern_dir: u8,
    pub capacity: u32,
    pub reader_wait_id: u64,
    pub writer_wait_id: u64,
    pub extern_name: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCopyPlan {
    pub src_domain: PieMemoryDomain,
    pub src_device_ordinal: u32,
    pub dst_domain: PieMemoryDomain,
    pub dst_device_ordinal: u32,
    pub src_page_ids: Vec<u32>,
    pub dst_page_ids: Vec<u32>,
    pub cells: Vec<PieKvMoveCell>,
}

impl Default for KvCopyPlan {
    fn default() -> Self {
        Self {
            src_domain: PIE_MEMORY_DOMAIN_HOST_PINNED,
            src_device_ordinal: 0,
            dst_domain: PIE_MEMORY_DOMAIN_HOST_PINNED,
            dst_device_ordinal: 0,
            src_page_ids: Vec::new(),
            dst_page_ids: Vec::new(),
            cells: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct StateCopyPlan {
    pub slot_ranges: Vec<PieStateCopyRange>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MediaEncodePlan {
    pub image_grids: Vec<u32>,
    pub image_pixels: Vec<u8>,
    pub image_pixel_indptr: Vec<u32>,
    pub image_patch_positions: Vec<u32>,
    pub image_anchor_rows: Vec<u32>,
    pub audio_features: Vec<u8>,
    pub audio_feature_indptr: Vec<u32>,
    pub audio_anchor_rows: Vec<u32>,
    pub output_rows: Vec<u8>,
    pub output_row_indptr: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PoolResizePlan {
    pub pool_id: u64,
    pub target_pages: u64,
    pub map_ranges: Vec<PiePoolRange>,
    pub unmap_ranges: Vec<PiePoolRange>,
}

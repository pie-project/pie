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
    pub masks: Vec<EncodedMask>,
    pub mask_indptr: Vec<u32>,
    pub sampling_indices: Vec<u32>,
    pub sampling_indptr: Vec<u32>,
    pub context_ids: Vec<u64>,
    pub single_token_mode: bool,
    pub device_resolved_geometry: bool,
    pub has_user_mask: bool,
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

#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgramRegistration {
    pub program_hash: u64,
    pub canonical_bytes: Vec<u8>,
    pub sidecar_bytes: Vec<u8>,
    /// Kernels the host generated for this driver's backend, empty unless the
    /// driver advertised a
    /// [`codegen_backend`](crate::capabilities::DriverCapabilities::codegen_backend).
    #[serde(default)]
    pub emitted_kernels: Vec<EmittedKernel>,
    /// The emitter version `emitted_kernels` was built with; part of the
    /// driver's compile-cache key, so a bump must miss rather than reuse.
    #[serde(default)]
    pub emitter_version: u32,
    /// Each stage's graph-cache identity, in plan order
    /// (`pie_plan::stage_identity`).
    ///
    /// The first field of the launch package: a decision the CUDA driver
    /// currently re-derives from the plan. Empty means "not supplied", so a
    /// driver that receives nothing keeps deriving.
    #[serde(default)]
    pub stage_identities: Vec<u64>,
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

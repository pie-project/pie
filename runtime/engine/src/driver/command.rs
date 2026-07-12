//! Runtime-owned command/plan types dispatched to a driver backend: the
//! per-verb payloads (`LaunchPlan`, `ProgramRegistration`,
//! `ChannelRegistrationPlan`, `KvCopyPlan`, `StateCopyPlan`,
//! `PoolResizePlan`) that [`super::abi`] marshals into the zero-copy ABI
//! descriptors a backend consumes. No IR semantics live here — these are
//! the driver-facing plans the scheduler dispatch facade builds and hands
//! to [`super::backend::DriverBackend`].

use pie_driver_abi::{
    PIE_MEMORY_DOMAIN_HOST_PINNED, PieKvMoveCell, PieMemoryDomain, PiePoolRange, PieStateCopyRange,
};

use pie_grammar::brle::RunMask;

pub const CHANNEL_TICKET_NONE: u64 = u64::MAX;

#[derive(Default, Debug, Clone, PartialEq)]
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
    pub masks: Vec<RunMask>,
    pub mask_indptr: Vec<u32>,
    pub sampling_indices: Vec<u32>,
    pub sampling_indptr: Vec<u32>,
    pub context_ids: Vec<u64>,
    pub single_token_mode: bool,
    pub has_user_mask: bool,
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
    pub kv_len: Vec<u32>,
    pub kv_len_device: Vec<u64>,
    /// This fire's WorkingSet page translation: entry `i` = the PHYSICAL KV
    /// page id backing WorkingSet-relative index `i` (committed mapping
    /// overlaid with the prepared write targets). The driver maps channel-
    /// resolved `Pages`/`WSlot` references through it; empty = no
    /// WorkingSet-relative geometry in this fire.
    pub kv_translation: Vec<u32>,
    /// Store mapping version that produced `kv_translation`. The direct ABI
    /// ships the complete translation every launch (drivers keep no mapping
    /// cache), but retaining the version here makes remap publication
    /// observable and prevents the runtime from silently discarding it.
    pub kv_translation_version: u64,
    /// Dense-channel-order immutable sequence tickets. `u64::MAX` means this
    /// fire does not consume/publish that channel.
    pub channel_expected_head: Vec<u64>,
    pub channel_expected_tail: Vec<u64>,
}

pub const RS_FLAG_RESET: u8 = 1;
pub const RS_FLAG_FOLD: u8 = 2;

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct ProgramRegistration {
    pub program_hash: u64,
    pub canonical_bytes: Vec<u8>,
    pub sidecar_bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

#[derive(Debug, Clone, PartialEq, Eq)]
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

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StateCopyPlan {
    pub slot_ranges: Vec<PieStateCopyRange>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PoolResizePlan {
    pub pool_id: u64,
    pub target_pages: u64,
    pub map_ranges: Vec<PiePoolRange>,
    pub unmap_ranges: Vec<PiePoolRange>,
}

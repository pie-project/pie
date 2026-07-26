//! The lane-table ABI: the structs the host fills and the kernels read.
//!
//! Declarations, not a compilation pass, which is why they sit at the crate
//! root rather than under `compile/`. Nothing here reads a stage or produces a
//! plan; every consumer is downstream — `pie_codegen::layout`, `header`, and
//! the two backends' emitters.
//!
//! The field layout is mirrored into the generated C header and both MSL
//! preambles by `pie_codegen::layout`, which pins these structs with
//! `offset_of!` so the copies cannot drift.

pub const LANE_TABLE_ABI_VERSION: u32 = 3;

/// Runtime extents never enter a stage signature.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RuntimeExtents {
    pub kv_len: u32,
    pub page_count: u32,
    pub row_count: u32,
    pub token_count: u32,
    pub sampled_rows: u32,
    pub query_len: u32,
    pub key_len: u32,
}

// `ExecutableCacheKey` used to live here: a struct of `{backend, device_arch,
// compiler_version, stage_signature, schedule_bucket, semantic_mode}` with an
// `encode() -> [u8; 22]`, described as "complete executable-cache identity".
//
// It was deleted because it was not one. Nothing called it -- not this crate,
// not codegen, not the drivers -- and it did not describe the cache it claimed
// to key. The CUDA driver builds its own key in
// `driver/cuda/src/pipeline/generated/module_cache.hpp` (`stage_cache_key`),
// and that key is a different shape in both directions: it adds
// `PTIR_REGION_PLAN_VERSION`, `PTIR_LANE_TABLE_ABI_VERSION`, the generated
// emitter version, the NVRTC major/minor, and the normalized op bytes
// themselves; it carries neither `schedule_bucket` nor `semantic_mode`. The
// two were never the same key, so there was no drift to repair -- only a
// description of a cache nobody had built.
//
// Worth recording, because it is the reason this file no longer worries about
// 64-bit signature collisions: the driver's key is the full material as a
// `std::string` map key, compared byte for byte, and `pipeline::program`'s
// registry likewise re-compares the container bytes on a hash hit and returns
// `RegisterError::HashCollision`. The FNV-1a hash is an index into those
// caches, never the identity.
//
// If a Rust-side key is ever wanted again, it has to be generated into the
// driver's header the way the op table now is -- otherwise it is a comment
// that compiles.

/// Stable grouped-dispatch header. Address fields in the lane records are
/// device virtual addresses represented as `u64` on both supported backends.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct LaneTableHeader {
    pub abi_version: u32,
    pub lane_count: u32,
    pub channel_slots_per_lane: u32,
    pub flags: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct LaneRecord {
    pub logits_base: u64,
    pub logits_row_offset: u32,
    pub logits_row_count: u32,
    pub kv_len: u32,
    pub page_count: u32,
    pub row_count: u32,
    pub token_count: u32,
    pub sampled_rows: u32,
    pub query_len: u32,
    pub key_len: u32,
    pub channel_slot_offset: u32,
    pub rng_state: u64,
    pub commit_slot: u64,
    /// Optional device bitset for active rows in a ragged lane; zero means all
    /// `logits_row_count` rows are active.
    pub active_row_mask: u64,
    /// Stage-local channel bits whose puts publish the sampled token value.
    pub sample_output_channel_mask: u64,
    /// Optional device byte mask for model rows. `row_valid_offset` selects the
    /// first row belonging to this program.
    pub row_valid: u64,
    pub row_valid_offset: u32,
    pub reserved0: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct LaneChannelSlot {
    pub committed_cell: u64,
    pub pending_cell: u64,
    pub expected_head: u64,
    pub expected_tail: u64,
}

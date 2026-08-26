//! The offset tables a plan build produces: every field is either a launch
//! shape fact or a byte offset into the int/float workspace the schedule was
//! staged for. These are the payloads the plan structs carry across the
//! prepare/capture boundary (design §6) — pure data, device-pointer-free.

/// The fa2 decode schedule's workspace map.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecodePlanInfo {
    pub padded_batch_size: i64,
    pub v_offset: i64,
    pub s_offset: i64,
    pub request_indices_offset: i64,
    pub kv_tile_indices_offset: i64,
    pub o_indptr_offset: i64,
    pub block_valid_mask_offset: i64,
    pub kv_chunk_size_ptr_offset: i64,
    pub enable_cuda_graph: bool,
    pub split_kv: bool,
}

/// The fa2 prefill schedule's workspace map.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefillPlanInfo {
    pub padded_batch_size: i64,
    pub total_num_rows: i64,
    pub total_num_rows_offset: i64,
    pub cta_tile_q: i64,
    pub request_indices_offset: i64,
    pub qo_tile_indices_offset: i64,
    pub kv_tile_indices_offset: i64,
    pub merge_indptr_offset: i64,
    pub o_indptr_offset: i64,
    pub kv_chunk_size_ptr_offset: i64,
    pub v_offset: i64,
    pub s_offset: i64,
    pub block_valid_mask_offset: i64,
    pub enable_cuda_graph: bool,
    pub split_kv: bool,
}

/// The sm90 prefill schedule's workspace map. The builder exists so the
/// `AttnPrefillPlanSm90` struct kind has an honest payload; the sm90
/// launcher itself was never part of this lattice (see `attn::prefill_sm90`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefillPlanSm90Info {
    pub qo_tile_indices_offset: i64,
    pub qo_indptr_offset: i64,
    pub kv_indptr_offset: i64,
    pub qo_len_offset: i64,
    pub kv_len_offset: i64,
    pub head_indices_offset: i64,
    pub work_indptr_offset: i64,
    pub batch_indices_offset: i64,
    pub same_schedule_for_all_heads: bool,
}

/// The latent-attention schedule's workspace map.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MlaPlanInfo {
    pub num_blks_x: i64,
    pub num_blks_y: i64,
    pub q_indptr_offset: i64,
    pub kv_indptr_offset: i64,
    pub partial_indptr_offset: i64,
    pub merge_packed_offset_start_offset: i64,
    pub merge_packed_offset_end_offset: i64,
    pub merge_partial_packed_offset_start_offset: i64,
    pub merge_partial_packed_offset_end_offset: i64,
    pub merge_partial_stride_offset: i64,
    pub q_len_offset: i64,
    pub kv_len_offset: i64,
    pub q_start_offset: i64,
    pub kv_start_offset: i64,
    pub kv_end_offset: i64,
    pub work_indptr_offset: i64,
    pub partial_o_offset: i64,
    pub partial_lse_offset: i64,
}

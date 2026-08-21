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

impl DecodePlanInfo {
    #[must_use]
    pub const fn to_vector(&self) -> [i64; 10] {
        [
            self.padded_batch_size,
            self.v_offset,
            self.s_offset,
            self.request_indices_offset,
            self.kv_tile_indices_offset,
            self.o_indptr_offset,
            self.block_valid_mask_offset,
            self.kv_chunk_size_ptr_offset,
            self.enable_cuda_graph as i64,
            self.split_kv as i64,
        ]
    }
}

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

impl PrefillPlanInfo {
    #[must_use]
    pub const fn to_vector(&self) -> [i64; 15] {
        [
            self.padded_batch_size,
            self.total_num_rows,
            self.total_num_rows_offset,
            self.cta_tile_q,
            self.request_indices_offset,
            self.qo_tile_indices_offset,
            self.kv_tile_indices_offset,
            self.merge_indptr_offset,
            self.o_indptr_offset,
            self.kv_chunk_size_ptr_offset,
            self.v_offset,
            self.s_offset,
            self.block_valid_mask_offset,
            self.enable_cuda_graph as i64,
            self.split_kv as i64,
        ]
    }
}

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

impl PrefillPlanSm90Info {
    #[must_use]
    pub const fn to_vector(&self) -> [i64; 9] {
        [
            self.qo_tile_indices_offset,
            self.qo_indptr_offset,
            self.kv_indptr_offset,
            self.qo_len_offset,
            self.kv_len_offset,
            self.head_indices_offset,
            self.work_indptr_offset,
            self.batch_indices_offset,
            self.same_schedule_for_all_heads as i64,
        ]
    }
}

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

impl MlaPlanInfo {
    #[must_use]
    pub const fn to_vector(&self) -> [i64; 18] {
        [
            self.num_blks_x,
            self.num_blks_y,
            self.q_indptr_offset,
            self.kv_indptr_offset,
            self.partial_indptr_offset,
            self.merge_packed_offset_start_offset,
            self.merge_packed_offset_end_offset,
            self.merge_partial_packed_offset_start_offset,
            self.merge_partial_packed_offset_end_offset,
            self.merge_partial_stride_offset,
            self.q_len_offset,
            self.kv_len_offset,
            self.q_start_offset,
            self.kv_start_offset,
            self.kv_end_offset,
            self.work_indptr_offset,
            self.partial_o_offset,
            self.partial_lse_offset,
        ]
    }
}

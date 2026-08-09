use core::mem::{align_of, offset_of, size_of};

/// `flashinfer::DecodePlanInfo` — the batch-decode descriptor.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecodePlanInfo {
    /// Work items the grid is sized for: `new_batch_size`, or under CUDA graphs
    pub padded_batch_size: i64,
    /// Partial outputs, in the float workspace. Carved only when splitting.
    pub v_offset: i64,
    /// Partial LSEs, in the float workspace. Carved only when splitting.
    pub s_offset: i64,
    /// `request_indices[work]` — which request each work item serves.
    pub request_indices_offset: i64,
    /// `kv_tile_indices[work]` — which KV chunk of it.
    pub kv_tile_indices_offset: i64,
    /// `o_indptr[request]` — where a request's work items start.
    pub o_indptr_offset: i64,
    /// `block_valid_mask[work]` — false for the padding a CUDA graph's fixed
    pub block_valid_mask_offset: i64,
    /// A single `IdType`: the KV chunk size **in tokens** (`pages * page_size`).
    pub kv_chunk_size_ptr_offset: i64,
    /// Whether the plan was built for graph capture, and so has a fixed grid.
    pub enable_cuda_graph: bool,
    /// Whether KV is partitioned across work items — the flag that says the
    pub split_kv: bool,
}

const _: () = assert!(size_of::<DecodePlanInfo>() == 72);
const _: () = assert!(align_of::<DecodePlanInfo>() == 8);
const _: () = assert!(offset_of!(DecodePlanInfo, padded_batch_size) == 0);
const _: () = assert!(offset_of!(DecodePlanInfo, v_offset) == 8);
const _: () = assert!(offset_of!(DecodePlanInfo, s_offset) == 16);
const _: () = assert!(offset_of!(DecodePlanInfo, request_indices_offset) == 24);
const _: () = assert!(offset_of!(DecodePlanInfo, kv_tile_indices_offset) == 32);
const _: () = assert!(offset_of!(DecodePlanInfo, o_indptr_offset) == 40);
const _: () = assert!(offset_of!(DecodePlanInfo, block_valid_mask_offset) == 48);
const _: () = assert!(offset_of!(DecodePlanInfo, kv_chunk_size_ptr_offset) == 56);
const _: () = assert!(offset_of!(DecodePlanInfo, enable_cuda_graph) == 64);
const _: () = assert!(offset_of!(DecodePlanInfo, split_kv) == 65);

impl DecodePlanInfo {
    /// `ToVector()` — ten `int64_t`, bools last, in the order upstream fixed.
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

/// `flashinfer::PrefillPlanInfo` — the batch-prefill (FA2) descriptor.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefillPlanInfo {
    /// Work items the grid is sized for.
    pub padded_batch_size: i64,
    /// Total QO rows in the batch — the merge's outer dimension.
    pub total_num_rows: i64,
    /// A single `uint32_t` holding `qo_indptr[batch_size]`, carved only under
    pub total_num_rows_offset: i64,
    /// The QO tile width the whole plan was built around; see
    pub cta_tile_q: i64,
    /// `request_indices[work]`.
    pub request_indices_offset: i64,
    /// `qo_tile_indices[work]`.
    pub qo_tile_indices_offset: i64,
    /// `kv_tile_indices[work]`.
    pub kv_tile_indices_offset: i64,
    /// `merge_indptr[row]` — where a row's partial outputs start. Split only.
    pub merge_indptr_offset: i64,
    /// `o_indptr[request]`.
    pub o_indptr_offset: i64,
    /// A single `IdType`: the KV chunk size in tokens.
    pub kv_chunk_size_ptr_offset: i64,
    /// Partial outputs, in the float workspace. Split only.
    pub v_offset: i64,
    /// Partial LSEs, in the float workspace. Split only.
    pub s_offset: i64,
    /// `block_valid_mask[work]`. Split only.
    pub block_valid_mask_offset: i64,
    /// Whether the plan was built for graph capture.
    pub enable_cuda_graph: bool,
    /// Whether KV is partitioned across work items.
    pub split_kv: bool,
}

const _: () = assert!(size_of::<PrefillPlanInfo>() == 112);
const _: () = assert!(align_of::<PrefillPlanInfo>() == 8);
const _: () = assert!(offset_of!(PrefillPlanInfo, padded_batch_size) == 0);
const _: () = assert!(offset_of!(PrefillPlanInfo, total_num_rows) == 8);
const _: () = assert!(offset_of!(PrefillPlanInfo, total_num_rows_offset) == 16);
const _: () = assert!(offset_of!(PrefillPlanInfo, cta_tile_q) == 24);
const _: () = assert!(offset_of!(PrefillPlanInfo, request_indices_offset) == 32);
const _: () = assert!(offset_of!(PrefillPlanInfo, qo_tile_indices_offset) == 40);
const _: () = assert!(offset_of!(PrefillPlanInfo, kv_tile_indices_offset) == 48);
const _: () = assert!(offset_of!(PrefillPlanInfo, merge_indptr_offset) == 56);
const _: () = assert!(offset_of!(PrefillPlanInfo, o_indptr_offset) == 64);
const _: () = assert!(offset_of!(PrefillPlanInfo, kv_chunk_size_ptr_offset) == 72);
const _: () = assert!(offset_of!(PrefillPlanInfo, v_offset) == 80);
const _: () = assert!(offset_of!(PrefillPlanInfo, s_offset) == 88);
const _: () = assert!(offset_of!(PrefillPlanInfo, block_valid_mask_offset) == 96);
const _: () = assert!(offset_of!(PrefillPlanInfo, enable_cuda_graph) == 104);
const _: () = assert!(offset_of!(PrefillPlanInfo, split_kv) == 105);

impl PrefillPlanInfo {
    /// `ToVector()` — fifteen `int64_t`.
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

/// `flashinfer::PrefillPlanSM90Info` — the FA3 (Hopper) prefill descriptor.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefillPlanSm90Info {
    /// `qo_tile_indices[work]`.
    pub qo_tile_indices_offset: i64,
    /// `qo_indptr[work]` — the request's QO base, copied per work item rather
    pub qo_indptr_offset: i64,
    /// `kv_indptr[work]`.
    pub kv_indptr_offset: i64,
    /// `qo_len[work]`.
    pub qo_len_offset: i64,
    /// `kv_len[work]`.
    pub kv_len_offset: i64,
    /// `head_indices[work]` — which QO head, when the schedule is per-head.
    pub head_indices_offset: i64,
    /// `work_indptr[cta]` — `num_sm + 1` entries delimiting each CTA's list.
    pub work_indptr_offset: i64,
    /// `batch_indices[work]`.
    pub batch_indices_offset: i64,
    /// Whether one schedule is shared by all QO heads, which is what the
    pub same_schedule_for_all_heads: bool,
}

const _: () = assert!(size_of::<PrefillPlanSm90Info>() == 72);
const _: () = assert!(align_of::<PrefillPlanSm90Info>() == 8);
const _: () = assert!(offset_of!(PrefillPlanSm90Info, qo_tile_indices_offset) == 0);
const _: () = assert!(offset_of!(PrefillPlanSm90Info, qo_indptr_offset) == 8);
const _: () = assert!(offset_of!(PrefillPlanSm90Info, kv_indptr_offset) == 16);
const _: () = assert!(offset_of!(PrefillPlanSm90Info, qo_len_offset) == 24);
const _: () = assert!(offset_of!(PrefillPlanSm90Info, kv_len_offset) == 32);
const _: () = assert!(offset_of!(PrefillPlanSm90Info, head_indices_offset) == 40);
const _: () = assert!(offset_of!(PrefillPlanSm90Info, work_indptr_offset) == 48);
const _: () = assert!(offset_of!(PrefillPlanSm90Info, batch_indices_offset) == 56);
const _: () = assert!(offset_of!(PrefillPlanSm90Info, same_schedule_for_all_heads) == 64);

impl PrefillPlanSm90Info {
    /// `ToVector()` — nine `int64_t`.
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

/// `flashinfer::MLAPlanInfo` — the MLA (DeepSeek-style) descriptor.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MlaPlanInfo {
    /// `gridDim.x`: the cluster size, 1 or 2.
    pub num_blks_x: i64,
    /// `gridDim.y`: `num_sm / cluster_size`.
    pub num_blks_y: i64,
    /// `q_indptr[work]`.
    pub q_indptr_offset: i64,
    /// `kv_indptr[work]`.
    pub kv_indptr_offset: i64,
    /// `partial_indptr[work]` — where this work item's partial output goes, or
    pub partial_indptr_offset: i64,
    /// Merge CTA's first packed output row.
    pub merge_packed_offset_start_offset: i64,
    /// Merge CTA's last packed output row.
    pub merge_packed_offset_end_offset: i64,
    /// Merge CTA's first partial row.
    pub merge_partial_packed_offset_start_offset: i64,
    /// Merge CTA's last partial row.
    pub merge_partial_packed_offset_end_offset: i64,
    /// Stride between a merge CTA's partial rows.
    pub merge_partial_stride_offset: i64,
    /// `q_len[work]`.
    pub q_len_offset: i64,
    /// `kv_len[work]`.
    pub kv_len_offset: i64,
    /// `q_start[work]`.
    pub q_start_offset: i64,
    /// `kv_start[work]`.
    pub kv_start_offset: i64,
    /// `kv_end[work]`.
    pub kv_end_offset: i64,
    /// `work_indptr[cluster]`.
    pub work_indptr_offset: i64,
    /// Partial outputs, in the float workspace.
    pub partial_o_offset: i64,
    /// Partial LSEs, in the float workspace.
    pub partial_lse_offset: i64,
}

const _: () = assert!(size_of::<MlaPlanInfo>() == 144);
const _: () = assert!(align_of::<MlaPlanInfo>() == 8);
const _: () = assert!(offset_of!(MlaPlanInfo, num_blks_x) == 0);
const _: () = assert!(offset_of!(MlaPlanInfo, num_blks_y) == 8);
const _: () = assert!(offset_of!(MlaPlanInfo, q_indptr_offset) == 16);
const _: () = assert!(offset_of!(MlaPlanInfo, kv_indptr_offset) == 24);
const _: () = assert!(offset_of!(MlaPlanInfo, partial_indptr_offset) == 32);
const _: () = assert!(offset_of!(MlaPlanInfo, merge_packed_offset_start_offset) == 40);
const _: () = assert!(offset_of!(MlaPlanInfo, merge_packed_offset_end_offset) == 48);
const _: () = assert!(offset_of!(MlaPlanInfo, merge_partial_packed_offset_start_offset) == 56);
const _: () = assert!(offset_of!(MlaPlanInfo, merge_partial_packed_offset_end_offset) == 64);
const _: () = assert!(offset_of!(MlaPlanInfo, merge_partial_stride_offset) == 72);
const _: () = assert!(offset_of!(MlaPlanInfo, q_len_offset) == 80);
const _: () = assert!(offset_of!(MlaPlanInfo, kv_len_offset) == 88);
const _: () = assert!(offset_of!(MlaPlanInfo, q_start_offset) == 96);
const _: () = assert!(offset_of!(MlaPlanInfo, kv_start_offset) == 104);
const _: () = assert!(offset_of!(MlaPlanInfo, kv_end_offset) == 112);
const _: () = assert!(offset_of!(MlaPlanInfo, work_indptr_offset) == 120);
const _: () = assert!(offset_of!(MlaPlanInfo, partial_o_offset) == 128);
const _: () = assert!(offset_of!(MlaPlanInfo, partial_lse_offset) == 136);

impl MlaPlanInfo {
    /// `ToVector()` — eighteen `int64_t`.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The vectors are the other half of the ABI, and their lengths are fixed
    #[test]
    fn the_vectors_are_the_lengths_from_vector_demands() {
        assert_eq!(DecodePlanInfo::default().to_vector().len(), 10);
        assert_eq!(PrefillPlanInfo::default().to_vector().len(), 15);
        assert_eq!(PrefillPlanSm90Info::default().to_vector().len(), 9);
        assert_eq!(MlaPlanInfo::default().to_vector().len(), 18);
    }

    /// A bool widens to 0/1 in the vector, and sits in the last slots.
    #[test]
    fn the_bools_widen_and_come_last() {
        let info =
            DecodePlanInfo { enable_cuda_graph: true, split_kv: false, ..Default::default() };
        assert_eq!(info.to_vector()[8], 1);
        assert_eq!(info.to_vector()[9], 0);
    }
}

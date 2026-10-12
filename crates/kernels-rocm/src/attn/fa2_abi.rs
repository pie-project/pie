use crate::attn::plan::{DecodePlan, PrefillPlan};

pub type DevicePtr = u64;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct UintFastdiv {
    words: [u64; 3],
}

impl UintFastdiv {
    #[must_use]
    pub const fn new(divisor: u32) -> Self {
        let d = if divisor == 0 { 1 } else { divisor } as u64;
        let q = u64::MAX / d;
        let r = u64::MAX % d;
        let carry = if r + 1 == d { 1 } else { 0 };
        Self {
            words: [d, q.wrapping_add(carry).wrapping_add(1), d],
        }
    }
}

const _: () = assert!(
    core::mem::size_of::<UintFastdiv>() == 24,
    "UintFastdiv: sizeof disagrees with the measured ::flashinfer::uint_fastdiv",
);
const _: () = assert!(
    core::mem::align_of::<UintFastdiv>() == 8,
    "UintFastdiv: alignof disagrees with the measured ::flashinfer::uint_fastdiv",
);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct PagedKv {
    pub page_size: UintFastdiv,
    pub num_heads: u32,
    pub head_dim: u32,
    pub batch_size: u32,
    pub stride_page: u32,
    pub stride_n: u32,
    pub stride_h: u32,
    pub k_data: DevicePtr,
    pub v_data: DevicePtr,
    pub indices: DevicePtr,
    pub indptr: DevicePtr,
    pub last_page_len: DevicePtr,
    pub rope_pos_offset: DevicePtr,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct DecodeParams {
    pub q: DevicePtr,
    pub q_rope_offset: DevicePtr,
    pub paged_kv: PagedKv,
    pub o: DevicePtr,
    pub lse: DevicePtr,
    pub maybe_alibi_slopes: DevicePtr,
    pub padded_batch_size: u32,
    pub num_qo_heads: u32,
    pub q_stride_n: i32,
    pub q_stride_h: i32,
    pub window_left: i32,
    pub logits_soft_cap: f32,
    pub sm_scale: f32,
    pub rope_rcp_scale: f32,
    pub rope_rcp_theta: f32,
    pub request_indices: DevicePtr,
    pub kv_tile_indices: DevicePtr,
    pub o_indptr: DevicePtr,
    pub kv_chunk_size_ptr: DevicePtr,
    pub block_valid_mask: DevicePtr,
    pub partition_kv: bool,
    pub q_indptr: DevicePtr,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct PrefillPagedParams {
    pub q: DevicePtr,
    pub paged_kv: PagedKv,
    pub maybe_custom_mask: DevicePtr,
    pub q_indptr: DevicePtr,
    pub maybe_mask_indptr: DevicePtr,
    pub maybe_q_rope_offset: DevicePtr,
    pub o: DevicePtr,
    pub lse: DevicePtr,
    pub maybe_alibi_slopes: DevicePtr,
    pub group_size: UintFastdiv,
    pub num_qo_heads: u32,
    pub q_stride_n: i32,
    pub q_stride_h: i32,
    pub k_sf_stride_page: u32,
    pub k_sf_stride_n: u32,
    pub k_sf_stride_h: u32,
    pub v_sf_stride_page: u32,
    pub v_sf_stride_n: u32,
    pub v_sf_stride_h: u32,
    pub window_left: i32,
    pub logits_soft_cap: f32,
    pub sm_scale: f32,
    pub rope_rcp_scale: f32,
    pub rope_rcp_theta: f32,
    pub request_indices: DevicePtr,
    pub qo_tile_indices: DevicePtr,
    pub kv_tile_indices: DevicePtr,
    pub merge_indptr: DevicePtr,
    pub o_indptr: DevicePtr,
    pub block_valid_mask: DevicePtr,
    pub kv_chunk_size_ptr: DevicePtr,
    pub max_total_num_rows: u32,
    pub total_num_rows: DevicePtr,
    pub padded_batch_size: u32,
    pub partition_kv: bool,
    pub maybe_prefix_len_ptr: DevicePtr,
    pub maybe_token_pos_in_items_ptr: DevicePtr,
    pub token_pos_in_items_len: u32,
    pub maybe_max_item_len_ptr: DevicePtr,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct PrefillRaggedParams {
    pub q: DevicePtr,
    pub k: DevicePtr,
    pub v: DevicePtr,
    pub maybe_custom_mask: DevicePtr,
    pub q_indptr: DevicePtr,
    pub kv_indptr: DevicePtr,
    pub maybe_mask_indptr: DevicePtr,
    pub maybe_q_rope_offset: DevicePtr,
    pub maybe_k_rope_offset: DevicePtr,
    pub o: DevicePtr,
    pub lse: DevicePtr,
    pub maybe_alibi_slopes: DevicePtr,
    pub group_size: UintFastdiv,
    pub num_qo_heads: u32,
    pub num_kv_heads: u32,
    pub q_stride_n: u32,
    pub q_stride_h: u32,
    pub k_stride_n: u32,
    pub k_stride_h: u32,
    pub v_stride_n: u32,
    pub v_stride_h: u32,
    pub k_sf_stride_page: u32,
    pub k_sf_stride_n: u32,
    pub k_sf_stride_h: u32,
    pub v_sf_stride_page: u32,
    pub v_sf_stride_n: u32,
    pub v_sf_stride_h: u32,
    pub window_left: i32,
    pub logits_soft_cap: f32,
    pub sm_scale: f32,
    pub rope_rcp_scale: f32,
    pub rope_rcp_theta: f32,
    pub request_indices: DevicePtr,
    pub qo_tile_indices: DevicePtr,
    pub kv_tile_indices: DevicePtr,
    pub merge_indptr: DevicePtr,
    pub o_indptr: DevicePtr,
    pub kv_chunk_size_ptr: DevicePtr,
    pub block_valid_mask: DevicePtr,
    pub max_total_num_rows: u32,
    pub total_num_rows: DevicePtr,
    pub padded_batch_size: u32,
    pub partition_kv: bool,
    pub maybe_prefix_len_ptr: DevicePtr,
    pub maybe_token_pos_in_items_ptr: DevicePtr,
    pub token_pos_in_items_len: u32,
    pub maybe_max_item_len_ptr: DevicePtr,
}

const _: () = assert!(
    core::mem::size_of::<PrefillRaggedParams>() == 312,
    "PrefillRaggedParams: sizeof disagrees with ::flashinfer::BatchPrefillRaggedParams",
);
const _: () = assert!(
    core::mem::align_of::<PrefillRaggedParams>() == 8,
    "PrefillRaggedParams: alignof disagrees with ::flashinfer::BatchPrefillRaggedParams",
);

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct PrefillRaggedRefParams {
    pub base: PrefillRaggedParams,
    pub ref_start: DevicePtr,
}

const _: () = assert!(
    core::mem::size_of::<PrefillRaggedRefParams>() == 320,
    "PrefillRaggedRefParams: sizeof disagrees with ::pie::attn::fa2::RaggedRefParams",
);

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct PrefillRaggedTagParams {
    pub base: PrefillRaggedParams,
    pub q_tags: DevicePtr,
    pub kv_tags: DevicePtr,
}

const _: () = assert!(
    core::mem::size_of::<PrefillRaggedTagParams>() == 328,
    "PrefillRaggedTagParams: sizeof disagrees with ::pie::attn::fa2::RaggedTagParams",
);

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct PrefillRaggedClassParams {
    pub base: PrefillRaggedParams,
    pub q_classes: DevicePtr,
    pub kv_classes: DevicePtr,
    pub table: DevicePtr,
    pub count: u32,
}

const _: () = assert!(
    core::mem::size_of::<PrefillRaggedClassParams>() == 344,
    "PrefillRaggedClassParams: sizeof disagrees with ::pie::attn::fa2::RaggedClassParams",
);

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct PrefillRaggedBiasParams {
    pub base: PrefillRaggedParams,
    pub bias: DevicePtr,
    pub max_len: u32,
}

const _: () = assert!(
    core::mem::size_of::<PrefillRaggedBiasParams>() == 328,
    "PrefillRaggedBiasParams: sizeof disagrees with ::pie::attn::fa2::RaggedBiasParams",
);

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct DecodeRelParams {
    pub base: DecodeParams,
    pub rel_bias: DevicePtr,
    pub rel_extent: u32,
    pub rel_heads: u32,
    pub log_floor: u32,
    pub log_alpha: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct PrefillRelParams {
    pub base: PrefillPagedParams,
    pub rel_bias: DevicePtr,
    pub rel_extent: u32,
    pub rel_heads: u32,
    pub log_floor: u32,
    pub log_alpha: f32,
}

const _: () = assert!(
    core::mem::size_of::<DecodeParams>().is_multiple_of(8)
        && core::mem::size_of::<PrefillPagedParams>().is_multiple_of(8),
    "a params base must end on its alignment for the derived block to start where C++ puts it",
);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Buffers {
    pub q: DevicePtr,
    pub k_pages: DevicePtr,
    pub v_pages: DevicePtr,
    pub o: DevicePtr,
    pub kv_page_indices: DevicePtr,
    pub kv_page_indptr: DevicePtr,
    pub kv_last_page_lens: DevicePtr,
    pub qo_indptr: DevicePtr,
    pub lse: DevicePtr,
    pub int_buffer: DevicePtr,
    pub float_buffer: DevicePtr,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Partials {
    pub tmp_v: DevicePtr,
    pub tmp_s: DevicePtr,
    pub indptr: DevicePtr,
    pub o: DevicePtr,
    pub lse: DevicePtr,
    pub max_seq_len: u32,
    pub seq_len: DevicePtr,
    pub num_heads: u32,
    pub head_dim: u32,
}

pub(crate) const fn resolve(base: DevicePtr, off: Option<u32>) -> DevicePtr {
    match off {
        Some(off) => base.saturating_add(off as u64),
        None => base,
    }
}

#[must_use]
pub fn sm_scale_or_default(sm_scale: f32, head_dim: u32) -> f32 {
    if sm_scale > 0.0 {
        sm_scale
    } else {
        1.0 / (head_dim as f32).sqrt()
    }
}

#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn make_paged_kv(
    num_heads: u32,
    page_size: u32,
    head_dim: u32,
    batch_size: u32,
    hnd_layout: bool,
    k_data: DevicePtr,
    v_data: DevicePtr,
    indices: DevicePtr,
    indptr: DevicePtr,
    last_page_len: DevicePtr,
) -> PagedKv {
    PagedKv {
        page_size: UintFastdiv::new(page_size),
        num_heads,
        head_dim,
        batch_size,
        stride_page: num_heads.wrapping_mul(page_size).wrapping_mul(head_dim),
        stride_n: if hnd_layout {
            head_dim
        } else {
            num_heads.wrapping_mul(head_dim)
        },
        stride_h: if hnd_layout {
            page_size.wrapping_mul(head_dim)
        } else {
            head_dim
        },
        k_data,
        v_data,
        indices,
        indptr,
        last_page_len,
        rope_pos_offset: 0,
    }
}

#[must_use]
pub fn make_decode_params(
    plan: &DecodePlan,
    bufs: &Buffers,
    window_left: i32,
    logits_soft_cap: f32,
    sm_scale: f32,
    broadcast_q: bool,
) -> (DecodeParams, Partials) {
    let info = &plan.info;
    let shape = &plan.shape;
    let mut p = DecodeParams {
        q: bufs.q,
        q_rope_offset: 0,
        paged_kv: make_paged_kv(
            shape.num_kv_heads,
            shape.page_size,
            shape.head_dim,
            shape.lane_offset + shape.num_requests,
            shape.hnd_layout,
            bufs.k_pages,
            bufs.v_pages,
            bufs.kv_page_indices,
            bufs.kv_page_indptr,
            bufs.kv_last_page_lens,
        ),
        o: bufs.o,
        lse: bufs.lse,
        maybe_alibi_slopes: 0,
        num_qo_heads: shape.num_q_heads,
        q_stride_n: if broadcast_q {
            0
        } else {
            (shape.num_q_heads * shape.head_dim) as i32
        },
        q_stride_h: shape.head_dim as i32,
        window_left,
        logits_soft_cap,
        sm_scale: sm_scale_or_default(sm_scale, shape.head_dim),
        rope_rcp_scale: 1.0,
        rope_rcp_theta: 1.0,
        ..DecodeParams::default()
    };

    let int_buf = bufs.int_buffer;
    p.request_indices = resolve(int_buf, info.request_indices_offset);
    p.kv_tile_indices = resolve(int_buf, info.kv_tile_indices_offset);
    p.o_indptr = resolve(int_buf, info.o_indptr_offset);
    p.kv_chunk_size_ptr = resolve(int_buf, info.kv_chunk_size_ptr_offset);
    p.q_indptr = bufs.qo_indptr;
    p.padded_batch_size = info.padded_batch_size as u32;
    p.partition_kv = info.split_kv;

    let mut partials = Partials::default();
    if info.split_kv {
        partials.tmp_v = resolve(bufs.float_buffer, info.v_offset);
        partials.tmp_s = resolve(bufs.float_buffer, info.s_offset);

        partials.o = p.o;
        partials.lse = p.lse;
        p.o = partials.tmp_v;
        p.lse = partials.tmp_s;

        partials.indptr = p.o_indptr;
        partials.max_seq_len = shape.num_requests;

        partials.seq_len = 0;
        partials.num_heads = shape.num_q_heads;
        partials.head_dim = shape.head_dim;

        if info.enable_cuda_graph {
            p.block_valid_mask = resolve(int_buf, info.block_valid_mask_offset);
        }
    }
    (p, partials)
}

#[must_use]
pub fn make_prefill_params(
    plan: &PrefillPlan,
    bufs: &Buffers,
    window_left: i32,
    logits_soft_cap: f32,
    sm_scale: f32,
) -> (PrefillPagedParams, Partials) {
    let info = &plan.info;
    let shape = &plan.shape;
    let mut p = PrefillPagedParams {
        q: bufs.q,
        paged_kv: make_paged_kv(
            shape.num_kv_heads,
            shape.page_size,
            shape.head_dim,
            shape.lane_offset + shape.num_requests,
            shape.hnd_layout,
            bufs.k_pages,
            bufs.v_pages,
            bufs.kv_page_indices,
            bufs.kv_page_indptr,
            bufs.kv_last_page_lens,
        ),
        maybe_custom_mask: 0,
        q_indptr: bufs.qo_indptr,
        maybe_mask_indptr: 0,
        maybe_q_rope_offset: 0,
        o: bufs.o,
        lse: bufs.lse,
        maybe_alibi_slopes: 0,
        group_size: UintFastdiv::new(shape.group_size()),
        num_qo_heads: shape.num_q_heads,
        q_stride_n: (shape.num_q_heads * shape.head_dim) as i32,
        q_stride_h: shape.head_dim as i32,
        window_left,
        logits_soft_cap,
        sm_scale: sm_scale_or_default(sm_scale, shape.head_dim),
        rope_rcp_scale: 1.0,
        rope_rcp_theta: 1.0,
        ..PrefillPagedParams::default()
    };

    let int_buf = bufs.int_buffer;
    p.request_indices = resolve(int_buf, info.request_indices_offset);
    p.qo_tile_indices = resolve(int_buf, info.qo_tile_indices_offset);
    p.kv_tile_indices = resolve(int_buf, info.kv_tile_indices_offset);
    p.o_indptr = resolve(int_buf, info.o_indptr_offset);
    p.kv_chunk_size_ptr = resolve(int_buf, info.kv_chunk_size_ptr_offset);
    p.padded_batch_size = info.padded_batch_size as u32;
    p.partition_kv = info.split_kv;

    p.max_total_num_rows = info.total_num_rows as u32;
    p.total_num_rows = match info.total_num_rows_offset {
        Some(off) => int_buf.saturating_add(u64::from(off)),
        None => 0,
    };

    let mut partials = Partials::default();
    if info.split_kv {
        p.merge_indptr = resolve(int_buf, info.merge_indptr_offset);
        partials.tmp_v = resolve(bufs.float_buffer, info.v_offset);
        partials.tmp_s = resolve(bufs.float_buffer, info.s_offset);

        partials.o = p.o;
        partials.lse = p.lse;
        p.o = partials.tmp_v;
        p.lse = partials.tmp_s;

        partials.indptr = p.merge_indptr;
        partials.max_seq_len = p.max_total_num_rows;

        partials.seq_len = p.total_num_rows;
        partials.num_heads = shape.num_q_heads;
        partials.head_dim = shape.head_dim;
        if info.enable_cuda_graph {
            p.block_valid_mask = resolve(int_buf, info.block_valid_mask_offset);
        }
    }
    (p, partials)
}

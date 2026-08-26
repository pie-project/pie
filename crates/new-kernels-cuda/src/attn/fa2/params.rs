//! The by-value parameter blocks the fa2 kernels take, `#[repr(C)]` against
//! the layouts `attn/fa2.cuh` declares, and the packers that fill them from
//! a plan + a buffer set. Device pointers travel as the `u64` the handles
//! carry ([`DevicePtr`]); the whole struct crosses the launch as one
//! `ArgValue::Bytes` argument.

use crate::attn::plan::{DecodePlan, PrefillPlan};

pub type DevicePtr = u64;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct FastModDiv {
    pub divisor: u32,
    pub magic: u64,
}

impl FastModDiv {
    #[must_use]
    pub const fn new(divisor: u32) -> Self {
        let d = if divisor == 0 { 1 } else { divisor };
        let all_ones = u64::MAX;
        let q = all_ones / d as u64;
        let r = all_ones % d as u64;
        let carry = if r + 1 == d as u64 { 1 } else { 0 };
        Self {
            divisor: d,
            magic: q.wrapping_add(carry).wrapping_add(1),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct UintFastdiv {
    pub magic: FastModDiv,
    pub d: u32,
}

impl UintFastdiv {
    #[must_use]
    pub const fn new(divisor: u32) -> Self {
        let d = if divisor == 0 { 1 } else { divisor };
        Self {
            magic: FastModDiv::new(d),
            d,
        }
    }
}

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

/// Every device address a decode or prefill fire touches, gathered by the
/// entry from `q`/`o`/the pool row/the plan's workspace.
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

/// What the cascade merge needs when the schedule split kv: the partial
/// planes the attention wrote and the final planes they fold into.
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

const fn offset_ptr(base: DevicePtr, off: i64) -> DevicePtr {
    if off < 0 {
        base
    } else {
        base.saturating_add(off as u64)
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
            shape.num_requests,
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
    p.request_indices = offset_ptr(int_buf, info.request_indices_offset);
    p.kv_tile_indices = offset_ptr(int_buf, info.kv_tile_indices_offset);
    p.o_indptr = offset_ptr(int_buf, info.o_indptr_offset);
    p.kv_chunk_size_ptr = offset_ptr(int_buf, info.kv_chunk_size_ptr_offset);
    p.padded_batch_size = info.padded_batch_size as u32;
    p.partition_kv = info.split_kv;

    let mut partials = Partials::default();
    if info.split_kv {
        partials.tmp_v = offset_ptr(bufs.float_buffer, info.v_offset);
        partials.tmp_s = offset_ptr(bufs.float_buffer, info.s_offset);

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
            p.block_valid_mask = offset_ptr(int_buf, info.block_valid_mask_offset);
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
            shape.num_requests,
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
    p.request_indices = offset_ptr(int_buf, info.request_indices_offset);
    p.qo_tile_indices = offset_ptr(int_buf, info.qo_tile_indices_offset);
    p.kv_tile_indices = offset_ptr(int_buf, info.kv_tile_indices_offset);
    p.o_indptr = offset_ptr(int_buf, info.o_indptr_offset);
    p.kv_chunk_size_ptr = offset_ptr(int_buf, info.kv_chunk_size_ptr_offset);
    p.padded_batch_size = info.padded_batch_size as u32;
    p.partition_kv = info.split_kv;

    p.max_total_num_rows = info.total_num_rows as u32;
    p.total_num_rows = 0;

    let mut partials = Partials::default();
    if info.split_kv {
        p.merge_indptr = offset_ptr(int_buf, info.merge_indptr_offset);
        partials.tmp_v = offset_ptr(bufs.float_buffer, info.v_offset);
        partials.tmp_s = offset_ptr(bufs.float_buffer, info.s_offset);

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
            p.block_valid_mask = offset_ptr(int_buf, info.block_valid_mask_offset);
        }
    }
    (p, partials)
}

//! Attention over the paged KV cache on AMD: the launches of
//! `kernels/attn/paged.hip`, fed by the work list a decode or prefill plan
//! staged. A split plan writes f32 partials per kv chunk, and
//! [`merge`] folds them into the output rows.

use crate::attn::kv;
use crate::attn::plan::{DecodePlan, PrefillPlan};
use crate::error::Error;
use crate::jit::{ArgValue, Ctx, Fire, Launch, refuse, symbol};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

pub const FILE: &str = "attn/paged.hip";

/// The prefill kernel's query tile: 64 packed (token, head) rows a block.
/// The prefill scheduler cuts its work list at this tile.
pub const PREFILL_TILE_Q: u32 = 128;

const LDS_BYTES: u32 = 64 * 1024;

/// `pie::attn::Paged`, field for field.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
struct Args {
    q: u64,
    k: u64,
    v: u64,
    page_indices: u64,
    page_indptr: u64,
    last_page_len: u64,
    q_indptr: u64,
    o_indptr: u64,
    request_indices: u64,
    qo_tile_indices: u64,
    kv_tile_indices: u64,
    kv_chunk_size: u64,
    block_valid: u64,
    o: u64,
    lse: u64,
    num_qo_heads: i32,
    num_kv_heads: i32,
    page_size: i32,
    q_stride_n: i32,
    q_stride_h: i32,
    window_left: i32,
    scale_log2: f32,
    partition: i32,
}

const _: () = assert!(core::mem::size_of::<Args>() == 152);

fn by_value(args: &Args) -> ArgValue {
    ArgValue::Bytes {
        ptr: core::ptr::from_ref(args).cast::<u8>(),
        len: core::mem::size_of::<Args>(),
    }
}

const fn at(base: u64, offset: Option<u32>) -> u64 {
    match offset {
        Some(off) => base + off as u64,
        None => 0,
    }
}

fn scale_log2(sm_scale: f32, head_dim: u32) -> f32 {
    let scale = if sm_scale > 0.0 {
        sm_scale
    } else {
        1.0 / (head_dim as f32).sqrt()
    };
    scale * core::f32::consts::LOG2_E
}

fn head_dim_built(op: &'static str, head_dim: u32) -> Result<(), Error> {
    if matches!(head_dim, 64 | 128 | 256 | 512) {
        return Ok(());
    }
    Err(refuse(
        op,
        format!(
            "no paged attention is built at head width {head_dim}; it is built at 64/128/256/512"
        ),
    ))
}

fn pool_args(
    q: RaggedTensor,
    pool: &KvPool,
    num_qo_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
) -> Args {
    // A quantized pool is read through its bf16 shadow, which the fire
    // refreshed (`kv::dequant_active`) before this one.
    let (k, v) = if kv::native_bf16(pool) {
        (pool.keys.ptr, pool.values.ptr)
    } else {
        (pool.bf16_keys.ptr, pool.bf16_values.ptr)
    };
    Args {
        q: q.data.ptr,
        k,
        v,
        page_indices: pool.page_indices.ptr,
        page_indptr: pool.page_indptr.ptr,
        last_page_len: pool.last_page_lens.ptr,
        q_indptr: q.indptr.ptr,
        num_qo_heads: num_qo_heads as i32,
        num_kv_heads: num_kv_heads as i32,
        page_size: pool.page_size,
        q_stride_n: (num_qo_heads * head_dim) as i32,
        q_stride_h: head_dim as i32,
        ..Args::default()
    }
}

/// The partials a split plan wrote, and where they fold to.
struct Partials {
    parts: u64,
    part_lse: u64,
    indptr: u64,
    o: u64,
    lse: u64,
    max_rows: u32,
    rows_ptr: u64,
    heads: u32,
    head_dim: u32,
}

/// Wavefronts per decode block: as many as four whose partial states, with
/// the query, fit in LDS.
fn decode_waves(group: u32, head_dim: u32) -> u32 {
    [4u32, 2, 1]
        .into_iter()
        .find(|w| (w + 1) * group * head_dim * 4 + 2 * w * group * 4 <= LDS_BYTES)
        .unwrap_or(1)
}

/// The grid a decode plan may fill without splitting the kv: a few blocks
/// per CU, over the kv heads each request launches.
#[must_use]
pub fn decode_max_grid_size(num_kv_heads: u32, num_cus: u32) -> u32 {
    const BLOCKS_PER_CU: u32 = 4;
    (num_cus.max(1) * BLOCKS_PER_CU / num_kv_heads.max(1)).max(1)
}

#[allow(clippy::too_many_arguments)]
pub fn decode(
    ctx: &Ctx,
    op: &'static str,
    q: RaggedTensor,
    plan: &DecodePlan,
    pool: &KvPool,
    window_left: i32,
    sm_scale: f32,
    o: &Tensor,
    lse: u64,
) -> Result<(), Error> {
    let shape = &plan.shape;
    head_dim_built(op, shape.head_dim)?;
    let group = shape.group_size();
    let info = &plan.info;
    let int = plan.workspace.int_ptr;
    let float = plan.workspace.float_ptr;
    let mut args = pool_args(
        q,
        pool,
        shape.num_q_heads,
        shape.num_kv_heads,
        shape.head_dim,
    );
    args.request_indices = at(int, info.request_indices_offset);
    args.kv_tile_indices = at(int, info.kv_tile_indices_offset);
    args.kv_chunk_size = at(int, info.kv_chunk_size_ptr_offset);
    args.window_left = window_left;
    args.scale_log2 = scale_log2(sm_scale, shape.head_dim);
    args.partition = i32::from(info.split_kv);
    let split = info.split_kv.then(|| Partials {
        parts: at(float, info.v_offset),
        part_lse: at(float, info.s_offset),
        indptr: at(int, info.o_indptr_offset),
        o: o.ptr,
        lse,
        max_rows: shape.num_requests,
        rows_ptr: 0,
        heads: shape.num_q_heads,
        head_dim: shape.head_dim,
    });
    match &split {
        Some(parts) => {
            args.o = parts.parts;
            args.lse = parts.part_lse;
            if info.enable_cuda_graph {
                args.block_valid = at(int, info.block_valid_mask_offset);
            }
        }
        None => {
            args.o = o.ptr;
            args.lse = lse;
        }
    }
    let hnd = if shape.hnd_layout { "true" } else { "false" };
    // The matrix-core decode takes a group within one 16-column tile; a
    // wider group takes the vector decode.
    let (entry, threads) = if group <= 16 && shape.head_dim.is_multiple_of(32) {
        (
            symbol(&format!(
                "::pie::attn::paged_decode_mfma<{}, {group}, {hnd}>",
                shape.head_dim
            )),
            512,
        )
    } else {
        let waves = decode_waves(group, shape.head_dim);
        (
            symbol(&format!(
                "::pie::attn::paged_decode<{}, {group}, {waves}, {hnd}>",
                shape.head_dim
            )),
            64 * waves,
        )
    };
    ctx.fire(
        op,
        Fire::at(FILE, entry).apply(Launch::grid(
            [info.padded_batch_size as u32, shape.num_kv_heads, 1],
            [threads, 1, 1],
        )),
        &[by_value(&args)],
    )?;
    match split {
        Some(parts) => merge(ctx, op, &parts),
        None => Ok(()),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prefill(
    ctx: &Ctx,
    op: &'static str,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    window_left: i32,
    sm_scale: f32,
    o: &Tensor,
    lse: u64,
) -> Result<(), Error> {
    let shape = &plan.shape;
    head_dim_built(op, shape.head_dim)?;
    let info = &plan.info;
    if info.cta_tile_q != i64::from(PREFILL_TILE_Q) {
        return Err(refuse(
            op,
            format!(
                "the plan cut {}-row query tiles and the kernel walks {PREFILL_TILE_Q}",
                info.cta_tile_q
            ),
        ));
    }
    let int = plan.workspace.int_ptr;
    let float = plan.workspace.float_ptr;
    let mut args = pool_args(
        q,
        pool,
        shape.num_q_heads,
        shape.num_kv_heads,
        shape.head_dim,
    );
    args.o_indptr = at(int, info.o_indptr_offset);
    args.request_indices = at(int, info.request_indices_offset);
    args.qo_tile_indices = at(int, info.qo_tile_indices_offset);
    args.kv_tile_indices = at(int, info.kv_tile_indices_offset);
    args.kv_chunk_size = at(int, info.kv_chunk_size_ptr_offset);
    args.window_left = window_left;
    args.scale_log2 = scale_log2(sm_scale, shape.head_dim);
    args.partition = i32::from(info.split_kv);
    let split = info.split_kv.then(|| Partials {
        parts: at(float, info.v_offset),
        part_lse: at(float, info.s_offset),
        indptr: at(int, info.merge_indptr_offset),
        o: o.ptr,
        lse,
        max_rows: info.total_num_rows as u32,
        rows_ptr: at(int, info.total_num_rows_offset),
        heads: shape.num_q_heads,
        head_dim: shape.head_dim,
    });
    match &split {
        Some(parts) => {
            args.o = parts.parts;
            args.lse = parts.part_lse;
            if info.enable_cuda_graph {
                args.block_valid = at(int, info.block_valid_mask_offset);
            }
        }
        None => {
            args.o = o.ptr;
            args.lse = lse;
        }
    }
    let hnd = if shape.hnd_layout { "true" } else { "false" };
    let causal = if plan.causal { "true" } else { "false" };
    let kernel = if matches!(shape.head_dim, 128 | 256) {
        "paged_prefill"
    } else {
        "paged_prefill_wide"
    };
    let entry = symbol(&format!(
        "::pie::attn::{kernel}<{}, {hnd}, {causal}>",
        shape.head_dim
    ));
    ctx.fire(
        op,
        Fire::at(FILE, entry).apply(Launch::grid(
            [info.padded_batch_size as u32, shape.num_kv_heads, 1],
            [512, 1, 1],
        )),
        &[by_value(&args)],
    )?;
    match split {
        Some(parts) => merge(ctx, op, &parts),
        None => Ok(()),
    }
}

fn merge(ctx: &Ctx, op: &'static str, parts: &Partials) -> Result<(), Error> {
    if parts.parts == 0 || parts.part_lse == 0 || parts.indptr == 0 {
        return Err(refuse(op, "the split plan staged no partials to fold"));
    }
    let blocks = parts.max_rows.saturating_mul(parts.heads);
    if blocks == 0 {
        return Ok(());
    }
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!("::pie::attn::merge_partials<{}>", parts.head_dim)),
        )
        .apply(Launch::grid([blocks, 1, 1], [256, 1, 1])),
        &[
            ArgValue::Ptr(parts.parts),
            ArgValue::Ptr(parts.part_lse),
            ArgValue::Ptr(parts.indptr),
            ArgValue::Ptr(parts.o),
            ArgValue::Ptr(parts.lse),
            ArgValue::I32(parts.max_rows as i32),
            ArgValue::Ptr(parts.rows_ptr),
            ArgValue::I32(parts.heads as i32),
            ctx.stage(),
        ],
    )
}

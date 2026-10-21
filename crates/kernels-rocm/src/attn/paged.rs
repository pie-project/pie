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
    rel_bias: u64,
    rel_extent: i32,
    rel_log_floor: i32,
    rel_log_alpha: f32,
    rel_pad: i32,
}

const _: () = assert!(core::mem::size_of::<Args>() == 176);

/// A learned relative-position bias and Inkling's logit growth, as the
/// matrix-core walks read it: `bias` f32 `[rows, heads · extent]`.
#[derive(Clone, Copy, Debug)]
pub struct Rel {
    pub bias: u64,
    pub extent: i32,
    pub log_floor: i32,
    pub log_alpha: f32,
}

impl Args {
    fn rel(&mut self, rel: Option<Rel>) {
        if let Some(rel) = rel {
            self.rel_bias = rel.bias;
            self.rel_extent = rel.extent;
            self.rel_log_floor = rel.log_floor;
            self.rel_log_alpha = rel.log_alpha;
        }
    }
}

/// Whether the matrix-core decode (and, at these widths, the tiled prefill)
/// takes a relative bias at this group and head width.
#[must_use]
pub fn rel_walks(group: u32, head_dim: u32) -> bool {
    group <= 16 && matches!(head_dim, 128 | 256)
}

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

/// The scores' scale in the log2 domain, with an e4m3 pool's unit folded
/// in (its keys read as half their value: `attn/kv.hip`'s `kE4m3Unit`).
fn scale_log2(sm_scale: f32, head_dim: u32, pool: &KvPool) -> f32 {
    let scale = if sm_scale > 0.0 {
        sm_scale
    } else {
        1.0 / (head_dim as f32).sqrt()
    };
    let unit = if kv::native_e4m3(pool) { 2.0 } else { 1.0 };
    scale * unit * core::f32::consts::LOG2_E
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
    op: &'static str,
    q: RaggedTensor,
    pool: &KvPool,
    num_qo_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
) -> Result<Args, Error> {
    if !kv::native_bf16(pool) && !kv::native_e4m3(pool) {
        return Err(refuse(op, "the paged attention reads bf16 or e4m3 pages"));
    }
    Ok(Args {
        q: q.data.ptr,
        k: pool.keys.ptr,
        v: pool.values.ptr,
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
    })
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
    rel: Option<Rel>,
) -> Result<(), Error> {
    let shape = &plan.shape;
    head_dim_built(op, shape.head_dim)?;
    let group = shape.group_size();
    if rel.is_some() && !rel_walks(group, shape.head_dim) {
        return Err(refuse(
            op,
            "a relative bias rides the matrix-core decode only",
        ));
    }
    let info = &plan.info;
    let int = plan.workspace.int_ptr;
    let float = plan.workspace.float_ptr;
    let mut args = pool_args(
        op,
        q,
        pool,
        shape.num_q_heads,
        shape.num_kv_heads,
        shape.head_dim,
    )?;
    args.request_indices = at(int, info.request_indices_offset);
    args.kv_tile_indices = at(int, info.kv_tile_indices_offset);
    args.kv_chunk_size = at(int, info.kv_chunk_size_ptr_offset);
    args.window_left = window_left;
    args.scale_log2 = scale_log2(sm_scale, shape.head_dim, pool);
    args.partition = i32::from(info.split_kv);
    args.rel(rel);
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
    // The blocks past the live work items exit on the staged mask.
    args.block_valid = at(int, info.block_valid_mask_offset);
    match &split {
        Some(parts) => {
            args.o = parts.parts;
            args.lse = parts.part_lse;
        }
        None => {
            args.o = o.ptr;
            args.lse = lse;
        }
    }
    let hnd = if shape.hnd_layout { "true" } else { "false" };
    let fp8 = kv::native_e4m3(pool);
    // The matrix-core decode takes a group within one 16-column tile; a
    // wider group takes the vector decode.
    let blocks = (info.padded_batch_size as u32).saturating_mul(shape.num_kv_heads);
    let (entry, threads) = if group <= 16 && shape.head_dim >= 512 && !info.split_kv {
        // An unsplit batch at the widest heads fills the grid: the decode
        // whose columns split over its wavefronts keeps more blocks a CU
        // (and their reads) in flight.
        (
            symbol(&format!(
                "::pie::attn::paged_decode_cols<{}, {group}, {hnd}, {fp8}>",
                shape.head_dim
            )),
            256,
        )
    } else if group <= 16 && shape.head_dim.is_multiple_of(32) {
        // A group of eight or more weighs its values on the matrix core too
        // (`PVM`), whose staged value rows hold four wavefronts' rounds at
        // 256-wide heads.
        let pvm = group >= 8 && shape.head_dim <= 256;
        // Four wavefronts at 512-wide heads; below, eight, or four (three
        // or four blocks a CU, not one) once the grid is two blocks a CU.
        let waves = if shape.head_dim >= 512
            || blocks >= 2 * plan.device.num_sm
            || (pvm && shape.head_dim > 128)
        {
            4
        } else {
            8
        };
        (
            symbol(&format!(
                "::pie::attn::paged_decode_mfma<{}, {group}, {hnd}, {waves}, {}, {pvm}, {fp8}>",
                shape.head_dim,
                shape.head_dim < 512
            )),
            64 * waves,
        )
    } else {
        let waves = decode_waves(group, shape.head_dim);
        (
            symbol(&format!(
                "::pie::attn::paged_decode<{}, {group}, {waves}, {hnd}, {fp8}>",
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
    rel: Option<Rel>,
) -> Result<(), Error> {
    prefill_walk(
        ctx,
        op,
        q,
        plan,
        pool,
        window_left,
        sm_scale,
        o,
        lse,
        None,
        rel,
    )
}

/// `pie::attn::PrefillSel`, field for field: the selection rows a selected
/// prefill masks its keys by.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PrefillSel {
    pub selection: u64,
    pub top_k: i32,
    pub ratio: i32,
    /// Nonzero: the row holds a run of `top_k` for each kv head.
    pub per_head: i32,
}

const _: () = assert!(core::mem::size_of::<PrefillSel>() == 24);

/// The tokens of a 128-row tile whose selections the selected prefill holds
/// (`kSelTokens`), and the pooled blocks a token's bitmap spans
/// (`kSelBlocks`).
pub const SEL_TOKENS: u32 = 16;
pub const SEL_BLOCKS: u32 = 2048;

/// The causal prefill over `plan` with every row's keys cut to the pooled
/// blocks its selection row names and its open block
/// (`paged_prefill_selected`).
#[allow(clippy::too_many_arguments)]
pub fn prefill_selected(
    ctx: &Ctx,
    op: &'static str,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    sm_scale: f32,
    o: &Tensor,
    sel: PrefillSel,
) -> Result<(), Error> {
    prefill_walk(ctx, op, q, plan, pool, -1, sm_scale, o, 0, Some(sel), None)
}

#[allow(clippy::too_many_arguments)]
fn prefill_walk(
    ctx: &Ctx,
    op: &'static str,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    window_left: i32,
    sm_scale: f32,
    o: &Tensor,
    lse: u64,
    sel: Option<PrefillSel>,
    rel: Option<Rel>,
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
        op,
        q,
        pool,
        shape.num_q_heads,
        shape.num_kv_heads,
        shape.head_dim,
    )?;
    args.o_indptr = at(int, info.o_indptr_offset);
    args.request_indices = at(int, info.request_indices_offset);
    args.qo_tile_indices = at(int, info.qo_tile_indices_offset);
    args.kv_tile_indices = at(int, info.kv_tile_indices_offset);
    args.kv_chunk_size = at(int, info.kv_chunk_size_ptr_offset);
    args.window_left = window_left;
    args.scale_log2 = scale_log2(sm_scale, shape.head_dim, pool);
    args.partition = i32::from(info.split_kv);
    if rel.is_some() && !rel_walks(shape.group_size(), shape.head_dim) {
        return Err(refuse(op, "a relative bias rides the tiled prefill only"));
    }
    args.rel(rel);
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
    if info.enable_cuda_graph {
        args.block_valid = at(int, info.block_valid_mask_offset);
    }
    match &split {
        Some(parts) => {
            args.o = parts.parts;
            args.lse = parts.part_lse;
        }
        None => {
            args.o = o.ptr;
            args.lse = lse;
        }
    }
    let hnd = if shape.hnd_layout { "true" } else { "false" };
    let causal = if plan.causal { "true" } else { "false" };
    let launch = Launch::grid(
        [info.padded_batch_size as u32, shape.num_kv_heads, 1],
        [512, 1, 1],
    );
    let fp8 = kv::native_e4m3(pool);
    if let Some(sel) = sel {
        if !matches!(shape.head_dim, 128 | 256) || !plan.causal || fp8 {
            return Err(refuse(
                op,
                "the selected prefill walks causal 128- or 256-wide heads of bf16 pages only",
            ));
        }
        let entry = symbol(&format!(
            "::pie::attn::paged_prefill_selected<{}, {hnd}>",
            shape.head_dim
        ));
        ctx.fire(
            op,
            Fire::at(FILE, entry).apply(launch),
            &[
                by_value(&args),
                ArgValue::Bytes {
                    ptr: core::ptr::from_ref(&sel).cast::<u8>(),
                    len: core::mem::size_of::<PrefillSel>(),
                },
            ],
        )?;
    } else {
        let kernel = if matches!(shape.head_dim, 128 | 256) {
            "paged_prefill"
        } else {
            "paged_prefill_wide"
        };
        let entry = symbol(&format!(
            "::pie::attn::{kernel}<{}, {hnd}, {causal}, {fp8}>",
            shape.head_dim
        ));
        ctx.fire(op, Fire::at(FILE, entry).apply(launch), &[by_value(&args)])?;
    }
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

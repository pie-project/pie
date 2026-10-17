//! DeepSeek-V4's attention at prefill widths on tcgen05, a block a row
//! (`attn/dsv4_sparse.cuh`): the window, the compressed pool (every visible
//! entry or those an index selected), their merge and the sinks in one pass,
//! for the shape the kernel is built for: 64 query heads of 512 over one
//! latent, a 128-key window, bf16 NHD pools. Anything else is the caller's to
//! run as its four ops.

use cudarc::driver::sys as dr;

use crate::attn::dsv4::Rows;
use crate::error::Error;
use crate::jit::{ArgValue, Ctx, Fire, Launch, aligned16, symbol};
use crate::linear::tc5::{Map, encode, supported};
use crate::tensor::{KvPool, Tensor};
use dtype::Dtype;

const SPARSE_FILE: &str = "attn/dsv4_sparse.cuh";

/// `pie::attn::dsv4s::Params`, field for field.
#[repr(C)]
struct SparseParams {
    o: u64,
    q: u64,
    positions: u64,
    requests: u64,
    floors: u64,
    win_indptr: u64,
    win_indices: u64,
    ent_indptr: u64,
    ent_indices: u64,
    selection: u64,
    sinks: u64,
    live: u64,
    win_raw: u64,
    ent_raw: u64,
    sinks_f32: i32,
    ent_runs: i32,
    win_page: i32,
    ent_page: i32,
    ent_compact: i32,
    top_k: i32,
    ratio: i32,
    rows: i32,
    q_stride: i32,
    o_stride: i32,
    scale_log2: f32,
    turn: crate::attn::dsv4::TurnArgs,
}

/// `pie::attn::dsv4s::smem_bytes` (a block a multiprocessor, walking rows):
/// three 64-key tiles, P twice, the score
/// exchange, three key lists, barriers, and the 1024-byte alignment.
const SPARSE_SMEM: u32 = 3 * 65536 + 2 * 8192 + 16384 + 6 * 64 * 4 + 512 + 1024;

const HEADS: u32 = 64;
const HEAD_DIM: u32 = 512;
const WINDOW: u32 = 128;

/// The compressed pool a row reads: its entries, how many positions an
/// entry closes, and the ids an index selected for each row (`top_k` a
/// row, ended early by a negative one), if it selects.
pub struct Pooled<'a> {
    pub entries: &'a KvPool,
    pub ratio: u32,
    pub selection: Option<(Tensor, u32)>,
}

/// The pooled entries as the pass reads them: a compact pool's view (an
/// entry a row of its width, `page_size / ratio` a page), and whether it is
/// one.
fn entries_of(pooled: &Pooled<'_>) -> (KvPool, bool) {
    if pooled.ratio == NO_ENTRY {
        return (*pooled.entries, false);
    }
    let row = crate::attn::dsv4::rows_packed(pooled.entries)
        .map_or(i64::from(HEAD_DIM), |(rows, _)| i64::from(rows.bytes()));
    crate::attn::dsv4::entries_view(pooled.entries, pooled.ratio, row)
}

fn plain_bf16(pool: &KvPool) -> bool {
    pool.keys.dtype == Dtype::Bf16
        && pool.layout == 0
        && pool.page_size > 0
        && pool.head_stride == i64::from(HEAD_DIM)
        && pool.seq_stride >= i64::from(HEAD_DIM)
        && aligned16(pool.keys.ptr)
        && (pool.seq_stride * 2) % 16 == 0
}

/// Whether [`prefill`] serves this attention.
#[must_use]
pub fn takes(
    heads: u32,
    head_dim: u32,
    window: Option<u32>,
    window_pool: &KvPool,
    pooled: &Pooled<'_>,
) -> bool {
    // DeepSeek's quantized rows (both planes), staged to bf16 pages first.
    let fp8 = crate::attn::dsv4::ds_rows(window_pool) && crate::attn::dsv4::ds_rows(pooled.entries);
    supported()
        && heads == HEADS
        && head_dim == HEAD_DIM
        && window == Some(WINDOW)
        && pooled.ratio > 0
        && (fp8 || plain_bf16(window_pool))
        && (fp8 || plain_bf16(&entries_of(pooled).0))
}

/// Whether [`window_only`] serves this window.
#[must_use]
pub fn takes_window(heads: u32, head_dim: u32, window: Option<u32>, window_pool: &KvPool) -> bool {
    let pooled = Pooled {
        entries: window_pool,
        ratio: NO_ENTRY,
        selection: None,
    };
    takes(heads, head_dim, window, window_pool, &pooled)
}

/// Lands the sinked attention of every row of `q` in `o`: the row's window
/// of `window_pool` and its compressed entries in one softmax, scaled by
/// `sigmoid(lse - sink)` a head; with `floors` (i32 a row) a row's window
/// reads no key below its floor. With `graph_safe` (a launch a CUDA graph
/// records, to replay against other page tables) nothing is sized by the
/// tables this fire states. The caller checks [`takes`] first.
#[allow(clippy::too_many_arguments)]
pub fn prefill(
    ctx: &Ctx,
    op: &'static str,
    q: Tensor,
    positions: Tensor,
    requests: Tensor,
    floors: Option<Tensor>,
    graph_safe: bool,
    window_pool: &KvPool,
    pooled: &Pooled<'_>,
    sinks: Tensor,
    sm_scale: f32,
    turn: Option<crate::attn::dsv4::Turn>,
    o: &mut Tensor,
) -> Result<(), Error> {
    let rows = o.rows;
    if rows == 0 {
        return Ok(());
    }
    // The pass turns at most 32 pairs (a lane's angle each) in whole
    // 32-dim chunks; another turn runs after it.
    if let Some(wide) = turn.filter(|t| t.dim > 64 || t.dim % 32 != 0) {
        prefill(
            ctx,
            op,
            q,
            positions,
            requests,
            floors,
            graph_safe,
            window_pool,
            pooled,
            sinks,
            sm_scale,
            None,
            o,
        )?;
        return crate::elemwise::rope::partial_last(
            ctx, o, positions, wide.dim, HEAD_DIM, wide.theta, true, true, wide.yarn,
        );
    }
    let (ent_view, ent_compact) = entries_of(pooled);
    // V4.1's rows (an MXFP8 window over NVFP4 entries, or the window alone)
    // below a prefill's width (`kernel.sdpa.dsv4_prefill_from`): the pass
    // widens them itself, no plane staged. A prefill's many rows a tile
    // read the staged bf16 pages faster.
    //
    // A pass a CUDA graph records (`graph_safe`) reads every quantized row
    // so, V4's too: the staged planes are sized by the page tables a fire
    // states, which a replay's fire outgrows.
    let from = ctx.tuning().dsv4_prefill_from;
    let wf = crate::attn::dsv4::rows_of(window_pool);
    if let Some(wf @ (Rows::Mx8 | Rows::Ds)) = wf
        && (graph_safe || wf == Rows::Mx8 && (from == 0 || rows < from))
    {
        let alone = pooled.entries.keys.ptr == window_pool.keys.ptr;
        let pf = if alone {
            wf
        } else {
            crate::attn::dsv4::rows_of(&ent_view).unwrap_or(wf)
        };
        let viewed = Pooled {
            entries: &ent_view,
            ratio: pooled.ratio,
            selection: pooled.selection,
        };
        return sparse(
            ctx,
            op,
            q,
            positions,
            requests,
            floors,
            window_pool,
            &viewed,
            ent_compact,
            Some((wf, pf)),
            sinks,
            sm_scale,
            turn,
            o,
        );
    }
    if let Some(wf) = crate::attn::dsv4::rows_of(window_pool) {
        // DeepSeek's quantized rows: every page of both pools' tables widened
        // into bf16 planes laid out in table order (as vLLM gathers its
        // prefill's rows to bf16), read through the same page_indptr. A
        // compact pool's pages widen a row an entry, as its view reads them.
        let window_bf16 = stage(ctx, op, window_pool, Staged::Window, wf)?;
        // A window-only layer's entries are its window (none of them seen).
        let entries_bf16 = if pooled.entries.keys.ptr == window_pool.keys.ptr {
            window_bf16
        } else {
            let pf = crate::attn::dsv4::rows_of(&ent_view).unwrap_or(wf);
            stage(ctx, op, &ent_view, Staged::Entries, pf)?
        };
        let staged = Pooled {
            entries: &entries_bf16,
            ratio: pooled.ratio,
            selection: pooled.selection,
        };
        return sparse(
            ctx,
            op,
            q,
            positions,
            requests,
            floors,
            &window_bf16,
            &staged,
            ent_compact,
            None,
            sinks,
            sm_scale,
            turn,
            o,
        );
    }
    let viewed = Pooled {
        entries: &ent_view,
        ratio: pooled.ratio,
        selection: pooled.selection,
    };
    sparse(
        ctx,
        op,
        q,
        positions,
        requests,
        floors,
        window_pool,
        &viewed,
        ent_compact,
        None,
        sinks,
        sm_scale,
        turn,
        o,
    )
}

/// The pass over bf16 planes, or with `quantized` (the window's and the
/// entries' formats) over DeepSeek's rows, widened as the tiles land:
/// `pooled.entries` a row a position (its entries at each group's last), or
/// with `ent_compact` a row an entry.
#[allow(clippy::too_many_arguments)]
fn sparse(
    ctx: &Ctx,
    op: &'static str,
    q: Tensor,
    positions: Tensor,
    requests: Tensor,
    floors: Option<Tensor>,
    window_pool: &KvPool,
    pooled: &Pooled<'_>,
    ent_compact: bool,
    quantized: Option<(Rows, Rows)>,
    sinks: Tensor,
    sm_scale: f32,
    turn: Option<crate::attn::dsv4::Turn>,
    o: &mut Tensor,
) -> Result<(), Error> {
    let rows = o.rows;
    let int = |v: u32| i32::try_from(v).unwrap_or(i32::MAX);
    let plane = |pool: &KvPool| {
        encode(Map {
            ptr: pool.keys.ptr,
            ty: dr::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            dims: [HEAD_DIM, pool.keys.rows, 0],
            strides: [u64::try_from(pool.seq_stride).unwrap_or(0) * 2, 0],
            boxes: [64, 1, 1],
            swizzle: dr::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        })
    };
    // Q a row: 64 heads of 512, eight boxes of 64 dims by the 64 heads.
    let map_q = encode(Map {
        ptr: q.ptr,
        ty: dr::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        dims: [HEAD_DIM, HEADS, q.rows],
        strides: [u64::from(HEAD_DIM) * 2, u64::from(q.width) * 2],
        boxes: [64, HEADS, 1],
        swizzle: dr::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
    })?;
    // A quantized pass reads no plane through a tensor map: Q's stands in.
    let (map_win, map_ent, map_win_run, map_ent_run, ent_runs) = if quantized.is_some() {
        (map_q, map_q, map_q, map_q, false)
    } else {
        let map_win = plane(window_pool)?;
        let map_ent = plane(pooled.entries)?;
        // A run of sixteen window positions: sixteen rows a box.
        let map_win_run = encode(Map {
            ptr: window_pool.keys.ptr,
            ty: dr::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            dims: [HEAD_DIM, window_pool.keys.rows, 0],
            strides: [u64::try_from(window_pool.seq_stride).unwrap_or(0) * 2, 0],
            boxes: [64, 16, 1],
            swizzle: dr::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        })?;
        // A page's entries sit `ratio` rows apart at its last row of each
        // stride: viewed as [ratio, cells / ratio] rows, sixteen of one phase a
        // box, where the page and the plane hold whole strides.
        // A compact pool's entries are rows of their own: a run is sixteen
        // consecutive rows, as the window's are.
        let ratio = if ent_compact { 1 } else { pooled.ratio };
        let ent_cells = pooled.entries.keys.rows;
        let ent_runs = ent_compact
            || (ratio > 1
                && ratio <= 16
                && u32::try_from(pooled.entries.page_size)
                    .is_ok_and(|page| page.is_multiple_of(ratio))
                && ent_cells.is_multiple_of(ratio));
        let ent_stride = u64::try_from(pooled.entries.seq_stride).unwrap_or(0) * 2;
        let map_ent_run = encode(Map {
            ptr: pooled.entries.keys.ptr,
            ty: dr::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            dims: if ent_runs {
                [HEAD_DIM, ratio, ent_cells / ratio.max(1)]
            } else {
                [HEAD_DIM, 1, ent_cells]
            },
            strides: if ent_runs {
                [ent_stride, ent_stride * u64::from(ratio)]
            } else {
                [ent_stride, ent_stride]
            },
            boxes: [64, 1, 16],
            swizzle: dr::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        })?;
        (map_win, map_ent, map_win_run, map_ent_run, ent_runs)
    };
    let (selection, top_k) = pooled
        .selection
        .map_or((0, 0), |(selection, top_k)| (selection.ptr, top_k));
    let live = match ctx.stage() {
        ArgValue::Ptr(at) => at,
        _ => 0,
    };
    let bytes = |m: &[u64; 16]| ArgValue::Bytes {
        ptr: m.as_ptr().cast(),
        len: 128,
    };
    let selected = pooled.selection.is_some();
    {
        // A block a row (`attn/dsv4_sparse.cuh`): Q in TMEM, whole 64-key
        // tiles, V the tile K brought.
        let sparse = SparseParams {
            o: o.ptr,
            q: q.ptr,
            positions: positions.ptr,
            requests: requests.ptr,
            floors: floors.map_or(0, |f| f.ptr),
            win_indptr: window_pool.page_indptr.ptr,
            win_indices: window_pool.page_indices.ptr,
            ent_indptr: pooled.entries.page_indptr.ptr,
            ent_indices: pooled.entries.page_indices.ptr,
            selection,
            sinks: sinks.ptr,
            live,
            win_raw: if quantized.is_some() {
                window_pool.keys.ptr
            } else {
                0
            },
            ent_raw: if quantized.is_some() {
                pooled.entries.keys.ptr
            } else {
                0
            },
            sinks_f32: i32::from(sinks.dtype == Dtype::F32),
            ent_runs: i32::from(ent_runs),
            win_page: window_pool.page_size,
            ent_page: pooled.entries.page_size,
            ent_compact: i32::from(ent_compact),
            top_k: int(top_k),
            ratio: int(pooled.ratio),
            rows: int(rows),
            q_stride: int(q.width),
            o_stride: int(o.width),
            scale_log2: sm_scale * core::f32::consts::LOG2_E,
            turn: crate::attn::dsv4::TurnArgs::of(turn),
        };
        ctx.fire(
            op,
            Fire::at(
                SPARSE_FILE,
                symbol(&match quantized {
                    Some((wf, pf)) => format!(
                        "::pie::attn::dsv4s::dsv4_sparse<{selected}, {}, {}>",
                        wf.code(),
                        pf.code()
                    ),
                    None => format!("::pie::attn::dsv4s::dsv4_sparse<{selected}>"),
                }),
            )
            .apply(
                Launch::grid(
                    [rows.min(ctx.multiprocessors().unwrap_or(148)), 1, 1],
                    [384, 1, 1],
                )
                .smem(SPARSE_SMEM),
            ),
            &[
                bytes(&map_win),
                bytes(&map_ent),
                bytes(&map_win_run),
                bytes(&map_ent_run),
                bytes(&map_q),
                ArgValue::Bytes {
                    ptr: (&raw const sparse).cast(),
                    len: core::mem::size_of::<SparseParams>(),
                },
            ],
        )
    }
}

/// [`prefill`] for a layer with a window and no pool: the window's own
/// pool stands in for the entries, at a ratio no position closes.
#[allow(clippy::too_many_arguments)]
pub fn window_only(
    ctx: &Ctx,
    op: &'static str,
    q: Tensor,
    positions: Tensor,
    requests: Tensor,
    graph_safe: bool,
    window_pool: &KvPool,
    sinks: Tensor,
    sm_scale: f32,
    o: &mut Tensor,
) -> Result<(), Error> {
    let pooled = Pooled {
        entries: window_pool,
        ratio: NO_ENTRY,
        selection: None,
    };
    prefill(
        ctx,
        op,
        q,
        positions,
        requests,
        None,
        graph_safe,
        window_pool,
        &pooled,
        sinks,
        sm_scale,
        None,
        o,
    )
}

/// A ratio past any position: a row sees none of the pool.
const NO_ENTRY: u32 = i32::MAX as u32;

/// Which pool a staged plane widens.
#[derive(Clone, Copy)]
enum Staged {
    Window,
    Entries,
}

/// `pool`'s pages, every one its table names, widened from DeepSeek's
/// quantized rows (`fmt`) into a bf16 plane in table order (`stage_pages`): the pool the pass
/// reads instead, its table the identity.
fn stage(
    ctx: &Ctx,
    op: &'static str,
    pool: &KvPool,
    which: Staged,
    fmt: crate::attn::dsv4::Rows,
) -> Result<KvPool, Error> {
    let (plane_name, table_name) = match which {
        Staged::Window => ("dsv4.stage.window", "dsv4.stage.window_table"),
        Staged::Entries => ("dsv4.stage.entries", "dsv4.stage.entries_table"),
    };
    let pages = pool.page_indices.rows;
    let page = u32::try_from(pool.page_size).unwrap_or(0);
    let cells = pages.saturating_mul(page);
    let plane =
        ctx.scratch_shared(op, plane_name, cells as usize * HEAD_DIM as usize * 2)? as usize as u64;
    let table = ctx.scratch_shared(op, table_name, pages as usize * 4)? as usize as u64;
    if pages > 0 {
        ctx.fire(
            op,
            Fire::at(
                SPARSE_FILE,
                symbol(&format!("::pie::attn::dsv4s::stage_pages<{}>", fmt.code())),
            )
            .apply(Launch::grid([pages, 1, 1], [256, 1, 1])),
            &[
                pool.keys.arg(),
                ArgValue::Ptr(plane),
                pool.page_indices.arg(),
                ArgValue::Ptr(table),
                ArgValue::I32(i32::try_from(pages).unwrap_or(i32::MAX)),
                ArgValue::I32(pool.page_size),
            ],
        )?;
    }
    Ok(KvPool {
        keys: Tensor::new(plane, cells, HEAD_DIM, Dtype::Bf16),
        values: Tensor::new(plane, cells, HEAD_DIM, Dtype::Bf16),
        page_indices: Tensor::new(table, pages, 1, Dtype::U32),
        seq_stride: i64::from(HEAD_DIM),
        head_stride: i64::from(HEAD_DIM),
        ..*pool
    })
}

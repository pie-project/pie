//! DeepSeek-V4's attention at prefill widths on tcgen05
//! (`attn/dsv4_prefill.cuh`): the window, the compressed pool (every visible
//! entry or those an index selected), their merge and the sinks in one pass,
//! for the shape the kernel is built for: 64 query heads of 512 over one
//! latent, a 128-key window, bf16 NHD pools. Anything else is the caller's to
//! run as its four ops.

use cudarc::driver::sys as dr;

use crate::error::Error;
use crate::jit::{ArgValue, Ctx, Fire, Launch, aligned16, symbol};
use crate::linear::tc5::{Map, encode, supported};
use crate::tensor::{KvPool, Tensor};
use dtype::Dtype;

const FILE: &str = "attn/dsv4_prefill.cuh";

/// `pie::attn::dsv4p::Params`, field for field.
#[repr(C)]
struct Params {
    o: u64,
    positions: u64,
    requests: u64,
    win_indptr: u64,
    win_indices: u64,
    ent_indptr: u64,
    ent_indices: u64,
    selection: u64,
    sinks: u64,
    live: u64,
    sinks_f32: i32,
    ent_runs: i32,
    win_page: i32,
    ent_page: i32,
    top_k: i32,
    ratio: i32,
    rows: i32,
    o_stride: i32,
    scale_log2: f32,
}

const SPARSE_FILE: &str = "attn/dsv4_sparse.cuh";

/// `pie::attn::dsv4s::Params`, field for field.
#[repr(C)]
struct SparseParams {
    o: u64,
    q: u64,
    positions: u64,
    requests: u64,
    win_indptr: u64,
    win_indices: u64,
    ent_indptr: u64,
    ent_indices: u64,
    selection: u64,
    sinks: u64,
    live: u64,
    sinks_f32: i32,
    ent_runs: i32,
    win_page: i32,
    ent_page: i32,
    top_k: i32,
    ratio: i32,
    rows: i32,
    q_stride: i32,
    o_stride: i32,
    scale_log2: f32,
}

/// `pie::attn::dsv4s::smem_bytes` (a block a multiprocessor, walking rows):
/// three 64-key tiles, P twice, the score
/// exchange, three key lists, barriers, and the 1024-byte alignment.
const SPARSE_SMEM: u32 = 3 * 65536 + 2 * 8192 + 16384 + 6 * 64 * 4 + 512 + 1024;

const HEADS: u32 = 64;
const HEAD_DIM: u32 = 512;
const WINDOW: u32 = 128;

/// `pie::attn::dsv4p::smem_bytes`: Q, three two-atom slices of 128 keys, two
/// key lists, barriers, and the 1024-byte alignment.
const SMEM: u32 = 8 * 128 * 128 + 3 * 2 * 128 * 128 + 3 * 128 * 4 + 512 + 1024;

/// The compressed pool a row reads: its entries, how many positions an
/// entry closes, and the ids an index selected for each row (`top_k` a
/// row, ended early by a negative one), if it selects.
pub struct Pooled<'a> {
    pub entries: &'a KvPool,
    pub ratio: u32,
    pub selection: Option<(Tensor, u32)>,
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
    supported()
        && heads == HEADS
        && head_dim == HEAD_DIM
        && window == Some(WINDOW)
        && pooled.ratio > 0
        && plain_bf16(window_pool)
        && plain_bf16(pooled.entries)
}

/// Lands the sinked attention of every row of `q` in `o`: the row's window
/// of `window_pool` and its compressed entries in one softmax, scaled by
/// `sigmoid(lse - sink)` a head. The caller checks [`takes`] first.
#[allow(clippy::too_many_arguments)]
pub fn prefill(
    ctx: &Ctx,
    op: &'static str,
    q: Tensor,
    positions: Tensor,
    requests: Tensor,
    window_pool: &KvPool,
    pooled: &Pooled<'_>,
    sinks: Tensor,
    sm_scale: f32,
    o: &mut Tensor,
) -> Result<(), Error> {
    let rows = o.rows;
    if rows == 0 {
        return Ok(());
    }
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
    let ratio = pooled.ratio;
    let ent_cells = pooled.entries.keys.rows;
    let ent_runs = ratio > 1
        && ratio * 16 <= 256
        && u32::try_from(pooled.entries.page_size).is_ok_and(|page| page.is_multiple_of(ratio))
        && ent_cells.is_multiple_of(ratio);
    let ent_stride = u64::try_from(pooled.entries.seq_stride).unwrap_or(0) * 2;
    let map_ent_run = encode(Map {
        ptr: pooled.entries.keys.ptr,
        ty: dr::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        dims: if ent_runs {
            [HEAD_DIM, ratio, ent_cells / ratio]
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
    // Two rows' 64 heads a box an atom: [64 columns, 64 heads, 2 rows].
    let map_q = encode(Map {
        ptr: q.ptr,
        ty: dr::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        dims: [HEAD_DIM, HEADS, q.rows],
        strides: [u64::from(HEAD_DIM) * 2, u64::from(q.width) * 2],
        boxes: [64, HEADS, 2],
        swizzle: dr::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
    })?;
    let (selection, top_k) = pooled
        .selection
        .map_or((0, 0), |(selection, top_k)| (selection.ptr, top_k));
    let live = match ctx.stage() {
        ArgValue::Ptr(at) => at,
        _ => 0,
    };
    let params = Params {
        o: o.ptr,
        positions: positions.ptr,
        requests: requests.ptr,
        win_indptr: window_pool.page_indptr.ptr,
        win_indices: window_pool.page_indices.ptr,
        ent_indptr: pooled.entries.page_indptr.ptr,
        ent_indices: pooled.entries.page_indices.ptr,
        selection,
        sinks: sinks.ptr,
        live,
        sinks_f32: i32::from(sinks.dtype == Dtype::F32),
        ent_runs: i32::from(ent_runs),
        win_page: window_pool.page_size,
        ent_page: pooled.entries.page_size,
        top_k: int(top_k),
        ratio: int(pooled.ratio),
        rows: int(rows),
        o_stride: int(o.width),
        scale_log2: sm_scale * core::f32::consts::LOG2_E,
    };
    let bytes = |m: &[u64; 16]| ArgValue::Bytes {
        ptr: m.as_ptr().cast(),
        len: 128,
    };
    let selected = pooled.selection.is_some();
    if ctx.tuning().dsv4_sparse {
        // A block a row (`attn/dsv4_sparse.cuh`): Q in TMEM, whole 64-key
        // tiles, V the tile K brought.
        let sparse = SparseParams {
            o: o.ptr,
            q: q.ptr,
            positions: positions.ptr,
            requests: requests.ptr,
            win_indptr: window_pool.page_indptr.ptr,
            win_indices: window_pool.page_indices.ptr,
            ent_indptr: pooled.entries.page_indptr.ptr,
            ent_indices: pooled.entries.page_indices.ptr,
            selection,
            sinks: sinks.ptr,
            live,
            sinks_f32: i32::from(sinks.dtype == Dtype::F32),
            ent_runs: i32::from(ent_runs),
            win_page: window_pool.page_size,
            ent_page: pooled.entries.page_size,
            top_k: int(top_k),
            ratio: int(pooled.ratio),
            rows: int(rows),
            q_stride: int(q.width),
            o_stride: int(o.width),
            scale_log2: sm_scale * core::f32::consts::LOG2_E,
        };
        return ctx.fire(
            op,
            Fire::at(
                SPARSE_FILE,
                symbol(&format!("::pie::attn::dsv4s::dsv4_sparse<{selected}>")),
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
                ArgValue::Bytes {
                    ptr: (&raw const sparse).cast(),
                    len: core::mem::size_of::<SparseParams>(),
                },
            ],
        );
    }
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!("::pie::attn::dsv4p::dsv4_prefill<{selected}>")),
        )
        .apply(
            Launch::grid([rows.div_ceil(2) * 2, 1, 1], [288, 1, 1])
                .smem(SMEM)
                .cluster(2),
        ),
        &[
            bytes(&map_q),
            bytes(&map_win),
            bytes(&map_ent),
            bytes(&map_win_run),
            bytes(&map_ent_run),
            ArgValue::Bytes {
                ptr: (&raw const params).cast(),
                len: core::mem::size_of::<Params>(),
            },
        ],
    )
}

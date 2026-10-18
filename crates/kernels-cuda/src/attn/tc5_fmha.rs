//! The tcgen05 prefill over a paged bf16 KV cache (`attn/tc5_fmha.cuh`), for
//! the shapes it serves: a causal or full prefill, sliding or not, without a
//! mask, a relative bias or a log-sum-exp, at a 64-, 128-, 256- or 512-wide
//! head, an NHD pool of 16-token pages, and rows long enough that a 128-row
//! tile of packed query heads is mostly full. Everything else stays on FA2.

use crate::attn::fa2_abi::sm_scale_or_default;
use crate::attn::kv;
use crate::attn::plan::PrefillPlan;
use crate::error::Error;
use crate::jit::{ArgValue, Ctx, Fire, Launch, aligned16, symbol};
use crate::linear::tc5::supported;
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const FILE: &str = "attn/tc5_fmha.cuh";

/// The pool's NHD layout code.
const NHD: i32 = 0;

/// `pie::attn::fmha::Params`, field for field.
#[repr(C)]
struct Params {
    q: u64,
    o: u64,
    qo_indptr: u64,
    kv_indptr: u64,
    kv_indices: u64,
    last_page_len: u64,
    live: u64,
    q_stride: i32,
    o_stride: i32,
    kv_heads: i32,
    group: i32,
    page: i32,
    scale_log2: f32,
    causal: i32,
    window_left: i32,
    lse: u64,
    sinks: u64,
}

/// The key tile and ring depth a head width runs at: three stages of K and
/// V fill shared memory, Q and P living in TMEM. A prefill `wide_of` names
/// runs `tc5_fmha_prefill_wide` instead.
fn tiling(head_dim: u32, sliding_wide: bool) -> (u32, u32) {
    if let Some(w) = wide_of(head_dim, sliding_wide) {
        return (w.bkv, w.slots);
    }
    match head_dim {
        256 => (64, 3),
        _ => (128, 3),
    }
}

/// `pie::attn::fmha::smem_bytes`.
const fn smem_bytes(d: u32, bkv: u32, stages: u32) -> u32 {
    2 * stages * (d / 64) * bkv * 128 + 1024 + 1024
}

/// `tc5_fmha_prefill_wide`'s shape: 128-key tiles; Q whole in shared
/// memory (`qt` of its columns in TMEM: none); K and V through a ring of
/// three slices of two 64-wide atoms.
#[derive(Clone, Copy)]
struct Wide {
    d: u32,
    bkv: u32,
    qt: u32,
    atoms: u32,
    slots: u32,
}

/// The prefills `tc5_fmha_prefill_wide` serves: every 512-wide head (O
/// would fill TMEM in the other kernel), and a 256-wide one without a window
/// or where `sliding_wide` says, whose 128-key tiles run S at twice the rate
/// of 64. A sliding one's pages, lying in one run a request, favour them
/// (Gemma-4's: 0.69 -> 0.64 ms a recorded 4 x 4096 fire); scattered, a
/// tile lands as twice the boxes a slice.
fn wide_of(head_dim: u32, sliding_wide: bool) -> Option<Wide> {
    match head_dim {
        512 => Some(Wide {
            d: 512,
            bkv: 128,
            qt: 0,
            atoms: 2,
            slots: 3,
        }),
        256 if sliding_wide => Some(Wide {
            d: 256,
            bkv: 128,
            qt: 0,
            atoms: 2,
            slots: 5,
        }),
        _ => None,
    }
}

/// `pie::attn::fmha::wide::smem_bytes`.
const fn wide_smem_bytes(w: &Wide) -> u32 {
    (w.d - w.qt) / 64 * 128 * 128 + w.slots * w.atoms * w.bkv * 128 + 512 + 1024
}

/// Whether this prefill runs on the tcgen05 kernel, and if so runs it,
/// landing each row's log2-sum-exp2 in `lse` and scaling it down by its
/// head's bf16 sink where given. `Ok(false)` leaves it to FA2.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_prefill(
    ctx: &Ctx,
    op: &'static str,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    window_left: i32,
    head_dim: u32,
    sm_scale: f32,
    o: &Tensor,
    lse: Option<&Tensor>,
    sinks: Option<Tensor>,
) -> Result<bool, Error> {
    let shape = &plan.shape;
    let group = shape.group_size();
    let rows = q.data.rows;
    let requests = shape.num_requests;
    let page = u32::try_from(pool.page_size).unwrap_or(0);
    // A sliding 256-wide prefill runs the 128-key kernel where the profile
    // says (`kernel.sdpa.prefill.wide_sliding`) and its window reaches past
    // two such tiles (MiMo's 128 reads fewer keys in 64s: 844 -> 767 us a
    // 4 x 4096 fire); without a window it always does.
    let sliding_wide = window_left < 0 || (ctx.tuning().prefill_wide_sliding && window_left >= 256);
    // A 64-wide head's block takes two tiles of 128 rows over 64-key tiles,
    // each its own S buffers, where the fire leaves a few such blocks an SM
    // and no short window has them on the IO path below (Muse's 4 x 4096
    // fire: sliding 1967 -> 1751 us, causal 2279 -> 2183).
    let sms = ctx.multiprocessors().unwrap_or(148);
    let pair = head_dim == 64
        && !(0..256).contains(&window_left)
        && (rows * group).div_ceil(256) * shape.num_kv_heads >= 4 * sms;
    let (bkv, stages) = if pair {
        (64, 4)
    } else {
        tiling(head_dim, sliding_wide)
    };
    // The profile's decision (`kernel.sdpa.prefill.tc5`, on for sm_100)
    // and the device's capability both: elsewhere FA2 serves as before.
    let fits = ctx.tuning().prefill_tc5
        && supported()
        && matches!(head_dim, 64 | 128 | 256 | 512)
        && requests > 0
        && kv::native_bf16(pool)
        && pool.layout == NHD
        && page >= 16
        && page.is_multiple_of(16)
        && bkv.is_multiple_of(page)
        && pool.head_stride == i64::from(head_dim)
        && [q.data.ptr, o.ptr, pool.keys.ptr, pool.values.ptr]
            .iter()
            .all(|&at| aligned16(at))
        // A tile is 128 packed query rows: short rows (a decode's) would
        // leave most of it empty and their keys unsplit.
        && u64::from(rows) * u64::from(group) >= 64 * u64::from(requests);
    let Some(live_at) = plan.info.live_lanes_offset else {
        return Ok(false);
    };
    if !fits {
        return Ok(false);
    }
    let kv_heads = shape.num_kv_heads;
    let cells = pool.keys.rows;
    // A box a 16-token page slice, and one a whole tile for a tile whose
    // pages lie in a run.
    let map = |ptr: u64, rows: u32| {
        crate::attn::bf16_map(
            ptr,
            [head_dim, kv_heads, cells],
            [
                u64::from(head_dim) * 2,
                u64::try_from(pool.seq_stride).unwrap_or(0) * 2,
            ],
            [64, 1, rows],
        )
    };
    let map_k = map(pool.keys.ptr, 16)?;
    let map_v = map(pool.values.ptr, 16)?;
    let map_k_run = map(pool.keys.ptr, bkv)?;
    let map_v_run = map(pool.values.ptr, bkv)?;
    let wide = wide_of(head_dim, sliding_wide);
    let int = |v: u32| i32::try_from(v).unwrap_or(i32::MAX);
    let params = Params {
        q: q.data.ptr,
        o: o.ptr,
        qo_indptr: q.indptr.ptr,
        kv_indptr: pool.page_indptr.ptr,
        kv_indices: pool.page_indices.ptr,
        last_page_len: pool.last_page_lens.ptr,
        live: plan.workspace.int_ptr + u64::from(live_at),
        q_stride: int(q.data.width),
        o_stride: int(o.width),
        kv_heads: int(kv_heads),
        group: int(group),
        page: int(page),
        scale_log2: sm_scale_or_default(sm_scale, head_dim) * core::f32::consts::LOG2_E,
        causal: i32::from(plan.causal),
        // The op's window, as FA2 reads it: a plan may serve layers of
        // several windows (Gemma's sliding and global layers share one).
        window_left,
        lse: lse.map_or(0, |t| t.ptr),
        sinks: sinks.map_or(0, |t| t.ptr),
    };
    // Every request's tiles, rounded up a tile each.
    let bound = (rows * group).div_ceil(if pair { 256 } else { 128 }) + requests;
    let tail = [
        ArgValue::of(&map_k),
        ArgValue::of(&map_v),
        ArgValue::of(&map_k_run),
        ArgValue::of(&map_v_run),
        ArgValue::of(&params),
        ArgValue::I32(int(bound)),
    ];
    let Some(w) = wide else {
        // A short window's tiles cost alike and read few keys: given a few a
        // block, a block an SM walks them, a warpgroup of its own taking each
        // tile's Q in and O out under the next's products (MiMo's 4 x 4096
        // fire 685 -> 531 us, GPT-OSS's 475 -> 447). A long window's (Muse's
        // 2048) are the softmax's: its registers go further without.
        let io = (0..256).contains(&window_left) && bound * kv_heads >= 4 * sms;
        let (grid, threads) = match io {
            true => ([sms, 1, 1], 320),
            false => ([bound, kv_heads, 1], if pair { 320 } else { 192 }),
        };
        ctx.fire(
            op,
            Fire::at(
                FILE,
                symbol(&format!(
                    "::pie::attn::fmha::tc5_fmha_prefill<{head_dim}, {bkv}, {stages}, {head_dim}, {io}, {pair}>"
                )),
            )
            .apply(
                Launch::grid(grid, [threads, 1, 1])
                    .smem(smem_bytes(head_dim, bkv, stages)),
            ),
            &tail,
        )?;
        return Ok(true);
    };
    // A 512-wide head's Q: a box an atom of a block's 128 packed rows, whole
    // tokens' `group` heads; when they are not whole tokens the kernel
    // stores Q itself and the map is never read.
    let whole_tokens = 128 % group == 0;
    let map_q = crate::attn::bf16_map(
        q.data.ptr,
        [head_dim, shape.num_q_heads, rows.max(1)],
        [u64::from(head_dim) * 2, u64::from(q.data.width) * 2],
        if whole_tokens {
            [64, group, 128 / group]
        } else {
            [64, 1, 1]
        },
    )?;
    let mut args = vec![ArgValue::of(&map_q)];
    args.extend(tail);
    // Two blocks a key head, each landing half of O, side by side in y.
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::attn::fmha::tc5_fmha_prefill_wide<{}, {}, {}, {}, {}>",
                w.d, w.bkv, w.qt, w.atoms, w.slots
            )),
        )
        .apply(
            Launch::grid([bound, kv_heads * (w.d / 256), 1], [192, 1, 1]).smem(wide_smem_bytes(&w)),
        ),
        &args,
    )?;
    Ok(true)
}

/// A prefill whose key and value heads are rows of planes of their own
/// (an expanded latent attention's: keys `dqk` wide, values `dv`, each at
/// its own head and token stride, the value rows perhaps inside a wider
/// plane), its pages named as a pool's are.
#[derive(Clone, Copy, Debug)]
pub struct Planes {
    /// Queries `[rows][heads * dqk]` at row stride `q_stride`.
    pub q: u64,
    pub q_stride: u32,
    /// Output `[rows][heads * dv]` at row stride `o_stride`.
    pub o: u64,
    pub o_stride: u32,
    /// Key rows: cell `c`, head `h` at `k + (c * k_cell + h * k_head) * 2`.
    pub k: u64,
    pub k_head: u32,
    pub k_cell: u32,
    /// Value rows likewise.
    pub v: u64,
    pub v_head: u32,
    pub v_cell: u32,
    /// Cells (tokens) the planes hold.
    pub cells: u32,
    pub heads: u32,
    pub dqk: u32,
    pub dv: u32,
    /// The ragged rows and the pages each request reads, as a pool's.
    pub qo_indptr: u64,
    pub kv_indptr: u64,
    pub kv_indices: u64,
    pub last_page_len: u64,
    pub page: u32,
    /// Device `[first lane, requests]`.
    pub live: u64,
    pub requests: u32,
    pub rows: u32,
    pub causal: bool,
    pub sm_scale: f32,
}

/// The widths [`prefill_planes`] runs: a 192-wide key over a 128-wide
/// value (DeepSeek-V3's and Kimi-K3's expanded latent heads), 256 over 256
/// (GLM-5.3's), or 128 or 64 for both.
#[must_use]
pub fn planes_take(dqk: u32, dv: u32) -> bool {
    supported() && matches!((dqk, dv), (192, 128) | (256, 256) | (128, 128) | (64, 64))
}

/// The key tile and ring depth a width pair runs at: TMEM holds two S
/// buffers, O and Q (`2 bkv + dv + dqk / 2` columns of 512), shared memory
/// the K and V rings.
const fn planes_tiling(dqk: u32, dv: u32) -> (u32, u32) {
    match (dqk, dv) {
        (256, 256) => (64, 3),
        (192, 128) => (128, 2),
        _ => (128, 3),
    }
}

/// One query head a key head, every head's rows in [`Planes`]: the tcgen05
/// prefill at a key width and a value width of their own.
pub fn prefill_planes(ctx: &Ctx, op: &'static str, at: &Planes) -> Result<(), Error> {
    use crate::jit::refuse;
    if !planes_take(at.dqk, at.dv) {
        return Err(refuse(
            op,
            format!("no tcgen05 prefill reads {} x {} heads", at.dqk, at.dv),
        ));
    }
    let (bkv, stages) = planes_tiling(at.dqk, at.dv);
    if at.page < 16 || !at.page.is_multiple_of(16) || !bkv.is_multiple_of(at.page) {
        return Err(refuse(
            op,
            format!(
                "a {}-token page is not whole 16-token slices of a {bkv}-key tile",
                at.page
            ),
        ));
    }
    if ![at.q, at.o, at.k, at.v].iter().all(|&p| aligned16(p)) {
        return Err(refuse(op, "a plane is not 16-byte aligned"));
    }
    let map = |ptr: u64, d: u32, head: u32, cell: u32, rows: u32| {
        crate::attn::bf16_map(
            ptr,
            [d, at.heads, at.cells.max(1)],
            [u64::from(head) * 2, u64::from(cell) * 2],
            [64, 1, rows],
        )
    };
    let map_k = map(at.k, at.dqk, at.k_head, at.k_cell, 16)?;
    let map_v = map(at.v, at.dv, at.v_head, at.v_cell, 16)?;
    let map_k_run = map(at.k, at.dqk, at.k_head, at.k_cell, bkv)?;
    let map_v_run = map(at.v, at.dv, at.v_head, at.v_cell, bkv)?;
    let int = |v: u32| i32::try_from(v).unwrap_or(i32::MAX);
    let params = Params {
        q: at.q,
        o: at.o,
        qo_indptr: at.qo_indptr,
        kv_indptr: at.kv_indptr,
        kv_indices: at.kv_indices,
        last_page_len: at.last_page_len,
        live: at.live,
        q_stride: int(at.q_stride),
        o_stride: int(at.o_stride),
        kv_heads: int(at.heads),
        group: 1,
        page: int(at.page),
        scale_log2: sm_scale_or_default(at.sm_scale, at.dqk) * core::f32::consts::LOG2_E,
        causal: i32::from(at.causal),
        window_left: -1,
        lse: 0,
        sinks: 0,
    };
    let bound = at.rows.div_ceil(128) + at.requests;
    let (dqk, dv) = (at.dqk, at.dv);
    // `pie::attn::fmha::smem_bytes(dqk, bkv, stages, dv)`.
    let smem = stages * (dqk / 64 + dv / 64) * bkv * 128 + 1024 + 1024;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::attn::fmha::tc5_fmha_prefill<{dqk}, {bkv}, {stages}, {dv}>"
            )),
        )
        .apply(Launch::grid([bound.max(1), at.heads, 1], [192, 1, 1]).smem(smem)),
        &[
            ArgValue::of(&map_k),
            ArgValue::of(&map_v),
            ArgValue::of(&map_k_run),
            ArgValue::of(&map_v_run),
            ArgValue::of(&params),
            ArgValue::I32(int(bound)),
        ],
    )
}

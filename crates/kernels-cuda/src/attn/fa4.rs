//! FlashAttention-4 forward — the host half of `attn/fa4.cuh`.
//!
//! The kernel's design, its numerics and the one register-layout coincidence
//! the whole thing rests on are `kernels/attn/fa4.cuh`'s long header; that
//! file is the source of truth and this one does not restate it. What is here
//! is the launch: the four instantiations upstream's `interface.py` tunes for
//! SM120, the compile-time shape each carries, and the twenty-three operands
//! bound in the exact order of the `__global__`'s parameter list.
//!
//! # One kernel per `(head_dim, causal)`, and no arm at run time
//!
//! `TILE_M`, `TILE_N` and the tile geometry are the same for every point —
//! `128 x 128`, `num_stages=1` — so the only things that vary across the four
//! are the head dimension and the causal flag, and both are template
//! parameters. There is therefore no host decision beyond selecting the
//! instantiation string: the raggedness a paged kernel would carry in a plan
//! is absent because this path is dense and non-paged.
//!
//! # The instantiation strings are literals on purpose
//!
//! `tests/every_instantiation_compiles.rs` scrapes `&'static str` template-ids
//! out of this crate's source and puts each through NVRTC. The four ids below
//! are written out rather than assembled so that test compiles exactly what
//! [`forward`] fires, and [`forward_instantiation`] is reached only from [`forward`],
//! which names one carried file — `attn/fa4.cuh` — so the test attributes all
//! four ids to it.

use kernels::{Bind, Fire};
use core::ptr::NonNull;

use crate::jit::abi::bf16;
use crate::jit::{Abi, Ctx, Launch};
use kernels::Refusal;

/// The Q-tile height for prefill, `interface.py`'s `FwdConfig(128, ...)` for
/// SM120.
///
/// The grid's first axis is `ceil_div(seqlen_q * pack, tile_m)`: one block per
/// row tile.
const TILE_M: u32 = 128;

/// The Q-tile height for decode.
///
/// A decode fills a handful of the M tile's rows and pads the rest, and the
/// MMA does not know the difference — a `TILE_M`-row tile holding four real
/// rows spends thirty-two times the arithmetic it needs. Halving the tile
/// halves that, and on this device it is exactly what moves a batch-1 decode
/// off compute and onto memory:
///
/// | b1 sk4096 hq32 hk8 | tile 128 | tile 64 | tile 32 | tile 16 |
/// |--------------------|----------|---------|---------|---------|
/// | d128               | 25.6us   | 18.9    | 19.0    | 19.2    |
/// | achieved bandwidth | 655 GB/s | 887     | 884     | 873     |
///
/// It stops at 64 because that is where the wall is: 32 and 16 rows buy
/// nothing, so the smallest useful tile is the largest one that reaches it.
/// Prefill keeps the 128 tile, which is 4-13% faster there — this is a
/// per-launch choice, not a retuning. See [`plan`].
const TILE_M_SMALL: u32 = 64;

/// The KV-tile width, the same `FwdConfig(_, 128, ...)`.
///
/// Only [`smem_bytes`] reads it — the launch does not tile K on the host —
/// but it is the other half of the shared-memory arithmetic and is spelled
/// once so the two cannot drift.
const TILE_N: u32 = 128;

/// What a head dim outside the tuned set gets.
///
/// The four points below are the ones measured on hardware; a fifth is a
/// `Traits` instantiation away rather than a fallback, exactly as
/// `cascade`'s `NO_ROW` is. Naming the supported dims makes the refusal
/// actionable instead of a driver fault at the first fire.
const NO_HEAD_DIM: Refusal = Refusal::Unstated {
    what: "an FA4 forward at this head dim -- 64 and 128 are here",
};

/// The template-id for `(head_dim, causal, packed, small)`, or `None` off the
/// tuned set.
///
/// Spelled fully as NVRTC matches a name expression: `__nv_bfloat16` and not
/// `::pie::bf16`, because `attn/fa4.cuh` does its arithmetic in the CUDA type
/// and never includes the prelude that would define the wrapper — the same
/// choice `attn/fa2.cuh:157` makes with `using DTypeQ = __nv_bfloat16`. The
/// `u` suffixes on the `uint32_t` parameters are load-bearing for the same
/// reason `mla_fa2::inst` documents: a name expression is matched as WRITTEN
/// before it is matched as parsed. `NUM_WARPS` is given explicitly rather than
/// left to `Traits`' default so the id the test compiles is the id fired.
///
/// `PACKED` is the seventh parameter. It has to be compile-time: see
/// `fa4.cuh`'s `Traits::PACKED`, where leaving it dynamic measured 4% on a
/// shape that does no packing at all, because the row-mapping table costs
/// registers a kernel already at 255 of them does not have.
///
/// `small` is the M tile: 128 rows over 8 warps at head dim 128 and 4 at 64,
/// or 64 rows over 4 warps for both. Sixteen ids, and a warp count that is a
/// function of the tile because warps split M and never N — 64 rows cannot
/// occupy 8 warps at 16 rows each. See [`plan`] for which is fired and why.
const fn forward_instantiation(
    head_dim: u32,
    causal: bool,
    packed: bool,
    small: bool,
) -> Option<&'static str> {
    Some(match (head_dim, causal, packed, small) {
        (64, false, false, false) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 64u, 128u, 128u, false, 4u, false>>"
        }
        (64, false, true, false) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 64u, 128u, 128u, false, 4u, true>>"
        }
        (64, true, false, false) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 64u, 128u, 128u, true, 4u, false>>"
        }
        (64, true, true, false) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 64u, 128u, 128u, true, 4u, true>>"
        }
        (64, false, false, true) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 64u, 64u, 128u, false, 4u, false>>"
        }
        (64, false, true, true) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 64u, 64u, 128u, false, 4u, true>>"
        }
        (64, true, false, true) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 64u, 64u, 128u, true, 4u, false>>"
        }
        (64, true, true, true) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 64u, 64u, 128u, true, 4u, true>>"
        }
        (128, false, false, false) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 128u, 128u, 128u, false, 8u, false>>"
        }
        (128, false, true, false) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 128u, 128u, 128u, false, 8u, true>>"
        }
        (128, true, false, false) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 128u, 128u, 128u, true, 8u, false>>"
        }
        (128, true, true, false) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 128u, 128u, 128u, true, 8u, true>>"
        }
        (128, false, false, true) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 128u, 64u, 128u, false, 4u, false>>"
        }
        (128, false, true, true) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 128u, 64u, 128u, false, 4u, true>>"
        }
        (128, true, false, true) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 128u, 64u, 128u, true, 4u, false>>"
        }
        (128, true, true, true) => {
            "::pie::attn::fa4::kernel<::pie::attn::fa4::Traits<__nv_bfloat16, 128u, 64u, 128u, true, 4u, true>>"
        }
        _ => return None,
    })
}

/// The combine pass's instantiation, one per head dim.
///
/// `combine` reads only `HEAD_DIM` and `DTypeIn` off its `Traits`, so the mask
/// and packing parameters are free — pinned here at the unpacked non-causal
/// spelling so two entries serve all eight forward variants rather than
/// eight more roots for the JIT to compile.
const fn combine_instantiation(head_dim: u32) -> Option<&'static str> {
    Some(match head_dim {
        64 => {
            "::pie::attn::fa4::combine<::pie::attn::fa4::Traits<__nv_bfloat16, 64u, 128u, 128u, false, 4u, false>>"
        }
        128 => {
            "::pie::attn::fa4::combine<::pie::attn::fa4::Traits<__nv_bfloat16, 128u, 128u, 128u, false, 8u, false>>"
        }
        _ => return None,
    })
}

const fn smem_bytes(head_dim: u32, small: bool) -> u32 {
    (tile_m(small) + 2 * TILE_N) * head_dim * core::mem::size_of::<u16>() as u32
}

/// The M tile of the chosen geometry, 128 rows or 64.
const fn tile_m(small: bool) -> u32 {
    if small { TILE_M_SMALL } else { TILE_M }
}

/// `(num_threads, smem_bytes, tile_m)` for the geometry, or `None` off the set.
///
/// `num_threads = NUM_WARPS * 32` — 128 everywhere but the large tile at head
/// dim 128, which is 256, because warps split M and 128 rows over 8 warps is
/// the same 16 rows each that 64 rows over 4 warps is — is the
/// block width the launch states and the `__launch_bounds__` the kernel was
/// compiled under, so the two cannot disagree. `tile_m` is returned so the
/// grid arithmetic reads the same constant the instantiation baked in.
const fn geometry(head_dim: u32, small: bool) -> Option<(u32, u32, u32)> {
    let num_warps = match (head_dim, small) {
        (64, _) | (128, true) => 4,
        (128, false) => 8,
        _ => return None,
    };
    Some((num_warps * 32, smem_bytes(head_dim, small), tile_m(small)))
}

/// Row-tile blocks for a query length: `ceil_div(seqlen_q * pack, TILE_M)`.
///
/// The grid's first axis. `pack` is the number of query heads folded into each
/// M row — `group_size` when packing and 1 when not — so this is the packed
/// row count, not the query length. A `const fn` so the launch and the test
/// read one definition of the arithmetic the kernel indexes against.
const fn blocks_m(seqlen_q: u32, pack: u32, small: bool) -> u32 {
    (seqlen_q * pack).div_ceil(tile_m(small))
}

/// The largest number of key-range splits a launch will ever ask for.
///
/// Bounds the scratch a caller has to size and the combine's inner loop. Eight
/// is already past the point where splitting pays on this device — the whole
/// machine is filled well before then — and the cap only binds on shapes that
/// would gain nothing from going further.
pub const MAX_SPLITS: u32 = 32;

/// What a block costs before it walks a single key tile, in key tiles.
///
/// Staging Q, the prologue, and — when split — writing a row of f32 scratch
/// the combine then reads back. Fitted, not derived, and pinned from both
/// sides by measurement: at 1 it splits batch 8 five ways, which is slower
/// than not splitting at all, and at 3 it stops splitting batch 32, which
/// costs the 12% that splitting there is worth.
const FIXED_TILES: u32 = 2;

/// How to lay this launch out: whether to pack the group, and how many ways to
/// split the key range.
///
/// These are one decision, not two. Packing folds a KV group's query heads
/// into the M tile, which divides the block count — and so the KV traffic — by
/// the group size, but at low batch that empties a machine the kernel can only
/// fill one block per SM. Splitting gives a block a contiguous RANGE of key
/// tiles instead of all of them, which multiplies the block count back up but
/// does nothing about the group re-reading the same keys. Chosen separately,
/// each looks like a loss:
///
/// | b1 sk4096 hq32 hk8 d128 | splits 1 | 2   | 4   | 8   | 16  |
/// |-------------------------|----------|-----|-----|-----|-----|
/// | unpacked                | 138us    | 74  | 77  | 81  | 74  |
/// | packed                  | 138us    | 72  | 40  | 23  | 25  |
///
/// The cost that predicts that table, and every other measurement taken, is
/// `waves * (key tiles per block + 1)`: the kernel is occupancy-limited, so a
/// grid costs one pass over the machine per wave, and a block costs the key
/// tiles it walks plus a fixed term for the ones it does not — staging Q,
/// and writing a row of scratch the combine then has to read. Packing alone
/// keeps the wave count at one and the tiles at 32; splitting alone cuts the
/// tiles but spends the saving on waves; together they reach one wave and
/// four tiles.
///
/// That `+ 1` is not a fudge, it is the whole difference between batch 8 and
/// batch 32. Without it the model splits until every block walks a single key
/// tile, which is where the fixed cost is all there is: at batch 8 that is
/// measured 215us against 191us for no splitting at all, while at batch 32 —
/// where 256 blocks is 3.1 waves and a quarter of the last one is idle —
/// splitting four ways really does pay, 736us against 822us. One term
/// separates the two.
///
/// Splitting is only offered when the query side is a SINGLE M tile. With more
/// than one the M dimension already supplies the parallelism splitting would
/// buy, while the scratch and the combine pass are sized by the output rows —
/// which is exactly what is large in that case. That gate is what keeps long
/// prefill on the unsplit path it was tuned for.
///
/// Ties go to the smaller split and then to unpacked, in that order: both are
/// the configuration that does not pay for scratch, and at equal modelled cost
/// more blocks hide latency better.
///
/// `num_sms` of zero means the device would not say. That is not a reason to
/// refuse a launch — it only feeds this comparison — so it falls back to
/// `(false, 1)`, which is correct for every shape and merely slower for some.
/// What [`plan`] decided: the packing, the split count and the M tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Plan {
    packed: bool,
    splits: u32,
    small: bool,
}

const fn plan(
    seqlen_q: u32,
    seqlen_k: u32,
    batch: u32,
    heads_q: u32,
    heads_kv: u32,
    num_sms: u32,
    may_split: bool,
) -> Plan {
    if num_sms == 0 || batch == 0 || heads_kv == 0 {
        return Plan { packed: false, splits: 1, small: false };
    }
    let group = heads_q / heads_kv;
    let tiles = if seqlen_k.div_ceil(TILE_N) == 0 { 1 } else { seqlen_k.div_ceil(TILE_N) };
    let mut best_cost = u32::MAX;
    let mut best = Plan { packed: false, splits: 1, small: false };

    let mut which = 0;
    while which < 2 {
        let packed = which == 1;
        if packed && group <= 1 {
            break;
        }
        let pack = if packed { group } else { 1 };
        // The small tile is worth its 4-13% prefill cost only where the tile
        // is mostly padding, which is exactly where it fits in one block: a
        // query side of 64 packed rows or fewer. Above that the large tile is
        // both faster per row and no more wasteful.
        let small = seqlen_q * pack <= TILE_M_SMALL;
        let m_tiles = blocks_m(seqlen_q, pack, small);
        let blocks = m_tiles * (heads_q / pack) * batch;
        let limit = if may_split && m_tiles == 1 {
            if tiles < MAX_SPLITS { tiles } else { MAX_SPLITS }
        } else {
            1
        };
        let mut splits = 1;
        while splits <= limit {
            let cost = (blocks * splits).div_ceil(num_sms) * (tiles.div_ceil(splits) + FIXED_TILES);
            if cost < best_cost {
                best_cost = cost;
                best = Plan { packed, splits, small };
            }
            splits += 1;
        }
        which += 1;
    }
    best
}

/// How much split-KV scratch a launch of this shape will use, in ELEMENTS.
///
/// `(o_partial, lse_partial)`, both `f32`. Zero for both means this shape will
/// not split whatever it is given, so a caller may skip the allocation and
/// bind nulls. The answer is exact rather than an upper bound: it plans
/// against the same numbers [`forward`] will, so allocating this much is
/// enough and no more.
///
/// `num_sms` comes from the same `ctx.multiprocessors()` the launch reads,
/// which is why this takes a `Ctx` rather than being a free arithmetic
/// function — the plan depends on the device.
pub fn split_scratch_elems(
    ctx: &Ctx<'_>,
    seqlen_q: u32,
    seqlen_k: u32,
    batch: u32,
    heads_q: u32,
    heads_kv: u32,
    head_dim: u32) -> (usize, usize) {
    let num_sms = ctx.multiprocessors().unwrap_or(0);
    let p = plan(seqlen_q, seqlen_k, batch, heads_q, heads_kv, num_sms, true);
    if p.splits == 1 {
        return (0, 0);
    }
    let rows = batch as usize * heads_q as usize * seqlen_q as usize * p.splits as usize;
    (rows * head_dim as usize, rows)
}

/// A required pointer, refused by name rather than faulted inside the kernel.
///
/// `q`, `k`, `v` and `o` are dereferenced by every thread; only `lse` is
/// nullable (`fa4.cuh` tests it), so a zero in one of the four is a bind error
/// the host can name — `cascade::null_check`'s discipline.
fn null_check(is_null: bool, which: &'static str) -> Result<(), Refusal> {
    if is_null {
        Err(Refusal::Null { what: which })
    } else {
        Ok(())
    }
}

/// One dense FA4 forward launch's operands.
///
/// One struct rather than twenty-three positional arguments: the strides alone
/// are fourteen `i32`s in four near-identical groups, and an order that
/// type-checks is not an order a reader can verify — `cascade::VarLen` bundles
/// for the same reason. Strides are ELEMENT strides, never byte strides
/// (`fa4.cuh` adds them to a typed `DTypeIn*` before any `sizeof`), so the
/// launcher must not scale them.
#[derive(Clone, Copy, Debug)]
pub struct Fa4 {
    /// `[batch, seqlen_q, heads_q, head_dim]` bf16 queries, `bshd`.
    pub q: *const bf16,
    /// `[batch, seqlen_k, heads_kv, head_dim]` bf16 keys.
    pub k: *const bf16,
    /// `[batch, seqlen_k, heads_kv, head_dim]` bf16 values.
    pub v: *const bf16,
    /// `[batch, seqlen_q, heads_q, head_dim]` bf16 output, written, q's shape.
    pub o: *mut bf16,
    /// `[batch, heads_q, seqlen_q]` f32 log-sum-exp, written — or null to
    /// skip it, which the kernel honours branch by branch.
    pub lse: *mut f32,

    /// Split-KV scratch, or a pair of nulls to forbid splitting.
    ///
    /// Sized by [`split_scratch_elems`], which answers for the same shape the
    /// launch will plan against. Both null is the whole feature declined: the
    /// launch then runs one block per output row as it always did, which is
    /// correct for every shape and slower only where splitting would have
    /// helped. One null and one not is a bind error rather than a silent
    /// half-measure.
    pub o_partial: *mut f32,
    pub lse_partial: *mut f32,

    /// Q element strides for the batch, sequence and head axes.
    pub q_stride_b: i32,
    pub q_stride_s: i32,
    pub q_stride_h: i32,
    /// K element strides, same three axes.
    pub k_stride_b: i32,
    pub k_stride_s: i32,
    pub k_stride_h: i32,
    /// V element strides, same three axes.
    pub v_stride_b: i32,
    pub v_stride_s: i32,
    pub v_stride_h: i32,
    /// O element strides, same three axes.
    pub o_stride_b: i32,
    pub o_stride_s: i32,
    pub o_stride_h: i32,
    /// LSE element strides for the batch and head axes; the sequence stride is
    /// 1 and the kernel assumes it. `heads_q * seqlen_q` and `seqlen_q` for a
    /// packed `[batch, heads_q, seqlen_q]`, ignored when `lse` is null.
    pub lse_stride_b: i32,
    pub lse_stride_h: i32,

    /// Batch size — the grid's third axis.
    pub batch: u32,
    /// Query heads — the grid's second axis. Must be a multiple of `heads_kv`.
    pub heads_q: u32,
    /// KV heads. `group_size = heads_q / heads_kv`; equal to `heads_q` is
    /// plain MHA and a smaller value is GQA.
    pub heads_kv: u32,
    /// Head dimension, which selects the instantiation: 64 or 128.
    pub head_dim: u32,
    /// Query and key sequence lengths. Causal alignment is bottom-right, so
    /// the kernel forms `causal_offset = seqlen_k - seqlen_q` itself.
    pub seqlen_q: u32,
    pub seqlen_k: u32,
    /// Whether the mask is causal — a template parameter, so it picks the
    /// kernel rather than a runtime branch.
    pub causal: bool,
    /// `softmax_scale * log2(e)`. Folded on the host (`fa4.cuh`'s exponentials
    /// are base-2), so the kernel multiplies by it and does not scale again.
    pub scale_log2: f32,
}

/// Fire one dense FA4 forward.
///
/// The host decisions, all of them plain `if`s:
///
/// 1. **Head dim.** [`forward_instantiation`] and [`geometry`] refuse anything but 64
///    or 128; a `Traits` point is the way to add a third, not a fallback.
/// 2. **GQA.** `heads_q % heads_kv != 0` has no `group_size`, and dividing
///    would silently truncate the ratio the kernel reads keys with.
/// 3. **Empty work.** A zero batch, head count or sequence length is nothing
///    to launch; the driver refuses a zero grid axis and this names which
///    extent was zero instead.
///
/// The grid is `(ceil_div(seqlen_q, TILE_M), heads_q, batch)` and the block is
/// `(num_threads, 1, 1)` with `smem_bytes` dynamic shared memory — exactly the
/// geometry the validated harness launched.
///
/// # Errors
///
/// [`Refusal::Null`] for a null `q`/`k`/`v`/`o`, [`Refusal::Unstated`] for an
/// unsupported head dim or a `heads_q` not divisible by `heads_kv`, and
/// [`Refusal::Empty`] for empty work.
///
/// # Safety
///
/// Every non-null pointer in `job` must address device memory of the extent
/// its shape and strides describe, live across the launch, and `ctx`'s stream
/// must outlive it — the obligation any `cudaLaunchKernel` carries.
pub unsafe fn forward(ctx: &Ctx<'_>, job: Fa4) -> Result<(), Refusal> {
    // The head dim is refused here, before any pointer is looked at, so an
    // unsupported one is named rather than reaching a plan that would have to
    // invent a geometry for it. Which tile it gets is decided below.
    geometry(job.head_dim, false).ok_or(NO_HEAD_DIM)?;

    null_check(job.q.is_null(), "q")?;
    null_check(job.k.is_null(), "k")?;
    null_check(job.v.is_null(), "v")?;
    null_check(job.o.is_null(), "o")?;

    if job.heads_kv == 0 {
        return Err(Refusal::Empty { what: "heads_kv" });
    }
    if !job.heads_q.is_multiple_of(job.heads_kv) {
        return Err(Refusal::Unstated {
            what: "a GQA group -- heads_q must be a multiple of heads_kv",
        });
    }
    if job.batch == 0 {
        return Err(Refusal::Empty { what: "batch" });
    }
    if job.heads_q == 0 {
        return Err(Refusal::Empty { what: "heads_q" });
    }
    if job.seqlen_q == 0 {
        return Err(Refusal::Empty { what: "seqlen_q" });
    }
    if job.seqlen_k == 0 {
        return Err(Refusal::Empty { what: "seqlen_k" });
    }

    let group_size = (job.heads_q / job.heads_kv) as i32;

    // Packing changes what a block means, so it changes both grid axes: the M
    // axis counts packed rows rather than query positions, and the head axis
    // walks KV heads rather than query heads. `multiprocessors` failing is not
    // a reason to refuse the launch -- it only feeds the wave comparison, and
    // zero there means "do not pack", which is the conservative answer.
    let num_sms = ctx.multiprocessors().unwrap_or(0);
    if job.o_partial.is_null() != job.lse_partial.is_null() {
        return Err(Refusal::Null {
            what: if job.o_partial.is_null() { "o_partial" } else { "lse_partial" },
        });
    }
    let Plan { packed, splits, small } = plan(
        job.seqlen_q,
        job.seqlen_k,
        job.batch,
        job.heads_q,
        job.heads_kv,
        num_sms,
        !job.o_partial.is_null(),
    );
    let instantiation =
        forward_instantiation(job.head_dim, job.causal, packed, small).ok_or(NO_HEAD_DIM)?;
    // The tile is a plan output, so the block width and the shared memory the
    // launch states have to be read back from it rather than from the head dim
    // alone -- the small tile is 4 warps at both head dims.
    let (num_threads, smem, _tile_m) = geometry(job.head_dim, small).ok_or(NO_HEAD_DIM)?;

    let pack = if packed { group_size as u32 } else { 1 };
    // The split rides the FAST axis of `z`, so splits of one batch row land on
    // nearby SMs and share the keys they pull into L2. `fa4.cuh` undoes it
    // with the same `num_splits`.
    let grid = [
        blocks_m(job.seqlen_q, pack, small),
        job.heads_q / pack,
        job.batch * splits,
    ];

    // The twenty-three operands, in the exact order of `fa4.cuh:364`'s
    // parameter list. `lse` binds through `Option<NonNull>` so a null skips
    // the log-sum-exp write, as `cascade` binds its nullable `s_merged`.
    //
    // SAFETY: the caller's contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    ctx.fire(Fire::at("attn/fa4.cuh", instantiation).apply(Launch::grid(grid, [num_threads, 1, 1]).smem(smem)), &[
                job.q.arg(),
                job.k.arg(),
                job.v.arg(),
                job.o.arg(),
                NonNull::new(job.lse).arg(),
                job.q_stride_b.arg(),
                job.q_stride_s.arg(),
                job.q_stride_h.arg(),
                job.k_stride_b.arg(),
                job.k_stride_s.arg(),
                job.k_stride_h.arg(),
                job.v_stride_b.arg(),
                job.v_stride_s.arg(),
                job.v_stride_h.arg(),
                job.o_stride_b.arg(),
                job.o_stride_s.arg(),
                job.o_stride_h.arg(),
                job.lse_stride_b.arg(),
                job.lse_stride_h.arg(),
                (job.seqlen_q as i32).arg(),
                (job.seqlen_k as i32).arg(),
                group_size.arg(),
                job.o_partial.arg(),
                job.lse_partial.arg(),
                (splits as i32).arg(),
                (job.heads_q as i32).arg(),
                job.scale_log2.arg(),
            ])?;

    if splits == 1 {
        return Ok(());
    }

    // The splits hold partial attentions, each normalised by its own range's
    // denominator. Only this pass knows the true one, so `o` and `lse` are
    // written here rather than above -- one block per output row, one thread
    // per channel.
    let rows = job.batch * job.heads_q * job.seqlen_q;
    let combine = combine_instantiation(job.head_dim).ok_or(NO_HEAD_DIM)?;
    ctx.fire(Fire::at("attn/fa4.cuh", combine).apply(Launch::grid([rows, 1, 1], [job.head_dim, 1, 1])), &[
                job.o_partial.cast_const().arg(),
                job.lse_partial.cast_const().arg(),
                job.o.arg(),
                NonNull::new(job.lse).arg(),
                (splits as i32).arg(),
                (job.heads_q as i32).arg(),
                (job.seqlen_q as i32).arg(),
                job.o_stride_b.arg(),
                job.o_stride_s.arg(),
                job.o_stride_h.arg(),
                job.lse_stride_b.arg(),
                job.lse_stride_h.arg(),
            ])
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_SPLITS, Plan, TILE_M, TILE_M_SMALL, TILE_N, blocks_m, forward_instantiation, geometry,
        plan, smem_bytes,
    };

    /// The tuned set is exactly `(64, 128) x (false, true) x (false, true)`.
    ///
    /// A head dim off the set must be `None` rather than silently reusing a
    /// neighbour's kernel — the failure `cascade`'s `throw std::invalid_argument`
    /// had, where an uninstantiated dim aborted with no message.
    #[test]
    fn only_the_four_tuned_points_instantiate() {
        assert!(
            forward_instantiation(96, false, false, false).is_none(),
            "96 is not tuned"
        );
        assert!(
            forward_instantiation(256, true, true, true).is_none(),
            "256 is not tuned"
        );
        assert!(
            forward_instantiation(0, false, false, false).is_none(),
            "0 is not a head dim"
        );
        for &head_dim in &[64u32, 128u32] {
            for &causal in &[false, true] {
                for &packed in &[false, true] {
                    for &small in &[false, true] {
                        assert!(
                            forward_instantiation(head_dim, causal, packed, small).is_some(),
                            "{head_dim} {causal} {packed} {small} is tuned"
                        );
                    }
                }
            }
        }
    }

    /// The sixteen ids are distinct: no two points share a kernel.
    ///
    /// If two were the same string the causal and full masks — or the two head
    /// dims, or the packed and unpacked forms, or the two M tiles — would fire
    /// one kernel, and the wrong answer that produces is one no assertion on
    /// the device would catch. Enumerated rather than spelled out because the
    /// ids now differ from each other in one trailing word in three different
    /// positions.
    #[test]
    fn the_sixteen_ids_are_distinct_and_nonempty() {
        let mut ids = Vec::new();
        for &head_dim in &[64u32, 128u32] {
            for &causal in &[false, true] {
                for &packed in &[false, true] {
                    for &small in &[false, true] {
                        ids.push(forward_instantiation(head_dim, causal, packed, small).unwrap());
                    }
                }
            }
        }
        assert_eq!(ids.len(), 16);
        for id in &ids {
            assert!(!id.is_empty());
        }
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                assert_ne!(ids[i], ids[j], "ids {i} and {j} collide");
            }
        }
    }

    /// The small tile is 64 rows over 4 warps, at BOTH head dims.
    ///
    /// Warps split M and never N, so a 64-row tile cannot occupy 8 warps at
    /// the 16 rows each the MMA fragment wants. Head dim 128 therefore changes
    /// its warp count with the tile, which is the one place the two geometry
    /// axes are not independent.
    #[test]
    fn the_small_tile_drops_head_dim_128_to_four_warps() {
        assert!(forward_instantiation(128, false, false, true).unwrap().contains("64u, 128u, false, 4u"));
        assert!(forward_instantiation(128, false, false, false).unwrap().contains("128u, 128u, false, 8u"));
        assert!(forward_instantiation(64, false, false, true).unwrap().contains("64u, 128u, false, 4u"));
        assert_eq!(geometry(128, true).unwrap(), (128, (64 + 256) * 128 * 2, 64));
        assert_eq!(geometry(128, false).unwrap(), (256, (128 + 256) * 128 * 2, 128));
    }

    /// The head dim and causal flag reach the right place in the id.
    ///
    /// A transposition — `true` written where `false` is meant, or `64u` in
    /// the `128` id — would type-check and fire the wrong kernel, so this pins
    /// the substrings the four differ by rather than trusting distinctness.
    #[test]
    fn each_id_names_its_own_shape() {
        assert!(
            forward_instantiation(64, false, false, false)
                .unwrap()
                .contains("64u, 128u, 128u, false, 4u, false")
        );
        assert!(
            forward_instantiation(64, true, false, false)
                .unwrap()
                .contains("64u, 128u, 128u, true, 4u, false")
        );
        assert!(
            forward_instantiation(128, false, false, false)
                .unwrap()
                .contains("128u, 128u, 128u, false, 8u, false")
        );
        assert!(
            forward_instantiation(128, true, true, false)
                .unwrap()
                .contains("128u, 128u, 128u, true, 8u, true")
        );
        assert!(
            forward_instantiation(64, false, true, false)
                .unwrap()
                .contains("64u, 128u, 128u, false, 4u, true")
        );
    }

    /// The host shared-memory figure is the kernel's formula, recomputed.
    ///
    /// `(TILE_M + 2 * TILE_N) * HEAD_DIM * 2` independently, against
    /// [`smem_bytes`], so a drift between this launcher and `fa4.cuh:204`
    /// fails here rather than as an out-of-resources launch on the GPU. The
    /// 98,304 B point is the one that matters: it is above the 48 KB default
    /// and the number `jit/launch.rs` must raise the cap to.
    #[test]
    fn shared_memory_matches_the_kernel_formula() {
        for &head_dim in &[64u32, 128u32] {
            let want = (TILE_M + 2 * TILE_N) * head_dim * 2;
            assert_eq!(smem_bytes(head_dim, false), want, "smem at {head_dim}");
            assert_eq!(
                geometry(head_dim, false).unwrap().1,
                want,
                "geometry smem at {head_dim}"
            );
            let want_small = (TILE_M_SMALL + 2 * TILE_N) * head_dim * 2;
            assert_eq!(smem_bytes(head_dim, true), want_small, "small smem at {head_dim}");
        }
        assert_eq!(smem_bytes(64, false), 49_152);
        assert_eq!(smem_bytes(128, false), 98_304);
        // The small tile is what makes head dim 128 fit under 96 KB with room
        // to spare, which is the only reason 4 warps can hold it.
        assert_eq!(smem_bytes(128, true), 81_920);
    }

    /// `num_threads` is `NUM_WARPS * 32`: 128 at head dim 64, 256 at 128.
    ///
    /// The block width and the `__launch_bounds__` the id was compiled under
    /// are one number; this pins it to the warp split the instantiation names.
    #[test]
    fn thread_count_tracks_the_warp_split() {
        assert_eq!(geometry(64, false).unwrap().0, 128);
        assert_eq!(geometry(128, false).unwrap().0, 256);
        assert_eq!(geometry(64, true).unwrap().0, 128);
        assert_eq!(geometry(128, true).unwrap().0, 128);
        assert!(geometry(96, false).is_none());
    }

    /// The row-tile grid is `ceil_div(seqlen_q, TILE_M)`, exact and one past.
    ///
    /// A floor here would drop the last partial tile and leave its query rows
    /// unwritten; the check is at a multiple of `TILE_M` and one row beyond,
    /// which is where a `/` instead of a `div_ceil` first diverges.
    #[test]
    fn the_grid_covers_every_row_tile() {
        assert_eq!(blocks_m(TILE_M, 1, false), 1, "an exact tile is one block");
        assert_eq!(blocks_m(TILE_M + 1, 1, false), 2, "one row past is a second block");
        assert_eq!(blocks_m(4 * TILE_M, 1, false), 4, "an exact multiple is that many");
        assert_eq!(blocks_m(4 * TILE_M - 1, 1, false), 4, "one short still rounds up");
        assert_eq!(blocks_m(1, 1, false), 1, "a single row is one block");

        // Packed, the M axis counts `seqlen_q * group_size` rows, which is the
        // whole point: four heads of one position are one tile, not four.
        assert_eq!(blocks_m(1, 4, false), 1, "a decode row's whole group is one tile");
        assert_eq!(blocks_m(TILE_M, 4, false), 4, "packing multiplies the row count");
        assert_eq!(blocks_m(33, 4, false), 2, "132 packed rows need two tiles");
    }

    /// The plan, on the shapes whose measurements produced it.
    ///
    /// Each row is a different reason for the answer rather than a different
    /// size: no GQA to pack, a long prefill where packing saves no tiles and
    /// splitting is not offered, the batch-1 decode that needs BOTH, and the
    /// batched decodes where packing alone already fills the machine.
    #[test]
    fn the_plan_matches_what_was_measured() {
        const SMS: u32 = 82;
        let p = |sq, sk, b, hq, hk| plan(sq, sk, b, hq, hk, SMS, true);
        let large = |packed, splits| Plan { packed, splits, small: false };
        let small = |packed, splits| Plan { packed, splits, small: true };

        assert_eq!(p(4096, 4096, 1, 32, 32), large(false, 1), "MHA has nothing to pack");
        assert_eq!(
            p(4096, 4096, 1, 32, 8),
            large(false, 1),
            "long prefill saves no tiles by packing, and is not offered splits"
        );
        assert_eq!(
            p(1, 4096, 1, 32, 8),
            small(true, 8),
            "batch-1 decode needs all three: the small tile, packing, and splits"
        );
        assert_eq!(
            p(1, 4096, 8, 32, 8),
            small(true, 1),
            "packing alone already fills the machine at batch 8"
        );
        assert_eq!(
            p(1, 4096, 32, 32, 8),
            small(true, 2),
            "at batch 32 the blocks are a ragged 3.1 waves, and splitting evens them"
        );

        // Splitting is a choice the caller can decline by binding no scratch,
        // and the answer then has to stay a legal one rather than an error.
        assert_eq!(
            plan(1, 4096, 1, 32, 8, SMS, false).splits,
            1,
            "no scratch means no splits, whatever the cost model wanted"
        );
        assert_eq!(
            plan(1, 4096, 8, 32, 8, 0, true),
            large(false, 1),
            "an unknown SM count neither packs nor splits nor retiles"
        );
    }

    /// The small tile is taken exactly when the query side fits in one.
    ///
    /// It is worth its 4-13% prefill cost only where the tile is mostly
    /// padding. The boundary is in PACKED rows, not query positions, which is
    /// why a 16-row decode of an 8-wide group is already past it.
    #[test]
    fn the_small_tile_is_taken_only_where_the_tile_is_padding() {
        const SMS: u32 = 82;
        assert!(plan(1, 4096, 8, 32, 8, SMS, true).small, "a decode row is 4 packed rows");
        assert!(plan(16, 4096, 8, 32, 8, SMS, true).small, "16 x 4 = 64 packed rows, exactly one");
        assert!(!plan(17, 4096, 8, 32, 8, SMS, true).small, "68 packed rows spill into a second");
        assert!(!plan(4096, 4096, 1, 32, 8, SMS, true).small, "prefill keeps the large tile");
        assert!(plan(64, 4096, 8, 32, 32, SMS, true).small, "MHA does not pack, so 64 rows fit");
    }

    /// The scratch is exactly what the plan will use, and nothing when unused.
    #[test]
    fn scratch_is_zero_when_nothing_will_split() {
        const SMS: u32 = 82;
        assert_eq!(plan(4096, 4096, 1, 32, 8, SMS, true).splits, 1);
        let p = plan(1, 4096, 1, 32, 8, SMS, true);
        assert!(p.packed && p.small);
        assert_eq!(p.splits, 8);
        assert_eq!(32 * p.splits as usize * 128, 32768);
    }

    /// Splits never exceed the key tiles there are to split.
    ///
    /// A split past the end contributes an all-masked row to the combine,
    /// which is defined but pure waste, and at short cache lengths the naive
    /// "fill the machine" answer would ask for many of them.
    #[test]
    fn splits_never_exceed_the_key_tiles() {
        const SMS: u32 = 82;
        for sk in [1u32, 64, 128, 129, 256, 1000] {
            let p = plan(1, sk, 1, 32, 8, SMS, true);
            let tiles = if sk.div_ceil(TILE_N) == 0 { 1 } else { sk.div_ceil(TILE_N) };
            assert!(p.splits <= tiles, "sk {sk} has {tiles} tiles but planned {} splits", p.splits);
            assert!(p.splits <= MAX_SPLITS);
        }
    }
}

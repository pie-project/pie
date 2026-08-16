//! The multimodal towers' kernels, and NOT a family.
//!
//! Gemma-4's vision and audio encoders and Qwen3-VL's vision tower are HOST
//! WALKS: a sequence of launches over a scratch arena with a handful of host
//! decisions between them. The walk stays in `driver-cuda/src/tower/`, because
//! what it reads is the driver's own vocabulary -- weight tables, arenas,
//! encode plans. What is here is the other half of every `<<<>>>` it wrote:
//! the instantiation NVRTC is handed and the grid the launch goes at.
//!
//! There is no `FAMILY` here, no `ROUTINES`, and no line in `lib.rs`. No trace
//! statement names a tower kernel -- the towers are reached through the encode
//! ABI, not through a lowered program -- so there is nothing for a symbol to
//! resolve to. [`crate::driver_internal`] is the same shape for the same
//! reason.
//!
//! # Five roots, because that is what NVRTC compiles
//!
//! One per header: `tower_naive_kernels` (the six more than one tower
//! launches), `gemma4_naive_kernels` (the one the two gemma-4 towers share),
//! and one for each of the three towers.
//!
//! # The geometry is not invented here
//!
//! Every routine below quotes the `<<<>>>` it reproduces, transcribed from the
//! launcher the walk replaced. The two tiles the towers launch under are
//! spelled once, at [`BLOCK`] and [`TILE`], rather than at forty call sites.

#![allow(clippy::too_many_arguments)]

use core::ffi::c_void;

use crate::jit::{Ctx, Launch};
use crate::jit::Abi;
use crate::jit::abi::{MaybeConst, bf16};
use kernels::Refusal;

/// The towers' pointwise block — every `<<<(t + 255) / 256, 256, 0, S>>>`, and
/// the block the two per-row kernels launch at.
const BLOCK: u32 = 256;

/// The square tile the towers walk a rectangle in, transcribed from the
/// launcher this module replaces:
///
/// ```text
/// dim3 B2(16,16); inline dim3 G2(int X,int Y){return dim3((X+15)/16,(Y+15)/16);}
/// ```
const TILE: u32 = 16;

/// `k_layernorm_relu`'s block — `<<<T*F, 128, 0, S>>>`, and 128 rather than
/// [`BLOCK`] is a numerics fact, not a spelling one: the fold sums
/// `(blockDim.x + 31) / 32` per-warp partials serially in thread 0, so 128 and
/// 256 add the same values in a different order.
const LAYERNORM_BLOCK: u32 = 128;

/// The head width `k_rope_axial2d` and `k_av` are compiled against.
///
/// Hard-coded in the device text (`k_av`'s `d >= 64`, `k_rope_axial2d`'s
/// `c >= 16` over a 64-wide head), which is why the walk checks
/// `hidden == 768 && heads == 12` before it starts and refuses otherwise.
const AXIAL_HEAD_DIM: u32 = 64;

/// The warp `k_rope_axial2d` launches, of which sixteen lanes work.
const WARP: u32 = 32;

/// `<<<G2(width, rows), B2>>>` — the grid every 2-D tower kernel takes.
const fn tile16(rows: u32, width: u32) -> Launch {
    Launch::grid([width.div_ceil(TILE), rows.div_ceil(TILE), 1], [TILE, TILE, 1])
}

/// A grid axis, as the `u32` it is. A non-positive extent has nothing to
/// launch and says so rather than sign-extending into one.
fn extent(what: &'static str, value: i32) -> Result<u32, Refusal> {
    if value <= 0 {
        return Err(Refusal::Empty { what });
    }
    Ok(value.unsigned_abs())
}

/// The product of two axes, refused rather than wrapped.
fn axes(what: &'static str, a: u32, b: u32) -> Result<u32, Refusal> {
    a.checked_mul(b).ok_or(Refusal::Wide {
        what,
        at: i64::from(a) * i64::from(b),
        max: i64::from(u32::MAX),
    })
}

/// `<<<(t + 255) / 256, 256, 0, S>>>` over a 64-bit element count.
///
/// The count is a `usize` on the device too (`usize = decltype(sizeof(0))`),
/// so the only narrowing is the grid's, and it is refused rather than taken.
fn flat(what: &'static str, t: usize) -> Result<Launch, Refusal> {
    if t == 0 {
        return Err(Refusal::Empty { what });
    }
    let blocks = t.div_ceil(BLOCK as usize);
    let Ok(blocks) = u32::try_from(blocks) else {
        return Err(Refusal::Wide {
            what,
            at: i64::try_from(t).unwrap_or(i64::MAX),
            max: i64::from(u32::MAX) * i64::from(BLOCK),
        });
    };
    Ok(Launch::grid([blocks, 1, 1], [BLOCK, 1, 1]))
}

// ── vision/tower_naive_kernels.cuh ──────────────────────────────────────────

/// RMSNorm with an OPTIONAL weight — `vision::k_rms_bf16`.
///
/// `k_rms<<<R, 256, 0, S>>>` at every one of its ten call sites. The shared
/// memory is STATIC (`__shared__ float warp[32], ss;`), so the launcher's zero
/// dynamic bytes is the whole of the contract.
///
/// # Safety
///
/// `x` addresses `rows * width` live bf16 elements and `o` that many writable
/// ones; `weight` is `width` live bf16 elements or null, which the kernel
/// reads as unit gain. All live on `ctx`'s stream.
pub fn k_rms_bf16(
    ctx: &Ctx,
    x: *const c_void,
    weight: *const c_void,
    o: *mut c_void,
    rows: i32,
    width: i32,
    eps: f32,
) -> Result<(), Refusal> {
    let blocks = extent("rows", rows)?;
    // The width is not a grid axis — one block walks the whole row — but a
    // non-positive one is a row the kernel reads past the end of.
    extent("width", width)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/tower_naive_kernels.cuh",
            "::pie::vision::k_rms<::pie::bf16>",
            Launch::per_row(blocks, BLOCK),
            &[
                x.cast::<bf16>().arg(),
                MaybeConst::new(weight.cast::<bf16>()).arg(),
                o.cast::<bf16>().arg(),
                rows.arg(),
                width.arg(),
                eps.arg(),
            ],
        )
    }
}

/// `a += b` — `vision::k_add_bf16`.
///
/// `k_add<<<(n+255)/256, 256, 0, S>>>`. `a` is read and written by the same
/// thread.
///
/// # Safety
///
/// `a` and `b` each address `n` live bf16 elements, `a` writable, both on
/// `ctx`'s stream.
pub fn k_add_bf16(ctx: &Ctx, a: *mut c_void, b: *const c_void, n: usize) -> Result<(), Refusal> {
    let launch = flat("n", n)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/tower_naive_kernels.cuh",
            "::pie::vision::k_add<::pie::bf16>",
            launch,
            &[a.cast::<bf16>().arg(), b.cast::<bf16>().arg(), n.arg()],
        )
    }
}

/// The f32 input plane narrowed to bf16 — `vision::k_f32_to_bf16_bf16`.
///
/// `k_f32_to_bf16<<<(n+255)/256, 256, 0, S>>>`. The SOURCE is float whatever
/// the destination's element type is, which is why `a` is `const float*` and
/// not the template's `T`.
///
/// # Safety
///
/// `a` addresses `n` live floats and `o` `n` writable bf16 elements, both on
/// `ctx`'s stream.
pub fn k_f32_to_bf16_bf16(
    ctx: &Ctx,
    a: *const c_void,
    o: *mut c_void,
    n: usize,
) -> Result<(), Refusal> {
    let launch = flat("n", n)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/tower_naive_kernels.cuh",
            "::pie::vision::k_f32_to_bf16<::pie::bf16>",
            launch,
            &[a.cast::<f32>().arg(), o.cast::<bf16>().arg(), n.arg()],
        )
    }
}

/// The exact erf GELU — `vision::k_gelu_erf_bf16`.
///
/// `k_gelu_erf<<<(t+255)/256, 256, 0, S>>>`. `nn.GELU()` with
/// `approximate='none'`, which is a different FUNCTION from
/// [`k_gelu_tanh_bf16`] and not a different spelling of one.
///
/// # Safety
///
/// `x` addresses `t` live bf16 elements and `o` `t` writable ones; they may
/// be the same address. Both live on `ctx`'s stream.
pub fn k_gelu_erf_bf16(
    ctx: &Ctx,
    x: *const c_void,
    o: *mut c_void,
    t: usize,
) -> Result<(), Refusal> {
    let launch = flat("t", t)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/tower_naive_kernels.cuh",
            "::pie::vision::k_gelu_erf<::pie::bf16>",
            launch,
            &[x.cast::<bf16>().arg(), o.cast::<bf16>().arg(), t.arg()],
        )
    }
}

/// LayerNorm with an optional scale and bias — `vision::k_layernorm_bf16`.
///
/// `k_layernorm<<<R, 256, 0, S>>>`. Two passes and two folds, in the
/// original's order: the mean is reduced, broadcast, and only then does the
/// variance pass start.
///
/// # Safety
///
/// `x` and `o` address `rows * width` live bf16 elements, `o` writable; `g`
/// and `beta` are `width` live bf16 elements each or null. All live on `ctx`'s
/// stream.
pub fn k_layernorm_bf16(
    ctx: &Ctx,
    x: *const c_void,
    g: *const c_void,
    beta: *const c_void,
    o: *mut c_void,
    rows: i32,
    width: i32,
    eps: f32,
) -> Result<(), Refusal> {
    let blocks = extent("rows", rows)?;
    // As [`k_rms_bf16`]: one block walks the whole row, and a non-positive
    // width is a read past the end of it.
    extent("width", width)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/tower_naive_kernels.cuh",
            "::pie::vision::k_layernorm<::pie::bf16>",
            Launch::per_row(blocks, BLOCK),
            &[
                x.cast::<bf16>().arg(),
                MaybeConst::new(g.cast::<bf16>()).arg(),
                MaybeConst::new(beta.cast::<bf16>()).arg(),
                o.cast::<bf16>().arg(),
                rows.arg(),
                width.arg(),
                eps.arg(),
            ],
        )
    }
}

/// The naive `[n, k] x [o, k]^T` matmul — `vision::k_matmul_bf16`.
///
/// `k_matmul<<<G2(O, N), B2, 0, S>>>`: the `k` extent is an operand and not a
/// grid axis, because it is the loop each thread runs.
///
/// # Safety
///
/// `x` is `[n, k]` bf16, `w` is `[o, k]` bf16, `y` is `[n, o]` bf16 and
/// writable. All live on `ctx`'s stream.
pub fn k_matmul_bf16(
    ctx: &Ctx,
    x: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    n: i32,
    k: i32,
    o: i32,
) -> Result<(), Refusal> {
    let rows = extent("n", n)?;
    let width = extent("o", o)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/tower_naive_kernels.cuh",
            "::pie::vision::k_matmul<::pie::bf16>",
            tile16(rows, width),
            &[
                x.cast::<bf16>().arg(),
                w.cast::<bf16>().arg(),
                y.cast::<bf16>().arg(),
                n.arg(),
                k.arg(),
                o.arg(),
            ],
        )
    }
}

// ── vision/gemma4_naive_kernels.cuh ─────────────────────────────────────────

/// The clipped linear's clamp — `vision::k_clamp_bf16`.
///
/// `k_clamp<<<(t+255)/256, 256, 0, S>>>`. `lo` and `hi` are DEVICE pointers to
/// single elements and both are nullable: the kernel's
/// `lo ? F(*lo) : neg_inf()` is what a null means, and reading them on the
/// host would be a synchronising copy per linear per layer.
///
/// # Safety
///
/// `x` and `o` address `t` live bf16 elements, `o` writable and possibly the
/// same address; `lo` and `hi` address one live bf16 element each, or are
/// null. All live on `ctx`'s stream.
pub fn k_clamp_bf16(
    ctx: &Ctx,
    x: *const c_void,
    o: *mut c_void,
    lo: *const c_void,
    hi: *const c_void,
    t: usize,
) -> Result<(), Refusal> {
    let launch = flat("t", t)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/gemma4_naive_kernels.cuh",
            "::pie::vision::k_clamp<::pie::bf16>",
            launch,
            &[
                x.cast::<bf16>().arg(),
                o.cast::<bf16>().arg(),
                MaybeConst::new(lo.cast::<bf16>()).arg(),
                MaybeConst::new(hi.cast::<bf16>()).arg(),
                t.arg(),
            ],
        )
    }
}

// ── vision/gemma4_vision.cuh ────────────────────────────────────────────────

/// The `[-1, 1]` patch rescale, `o = 2*(p - 0.5)` — `vision::k_scale_bf16`.
///
/// `k_scale<<<((long)N*Hd+255)/256, 256, 0, S>>>`.
///
/// # Safety
///
/// `p` addresses `t` live bf16 elements and `o` `t` writable ones, both on
/// `ctx`'s stream.
pub fn k_scale_bf16(ctx: &Ctx, p: *const c_void, o: *mut c_void, t: usize) -> Result<(), Refusal> {
    let launch = flat("t", t)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/gemma4_vision.cuh",
            "::pie::vision::k_scale<::pie::bf16>",
            launch,
            &[p.cast::<bf16>().arg(), o.cast::<bf16>().arg(), t.arg()],
        )
    }
}

/// The per-row score softmax — `vision::k_softmax_bf16`.
///
/// `k_softmax<<<N, 256, 0, S>>>`, in place over the `[n, n]` f32 score matrix.
/// The kernel's `__shared__` is static, so zero dynamic bytes is the whole
/// contract.
///
/// # Safety
///
/// `s` addresses `n * n` live floats and is writable, on `ctx`'s stream.
pub fn k_softmax_bf16(ctx: &Ctx, s: *mut c_void, n: i32) -> Result<(), Refusal> {
    let rows = extent("n", n)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/gemma4_vision.cuh",
            "::pie::vision::k_softmax<::pie::bf16>",
            Launch::per_row(rows, BLOCK),
            &[s.cast::<f32>().arg(), n.arg()],
        )
    }
}

/// The pooling accumulator, scaled and narrowed —
/// `vision::k_pool_finish_bf16`.
///
/// `k_pool_finish<<<((long)OUTL*Hd+255)/256, 256, 0, S>>>`. `s` is
/// `sqrtf((float)Hd)` computed on the HOST: an operand, not an extent, and no
/// grid recovers it.
///
/// # Safety
///
/// `input` addresses `t` live floats and `o` `t` writable bf16 elements, both
/// on `ctx`'s stream.
pub fn k_pool_finish_bf16(
    ctx: &Ctx,
    input: *const c_void,
    o: *mut c_void,
    s: f32,
    t: usize,
) -> Result<(), Refusal> {
    let launch = flat("t", t)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/gemma4_vision.cuh",
            "::pie::vision::k_pool_finish<::pie::bf16>",
            launch,
            &[input.cast::<f32>().arg(), o.cast::<bf16>().arg(), s.arg(), t.arg()],
        )
    }
}

/// The two axial position-table rows, added — `vision::k_addpos_grid2d_bf16`.
///
/// `k_addpos_grid2d<<<G2(Hd, N), B2, 0, S>>>`, in place on `y`.
///
/// `pos` is float and not int because it is the same buffer
/// [`k_rope_axial2d_bf16`] consumes, where the values are trigonometric
/// arguments; the kernel's `llrintf` and two clamps are what a grid index
/// costs for sharing it.
///
/// # Safety
///
/// `y` is `[n, o]` bf16 and writable, `tb` is `[2, p, o]` bf16, `pos` is
/// `[n, 2]` floats. All live on `ctx`'s stream.
pub fn k_addpos_grid2d_bf16(
    ctx: &Ctx,
    y: *mut c_void,
    tb: *const c_void,
    pos: *const c_void,
    n: i32,
    o: i32,
    p: i32,
) -> Result<(), Refusal> {
    let rows = extent("n", n)?;
    let width = extent("o", o)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/gemma4_vision.cuh",
            "::pie::vision::k_addpos_grid2d<::pie::bf16>",
            tile16(rows, width),
            &[
                y.cast::<bf16>().arg(),
                tb.cast::<bf16>().arg(),
                pos.cast::<f32>().arg(),
                n.arg(),
                o.arg(),
                p.arg(),
            ],
        )
    }
}

/// 2-D axial RoPE over a 64-wide head — `vision::k_rope_axial2d_bf16`.
///
/// `dim3 rg(1, NH, N); k_rope_axial2d<<<rg, 32, 0, S>>>` — a 3-D grid, and the
/// kernel reads all three of `blockIdx.{x,y,z}`. One WARP per (head, patch),
/// of which sixteen lanes work: the half-width is [`AXIAL_HEAD_DIM`] / 4,
/// hard-coded in the device text.
///
/// # Safety
///
/// `q` is `[n, h, 64]` bf16 and writable and `pos` is `[n, 2]` floats, both on
/// `ctx`'s stream.
pub fn k_rope_axial2d_bf16(
    ctx: &Ctx,
    q: *mut c_void,
    pos: *const c_void,
    n: i32,
    h: i32,
    theta: f32,
) -> Result<(), Refusal> {
    let rows = extent("n", n)?;
    let heads = extent("h", h)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/gemma4_vision.cuh",
            "::pie::vision::k_rope_axial2d<::pie::bf16>",
            Launch::grid([1, heads, rows], [WARP, 1, 1]),
            &[q.cast::<bf16>().arg(), pos.cast::<f32>().arg(), n.arg(), h.arg(), theta.arg()],
        )
    }
}

/// One head's `QK^T` — `vision::k_qk_bf16`.
///
/// `k_qk<<<G2(N, N), B2, 0, S>>>`: the SCORE matrix's rectangle, square in
/// `n`. The head axis is walked by the caller's loop, which is why `head` is
/// an operand.
///
/// # Safety
///
/// `q` and `k` are `[n, h, 64]` bf16, `s` is `[n, n]` floats and writable. All
/// live on `ctx`'s stream.
pub fn k_qk_bf16(
    ctx: &Ctx,
    q: *const c_void,
    k: *const c_void,
    s: *mut c_void,
    n: i32,
    h: i32,
    head: i32,
    scale: f32,
) -> Result<(), Refusal> {
    let rows = extent("n", n)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/gemma4_vision.cuh",
            "::pie::vision::k_qk<::pie::bf16>",
            tile16(rows, rows),
            &[
                q.cast::<bf16>().arg(),
                k.cast::<bf16>().arg(),
                s.cast::<f32>().arg(),
                n.arg(),
                h.arg(),
                head.arg(),
                scale.arg(),
            ],
        )
    }
}

/// One head's `softmax(QK^T) V` — `vision::k_av_bf16`.
///
/// `k_av<<<G2(64, N), B2, 0, S>>>`: the width is ONE HEAD's
/// [`AXIAL_HEAD_DIM`] and not the tower's 768, because the kernel's own guard
/// is `d >= 64`. The head axis is walked by the caller's loop.
///
/// # Safety
///
/// `s` is `[n, n]` floats, `v` is `[n, h, 64]` bf16, `o` is `[n, h, 64]` bf16
/// and writable. All live on `ctx`'s stream.
pub fn k_av_bf16(
    ctx: &Ctx,
    s: *const c_void,
    v: *const c_void,
    o: *mut c_void,
    n: i32,
    h: i32,
    head: i32,
) -> Result<(), Refusal> {
    let rows = extent("n", n)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/gemma4_vision.cuh",
            "::pie::vision::k_av<::pie::bf16>",
            tile16(rows, AXIAL_HEAD_DIM),
            &[
                s.cast::<f32>().arg(),
                v.cast::<bf16>().arg(),
                o.cast::<bf16>().arg(),
                n.arg(),
                h.arg(),
                head.arg(),
            ],
        )
    }
}

/// The pooling scatter-accumulate — `vision::k_pool_bf16`.
///
/// `k_pool<<<G2(Hd, N), B2, 0, S>>>` — the INPUT rectangle: the grid covers
/// the patches being scattered. The accumulator is FLOAT and the add is atomic
/// because the group map `grp` is data: several patches land on one output row
/// and the order they land in is the scheduler's, which is also why
/// [`k_pool_finish_bf16`] exists.
///
/// # Safety
///
/// `h` is `[n, d]` bf16, `grp` is `[n]` `int`s each addressing a live row of
/// `o`, and `o` is that accumulator, writable. All live on `ctx`'s stream.
pub fn k_pool_bf16(
    ctx: &Ctx,
    h: *const c_void,
    grp: *const c_void,
    o: *mut c_void,
    n: i32,
    d: i32,
    k2: f32,
) -> Result<(), Refusal> {
    let rows = extent("n", n)?;
    let width = extent("d", d)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/gemma4_vision.cuh",
            "::pie::vision::k_pool<::pie::bf16>",
            tile16(rows, width),
            &[
                h.cast::<bf16>().arg(),
                grp.cast::<i32>().arg(),
                o.cast::<f32>().arg(),
                n.arg(),
                d.arg(),
                k2.arg(),
            ],
        )
    }
}

// ── vision/gemma4_audio.cuh ─────────────────────────────────────────────────

/// `o = x * sigmoid(x)` — `vision::k_silu_bf16`.
///
/// `k_silu<<<(t+255)/256, 256, 0, S>>>`.
///
/// # Safety
///
/// `x` addresses `t` live bf16 elements and `o` `t` writable ones, possibly
/// the same address. Both live on `ctx`'s stream.
pub fn k_silu_bf16(ctx: &Ctx, x: *const c_void, o: *mut c_void, t: usize) -> Result<(), Refusal> {
    let launch = flat("t", t)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/gemma4_audio.cuh",
            "::pie::vision::k_silu<::pie::bf16>",
            launch,
            &[x.cast::<bf16>().arg(), o.cast::<bf16>().arg(), t.arg()],
        )
    }
}

/// `a += scale * b` — `vision::k_axpy_bf16`.
///
/// `k_axpy<<<(t+255)/256, 256, 0, S>>>`. `scale` is the macaron half-step's
/// residual weight.
///
/// # Safety
///
/// `a` and `b` each address `t` live bf16 elements, `a` writable. Both live on
/// `ctx`'s stream.
pub fn k_axpy_bf16(
    ctx: &Ctx,
    a: *mut c_void,
    b: *const c_void,
    scale: f32,
    t: usize,
) -> Result<(), Refusal> {
    let launch = flat("t", t)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/gemma4_audio.cuh",
            "::pie::vision::k_axpy<::pie::bf16>",
            launch,
            &[a.cast::<bf16>().arg(), b.cast::<bf16>().arg(), scale.arg(), t.arg()],
        )
    }
}

/// The naive matmul with an optional bias — `vision::k_matmul_bias_bf16`.
///
/// `k_matmul_bias<<<G2(O, N), B2, 0, S>>>`. `b` is `[o]` or null.
///
/// # Safety
///
/// `x` is `[n, k]` bf16, `w` is `[o, k]` bf16, `b` is `[o]` bf16 or null, `y`
/// is `[n, o]` bf16 and writable. All live on `ctx`'s stream.
pub fn k_matmul_bias_bf16(
    ctx: &Ctx,
    x: *const c_void,
    w: *const c_void,
    b: *const c_void,
    y: *mut c_void,
    n: i32,
    k: i32,
    o: i32,
) -> Result<(), Refusal> {
    let rows = extent("n", n)?;
    let width = extent("o", o)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/gemma4_audio.cuh",
            "::pie::vision::k_matmul_bias<::pie::bf16>",
            tile16(rows, width),
            &[
                x.cast::<bf16>().arg(),
                w.cast::<bf16>().arg(),
                MaybeConst::new(b.cast::<bf16>()).arg(),
                y.cast::<bf16>().arg(),
                n.arg(),
                k.arg(),
                o.arg(),
            ],
        )
    }
}

/// The conv module's gated linear unit — `vision::k_glu_bf16`.
///
/// `k_glu<<<G2(D, N), B2, 0, S>>>`, where `d` is the OUTPUT width: the input
/// row is `2 * d` wide and the kernel reads both halves of it.
///
/// # Safety
///
/// `x` is `[n, 2*d]` bf16 and `o` is `[n, d]` bf16 and writable, both on
/// `ctx`'s stream.
pub fn k_glu_bf16(
    ctx: &Ctx,
    x: *const c_void,
    o: *mut c_void,
    n: i32,
    d: i32,
) -> Result<(), Refusal> {
    let rows = extent("n", n)?;
    let width = extent("d", d)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/gemma4_audio.cuh",
            "::pie::vision::k_glu<::pie::bf16>",
            tile16(rows, width),
            &[x.cast::<bf16>().arg(), o.cast::<bf16>().arg(), n.arg(), d.arg()],
        )
    }
}

/// LayerNorm over the channel axis, then ReLU —
/// `vision::k_layernorm_relu_bf16`.
///
/// `k_layernorm_relu<<<R, 128, 0, S>>>` at both SSCP call sites. The block is
/// [`LAYERNORM_BLOCK`] and not [`BLOCK`], and that is a numerics fact rather
/// than a spelling one — see the constant.
///
/// # Safety
///
/// `x` and `o` address `r * c` live bf16 elements, `o` writable and possibly
/// the same address; `w` is `[c]` bf16 or null. All live on `ctx`'s stream.
pub fn k_layernorm_relu_bf16(
    ctx: &Ctx,
    x: *const c_void,
    w: *const c_void,
    o: *mut c_void,
    r: i32,
    c: i32,
    eps: f32,
) -> Result<(), Refusal> {
    let rows = extent("r", r)?;
    // As [`k_rms_bf16`]: the channel count is the row this block walks, not a
    // grid axis, and a non-positive one is a read past the end of it.
    extent("c", c)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/gemma4_audio.cuh",
            "::pie::vision::k_layernorm_relu<::pie::bf16>",
            Launch::per_row(rows, LAYERNORM_BLOCK),
            &[
                x.cast::<bf16>().arg(),
                MaybeConst::new(w.cast::<bf16>()).arg(),
                o.cast::<bf16>().arg(),
                r.arg(),
                c.arg(),
                eps.arg(),
            ],
        )
    }
}

/// `[oc, t_out, f_out]` flattened to `[t_out, f_out*oc]` —
/// `vision::k_sscp_flatten_bf16`.
///
/// `dim3 g((FLAT+15)/16, (To+15)/16); k_sscp_flatten<<<g, B2, 0, S>>>` with
/// `FLAT = f_out * oc`, which is the width this recovers rather than takes.
///
/// # Safety
///
/// `input` is `[oc, t_out, f_out]` bf16 and `out` is `[t_out, f_out*oc]` bf16
/// and writable, both on `ctx`'s stream.
pub fn k_sscp_flatten_bf16(
    ctx: &Ctx,
    input: *const c_void,
    out: *mut c_void,
    oc: i32,
    t_out: i32,
    f_out: i32,
) -> Result<(), Refusal> {
    let rows = extent("t_out", t_out)?;
    let width = axes("f_out * oc", extent("f_out", f_out)?, extent("oc", oc)?)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/gemma4_audio.cuh",
            "::pie::vision::k_sscp_flatten<::pie::bf16>",
            tile16(rows, width),
            &[
                input.cast::<bf16>().arg(),
                out.cast::<bf16>().arg(),
                oc.arg(),
                t_out.arg(),
                f_out.arg(),
            ],
        )
    }
}

/// The in-place q/k pre-scale — `vision::k_qkv_scale_bf16`.
///
/// `k_qkv_scale<<<G2(H*hd, N), B2, 0, S>>>`. Both scales are host constants —
/// `q_scale = hd^-0.5 / ln 2`, `k_scale = ln(1+e) / ln 2` — computed once per
/// encode: they look like extents and are not, because nothing about the
/// rectangle determines them.
///
/// # Safety
///
/// `q` and `k` are `[n, h*hd]` bf16 and writable, `pds` is `[hd]` bf16. All
/// live on `ctx`'s stream.
pub fn k_qkv_scale_bf16(
    ctx: &Ctx,
    q: *mut c_void,
    k: *mut c_void,
    pds: *const c_void,
    n: i32,
    h: i32,
    hd: i32,
    q_scale: f32,
    k_scale: f32,
) -> Result<(), Refusal> {
    let rows = extent("n", n)?;
    let width = axes("h * hd", extent("h", h)?, extent("hd", hd)?)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/gemma4_audio.cuh",
            "::pie::vision::k_qkv_scale<::pie::bf16>",
            tile16(rows, width),
            &[
                q.cast::<bf16>().arg(),
                k.cast::<bf16>().arg(),
                pds.cast::<bf16>().arg(),
                n.arg(),
                h.arg(),
                hd.arg(),
                q_scale.arg(),
                k_scale.arg(),
            ],
        )
    }
}

/// The sinusoidal relative-position encoding — `vision::k_rel_pos_enc_bf16`.
///
/// `dim3 g((hidden+15)/16, (P+15)/16); k_rel_pos_enc<<<g, B2, 0, S>>>`. Row
/// `r` holds position id `(p-1) - r`, so the table is shared across layers
/// while `relative_k_proj` of it is not.
///
/// # Safety
///
/// `pe` is `[p, hidden]` bf16 and writable, on `ctx`'s stream.
pub fn k_rel_pos_enc_bf16(ctx: &Ctx, pe: *mut c_void, p: i32, hidden: i32) -> Result<(), Refusal> {
    let rows = extent("p", p)?;
    let width = extent("hidden", hidden)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/gemma4_audio.cuh",
            "::pie::vision::k_rel_pos_enc<::pie::bf16>",
            tile16(rows, width),
            &[pe.cast::<bf16>().arg(), p.arg(), hidden.arg()],
        )
    }
}

/// `Conv2d(k3, s2, p1)` over (time, frequency) — `vision::k_conv2d_s2_bf16`.
///
/// `dim3 g((Fo+15)/16, (To+15)/16, OC); k_conv2d_s2<<<g, B2, 0, S>>>` — a 3-D
/// grid AND a 2-D block, with the output channel on `grid.z`, which is why
/// this geometry never had a rule.
///
/// # Safety
///
/// `input` is `[ic, t_in, f_in]` bf16, `w` is `[oc, ic, 3, 3]` bf16, `out` is
/// `[oc, t_out, f_out]` bf16 and writable. All live on `ctx`'s stream.
pub fn k_conv2d_s2_bf16(
    ctx: &Ctx,
    input: *const c_void,
    w: *const c_void,
    out: *mut c_void,
    ic: i32,
    t_in: i32,
    f_in: i32,
    oc: i32,
    t_out: i32,
    f_out: i32,
) -> Result<(), Refusal> {
    let launch = channelled(oc, t_out, f_out)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/gemma4_audio.cuh",
            "::pie::vision::k_conv2d_s2<::pie::bf16>",
            launch,
            &[
                input.cast::<bf16>().arg(),
                w.cast::<bf16>().arg(),
                out.cast::<bf16>().arg(),
                ic.arg(),
                t_in.arg(),
                f_in.arg(),
                oc.arg(),
                t_out.arg(),
                f_out.arg(),
            ],
        )
    }
}

/// `[oc, t_out, f_out]` to `[t_out*f_out, oc]` — `vision::k_chlast_bf16`.
///
/// [`k_conv2d_s2_bf16`]'s grid. A pure permutation: the LayerNorm above runs
/// over the channel axis and the conv's output is channels-first, which is the
/// whole reason the transpose pair exists.
///
/// # Safety
///
/// `input` is `[oc, t_out, f_out]` bf16 and `out` is `[t_out*f_out, oc]` bf16
/// and writable, both on `ctx`'s stream.
pub fn k_chlast_bf16(
    ctx: &Ctx,
    input: *const c_void,
    out: *mut c_void,
    oc: i32,
    t_out: i32,
    f_out: i32,
) -> Result<(), Refusal> {
    let launch = channelled(oc, t_out, f_out)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/gemma4_audio.cuh",
            "::pie::vision::k_chlast<::pie::bf16>",
            launch,
            &[
                input.cast::<bf16>().arg(),
                out.cast::<bf16>().arg(),
                oc.arg(),
                t_out.arg(),
                f_out.arg(),
            ],
        )
    }
}

/// [`k_chlast_bf16`]'s inverse — `vision::k_chfirst_bf16`.
///
/// # Safety
///
/// [`k_chlast_bf16`]'s, with the two tensors' roles swapped.
pub fn k_chfirst_bf16(
    ctx: &Ctx,
    input: *const c_void,
    out: *mut c_void,
    oc: i32,
    t_out: i32,
    f_out: i32,
) -> Result<(), Refusal> {
    let launch = channelled(oc, t_out, f_out)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/gemma4_audio.cuh",
            "::pie::vision::k_chfirst<::pie::bf16>",
            launch,
            &[
                input.cast::<bf16>().arg(),
                out.cast::<bf16>().arg(),
                oc.arg(),
                t_out.arg(),
                f_out.arg(),
            ],
        )
    }
}

/// `dim3((f_out+15)/16, (t_out+15)/16, oc)` over `B2` — the SSCP stage's grid,
/// shared by the conv and both transposes.
fn channelled(oc: i32, t_out: i32, f_out: i32) -> Result<Launch, Refusal> {
    let channels = extent("oc", oc)?;
    let time = extent("t_out", t_out)?;
    let freq = extent("f_out", f_out)?;
    Ok(Launch::grid([freq.div_ceil(TILE), time.div_ceil(TILE), channels], [TILE, TILE, 1]))
}

/// Causal-sliding-window attention with a relative-position bias —
/// `vision::k_local_attn_bf16`.
///
/// `dim3 g((N+127)/128, NH); k_local_attn<<<g, 128, 0, S>>>` — a TILE count on
/// `grid.x` where every ported rule put a count of things.
///
/// The masking: HF's blocked 5-D path (chunk 12 / past 12 / future 0) plus
/// `_rel_shift` collapses, for this configuration, to a plain causal sliding
/// window — query `t` attends keys `j` with `0 <= t-j < p-1` — and the
/// rel_shift gather collapses to `matrix_bd[t,j]` reading `pe` row
/// `(p-1)-(t-j)`.
///
/// # Safety
///
/// `q`, `k` and `v` are `[n, h, hd]` bf16, `relk` is `[p, h, hd]` bf16, `out`
/// is `[n, h, hd]` bf16 and writable. All live on `ctx`'s stream.
pub fn k_local_attn_bf16(
    ctx: &Ctx,
    q: *const c_void,
    k: *const c_void,
    v: *const c_void,
    relk: *const c_void,
    out: *mut c_void,
    n: i32,
    h: i32,
    hd: i32,
    p: i32,
    cap: f32,
) -> Result<(), Refusal> {
    /// `k_local_attn`'s block — `dim3 g((N+127)/128, NH); <<<g, 128, 0, S>>>`.
    /// A TILE count on `grid.x`, which is why the walk never had a rule for it.
    const LOCAL_ATTN_BLOCK: u32 = 128;

    let tiles = extent("n", n)?.div_ceil(LOCAL_ATTN_BLOCK);
    let heads = extent("h", h)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/gemma4_audio.cuh",
            "::pie::vision::k_local_attn<::pie::bf16>",
            Launch::grid([tiles, heads, 1], [LOCAL_ATTN_BLOCK, 1, 1]),
            &[
                q.cast::<bf16>().arg(),
                k.cast::<bf16>().arg(),
                v.cast::<bf16>().arg(),
                relk.cast::<bf16>().arg(),
                out.cast::<bf16>().arg(),
                n.arg(),
                h.arg(),
                hd.arg(),
                p.arg(),
                cap.arg(),
            ],
        )
    }
}

// ── vision/qwen3_vl_tower.cuh ───────────────────────────────────────────────

/// A projection's bias — `vision::k_bias_bf16`.
///
/// `k_bias<<<((long)M*N+255)/256, 256, 0, S>>>`, in place on `y`. `m` crosses
/// as a `usize` and `n` as an `int`, which is the kernel's own asymmetry: the
/// bias index is `i % n` on a 64-bit `i`.
///
/// # Safety
///
/// `y` is `[m, n]` bf16 and writable and `b` is `[n]` bf16, both on `ctx`'s
/// stream.
pub fn k_bias_bf16(
    ctx: &Ctx,
    y: *mut c_void,
    b: *const c_void,
    m: usize,
    n: i32,
) -> Result<(), Refusal> {
    let width = usize::try_from(extent("n", n)?).unwrap_or(usize::MAX);
    let count =
        m.checked_mul(width).ok_or(Refusal::Wide { what: "m * n", at: i64::MAX, max: i64::MAX })?;
    let launch = flat("m * n", count)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/qwen3_vl_tower.cuh",
            "::pie::vision::k_bias<::pie::bf16>",
            launch,
            &[y.cast::<bf16>().arg(), b.cast::<bf16>().arg(), m.arg(), n.arg()],
        )
    }
}

/// The interpolated position embedding, added — `vision::k_add_pe_bf16`.
///
/// `k_add_pe<<<((long)N*Hd+255)/256, 256, 0, S>>>`, in place on `h`. `pe` is
/// the HOST-interpolated table.
///
/// # Safety
///
/// `h` addresses `t` live bf16 elements and is writable, `pe` `t` live ones.
/// Both live on `ctx`'s stream.
pub fn k_add_pe_bf16(
    ctx: &Ctx,
    h: *mut c_void,
    pe: *const c_void,
    t: usize,
) -> Result<(), Refusal> {
    let launch = flat("t", t)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/qwen3_vl_tower.cuh",
            "::pie::vision::k_add_pe<::pie::bf16>",
            launch,
            &[h.cast::<bf16>().arg(), pe.cast::<bf16>().arg(), t.arg()],
        )
    }
}

/// `gelu_pytorch_tanh` — `vision::k_gelu_tanh_bf16`.
///
/// `k_gelu_tanh<<<(t+255)/256, 256, 0, S>>>`. A different FUNCTION from
/// [`k_gelu_erf_bf16`]: merging them by name once changed numerics silently.
///
/// # Safety
///
/// [`k_gelu_erf_bf16`]'s.
pub fn k_gelu_tanh_bf16(
    ctx: &Ctx,
    x: *const c_void,
    o: *mut c_void,
    t: usize,
) -> Result<(), Refusal> {
    let launch = flat("t", t)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/qwen3_vl_tower.cuh",
            "::pie::vision::k_gelu_tanh<::pie::bf16>",
            launch,
            &[x.cast::<bf16>().arg(), o.cast::<bf16>().arg(), t.arg()],
        )
    }
}

/// The same, with fc1's optional bias folded in —
/// `vision::k_gelu_bias_bf16`.
///
/// `k_gelu_bias<<<((long)N*D+255)/256, 256, 0, S>>>`, in place on `x`. Fired
/// whether or not there is a bias: the kernel's `b ? F(b[i % D]) : 0.f` is
/// what a null means.
///
/// # Safety
///
/// `x` is `[n, d]` bf16 and writable and `b` is `[d]` bf16 or null, both on
/// `ctx`'s stream.
pub fn k_gelu_bias_bf16(
    ctx: &Ctx,
    x: *mut c_void,
    b: *const c_void,
    n: i32,
    d: i32,
) -> Result<(), Refusal> {
    /// A `rows * width` element count, as the `usize` the kernel takes.
    fn elements(what: &'static str, rows: i32, width: i32) -> Result<usize, Refusal> {
    let rows = usize::try_from(extent(what, rows)?).unwrap_or(usize::MAX);
    let width = usize::try_from(extent(what, width)?).unwrap_or(usize::MAX);
    rows.checked_mul(width).ok_or(Refusal::Wide { what, at: i64::MAX, max: i64::MAX })
    }

    let launch = flat("n * d", elements("n * d", n, d)?)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/qwen3_vl_tower.cuh",
            "::pie::vision::k_gelu_bias<::pie::bf16>",
            launch,
            &[x.cast::<bf16>().arg(), MaybeConst::new(b.cast::<bf16>()).arg(), n.arg(), d.arg()],
        )
    }
}

/// The patch merger's 2x2 gather — `vision::k_merge_gather_bf16`.
///
/// `k_merge_gather<<<G2(U*C, n_token), B2, 0, S>>>`: the output row is `u * c`
/// wide, which this recovers rather than takes.
///
/// # Safety
///
/// `h` is `[n_token * u, c]` bf16 and `g` is `[n_token, u*c]` bf16 and
/// writable, both on `ctx`'s stream.
pub fn k_merge_gather_bf16(
    ctx: &Ctx,
    h: *const c_void,
    g: *mut c_void,
    n_token: i32,
    u: i32,
    c: i32,
) -> Result<(), Refusal> {
    let rows = extent("n_token", n_token)?;
    let width = axes("u * c", extent("u", u)?, extent("c", c)?)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/qwen3_vl_tower.cuh",
            "::pie::vision::k_merge_gather<::pie::bf16>",
            tile16(rows, width),
            &[h.cast::<bf16>().arg(), g.cast::<bf16>().arg(), n_token.arg(), u.arg(), c.arg()],
        )
    }
}

/// The qkv split, its bias and the ViT rope, fused —
/// `vision::k_split_rope_qkv_bf16`.
///
/// `k_split_rope_qkv<<<dim3(NH, N), HEAD/2, 0, S>>>`. The GRID is one block
/// per (head, row) and the BLOCK is HALF A HEAD — `head / 2` and not a fixed
/// 128, because widening it would be a performance decision taken on the
/// tower's behalf: 96 idle lanes over a 32-wide half, correct and four times
/// the launch.
///
/// # Safety
///
/// `qkv` is `[n, 3, nh*head]` bf16, `b` is `[3, nh*head]` bf16 or null, `q`,
/// `k` and `v` are `[n, nh, head]` bf16 and writable, `pos` is the rope
/// position table as floats. All live on `ctx`'s stream.
pub fn k_split_rope_qkv_bf16(
    ctx: &Ctx,
    qkv: *const c_void,
    b: *const c_void,
    q: *mut c_void,
    k: *mut c_void,
    v: *mut c_void,
    pos: *const c_void,
    n: i32,
    nh: i32,
    head: i32,
    theta: f32,
) -> Result<(), Refusal> {
    let rows = extent("n", n)?;
    let heads = extent("nh", nh)?;
    let half = extent("head / 2", head / 2)?;
    // SAFETY: the caller's assertion, forwarded.
    unsafe {
        ctx.launch(
            "vision/qwen3_vl_tower.cuh",
            "::pie::vision::k_split_rope_qkv<::pie::bf16>",
            Launch::grid([heads, rows, 1], [half, 1, 1]),
            &[
                qkv.cast::<bf16>().arg(),
                MaybeConst::new(b.cast::<bf16>()).arg(),
                q.cast::<bf16>().arg(),
                k.cast::<bf16>().arg(),
                v.cast::<bf16>().arg(),
                pos.cast::<f32>().arg(),
                n.arg(),
                nh.arg(),
                head.arg(),
                theta.arg(),
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::jit::Root;

    /// The five carried files this family compiles, so the two tests below
    /// walk the same set.
    const ROOTS: [&str; 5] = [
        "vision/tower_naive_kernels.cuh",
        "vision/gemma4_naive_kernels.cuh",
        "vision/gemma4_vision.cuh",
        "vision/gemma4_audio.cuh",
        "vision/qwen3_vl_tower.cuh",
    ];

    /// Every `#include` these roots reach is CARRIED, so NVRTC never goes
    /// looking for a header on disk.
    ///
    /// `source::every_include_reachable_from_a_unit_resolves` walked the
    /// `vision` units for this. There is no unit to walk now, so the walk is
    /// here, over the same five files and the header set each one states.
    #[test]
    fn every_include_the_roots_reach_is_carried() {
        for file in ROOTS {
            let root = Root::new(file);
            let reached = crate::source::reachable(root.name, root.text, root.header_set())
                .unwrap_or_else(|why| panic!("{}: {why}", root.name));
            assert!(!reached.is_empty(), "{} reaches no carried header at all", root.name);
        }
    }

    /// The five files are five different roots, so a diagnostic points at the
    /// header the symbol came out of and no two share a cache key.
    #[test]
    fn the_five_roots_are_five_headers() {
        let roots = ROOTS.map(Root::new);
        for root in &roots {
            assert_eq!(format!("{}.cuh", root.name), root.file);
            assert!(root.options.is_empty(), "{} asks for no NVRTC option", root.name);
            assert!(root.floor.is_any(), "{} compiles under any NVRTC", root.name);
            assert!(!root.text.is_empty());
        }
        for (i, a) in roots.iter().enumerate() {
            for b in &roots[i + 1..] {
                assert_ne!(a.name, b.name);
                assert_ne!(a.key("x", "sm_90"), b.key("x", "sm_90"), "two roots, two keys");
            }
        }
    }
}

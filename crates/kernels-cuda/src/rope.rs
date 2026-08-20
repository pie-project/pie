//! Rope family: cos/sin table generation, plain/fused/mrope rotation, YaRN
//! scaling, partial rotary, and the paged-KV write-through variant.
//!
//! Parameters are typed regions (`In<N,_>`/`Out<N,_>`/`Bank<N,_>`) or
//! `Env<keys::X>` facts; a bare `#[source(...)]` attribute remains only
//! where no type can carry the source.
#![allow(clippy::too_many_arguments)]

use kernels::{Bind, Fire};
use kernels::routine::{Asks, Const, In, InOut, Out};
use kernels_macros::routine;
use crate::jit::Abi;
use crate::jit::abi::{MaybeConst, bf16, f16};
use crate::jit::abi::Tensor;
use crate::jit::{Ctx, Launch};
use kernels::Refusal;
use kernels::keys;

// Three same-shaped pairs recur and must not be merged: `keys::Theta`
// (per-layer) vs. `keys::RopeTheta` (fire-wide), `keys::HeadDim` (the fire's)
// vs. `keys::KvHeadDim` (the cache's), and `keys::Rows` vs. `keys::RowsTotal`.
// Each pair agrees on most checkpoints and differs silently where it doesn't.
// `Bank<N, T>` and `Weight<N, T>` are the same hazard for the word "weight".

use core::ptr::NonNull;

/// `rope.cu`'s `constexpr int BLOCK = 256;`.
pub const ROTATE_BLOCK: i32 = 256;

/// `rope.cu`'s `constexpr int BLOCK = 128;`.
pub const FUSED_BLOCK: u32 = 128;

/// `rope.cu`'s `constexpr int kMaxCachedPairs = 4096;`.
pub const MAX_CACHED_PAIRS: i32 = 4096;

/// `half >= BLOCK ? 1 : (BLOCK / half)`, mirrored from `rope.cu`.
#[must_use]
pub const fn heads_per_block(half: i32) -> i32 {
    if half >= ROTATE_BLOCK { 1 } else { ROTATE_BLOCK / half }
}

/// `half <= kMaxCachedPairs ? half : 0`, mirrored from `rope.cu`.
#[must_use]
pub const fn cache_pairs(half: i32) -> i32 {
    if half <= MAX_CACHED_PAIRS { half } else { 0 }
}
/// The fused grid, mirrored from `rope.cu`.
#[must_use]
const fn fused_launch(rows: i32, total_heads: i32) -> Launch {
    Launch::grid([rows.unsigned_abs(), total_heads.unsigned_abs(), 1], [FUSED_BLOCK, 1, 1])
}

/// The rotation grid, at `smem` bytes of cached cos/sin pairs.
#[must_use]
const fn rotate_launch(num_tokens: i32, total_heads: i32, per_block: i32, smem: u32) -> Launch {
    /// The two-axis grid the head split produces, mirrored from `rope.cu`.
    #[must_use]
    const fn rotate_grid(num_tokens: i32, total_heads: i32, per_block: i32) -> [u32; 3] {
    [
    num_tokens.unsigned_abs(),
    (total_heads + per_block - 1).unsigned_abs() / per_block.unsigned_abs(),
    1,
    ]
    }

    Launch::grid(
        rotate_grid(num_tokens, total_heads, per_block),
        [ROTATE_BLOCK.unsigned_abs(), 1, 1],
    )
    .smem(smem)
}

/// `rope_device.cuh`'s `yarn_original_ramp_bounds`, on the host.
#[must_use]
pub fn ramp_bounds(
    span: i32,
    theta: f32,
    beta_fast: f32,
    beta_slow: f32,
    original_max_position: i32,
) -> (f32, f32) {
    /// Matches the CUDA literal `6.283185307179586` bit-for-bit in `f32`.
    const TWO_PI: f32 = core::f32::consts::TAU;
    let ln_theta = theta.ln();
    #[allow(clippy::cast_precision_loss)]
    let corr_dim = |rot: f32| -> f32 {
        span as f32 * (original_max_position as f32 / (rot * TWO_PI)).ln() / (2.0 * ln_theta)
    };
    let mut low_dim = corr_dim(beta_fast).floor();
    let mut high_dim = corr_dim(beta_slow).ceil();
    if low_dim < 0.0 {
        low_dim = 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let max_pair = (span / 2) as f32 - 1.0;
    if high_dim > max_pair {
        high_dim = max_pair;
    }
    if high_dim < low_dim {
        high_dim = low_dim;
    }
    (low_dim, high_dim)
}

// A width here is never a `Stride`: every rope kernel takes head counts and
// rebuilds its own address (`q + (n * num_q_heads + local_head) * head_dim`),
// so `q.width`/`k.width` stay plain extents that [`heads`] divides by
// `head_dim`. A zero `k.width` is this file's spelling for "there is no k";
// [`k_heads`] tells that apart from a malformed statement by the address,
// not the width.

/// How many heads a row of `width` elements holds, at `head_dim` each.
///
/// `head_dim <= 0` is [`Refusal::Empty`]; a width that isn't a whole number
/// of heads is [`Refusal::Narrow`]. `heads(0, head_dim)` is `Ok(0)` on
/// purpose — this file's spelling for "there is no k".
fn heads(width: i32, head_dim: i32) -> Result<i32, Refusal> {
    if head_dim <= 0 {
        return Err(Refusal::Empty { what: "head_dim" });
    }
    if width % head_dim != 0 {
        return Err(Refusal::Narrow {
            what: "the row is not a whole number of heads",
            at: i64::from(width),
        });
    }
    Ok(width / head_dim)
}

/// [`heads`] for `q`, whose width may never legitimately be zero (unlike
/// `k`'s — see [`k_heads`]).
fn q_heads(width: i32, head_dim: i32) -> Result<i32, Refusal> {
    if width <= 0 {
        return Err(Refusal::Empty { what: "the q region's width" });
    }
    heads(width, head_dim)
}

/// [`heads`] for `k`, where a zero width is malformed unless `k` is null or
/// aliases `q`'s address — this file's convention for "there is no k". The
/// address is the discriminator, not the width.
fn k_heads<T>(q: *mut T, k: *mut T, width: i32, head_dim: i32) -> Result<i32, Refusal> {
    if width <= 0 && !k.is_null() && !core::ptr::eq(k.cast_const(), q.cast_const()) {
        return Err(Refusal::Empty { what: "the k region's width" });
    }
    heads(width, head_dim)
}

/// `rope::rope_standard_table` — the cos/sin table `attn`'s fused prepare
/// launches from.
///
/// # Safety
///
/// `positions` addresses `table.rows` live `i32`s and `table` itself
/// `table.rows * head_dim` live floats; `stream` must be live across the
/// launch.
#[routine]
pub fn rope_standard_table(
    ctx: &Ctx<'_>,
    // The region also carries the row count `Launch::per_row` uses.
    table: Out<Tensor<f32>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.
    let head_dim = ctx.ask::<i32, keys::HeadDim>()?;
    let theta = ctx.ask::<f32, keys::RopeTheta>()?;

    let positions = ctx.ask::<*const i32, keys::Positions>()?;
    if head_dim / 2 <= 0 {
        return Err(Refusal::Empty { what: "head_dim / 2" });
    }
    ctx.fire(Fire::at("rope/rope.cuh", "::pie::rope::standard_table<::pie::i32>").apply(Launch::per_row(table.rows.unsigned_abs(), ROTATE_BLOCK.unsigned_abs())), &[positions.arg(), table.arg(), head_dim.arg(), theta.arg()])
}

/// `rope.cu`'s `rope::rope_bf16`.
///
/// # Safety
///
/// `q` and `k` address `q.rows * num_q_heads * head_dim` and
/// `k.rows * num_kv_heads * head_dim` live bf16 elements, `positions`
/// `q.rows` live `i32`s, and `stream` must be live across the launch.
#[routine]
pub fn rope_bf16(
    ctx: &Ctx<'_>,
    // Head counts stay stated facts rather than `heads(q.width, ..)`: this
    // kernel's arm never divided a width.
    q: InOut<Tensor<bf16>>,
    k: InOut<Tensor<bf16>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.
    let num_q_heads = ctx.ask::<i32, keys::NumQHeads>()?;
    let num_kv_heads = ctx.ask::<i32, keys::NumKvHeads>()?;
    let head_dim = ctx.ask::<i32, keys::HeadDim>()?;

    // BACK TO ASKS, AND THE REASON IS WRITTEN DOWN ELSEWHERE. Both were
    // `Env<keys::RopeTheta>` and `Env<keys::RopeInterleaved>`, and
    // `driver-cuda`'s `launch_context_is_stated` files `rope_interleaved`
    // under `VARIED_BY_A_ROW_WITH_NO_TEXT` with the sentence that settles it:
    // *"NOTHING ON `Deployment` STATES IT"*. A `Const` mark is a promise the
    // STATEMENT carries the number, and no trace text can keep it.
    //
    // `RopeTheta` is FIRE-WIDE; gemma-4's per-layer `keys::Theta` differs on
    // sliding layers, so the two must not be confused.
    let theta = ctx.ask::<f32, keys::RopeTheta>()?;
    let interleaved = ctx.ask::<bool, keys::RopeInterleaved>()?;
    let positions = ctx.ask::<*const i32, keys::Positions>()?;
    let half = head_dim / 2;
    let pairs = cache_pairs(half);
    let smem = pairs.unsigned_abs() * 2 * 4;
    let total_heads = num_q_heads + num_kv_heads;
    let per_block = heads_per_block(half);
    ctx.fire(Fire::at("rope/rope.cuh", "::pie::rope::rotate<::pie::false_type::value, false>").apply(rotate_launch(q.rows, total_heads, per_block, smem)), &[
                q.arg(),
                k.arg(),
                positions.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                theta.arg(),
                interleaved.arg(),
                pairs.arg(),
                per_block.arg(),
                MaybeConst::<bf16>::none().arg(),
                None::<NonNull<bf16>>.arg(),
                None::<NonNull<bf16>>.arg(),
                MaybeConst::<u32>::none().arg(),
                MaybeConst::<u32>::none().arg(),
                MaybeConst::<u32>::none().arg(),
                MaybeConst::<u32>::none().arg(),
                MaybeConst::<u8>::none().arg(),
                0_i32.arg(),
                0_i32.arg(),
            ])
}

/// `rope.cu`'s `rope::rope_write_kv_bf16`.
///
/// # Safety
///
/// Every pointer must address live device memory of the extent the paged-KV
/// descriptors describe, and `stream` must be live across the launch.
#[routine(whole)]
pub fn rope_write_kv_bf16(
    ctx: &Ctx<'_>,
    // ONE ADDRESS IN BOTH RUNS: the statement places `q` as input 0 and
    // declares the rotated `q` as its one result. `Out` alone would leave
    // input 0 unclaimed and hand `k` the query's buffer.
    q: InOut<Tensor<bf16>>,
    // Through the statement's second input, not `Out(1)` (no such slot):
    // `k` rotates in place and no result is declared for it.
    k: In<Tensor<bf16>>,
    v: In<Tensor<bf16>>,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<bool, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    interleaved: bool) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.
    let page_size = ctx.ask::<i32, keys::KvPageSize>()?;
    let num_q_heads = ctx.ask::<i32, keys::NumQHeads>()?;
    let num_kv_heads = ctx.ask::<i32, keys::NumKvHeads>()?;
    let head_dim = ctx.ask::<i32, keys::HeadDim>()?;
    let theta = ctx.ask::<f32, keys::RopeTheta>()?;
    let hnd_layout = ctx.ask::<bool, keys::KvHndLayout>()?;

    let k_pages = ctx.ask::<*mut bf16, keys::KvKeys>()?;
    let qo_indptr = ctx.ask::<*const u32, keys::QoIndptr>()?;
    let row_valid = ctx.ask::<*const u8, keys::RowValid>()?;
    let num_requests = ctx.ask::<i32, keys::RequestCount>()?;
    let positions = ctx.ask::<*const i32, keys::Positions>()?;
    let v_pages = ctx.ask::<*mut bf16, keys::KvValues>()?;
    let kv_page_indices = ctx.ask::<*const u32, keys::KvPageIndices>()?;
    let kv_page_indptr = ctx.ask::<*const u32, keys::KvPageIndptr>()?;
    let kv_last_page_lens = ctx.ask::<*const u32, keys::KvLastPageLens>()?;
    let half = head_dim / 2;
    let pairs = cache_pairs(half);
    let smem = pairs.unsigned_abs() * 2 * 4;
    let total_heads = num_q_heads + num_kv_heads;
    let per_block = heads_per_block(half);
    let instantiation =
        if hnd_layout { "::pie::rope::rotate<\
                             ::pie::true_type::value, true>" } else { "::pie::rope::rotate<::pie::true_type::value, false>" };
    ctx.fire(Fire::at("rope/rope.cuh", instantiation).apply(rotate_launch(q.rows, total_heads, per_block, smem)), &[
                q.arg(),
                k.arg(),
                positions.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                theta.arg(),
                interleaved.arg(),
                pairs.arg(),
                per_block.arg(),
                MaybeConst::new(v.ptr).arg(),
                NonNull::new(k_pages).arg(),
                NonNull::new(v_pages).arg(),
                MaybeConst::new(qo_indptr).arg(),
                MaybeConst::new(kv_page_indices).arg(),
                MaybeConst::new(kv_page_indptr).arg(),
                MaybeConst::new(kv_last_page_lens).arg(),
                MaybeConst::new(row_valid).arg(),
                num_requests.arg(),
                page_size.arg(),
            ])
}

/// `rope.cu`'s `rope::qk_rmsnorm_rope_bf16`.
///
/// # Errors
///
/// [`q_heads`]'s and [`heads`]'s.
///
/// # Safety
///
/// [`rope_bf16`]'s, plus `q_weight`/`k_weight` addressing `head_dim` live
/// bf16 elements each.
#[routine]
pub fn qk_rmsnorm_rope_bf16(
    ctx: &Ctx<'_>,
    q: InOut<Tensor<bf16>>,
    k: InOut<Tensor<bf16>>,
    // Positional bank, not `Weight<N, T>` (same word, different table).
    q_weight: Const<Tensor<bf16>>,
    k_weight: Const<Tensor<bf16>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: every one of these was `Env<keys::_>` before the
    // four marks, and no builder ever began stating them. A `Const` mark
    // PROMISES the statement carries the number at its slot in the params
    // run; where nothing states one the promise is broken at the fire, not
    // at the type. See `.wiki/migration.md` §11.20.
    let head_dim = ctx.ask::<i32, keys::HeadDim>()?;
    let theta = ctx.ask::<f32, keys::Theta>()?;
    let eps = ctx.ask::<f32, keys::RmsEps>()?;

    let positions = ctx.ask::<*const i32, keys::Positions>()?;
    let (num_q_heads, num_kv_heads) =
        (q_heads(q.width, head_dim)?, k_heads(q.ptr, k.ptr, k.width, head_dim)?);
    let total_heads = num_q_heads + num_kv_heads;
    ctx.fire(Fire::at("rope/rope.cuh", "::pie::rope::qk_rmsnorm_rotate<::pie::i32(128)>").apply(fused_launch(q.rows, total_heads)), &[
                q.arg(),
                k.arg(),
                q_weight.arg(),
                k_weight.arg(),
                positions.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                theta.arg(),
                eps.arg(),
            ])
}

/// `rope.cu`'s `rope::qk_rmsnorm_rope_bf16_devwin`.
///
/// # Errors
///
/// [`q_heads`]'s and [`heads`]'s, per [`qk_rmsnorm_rope_bf16`].
///
/// # Safety
///
/// `win` addresses two live `u32`s on the device; the rest is
/// [`rope_bf16`]'s obligation with `n_max` for the row count.
#[routine(whole)]
pub fn qk_rmsnorm_rope_bf16_devwin(
    ctx: &Ctx<'_>,
    q: InOut<Tensor<bf16>>,
    k: InOut<Tensor<bf16>>,
    q_weight: Const<Tensor<bf16>>,
    k_weight: Const<Tensor<bf16>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.
    let head_dim = ctx.ask::<i32, keys::HeadDim>()?;
    let theta = ctx.ask::<f32, keys::Theta>()?;
    let eps = ctx.ask::<f32, keys::RmsEps>()?;

    let win = ctx.ask::<*mut u32, keys::PeelWindow>()?;
    let n_max = ctx.ask::<i32, keys::RowsTotal>()?;
    let positions = ctx.ask::<*const i32, keys::Positions>()?;
    let (num_q_heads, num_kv_heads) =
        (q_heads(q.width, head_dim)?, k_heads(q.ptr, k.ptr, k.width, head_dim)?);
    let total_heads = num_q_heads + num_kv_heads;
    ctx.fire(Fire::at("rope/rope.cuh", "::pie::rope::qk_rmsnorm_rotate_devwin<::pie::i32(128)>").apply(fused_launch(n_max, total_heads)), &[
                q.arg(),
                k.arg(),
                q_weight.arg(),
                k_weight.arg(),
                positions.arg(),
                win.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                theta.arg(),
                eps.arg(),
            ])
}

/// `rope.cu`'s `rope::qk_rmsnorm_mrope_bf16`.
///
/// # Safety
///
/// [`qk_rmsnorm_rope_bf16_devwin`]'s, without `win`, and `positions` must
/// address `q.rows * 3` live `i32`s rather than `q.rows`.
#[routine]
pub fn qk_rmsnorm_mrope_bf16(
    ctx: &Ctx<'_>,
    q: InOut<Tensor<bf16>>,
    k: InOut<Tensor<bf16>>,
    // Head counts stay stated facts; this launcher never divides a width.
    q_weight: Const<Tensor<bf16>>,
    k_weight: Const<Tensor<bf16>>,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    mrope_section_t: i32,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    mrope_section_h: i32,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    mrope_section_w: i32) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.
    let num_q_heads = ctx.ask::<i32, keys::NumQHeads>()?;
    let num_kv_heads = ctx.ask::<i32, keys::NumKvHeads>()?;
    let head_dim = ctx.ask::<i32, keys::HeadDim>()?;
    let theta = ctx.ask::<f32, keys::Theta>()?;
    let eps = ctx.ask::<f32, keys::RmsEps>()?;

    let positions = ctx.ask::<*const i32, keys::Positions>()?;
    let total_heads = num_q_heads + num_kv_heads;
    ctx.fire(Fire::at("rope/rope.cuh", "::pie::rope::qk_rmsnorm_rotate_mrope<::pie::i32(128)>").apply(fused_launch(q.rows, total_heads)), &[
                q.arg(),
                k.arg(),
                q_weight.arg(),
                k_weight.arg(),
                positions.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                theta.arg(),
                eps.arg(),
                mrope_section_t.arg(),
                mrope_section_h.arg(),
                mrope_section_w.arg(),
            ])
}

/// `rope.cu`'s `rope::qk_rmsnorm_rope_bf16_rounded`.
///
/// # Errors
///
/// [`q_heads`]'s and [`heads`]'s, per [`qk_rmsnorm_rope_bf16`].
///
/// # Safety
///
/// [`qk_rmsnorm_mrope_bf16`]'s. `k.ptr` and `k_weight` may be null together,
/// and the kernel reads the pair as "there is no k".
#[routine]
pub fn qk_rmsnorm_rope_bf16_rounded(
    ctx: &Ctx<'_>,
    q: InOut<Tensor<bf16>>,
    // A resolved `Out(1)` with no width is this file's "there is no k".
    k: InOut<Tensor<bf16>>,
    q_weight: Const<Tensor<bf16>>,
    // Optional: the no-K caller supplies `Const { v: Or(core::ptr::null())
    // }` by hand; `num_kv_heads = 0` keeps the kernel from reading it.
    k_weight: Const<Tensor<bf16>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: every one of these was `Env<keys::_>` before the
    // four marks, and no builder ever began stating them. A `Const` mark
    // PROMISES the statement carries the number at its slot in the params
    // run; where nothing states one the promise is broken at the fire, not
    // at the type. See `.wiki/migration.md` §11.20.
    let head_dim = ctx.ask::<i32, keys::KvHeadDim>()?;
    let theta = ctx.ask::<f32, keys::Theta>()?;
    let eps = ctx.ask::<f32, keys::RmsEps>()?;

    let positions = ctx.ask::<*const i32, keys::Positions>()?;
    let (num_q_heads, num_kv_heads) =
        (q_heads(q.width, head_dim)?, k_heads(q.ptr, k.ptr, k.width, head_dim)?);
    let total_heads = num_q_heads + num_kv_heads;
    ctx.fire(Fire::at("rope/rope.cuh", "::pie::rope::qk_rmsnorm_rotate_rounded<::pie::i32(128)>").apply(fused_launch(q.rows, total_heads)), &[
                q.arg(),
                k.arg(),
                q_weight.arg(),
                k_weight.arg(),
                positions.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                theta.arg(),
                eps.arg(),
            ])
}

/// `rope::q_rmsnorm_rope_bf16_rounded` -- [`qk_rmsnorm_rope_bf16_rounded`]
/// over Q alone.
///
/// gemma-4's shared sliding layer, whose K was normed and rotated at the
/// layer that owns the cache entry, so only Q is rotated here.
///
/// # Errors
///
/// [`qk_rmsnorm_rope_bf16_rounded`]'s.
///
/// # Safety
///
/// [`qk_rmsnorm_rope_bf16_rounded`]'s, less `k` and `k_weight`.
#[routine]
pub fn q_rmsnorm_rope_bf16_rounded(
    ctx: &Ctx<'_>,
    q: InOut<Tensor<bf16>>,
    q_weight: Const<Tensor<bf16>>) -> Result<(), Refusal> {
    // THE CALLEE ASKS FOR ALL OF IT ITSELF, so this forwards none of it: a
    // fact only the fire can answer reaches a body through its own context,
    // not through the argument list of whoever called it. The positions, the
    // head pitch, the theta and the epsilon all left this call for that
    // reason.
    //
    // AND THE ASKS LEFT WITH THEM. Three `ctx.ask` lines stood here after the
    // arguments they fed were gone -- `KvHeadDim`, `Theta`, `RmsEps` --
    // reading facts into names nothing read. Removing them changes no
    // refusal: `qk_rmsnorm_rope_bf16_rounded` asks the same three keys with
    // the same `?`, so an unstated one refuses one frame later and with the
    // same words. What they cost was a reader believing this body reads them.
    qk_rmsnorm_rope_bf16_rounded(
        ctx,
        q,
        // Zero-width operand the statement never places; `rows: q.rows`
        // since `rows` is the launch's, not this absent operand's.
        InOut {
            ptr: core::ptr::null_mut(),
            rows: q.rows,
            width: 0,
        },
        q_weight,
        // The callee's `k_weight` fallback: no bank slot 1 to forward.
        Const { v: core::ptr::null() },
    )
}

/// `rope.cu`'s `rope::rope_yarn_bf16`.
///
/// # Safety
///
/// [`rope_bf16`]'s.
#[routine]
pub fn rope_yarn_bf16(
    ctx: &Ctx<'_>,
    q: InOut<Tensor<bf16>>,
    k: InOut<Tensor<bf16>>,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<f32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    factor: f32,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<f32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    low_freq_factor: f32,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<f32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    high_freq_factor: f32,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    original_max_position: i32) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.
    let num_q_heads = ctx.ask::<i32, keys::NumQHeads>()?;
    let num_kv_heads = ctx.ask::<i32, keys::NumKvHeads>()?;
    let head_dim = ctx.ask::<i32, keys::HeadDim>()?;
    let theta = ctx.ask::<f32, keys::RopeTheta>()?;

    let positions = ctx.ask::<*const i32, keys::Positions>()?;
    let half = head_dim / 2;
    let total_heads = num_q_heads + num_kv_heads;
    let per_block = heads_per_block(half);
    #[allow(clippy::cast_precision_loss)]
    let orig_max_pos = original_max_position as f32;
    ctx.fire(Fire::at("rope/rope.cuh", "::pie::rope::rotate_yarn").apply(rotate_launch(q.rows, total_heads, per_block, 0)), &[
                q.arg(),
                k.arg(),
                positions.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                theta.arg(),
                factor.arg(),
                low_freq_factor.arg(),
                high_freq_factor.arg(),
                orig_max_pos.arg(),
                per_block.arg(),
            ])
}

/// `rope.cu`'s `rope::rope_yarn_original_bf16` (OLMo-3, gpt-oss).
///
/// # Errors
///
/// [`q_heads`]'s and [`heads`]'s, per [`qk_rmsnorm_rope_bf16`], and
/// [`Refusal::Unstated`] for a checkpoint with no YaRN block — see the body.
///
/// # Safety
///
/// [`rope_bf16`]'s.
#[routine]
pub fn rope_yarn_original_bf16(
    ctx: &Ctx<'_>,
    q: InOut<Tensor<bf16>>,
    k: InOut<Tensor<bf16>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: every one of these was `Env<keys::_>` before the
    // four marks, and no builder ever began stating them. A `Const` mark
    // PROMISES the statement carries the number at its slot in the params
    // run; where nothing states one the promise is broken at the fire, not
    // at the type. See `.wiki/migration.md` §11.20.
    let head_dim = ctx.ask::<i32, keys::HeadDim>()?;
    let theta = ctx.ask::<f32, keys::RopeTheta>()?;
    let factor = ctx.ask::<f32, keys::YarnFactor>()?;
    let beta_fast = ctx.ask::<f32, keys::YarnBetaFast>()?;
    let beta_slow = ctx.ask::<f32, keys::YarnBetaSlow>()?;
    let attention_factor = ctx.ask::<f32, keys::YarnAttentionFactor>()?;
    let original_max_position = ctx.ask::<i32, keys::YarnOriginalMaxPosition>()?;
    let interleaved = ctx.ask::<bool, keys::RopeInterleaved>()?;

    let positions = ctx.ask::<*const i32, keys::Positions>()?;
    // A checkpoint with no YaRN block reaches here as `Yarn::NONE`;
    // unguarded, `ramp_bounds` would compute `(0.0 / 0.0).ln()` and rotate
    // against NaN ramps.
    if original_max_position <= 0 {
        return Err(Refusal::Unstated { what: "the checkpoint's YaRN block" });
    }
    let (num_q_heads, num_kv_heads) =
        (q_heads(q.width, head_dim)?, k_heads(q.ptr, k.ptr, k.width, head_dim)?);
    let (low_dim, high_dim) = ramp_bounds(
        head_dim,
        theta,
        beta_fast,
        beta_slow,
        original_max_position,
    );
    let half = head_dim / 2;
    let pairs = cache_pairs(half);
    let smem = pairs.unsigned_abs() * 8;
    let total_heads = num_q_heads + num_kv_heads;
    let per_block = heads_per_block(half);
    ctx.fire(Fire::at("rope/rope.cuh", "::pie::rope::rotate_yarn_original").apply(rotate_launch(q.rows, total_heads, per_block, smem)), &[
                q.arg(),
                k.arg(),
                positions.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                theta.arg(),
                factor.arg(),
                low_dim.arg(),
                high_dim.arg(),
                attention_factor.arg(),
                interleaved.arg(),
                per_block.arg(),
                pairs.arg(),
            ])
}

/// `rope::rope_partial_bf16` — partial rotary over the first `rotary_dim`.
///
/// One `fn` for all three concrete forms below, so the head-count division
/// lives in one place.
///
/// # Errors
///
/// [`q_heads`]'s and [`heads`]'s.
///
/// # Safety
///
/// [`rope_bf16`]'s.
fn rope_partial<T>(
    ctx: &Ctx<'_>,
    instantiation: &'static str,
    q: *mut T,
    k: *mut T,
    // Plain `*const i32`, not `Env`: this private `fn` has no table row.
    // Callers deref `*positions` to forward.
    positions: *const i32,
    // Plain `i32`/`f32`s for the same reason.
    num_tokens: i32,
    q_width: i32,
    k_width: i32,
    head_dim: i32,
    rotary_dim: i32,
    theta: f32) -> Result<(), Refusal>
where
    T: kernels::Elem,
    // BOTH BOUNDS, BECAUSE THIS HELPER TAKES RAW POINTERS AND SPENDS THEM.
    // `Abi` is what makes `*mut T` a CUDA argument at all; `Bind` is what
    // lets the body write `q.arg()` -- the one spelling every plane's body
    // uses, which `arg_via_abi!` stamps per pointee rather than deriving.
    // AND THE WRITE SIDE NAMES ITS OWN TRAIT. `Out`/`InOut` bind through
    // `BindMut`, not `Bind`, so that a plane whose read and write carriers
    // are ONE TYPE can still say which way a slot is driven -- see
    // `kernels::routine::BindMut`. Here the two carriers already differ, so
    // the blanket impl over `*mut T` makes this the same obligation twice;
    // it is spelled because `<T as Elem>::Write` is opaque under a generic
    // `T` and the compiler cannot see that it is this pointer.
    *mut T: Abi + kernels::Bind<crate::jit::ArgValue>,
    T: kernels::Elem<Write = *mut T>,
    <T as kernels::Elem>::Write: Abi,
{
    let (num_q_heads, num_kv_heads) =
        (q_heads(q_width, head_dim)?, k_heads(q, k, k_width, head_dim)?);
    ctx.fire(Fire::at("rope/rope.cuh", instantiation).apply(Launch::per_row(num_tokens.unsigned_abs(), ROTATE_BLOCK.unsigned_abs())), &[
                q.arg(),
                k.arg(),
                positions.arg(),
                // Every caller passes the literal `0`; no nonzero-delta
                // symbol has ever had a launcher.
                0i32.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                rotary_dim.arg(),
                theta.arg(),
            ])
}

/// `rope::rope_partial_bf16` — [`rope_partial`] over bf16.
///
/// # Errors
///
/// [`rope_partial`]'s.
#[routine]
pub fn rope_partial_bf16(
    ctx: &Ctx<'_>,
    q: InOut<Tensor<bf16>>,
    k: InOut<Tensor<bf16>>,
    // `head_dim` is the cache's -- see [`qk_rmsnorm_rope_bf16_rounded`].
    rotary_dim: Const<i32>) -> Result<(), Refusal> {
    // HEAD ASKED FOR BOTH, and the statement carries neither: `dsl::cuda::
    // rope_partial` states `[rotary_dim]` alone, which is HEAD's `Param<0>`.
    // `keys::KvHeadDim` and `keys::Theta` are answered by `driver-cuda`, and
    // a `Const` mark here promises a number no trace text passes.
    let head_dim = ctx.ask::<i32, keys::KvHeadDim>()?;
    let theta = ctx.ask::<f32, keys::Theta>()?;
    let positions = ctx.ask::<*const i32, keys::Positions>()?;
    rope_partial(
        ctx,
        "::pie::rope::rotate_partial<::pie::bf16>",
        q.ptr,
        k.ptr,
        positions,
        q.rows,
        q.width,
        k.width,
        // Starred: the callee takes a plain number, not a fact type.
        head_dim,
        *rotary_dim,
        theta,
    )
}

/// `rope::rope_partial_q_bf16` — [`rope_partial_bf16`] rotating Q alone.
///
/// K was already rotated at its source layer, so "no k" here is a null
/// pair with a zero width ([`heads`] reads it back as `num_kv_heads = 0`).
///
/// # Errors
///
/// [`rope_partial`]'s.
///
/// # Safety
///
/// [`rope_partial_bf16`]'s, less `k`.
#[routine]
pub fn rope_partial_q_bf16(
    ctx: &Ctx<'_>,
    q: InOut<Tensor<bf16>>,
    // The cache's, per [`rope_partial_bf16`].
    rotary_dim: Const<i32>) -> Result<(), Refusal> {
    // HEAD ASKED FOR BOTH, and the statement carries neither: `dsl::cuda::
    // rope_partial` states `[rotary_dim]` alone, which is HEAD's `Param<0>`.
    // `keys::KvHeadDim` and `keys::Theta` are answered by `driver-cuda`, and
    // a `Const` mark here promises a number no trace text passes.
    let head_dim = ctx.ask::<i32, keys::KvHeadDim>()?;
    let theta = ctx.ask::<f32, keys::Theta>()?;
    let positions = ctx.ask::<*const i32, keys::Positions>()?;
    rope_partial(
        ctx,
        "::pie::rope::rotate_partial<::pie::bf16>",
        q.ptr,
        // Q's own address stands in for K; never dereferenced since
        // `num_kv_heads = 0` keeps the kernel off it.
        q.ptr,
        positions,
        q.rows,
        q.width,
        0,
        head_dim,
        *rotary_dim,
        theta,
    )
}

/// `rope::rope_partial_f16` — [`rope_partial`] over f16.
///
/// No arm binds this symbol — the split it would need does not exist for
/// f16.
///
/// # Errors
///
/// [`rope_partial`]'s.
#[routine(internal)]
pub fn rope_partial_f16(
    ctx: &Ctx<'_>,
    q: InOut<Tensor<f16>>,
    k: InOut<Tensor<f16>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.
    let head_dim = ctx.ask::<i32, keys::KvHeadDim>()?;
    let rotary_dim = ctx.ask::<i32, keys::RotaryWidth>()?;
    let theta = ctx.ask::<f32, keys::Theta>()?;

    let positions = ctx.ask::<*const i32, keys::Positions>()?;
    rope_partial(
        ctx,
        "::pie::rope::rotate_partial<::pie::f16>",
        q.ptr,
        k.ptr,
        positions,
        q.rows,
        q.width,
        k.width,
        head_dim,
        rotary_dim,
        theta,
    )
}

/// `rope.cu`'s `rope::rope_partial_last_bf16` (deepseek-v4).
///
/// # Errors
///
/// [`q_heads`]'s and [`heads`]'s, per [`qk_rmsnorm_rope_bf16`].
///
/// # Safety
///
/// [`rope_bf16`]'s.
#[routine]
pub fn rope_partial_last_bf16(
    ctx: &Ctx<'_>,
    // K is declared here; the Q-alone form is the separate symbol below.
    q: Out<Tensor<bf16>>,
    k: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.
    let head_dim = ctx.ask::<i32, keys::KvHeadDim>()?;
    let rotary_dim = ctx.ask::<i32, keys::RotaryWidth>()?;
    let theta = ctx.ask::<f32, keys::Theta>()?;
    let interleaved = ctx.ask::<bool, keys::RopeInterleaved>()?;
    let yarn_factor = ctx.ask::<f32, keys::YarnFactor>()?;
    let yarn_beta_fast = ctx.ask::<f32, keys::YarnBetaFast>()?;
    let yarn_beta_slow = ctx.ask::<f32, keys::YarnBetaSlow>()?;
    let yarn_original_max_position = ctx.ask::<i32, keys::YarnOriginalMaxPosition>()?;

    let positions = ctx.ask::<*const i32, keys::Positions>()?;
    let (num_q_heads, num_kv_heads) = (q_heads(q.width, head_dim)?, heads(k.width, head_dim)?);
    let (low_dim, high_dim) = if yarn_factor > 1.0 && yarn_original_max_position > 0 {
        ramp_bounds(
            rotary_dim,
            theta,
            yarn_beta_fast,
            yarn_beta_slow,
            yarn_original_max_position,
        )
    } else {
        (0.0, 0.0)
    };
    ctx.fire(Fire::at("rope/rope.cuh", "::pie::rope::rotate_partial_last").apply(Launch::per_row(q.rows.unsigned_abs(), ROTATE_BLOCK.unsigned_abs())), &[
                q.arg(),
                k.arg(),
                positions.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                rotary_dim.arg(),
                theta.arg(),
                // Always `false`: the caller-facing flag was removed, but
                // `rotate_partial_last` still takes it.
                false.arg(),
                interleaved.arg(),
                yarn_factor.arg(),
                low_dim.arg(),
                high_dim.arg(),
            ])
}

/// `rope::rope_partial_last_q_bf16` -- [`rope_partial_last_bf16`] rotating Q
/// alone.
///
/// The built `k` uses `q.ptr` (not null) with a zero width: unlike
/// [`q_rmsnorm_rope_bf16_rounded`]'s callee, this kernel indexes both
/// pointers before testing the head count.
///
/// # Errors
///
/// [`rope_partial_last_bf16`]'s.
///
/// # Safety
///
/// [`rope_partial_last_bf16`]'s, less `k`.
#[expect(clippy::too_many_arguments, reason = "D1: a routine takes fields, never a struct")]
#[routine]
pub fn rope_partial_last_q_bf16(
    ctx: &Ctx<'_>,
    q: InOut<Tensor<bf16>>) -> Result<(), Refusal> {
    // THE CALLEE ASKS FOR ALL OF IT ITSELF -- the positions and the eight rope
    // and YaRN numbers alike.
    //
    // EIGHT `ctx.ask` LINES STOOD HERE and fed nothing: `KvHeadDim`,
    // `RotaryWidth`, `Theta`, `RopeInterleaved` and the four YaRN numbers,
    // read into names no line below mentions. `rope_partial_last_bf16` asks
    // for the same eight with the same `?`, so this refuses on exactly the
    // same unstated facts as before, one frame later. Eight is enough of them
    // that the block read like the body's own vocabulary.
    //
    // The two marks differ and the ADDRESS does not: this form takes `q` as
    // `InOut` because the statement places it once and declares it once,
    // while the K-carrying form declares both halves as results. Forwarding
    // is a re-mark, not a copy.
    rope_partial_last_bf16(
        ctx,
        Out { ptr: q.ptr, rows: q.rows, width: q.width },
        // Q's own address with a zero width; must be real, not null (see
        // the doc above).
        Out {
            ptr: q.ptr,
            rows: q.rows,
            width: 0,
        },
    )
}


// `Yarn` can't arrive from a statement, so `Env<Yarn>`-style types group by
// that property rather than by family — a scheme shared with attention/ssm.

/// The YaRN quartet, as a checkpoint states it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Yarn {
    /// The context scale factor.
    pub factor: f32,
    /// The ramp's high-frequency bound, in rotations.
    pub beta_fast: f32,
    /// The ramp's low-frequency bound, in rotations.
    pub beta_slow: f32,
    /// The attention temperature.
    pub attention_factor: f32,
    /// The position count the checkpoint was trained at.
    pub original_max_position: i32,
}
impl Yarn {
    /// A checkpoint with no YaRN block.
    pub const NONE: Self = Self {
        factor: 1.0,
        beta_fast: 0.0,
        beta_slow: 0.0,
        attention_factor: 1.0,
        original_max_position: 0,
    };
}
// `rope_write_kv_bf16`'s `q` is one address in both runs -- the statement
// places it as input 0 and takes the rotated query back as its one result --
// and `k` writes in place through the statement's SECOND input, so its source
// is `Slot(In, 1)` and `v`'s is `Slot(In, 2)`. `qo_indptr`,
// `kv_last_page_lens` and `row_valid` moved from `Unbound` to asked facts —
// invisible to `arity_problem` either way — so `[0..=2]` below must still be
// the same three operand slots.
const _: () = {
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(rope_write_kv_bf16);
    assert!(d.len() == 4);
    assert!(matches!(d[0], Some(kernels::Source::Alias(0, 0))));
    assert!(matches!(d[1], Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(d[2], Some(kernels::Source::Slot(kernels::Kind::In, 2))));
    // THE FIVE PAGED TABLES ARE NOT PARAMETERS ANY MORE. The query-side CSR,
    // the page indices, the page indptr and the last-page lengths were four
    // `Source::Named` entries here -- one index apart and each easy to swap
    // for its neighbour, which is why they were pinned by index at all. They
    // are §6.2's plan leaves: what a fire's scheduler built, which the body
    // asks its context for. There is no parameter left to swap.
    //
    // What the statement carries is the layer geometry, and the params run is
    // pinned instead, in the order the signature declares it.
    // `interleaved` used to be pinned here as the one `#[unbound]` entry.
    // The column is four entries long now -- the eight YaRN/rope numbers are
    // asked for, `interleaved` among them -- so there is no index left to
    // pin, and nothing positional left to swap.
};

// `rotary_dim` IS THE STATEMENT'S, FULL STOP.
//
// It was a chain -- the statement's scalar where a checkpoint states one and
// the fire's `keys::RotaryWidth` otherwise -- because `model-dsl`'s
// `rope_launch` passed `vec![0]` for full rope and `vec![rotary_dim]` for
// partial, so a SENTINEL in the value carried a distinction the key already
// named. §3.1 puts `RotaryWidth` among the checkpoint's constants: it is a
// property of the layer, not of the fire, and `Const<i32>` carries it in the
// params run with no sentinel and no fallback.
//
// AND IT IS SLOT 0, which is `Param<0>` again. The head dim and the theta sat
// in front of it as `Const` marks for a while and pushed it to 1 -- a slot
// `dsl::cuda::rope_partial`, which states `[rotary_dim]` alone, never filled.
// Both are asked for in the body now, so the width is the only scalar and the
// run is the one the statement has always carried.
const _: () = {
    let partial = kernels::routine::sources::<crate::jit::Cuda, _, _>(rope_partial_bf16);
    assert!(matches!(partial[2], Some(kernels::Source::Slot(kernels::Kind::Param, 0))));
    let partial_q = kernels::routine::sources::<crate::jit::Cuda, _, _>(rope_partial_q_bf16);
    assert!(matches!(partial_q[1], Some(kernels::Source::Slot(kernels::Kind::Param, 0))));
};

// `interleaved`/`win` were bare `bool`/`Unbound` (`source: None`); now
// A silently-bound bool picks the wrong `rotate` instantiation with no type
// error, so each row is pinned by index -- and the index is what changed.
//
// `interleaved` IS `Source::Named("rope_interleaved")` AGAIN, and this pin
// used to argue the opposite: §3.2 puts the flag among the checkpoint's rope
// constants -- a deployment either interleaves its pairs or does not -- so a
// `Const<bool>` looked right. The premise it needed is false. `driver-cuda`'s
// `launch_context_is_stated` files the same flag under
// `VARIED_BY_A_ROW_WITH_NO_TEXT` with the sentence that settles it: *"NOTHING
// ON `Deployment` STATES IT"*. A `Const` mark promises the STATEMENT carries
// the number, and no trace text can keep that promise, so `rope_bf16` could
// not fire while it made one.
//
// What the column says now: three `Const` extents and no scalar past them,
// because the theta and the flag are asked for in the body.
const _: () = {
    let plain = kernels::routine::sources::<crate::jit::Cuda, _, _>(rope_bf16);
    assert!(plain.len() == 2);
    // `rope_yarn_original_bf16` was bound here and asserted nothing. It is
    // walked properly at the bottom of this file, where `whole` reads the
    // source column.
};

// These four rows have no hand arm left, so this is the only thing that
// catches a slot regressing to `None` (silently falling back to
// `Refusal::Unstated`) under a `cargo check`-only regime.
// THE FOUR ROWS THIS WALKS WERE WALKED TWICE, and the other walk asserted
// nothing: a block above bound `qk_rmsnorm_rope_bf16_devwin`,
// `rope_partial_last_bf16`, `rope_partial_last_q_bf16` and
// `rope_yarn_original_bf16` to four names and stopped, under a comment saying
// it was *"the only thing that catches a slot regressing to `None`"*.
// `sources` returns `F::SOURCES` and inspects nothing, so what it caught was
// that the four routines still EXIST and still satisfy `KernelFn` -- worth
// something, and not what it said. This block is where the claim is kept.
//
// Its prose is worth carrying over, because it says what a `None` here would
// mean. The three YaRN lists don't line up -- the `_last` pair omits
// `attention_factor` -- so a reorder cannot be caught by eye. THE FIVE YARN
// NUMBERS ARE THE STATEMENT'S NOW, so there are no keys to hold the slots
// apart by: §3.2 puts the whole YaRN block among the checkpoint's rope
// constants, and records why they belong together -- they *"always arrive
// together and are always absent together"*, which as five `Env` parameters
// was an invariant a body had to guard and as five `Const` parameters is
// arity. A checkpoint with no YaRN block emits no statement carrying them, so
// `arity_problem` refuses it before a body runs; what is pinned is their ORDER
// in the params run, which is the thing a swap would still get wrong. And
// `RowsTotal` LEFT the parameter list -- the fire's total token count, which
// no region can stand in for -- because §6.1 keeps it in `Env`'s successor
// and the body asks for it.
const _: () = {
    // `Derived` CARRIES NO SOURCE, and never did: the two columns are
    // computed differently -- a `Derived` row is what `#[routine]` reads off
    // the SYNTAX, a source is what `resolve` walks out of the TYPES -- and
    // keeping them apart is what stopped the two disagreeing. So this walks
    // the source column directly.
    const fn whole(d: &[Option<kernels::Source>]) -> bool {
        let mut i = 0;
        while i < d.len() {
            if d[i].is_none() {
                return false;
            }
            i += 1;
        }
        true
    }
    let devwin = <qk_rmsnorm_rope_bf16_devwin as ::kernels::Derivation>::SOURCES;
    assert!(devwin.len() == 4 && whole(devwin));
    let last = <rope_partial_last_bf16 as ::kernels::Derivation>::SOURCES;
    assert!(last.len() == 2 && whole(last));
    let last_q = <rope_partial_last_q_bf16 as ::kernels::Derivation>::SOURCES;
    assert!(last_q.len() == 1 && whole(last_q));
    let orig = <rope_yarn_original_bf16 as ::kernels::Derivation>::SOURCES;
    assert!(orig.len() == 2 && whole(orig));
};

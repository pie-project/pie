//! Rope family: cos/sin table generation, plain/fused/mrope rotation, YaRN
//! scaling, partial rotary, and the paged-KV write-through variant.
//!
//! Parameters are typed regions (`In<N,_>`/`Out<N,_>`/`Bank<N,_>`) or
//! `Env<keys::X>` facts; a bare `#[source(...)]` attribute remains only
//! where no type can carry the source.
#![allow(clippy::too_many_arguments)]

use crate::jit::Abi;
use crate::jit::abi::{MaybeConst, bf16, f16};
use crate::jit::{Ctx, Family, Launch, Routine};
use crate::routine;
use kernels::Refusal;
use kernels::keys;
use kernels::routine::Bank;
use kernels::routine::InOut;
use kernels::routine::In;
use kernels::routine::Env;
use kernels::routine::Out;
use kernels::routine::Param;

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
#[kernels_macros::routine]
pub fn rope_standard_table(
    ctx: &Ctx,
    positions: Env<keys::Positions>,
    // The region also carries the row count `Launch::per_row` uses.
    table: Out<0, f32>,
    head_dim: Env<keys::HeadDim>,
    // The fire-wide base (`Cx::rope_theta`), not the per-layer `keys::Theta`
    // gemma-4 splits by sliding/full layer kind; the two agree elsewhere.
    theta: Env<keys::RopeTheta>,
) -> Result<(), Refusal> {
    if **head_dim / 2 <= 0 {
        return Err(Refusal::Empty { what: "head_dim / 2" });
    }
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "rope/rope.cuh",
            "::pie::rope::standard_table<::pie::i32>",
            Launch::per_row(table.rows.unsigned_abs(), ROTATE_BLOCK.unsigned_abs()),
            &[positions.arg(), table.ptr.arg(), head_dim.arg(), theta.arg()],
        )
    }
}

/// `rope.cu`'s `rope::rope_bf16`.
///
/// # Safety
///
/// `q` and `k` address `q.rows * num_q_heads * head_dim` and
/// `k.rows * num_kv_heads * head_dim` live bf16 elements, `positions`
/// `q.rows` live `i32`s, and `stream` must be live across the launch.
#[kernels_macros::routine]
pub fn rope_bf16(
    ctx: &Ctx,
    // Head counts stay stated facts rather than `heads(q.width, ..)`: this
    // kernel's arm never divided a width.
    q: Out<0, bf16>,
    k: Out<1, bf16>,
    positions: Env<keys::Positions>,
    num_q_heads: Env<keys::NumQHeads>,
    num_kv_heads: Env<keys::NumKvHeads>,
    head_dim: Env<keys::HeadDim>,
    // Fire-wide; gemma-4's per-layer `keys::Theta` differs on sliding
    // layers, so the two must not be confused.
    theta: Env<keys::RopeTheta>,
    interleaved: Env<keys::RopeInterleaved>,
) -> Result<(), Refusal> {
    let half = **head_dim / 2;
    let pairs = cache_pairs(half);
    let smem = pairs.unsigned_abs() * 2 * 4;
    let total_heads = **num_q_heads + **num_kv_heads;
    let per_block = heads_per_block(half);
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "rope/rope.cuh",
            "::pie::rope::rotate<::pie::false_type::value, false>",
            rotate_launch(q.rows, total_heads, per_block, smem),
            &[
                q.ptr.arg(),
                k.ptr.arg(),
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
            ],
        )
    }
}

/// `rope.cu`'s `rope::rope_write_kv_bf16`.
///
/// # Safety
///
/// Every pointer must address live device memory of the extent the paged-KV
/// descriptors describe, and `stream` must be live across the launch.
#[kernels_macros::routine]
pub fn rope_write_kv_bf16(
    ctx: &Ctx,
    // Declares one result; `k` rotates in place through input 1, an arity
    // mismatch this file doesn't fix.
    q: Out<0, bf16>,
    // Through the statement's second input, not `Out(1)` (no such slot).
    k: InOut<1, bf16>,
    v: In<2, bf16>,
    positions: Env<keys::Positions>,
    // Reached via `state:`, hence `#[source(..)]`; `*mut u8` because the
    // element type varies by layer dtype and this launcher casts to bf16.
    #[source(KvKeys)] k_pages: *mut bf16,
    #[source(KvValues)] v_pages: *mut bf16,
    // The query-side CSR, distinct from `kv_page_indptr` below.
    qo_indptr: Env<keys::QoIndptr>,
    kv_page_indices: Env<keys::KvPageIndices>,
    kv_page_indptr: Env<keys::KvPageIndptr>,
    kv_last_page_lens: Env<keys::KvLastPageLens>,
    // Per-row, not `keys::AttentionMaskEnabled` (per-lane).
    row_valid: Env<keys::RowValid>,
    // The true request count, not a row count (the two agree only in
    // one-token decode).
    num_requests: Env<keys::RequestCount>,
    page_size: Env<keys::KvPageSize>,
    num_q_heads: Env<keys::NumQHeads>,
    // The fire's head count, not the cache's; swapping them is an
    // out-of-bounds.
    num_kv_heads: Env<keys::NumKvHeads>,
    head_dim: Env<keys::HeadDim>,
    // Fire-wide; see `rope_standard_table`.
    theta: Env<keys::RopeTheta>,
    // The cache's page layout flag (`[head,page,dim]` vs. `[page,head,dim]`).
    hnd_layout: Env<keys::KvHndLayout>,
    interleaved: bool,
) -> Result<(), Refusal> {
    let half = **head_dim / 2;
    let pairs = cache_pairs(half);
    let smem = pairs.unsigned_abs() * 2 * 4;
    let total_heads = **num_q_heads + **num_kv_heads;
    let per_block = heads_per_block(half);
    let instantiation =
        if **hnd_layout { "::pie::rope::rotate<\
                             ::pie::true_type::value, true>" } else { "::pie::rope::rotate<::pie::true_type::value, false>" };
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "rope/rope.cuh",
            instantiation,
            rotate_launch(q.rows, total_heads, per_block, smem),
            &[
                q.ptr.arg(),
                k.ptr.arg(),
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
                MaybeConst::new(**qo_indptr).arg(),
                MaybeConst::new(**kv_page_indices).arg(),
                MaybeConst::new(**kv_page_indptr).arg(),
                MaybeConst::new(**kv_last_page_lens).arg(),
                MaybeConst::new(**row_valid).arg(),
                num_requests.arg(),
                page_size.arg(),
            ],
        )
    }
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
#[kernels_macros::routine]
pub fn qk_rmsnorm_rope_bf16(
    ctx: &Ctx,
    q: Out<0, bf16>,
    k: Out<1, bf16>,
    // Positional bank, not `Weight<N, T>` (same word, different table).
    q_weight: Bank<0, bf16>,
    k_weight: Bank<1, bf16>,
    positions: Env<keys::Positions>,
    // The layer's (`keys::Theta`), not the fire-wide `keys::RopeTheta`.
    head_dim: Env<keys::HeadDim>,
    theta: Env<keys::Theta>,
    eps: Env<keys::RmsEps>,
) -> Result<(), Refusal> {
    let (num_q_heads, num_kv_heads) =
        (q_heads(q.width, **head_dim)?, k_heads(q.ptr, k.ptr, k.width, **head_dim)?);
    let total_heads = num_q_heads + num_kv_heads;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "rope/rope.cuh",
            "::pie::rope::qk_rmsnorm_rotate<::pie::i32(128)>",
            fused_launch(q.rows, total_heads),
            &[
                q.ptr.arg(),
                k.ptr.arg(),
                q_weight.ptr.arg(),
                k_weight.ptr.arg(),
                positions.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                theta.arg(),
                eps.arg(),
            ],
        )
    }
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
#[kernels_macros::routine]
pub fn qk_rmsnorm_rope_bf16_devwin(
    ctx: &Ctx,
    q: Out<0, bf16>,
    k: Out<1, bf16>,
    q_weight: Bank<0, bf16>,
    k_weight: Bank<1, bf16>,
    positions: Env<keys::Positions>,
    // `[start, count]`; rewritten between trace replays, so the bind is a
    // stable address, not the values.
    win: Env<keys::PeelWindow>,
    // `RowsTotal`, not `keys::Rows`: covers the fire's whole row space.
    n_max: Env<keys::RowsTotal>,
    head_dim: Env<keys::HeadDim>,
    theta: Env<keys::Theta>,
    eps: Env<keys::RmsEps>,
) -> Result<(), Refusal> {
    let (num_q_heads, num_kv_heads) =
        (q_heads(q.width, **head_dim)?, k_heads(q.ptr, k.ptr, k.width, **head_dim)?);
    let total_heads = num_q_heads + num_kv_heads;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "rope/rope.cuh",
            "::pie::rope::qk_rmsnorm_rotate_devwin<::pie::i32(128)>",
            // `n_max`, not `q.rows`: a `_devwin` grid is sized by the
            // fire's row total.
            fused_launch(**n_max, total_heads),
            &[
                q.ptr.arg(),
                k.ptr.arg(),
                q_weight.ptr.arg(),
                k_weight.ptr.arg(),
                positions.arg(),
                win.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                theta.arg(),
                eps.arg(),
            ],
        )
    }
}

/// `rope.cu`'s `rope::qk_rmsnorm_mrope_bf16`.
///
/// # Safety
///
/// [`qk_rmsnorm_rope_bf16_devwin`]'s, without `win`, and `positions` must
/// address `q.rows * 3` live `i32`s rather than `q.rows`.
#[kernels_macros::routine]
pub fn qk_rmsnorm_mrope_bf16(
    ctx: &Ctx,
    q: Out<0, bf16>,
    k: Out<1, bf16>,
    // Head counts stay stated facts; this launcher never divides a width.
    q_weight: Bank<0, bf16>,
    k_weight: Bank<1, bf16>,
    positions: Env<keys::Positions>,
    num_q_heads: Env<keys::NumQHeads>,
    num_kv_heads: Env<keys::NumKvHeads>,
    head_dim: Env<keys::HeadDim>,
    theta: Env<keys::Theta>,
    eps: Env<keys::RmsEps>,
    mrope_section_t: i32,
    mrope_section_h: i32,
    mrope_section_w: i32,
) -> Result<(), Refusal> {
    let total_heads = **num_q_heads + **num_kv_heads;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "rope/rope.cuh",
            "::pie::rope::qk_rmsnorm_rotate_mrope<::pie::i32(128)>",
            fused_launch(q.rows, total_heads),
            &[
                q.ptr.arg(),
                k.ptr.arg(),
                q_weight.ptr.arg(),
                k_weight.ptr.arg(),
                positions.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                theta.arg(),
                eps.arg(),
                mrope_section_t.arg(),
                mrope_section_h.arg(),
                mrope_section_w.arg(),
            ],
        )
    }
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
#[kernels_macros::routine]
pub fn qk_rmsnorm_rope_bf16_rounded(
    ctx: &Ctx,
    q: Out<0, bf16>,
    // A resolved `Out(1)` with no width is this file's "there is no k".
    k: Out<1, bf16>,
    q_weight: Bank<0, bf16>,
    // Optional: the no-K caller supplies `Bank { ptr: Or(core::ptr::null())
    // }` by hand; `num_kv_heads = 0` keeps the kernel from reading it.
    k_weight: Bank<1, bf16>,
    positions: Env<keys::Positions>,
    // The cache's, not `keys::HeadDim`: a rounded layer whose cache differs
    // in head width would norm with one and index with the other.
    head_dim: Env<keys::KvHeadDim>,
    theta: Env<keys::Theta>,
    eps: Env<keys::RmsEps>,
) -> Result<(), Refusal> {
    let (num_q_heads, num_kv_heads) =
        (q_heads(q.width, **head_dim)?, k_heads(q.ptr, k.ptr, k.width, **head_dim)?);
    let total_heads = num_q_heads + num_kv_heads;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "rope/rope.cuh",
            "::pie::rope::qk_rmsnorm_rotate_rounded<::pie::i32(128)>",
            fused_launch(q.rows, total_heads),
            &[
                q.ptr.arg(),
                k.ptr.arg(),
                q_weight.ptr.arg(),
                k_weight.ptr.arg(),
                positions.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                theta.arg(),
                eps.arg(),
            ],
        )
    }
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
#[kernels_macros::routine]
pub fn q_rmsnorm_rope_bf16_rounded(
    ctx: &Ctx,
    q: Out<0, bf16>,
    q_weight: Bank<0, bf16>,
    positions: Env<keys::Positions>,
    // The cache's, per the routine above.
    head_dim: Env<keys::KvHeadDim>,
    theta: Env<keys::Theta>,
    eps: Env<keys::RmsEps>,
) -> Result<(), Refusal> {
    qk_rmsnorm_rope_bf16_rounded(
        ctx,
        q,
        // Zero-width result the statement never declares; `rows: q.rows`
        // since `rows` is the launch's, not this absent operand's.
        Out {
            ptr: core::ptr::null_mut(),
            rows: q.rows,
            width: 0,
        },
        q_weight,
        // The callee's `k_weight` fallback: no bank slot 1 to forward.
        Bank { ptr: core::ptr::null() },
        positions,
        head_dim,
        theta,
        eps,
    )
}

/// `rope.cu`'s `rope::rope_yarn_bf16`.
///
/// # Safety
///
/// [`rope_bf16`]'s.
#[kernels_macros::routine]
pub fn rope_yarn_bf16(
    ctx: &Ctx,
    q: Out<0, bf16>,
    k: Out<1, bf16>,
    positions: Env<keys::Positions>,
    num_q_heads: Env<keys::NumQHeads>,
    num_kv_heads: Env<keys::NumKvHeads>,
    head_dim: Env<keys::HeadDim>,
    // Fire-wide, not gemma-4's per-layer `keys::Theta`.
    theta: Env<keys::RopeTheta>,
    factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_position: i32,
) -> Result<(), Refusal> {
    let half = **head_dim / 2;
    let total_heads = **num_q_heads + **num_kv_heads;
    let per_block = heads_per_block(half);
    #[allow(clippy::cast_precision_loss)]
    let orig_max_pos = original_max_position as f32;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "rope/rope.cuh",
            "::pie::rope::rotate_yarn",
            rotate_launch(q.rows, total_heads, per_block, 0),
            &[
                q.ptr.arg(),
                k.ptr.arg(),
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
            ],
        )
    }
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
#[kernels_macros::routine]
pub fn rope_yarn_original_bf16(
    ctx: &Ctx,
    q: Out<0, bf16>,
    k: Out<1, bf16>,
    positions: Env<keys::Positions>,
    head_dim: Env<keys::HeadDim>,
    // Fire-wide (`RopeTheta`), not gemma-4's per-layer `keys::Theta`.
    theta: Env<keys::RopeTheta>,
    factor: Env<keys::YarnFactor>,
    beta_fast: Env<keys::YarnBetaFast>,
    beta_slow: Env<keys::YarnBetaSlow>,
    attention_factor: Env<keys::YarnAttentionFactor>,
    original_max_position: Env<keys::YarnOriginalMaxPosition>,
    interleaved: Env<keys::RopeInterleaved>,
) -> Result<(), Refusal> {
    // A checkpoint with no YaRN block reaches here as `Yarn::NONE`;
    // unguarded, `ramp_bounds` would compute `(0.0 / 0.0).ln()` and rotate
    // against NaN ramps.
    if **original_max_position <= 0 {
        return Err(Refusal::Unstated { what: "the checkpoint's YaRN block" });
    }
    let (num_q_heads, num_kv_heads) =
        (q_heads(q.width, **head_dim)?, k_heads(q.ptr, k.ptr, k.width, **head_dim)?);
    let (low_dim, high_dim) = ramp_bounds(
        **head_dim,
        **theta,
        **beta_fast,
        **beta_slow,
        **original_max_position,
    );
    let half = **head_dim / 2;
    let pairs = cache_pairs(half);
    let smem = pairs.unsigned_abs() * 8;
    let total_heads = num_q_heads + num_kv_heads;
    let per_block = heads_per_block(half);
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "rope/rope.cuh",
            "::pie::rope::rotate_yarn_original",
            rotate_launch(q.rows, total_heads, per_block, smem),
            &[
                q.ptr.arg(),
                k.ptr.arg(),
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
            ],
        )
    }
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
    ctx: &Ctx,
    instantiation: &'static str,
    q: *mut T,
    k: *mut T,
    // Plain `*const i32`, not `Env`: this private `fn` has no table row.
    // Callers deref `**positions` to forward.
    positions: *const i32,
    // Plain `i32`/`f32`s for the same reason.
    num_tokens: i32,
    q_width: i32,
    k_width: i32,
    head_dim: i32,
    rotary_dim: i32,
    theta: f32,
) -> Result<(), Refusal>
where
    *mut T: Abi,
{
    let (num_q_heads, num_kv_heads) =
        (q_heads(q_width, head_dim)?, k_heads(q, k, k_width, head_dim)?);
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "rope/rope.cuh",
            instantiation,
            Launch::per_row(num_tokens.unsigned_abs(), ROTATE_BLOCK.unsigned_abs()),
            &[
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
            ],
        )
    }
}

/// `rope::rope_partial_bf16` — [`rope_partial`] over bf16.
///
/// # Errors
///
/// [`rope_partial`]'s.
#[kernels_macros::routine]
pub fn rope_partial_bf16(
    ctx: &Ctx,
    q: Out<0, bf16>,
    k: Out<1, bf16>,
    positions: Env<keys::Positions>,
    // `head_dim` is the cache's -- see [`qk_rmsnorm_rope_bf16_rounded`].
    head_dim: Env<keys::KvHeadDim>,
    rotary_dim: Param<0, i32>,
    theta: Env<keys::Theta>,
) -> Result<(), Refusal> {
    rope_partial(
        ctx,
        "::pie::rope::rotate_partial<::pie::bf16>",
        q.ptr,
        k.ptr,
        **positions,
        q.rows,
        q.width,
        k.width,
        // Starred: the callee takes a plain number, not a fact type.
        **head_dim,
        *rotary_dim,
        **theta,
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
#[kernels_macros::routine]
pub fn rope_partial_q_bf16(
    ctx: &Ctx,
    q: Out<0, bf16>,
    positions: Env<keys::Positions>,
    // The cache's, per [`rope_partial_bf16`].
    head_dim: Env<keys::KvHeadDim>,
    rotary_dim: Param<0, i32>,
    theta: Env<keys::Theta>,
) -> Result<(), Refusal> {
    rope_partial(
        ctx,
        "::pie::rope::rotate_partial<::pie::bf16>",
        q.ptr,
        // Q's own address stands in for K; never dereferenced since
        // `num_kv_heads = 0` keeps the kernel off it.
        q.ptr,
        **positions,
        q.rows,
        q.width,
        0,
        **head_dim,
        *rotary_dim,
        **theta,
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
#[kernels_macros::routine]
pub fn rope_partial_f16(
    ctx: &Ctx,
    q: Out<0, f16>,
    k: Out<1, f16>,
    positions: Env<keys::Positions>,
    // The cache's, per the bf16 twin.
    head_dim: Env<keys::KvHeadDim>,
    rotary_dim: Env<keys::RotaryWidth>,
    theta: Env<keys::Theta>,
) -> Result<(), Refusal> {
    rope_partial(
        ctx,
        "::pie::rope::rotate_partial<::pie::f16>",
        q.ptr,
        k.ptr,
        **positions,
        q.rows,
        q.width,
        k.width,
        **head_dim,
        **rotary_dim,
        **theta,
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
#[kernels_macros::routine]
pub fn rope_partial_last_bf16(
    ctx: &Ctx,
    // K is declared here; the Q-alone form is the separate symbol below.
    q: Out<0, bf16>,
    k: Out<1, bf16>,
    positions: Env<keys::Positions>,
    // The cache's, per [`qk_rmsnorm_rope_bf16_rounded`].
    head_dim: Env<keys::KvHeadDim>,
    rotary_dim: Env<keys::RotaryWidth>,
    theta: Env<keys::Theta>,
    interleaved: Env<keys::RopeInterleaved>,
    yarn_factor: Env<keys::YarnFactor>,
    yarn_beta_fast: Env<keys::YarnBetaFast>,
    yarn_beta_slow: Env<keys::YarnBetaSlow>,
    yarn_original_max_position: Env<keys::YarnOriginalMaxPosition>,
) -> Result<(), Refusal> {
    let (num_q_heads, num_kv_heads) = (q_heads(q.width, **head_dim)?, heads(k.width, **head_dim)?);
    let (low_dim, high_dim) = if **yarn_factor > 1.0 && **yarn_original_max_position > 0 {
        ramp_bounds(
            **rotary_dim,
            **theta,
            **yarn_beta_fast,
            **yarn_beta_slow,
            **yarn_original_max_position,
        )
    } else {
        (0.0, 0.0)
    };
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "rope/rope.cuh",
            "::pie::rope::rotate_partial_last",
            Launch::per_row(q.rows.unsigned_abs(), ROTATE_BLOCK.unsigned_abs()),
            &[
                q.ptr.arg(),
                k.ptr.arg(),
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
            ],
        )
    }
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
#[kernels_macros::routine]
pub fn rope_partial_last_q_bf16(
    ctx: &Ctx,
    q: Out<0, bf16>,
    positions: Env<keys::Positions>,
    // The cache's, per [`rope_partial_last_bf16`].
    head_dim: Env<keys::KvHeadDim>,
    rotary_dim: Env<keys::RotaryWidth>,
    theta: Env<keys::Theta>,
    interleaved: Env<keys::RopeInterleaved>,
    yarn_factor: Env<keys::YarnFactor>,
    yarn_beta_fast: Env<keys::YarnBetaFast>,
    yarn_beta_slow: Env<keys::YarnBetaSlow>,
    yarn_original_max_position: Env<keys::YarnOriginalMaxPosition>,
) -> Result<(), Refusal> {
    rope_partial_last_bf16(
        ctx,
        q,
        // Q's own address with a zero width; must be real, not null (see
        // the doc above).
        Out {
            ptr: q.ptr,
            rows: q.rows,
            width: 0,
        },
        positions,
        head_dim,
        rotary_dim,
        theta,
        interleaved,
        yarn_factor,
        yarn_beta_fast,
        yarn_beta_slow,
        yarn_original_max_position,
    )
}

/// This family's routines, and what a trace may say about each.
///
/// Argument lists are derived from the `fn`s above; what's stated here is
/// what no signature carries — whether a statement consumes its whole
/// operand, and which operands must share an address.
pub static ROUTINES: &[Routine] = &[
    routine!(rope_standard_table, ),
    routine!(rope_bf16, in_place = &[(0, 0), (1, 1)], ),
    routine!(rope_write_kv_bf16, whole, ),
    routine!(qk_rmsnorm_rope_bf16, in_place = &[(0, 0), (1, 1)], ),
    routine!(qk_rmsnorm_rope_bf16_devwin, whole, in_place = &[(0, 0), (1, 1)], ),
    routine!(qk_rmsnorm_rope_bf16_rounded, in_place = &[(0, 0), (1, 1)], ),
    // Each is the Q-alone form of the routine above it, so no symbol's
    // operand count decides what it does.
    routine!(q_rmsnorm_rope_bf16_rounded, in_place = &[(0, 0)], ),
    // `in_place` is read off the device `.cuh` text, not guessed from
    // `Out<N,_>`. [`rope_partial_last_bf16`] has none: it has no DSL
    // statement at all.
    routine!(qk_rmsnorm_mrope_bf16, in_place = &[(0, 0), (1, 1)], ),
    routine!(rope_yarn_bf16, in_place = &[(0, 0), (1, 1)], ),
    routine!(rope_yarn_original_bf16, in_place = &[(0, 0), (1, 1)], ),
    routine!(rope_partial_bf16, in_place = &[(0, 0), (1, 1)], ),
    routine!(rope_partial_q_bf16, in_place = &[(0, 0)], ),
    routine!(rope_partial_f16, in_place = &[(0, 0), (1, 1)], ),
    routine!(rope_partial_last_bf16, ),
    routine!(rope_partial_last_q_bf16, in_place = &[(0, 0)], ),
];

/// `rope`, as a trace names it.
pub static FAMILY: Family = crate::family!(ROUTINES);

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
// `rope_write_kv_bf16`'s `k` writes in place through the statement's second
// input, so its source stays `Slot(In, 1)`. `qo_indptr`, `kv_last_page_lens`
// and `row_valid` moved from `Unbound` to `Env<keys::_>` — invisible to
// `arity_problem` either way — so `[0..=2]` below must still be the same
// three operand slots.
const _: () = {
    let d = <rope_write_kv_bf16 as kernels::Derivation>::DERIVED;
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(d[2].source, Some(kernels::Source::Slot(kernels::Kind::In, 2))));
    // The query-side CSR; one index from the KV-side `kv_page_indptr` at
    // `[8]` and easy to swap for it (same type, wrong table).
    assert!(kernels::source_is_named(
        &d[6].source,
        <kernels::keys::QoIndptr as kernels::keys::Fact>::KEY
    ));
    assert!(kernels::source_is_named(
        &d[9].source,
        <kernels::keys::KvLastPageLens as kernels::keys::Fact>::KEY
    ));
    assert!(kernels::source_is_named(
        &d[10].source,
        <kernels::keys::RowValid as kernels::keys::Fact>::KEY
    ));
};

// `rotary_dim` is the statement's where a checkpoint states it, and the
// fire's otherwise (deepseek-v4's `_last_q` form, and `rope_partial_f16`,
// which has no DSL site at all).
const _: () = {
    assert!(matches!(
        <rope_partial_bf16 as kernels::Derivation>::DERIVED[4].source,
        Some(kernels::Source::Slot(kernels::Kind::Param, 0))
    ));
    assert!(matches!(
        <rope_partial_q_bf16 as kernels::Derivation>::DERIVED[3].source,
        Some(kernels::Source::Slot(kernels::Kind::Param, 0))
    ));
    assert!(kernels::source_is_named(
        &<rope_partial_last_q_bf16 as kernels::Derivation>::DERIVED[3].source,
        <kernels::keys::RotaryWidth as kernels::keys::Fact>::KEY
    ));
};

// `interleaved`/`win` were bare `bool`/`Unbound` (`source: None`); now
// `Named`. A silently-bound bool picks the wrong `rotate` instantiation
// with no type error, so each row is pinned by index.
const _: () = {
    let key = <kernels::keys::RopeInterleaved as kernels::keys::Fact>::KEY;
    assert!(kernels::source_is_named(
        &<rope_bf16 as kernels::Derivation>::DERIVED[7].source,
        key
    ));
    assert!(kernels::source_is_named(
        &<rope_yarn_original_bf16 as kernels::Derivation>::DERIVED[10].source,
        key
    ));
    assert!(kernels::source_is_named(
        &<rope_partial_last_bf16 as kernels::Derivation>::DERIVED[6].source,
        key
    ));
    assert!(kernels::source_is_named(
        &<rope_partial_last_q_bf16 as kernels::Derivation>::DERIVED[5].source,
        key
    ));
    assert!(kernels::source_is_named(
        &<qk_rmsnorm_rope_bf16_devwin as kernels::Derivation>::DERIVED[5].source,
        <kernels::keys::PeelWindow as kernels::keys::Fact>::KEY
    ));
};

// The three YaRN lists don't line up (the `_last` pair omits
// `attention_factor`), so a reorder can't be caught by eye. `n_max` is
// `RowsTotal`, not `keys::Rows`: no region can stand in for the total.
const _: () = {
    let (f, bf, bs, af, omp) = (
        <kernels::keys::YarnFactor as kernels::keys::Fact>::KEY,
        <kernels::keys::YarnBetaFast as kernels::keys::Fact>::KEY,
        <kernels::keys::YarnBetaSlow as kernels::keys::Fact>::KEY,
        <kernels::keys::YarnAttentionFactor as kernels::keys::Fact>::KEY,
        <kernels::keys::YarnOriginalMaxPosition as kernels::keys::Fact>::KEY,
    );
    assert!(kernels::source_is_named(
        &<qk_rmsnorm_rope_bf16_devwin as kernels::Derivation>::DERIVED[6].source,
        <kernels::keys::RowsTotal as kernels::keys::Fact>::KEY
    ));
    let d = <rope_partial_last_bf16 as kernels::Derivation>::DERIVED;
    assert!(kernels::source_is_named(&d[7].source, f));
    assert!(kernels::source_is_named(&d[8].source, bf));
    assert!(kernels::source_is_named(&d[9].source, bs));
    assert!(kernels::source_is_named(&d[10].source, omp));
    let q = <rope_partial_last_q_bf16 as kernels::Derivation>::DERIVED;
    assert!(kernels::source_is_named(&q[6].source, f));
    assert!(kernels::source_is_named(&q[7].source, bf));
    assert!(kernels::source_is_named(&q[8].source, bs));
    assert!(kernels::source_is_named(&q[9].source, omp));
    let y = <rope_yarn_original_bf16 as kernels::Derivation>::DERIVED;
    assert!(kernels::source_is_named(&y[5].source, f));
    assert!(kernels::source_is_named(&y[6].source, bf));
    assert!(kernels::source_is_named(&y[7].source, bs));
    assert!(kernels::source_is_named(&y[8].source, af));
    assert!(kernels::source_is_named(&y[9].source, omp));
};

// These four rows have no hand arm left, so this is the only thing that
// catches a slot regressing to `None` (silently falling back to
// `Refusal::Unstated`) under a `cargo check`-only regime.
const _: () = {
    const fn whole(d: &[kernels::Derived]) -> bool {
        let mut i = 0;
        while i < d.len() {
            if d[i].source.is_none() {
                return false;
            }
            i += 1;
        }
        true
    }
    let devwin = <qk_rmsnorm_rope_bf16_devwin as kernels::Derivation>::DERIVED;
    assert!(devwin.len() == 10 && whole(devwin));
    let last = <rope_partial_last_bf16 as kernels::Derivation>::DERIVED;
    assert!(last.len() == 11 && whole(last));
    let last_q = <rope_partial_last_q_bf16 as kernels::Derivation>::DERIVED;
    assert!(last_q.len() == 10 && whole(last_q));
    let orig = <rope_yarn_original_bf16 as kernels::Derivation>::DERIVED;
    assert!(orig.len() == 11 && whole(orig));
};

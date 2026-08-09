#![allow(clippy::too_many_arguments)]

use crate::jit::{Ctx, Family, Launch, Root, Routine};
use crate::routine;
use crate::x::Abi;
use crate::x::abi::{MaybeConst, bf16, f16};
use kernels::Refusal;

use core::ptr::NonNull;

/// `rope/rope.cuh` — the root every routine here compiles a symbol out of.
pub static ROOT: Root =
    Root::new("rope/rope", include_str!("../../csrc/src/rope/rope.cuh"), "rope/rope.cuh");

/// The template-ids NVRTC is handed, spelled as it is handed them.
///
/// Absolute, because a routine body names the instantiation itself rather
/// than a label some other table maps to one. The `<...>` arguments are what
/// used to be a row's `elem`.
mod inst {
    /// `rope.cuh:127` — the cos/sin table.
    pub const STANDARD_TABLE: &str = "::pie_cuda_driver::kernels::rope::device::standard_table\
         <::pie_cuda_driver::kernels::device::i32>";
    /// `rope.cuh:163` — the plain rotation, writing no KV.
    pub const ROTATE: &str = "::pie_cuda_driver::kernels::rope::device::rotate\
         <::pie_cuda_driver::kernels::device::false_type::value, false>";
    /// The same, fused with the paged-KV write, `[n, h, d]` pages.
    pub const ROTATE_WRITE_KV_NHD: &str = "::pie_cuda_driver::kernels::rope::device::rotate\
         <::pie_cuda_driver::kernels::device::true_type::value, false>";
    /// The same, `[h, n, d]` pages.
    pub const ROTATE_WRITE_KV_HND: &str = "::pie_cuda_driver::kernels::rope::device::rotate\
         <::pie_cuda_driver::kernels::device::true_type::value, true>";
    /// `rope.cuh:321` — per-head q/k RMS norms fused with the rotation.
    pub const QK_RMSNORM_ROTATE: &str = "::pie_cuda_driver::kernels::rope::device::qk_rmsnorm_rotate\
         <::pie_cuda_driver::kernels::device::i32(128)>";
    /// `rope.cuh:375` — the same, intermediate rounded to bf16.
    pub const QK_RMSNORM_ROTATE_ROUNDED: &str = "::pie_cuda_driver::kernels::rope::device::qk_rmsnorm_rotate_rounded\
         <::pie_cuda_driver::kernels::device::i32(128)>";
    /// `rope.cuh:442` — MROPE, over `[num_tokens, 3]` positions.
    pub const QK_RMSNORM_ROTATE_MROPE: &str = "::pie_cuda_driver::kernels::rope::device::qk_rmsnorm_rotate_mrope\
         <::pie_cuda_driver::kernels::device::i32(128)>";
    /// `rope.cuh:530` — the same over a device-resident window.
    pub const QK_RMSNORM_ROTATE_DEVWIN: &str = "::pie_cuda_driver::kernels::rope::device::qk_rmsnorm_rotate_devwin\
         <::pie_cuda_driver::kernels::device::i32(128)>";
    /// `rope.cuh:610` — llama-3-style YaRN.
    pub const ROTATE_YARN: &str = "::pie_cuda_driver::kernels::rope::device::rotate_yarn";
    /// `rope.cuh:656` — YaRN as its paper spells it.
    pub const ROTATE_YARN_ORIGINAL: &str =
        "::pie_cuda_driver::kernels::rope::device::rotate_yarn_original";
    /// `rope.cuh:733` — partial rotary over the FIRST `rotary_dim` lanes.
    pub const ROTATE_PARTIAL_BF16: &str = "::pie_cuda_driver::kernels::rope::device::rotate_partial\
         <::pie_cuda_driver::kernels::device::bf16>";
    /// The same, over `f16`.
    pub const ROTATE_PARTIAL_F16: &str = "::pie_cuda_driver::kernels::rope::device::rotate_partial\
         <::pie_cuda_driver::kernels::device::f16>";
    /// `rope.cuh:792` — partial rotary over the LAST `rotary_dim` lanes.
    pub const ROTATE_PARTIAL_LAST: &str =
        "::pie_cuda_driver::kernels::rope::device::rotate_partial_last";
}

/// `rope.cu:82,119,236,276,314,337,382` — `constexpr int BLOCK = 256;`
pub const ROTATE_BLOCK: i32 = 256;

/// `rope.cu:45,66,162,189,213` — `constexpr int BLOCK = 128;`
pub const FUSED_BLOCK: u32 = 128;

/// `rope.cu:84,120,282` — `constexpr int kMaxCachedPairs = 4096;`
pub const MAX_CACHED_PAIRS: i32 = 4096;

/// `rope.cu:92,127,240,287` — `half >= BLOCK ? 1 : (BLOCK / half)`.
#[must_use]
pub const fn heads_per_block(half: i32) -> i32 {
    if half >= ROTATE_BLOCK { 1 } else { ROTATE_BLOCK / half }
}

/// `rope.cu:87,123,285` — `half <= kMaxCachedPairs ? half : 0`.
#[must_use]
pub const fn cache_pairs(half: i32) -> i32 {
    if half <= MAX_CACHED_PAIRS { half } else { 0 }
}

/// `rope.cu:93,128,241,288` — the two-axis grid the head split produces.
#[must_use]
const fn rotate_grid(num_tokens: i32, total_heads: i32, per_block: i32) -> [u32; 3] {
    [
        num_tokens.unsigned_abs(),
        (total_heads + per_block - 1).unsigned_abs() / per_block.unsigned_abs(),
        1,
    ]
}

/// `rope.cu:189-191`, `:45-47`, `:162-164`, `:213-215` — the fused grid.
#[must_use]
const fn fused_launch(rows: i32, total_heads: i32) -> Launch {
    Launch::grid([rows.unsigned_abs(), total_heads.unsigned_abs(), 1], [FUSED_BLOCK, 1, 1])
}

/// The rotation grid, at `smem` bytes of cached cos/sin pairs.
#[must_use]
const fn rotate_launch(num_tokens: i32, total_heads: i32, per_block: i32, smem: u32) -> Launch {
    Launch::grid(
        rotate_grid(num_tokens, total_heads, per_block),
        [ROTATE_BLOCK.unsigned_abs(), 1, 1],
    )
    .smem(smem)
}

/// `rope_device.cuh:112` — `yarn_original_ramp_bounds`, on the host.
#[must_use]
pub fn ramp_bounds(
    span: i32,
    theta: f32,
    beta_fast: f32,
    beta_slow: f32,
    original_max_position: i32,
) -> (f32, f32) {
    const TWO_PI: f32 = 6.283_185_307_179_586_5_f32;
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

/// `rope::rope_standard_table` — the cos/sin table `attn`'s fused prepare
///
/// # Safety
///
/// `positions` must address `num_tokens` live `i32`s and `table`
/// `num_tokens * head_dim` live floats; `stream` must be live across the
/// launch.
pub fn rope_standard_table(
    ctx: &Ctx,
    positions: *const i32,
    table: *mut f32,
    num_tokens: i32,
    head_dim: i32,
    theta: f32,
) -> Result<(), Refusal> {
    if num_tokens <= 0 {
        return Err(Refusal::Empty { what: "num_tokens" });
    }
    if head_dim / 2 <= 0 {
        return Err(Refusal::Empty { what: "head_dim / 2" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &ROOT,
            inst::STANDARD_TABLE,
            Launch::per_row(num_tokens.unsigned_abs(), ROTATE_BLOCK.unsigned_abs()),
            &[positions.arg(), table.arg(), head_dim.arg(), theta.arg()],
        )
    }
}

/// `rope.cu:71` — `rope::rope_bf16`.
///
/// # Safety
///
/// `q` and `k` must address `num_tokens * num_q_heads * head_dim` and
/// `num_tokens * num_kv_heads * head_dim` live bf16 elements, `positions`
/// `num_tokens` live `i32`s, and `stream` must be live across the launch.
pub fn rope_bf16(
    ctx: &Ctx,
    q: *mut bf16,
    k: *mut bf16,
    positions: *const i32,
    num_tokens: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    interleaved: bool,
) -> Result<(), Refusal> {
    let half = head_dim / 2;
    if half <= 0 {
        return Err(Refusal::Empty { what: "head_dim / 2" });
    }
    if num_tokens <= 0 {
        return Err(Refusal::Empty { what: "num_tokens" });
    }
    let pairs = cache_pairs(half);
    let smem = pairs.unsigned_abs() * 2 * 4;
    let total_heads = num_q_heads + num_kv_heads;
    let per_block = heads_per_block(half);
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &ROOT,
            inst::ROTATE,
            rotate_launch(num_tokens, total_heads, per_block, smem),
            &[
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
            ],
        )
    }
}

/// `rope.cu:105` — `rope::rope_write_kv_bf16`.
///
/// # Safety
///
/// Every pointer must address live device memory of the extent the paged-KV
/// descriptors describe, and `stream` must be live across the launch.
pub fn rope_write_kv_bf16(
    ctx: &Ctx,
    q: *mut bf16,
    k: *mut bf16,
    v: *const bf16,
    positions: *const i32,
    k_pages: *mut bf16,
    v_pages: *mut bf16,
    qo_indptr: *const u32,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    row_valid: *const u8,
    num_tokens: i32,
    num_requests: i32,
    page_size: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    hnd_layout: bool,
    interleaved: bool,
) -> Result<(), Refusal> {
    let half = head_dim / 2;
    if half <= 0 {
        return Err(Refusal::Empty { what: "head_dim / 2" });
    }
    if num_tokens <= 0 {
        return Err(Refusal::Empty { what: "num_tokens" });
    }
    let pairs = cache_pairs(half);
    let smem = pairs.unsigned_abs() * 2 * 4;
    let total_heads = num_q_heads + num_kv_heads;
    let per_block = heads_per_block(half);
    let instantiation =
        if hnd_layout { inst::ROTATE_WRITE_KV_HND } else { inst::ROTATE_WRITE_KV_NHD };
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &ROOT,
            instantiation,
            rotate_launch(num_tokens, total_heads, per_block, smem),
            &[
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
                MaybeConst::new(v).arg(),
                NonNull::new(k_pages).arg(),
                NonNull::new(v_pages).arg(),
                MaybeConst::new(qo_indptr).arg(),
                MaybeConst::new(kv_page_indices).arg(),
                MaybeConst::new(kv_page_indptr).arg(),
                MaybeConst::new(kv_last_page_lens).arg(),
                MaybeConst::new(row_valid).arg(),
                num_requests.arg(),
                page_size.arg(),
            ],
        )
    }
}

/// `rope/rope.cu:189-191` — `rope::qk_rmsnorm_rope_bf16`.
///
/// # Safety
///
/// [`rope_bf16`]'s, plus `q_weight`/`k_weight` addressing `head_dim` live
/// bf16 elements each.
pub fn qk_rmsnorm_rope_bf16(
    ctx: &Ctx,
    q: *mut bf16,
    k: *mut bf16,
    q_weight: *const bf16,
    k_weight: *const bf16,
    positions: *const i32,
    num_tokens: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    eps: f32,
) -> Result<(), Refusal> {
    let total_heads = num_q_heads + num_kv_heads;
    if num_tokens <= 0 {
        return Err(Refusal::Empty { what: "num_tokens" });
    }
    if total_heads <= 0 {
        return Err(Refusal::Empty { what: "num_q_heads + num_kv_heads" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &ROOT,
            inst::QK_RMSNORM_ROTATE,
            fused_launch(num_tokens, total_heads),
            &[
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
            ],
        )
    }
}

/// `rope.cu:148` — `rope::qk_rmsnorm_rope_bf16_devwin`.
///
/// # Safety
///
/// `win` must address two live `u32`s on the device; the rest is
/// [`rope_bf16`]'s obligation with `n_max` for `num_tokens`.
pub fn qk_rmsnorm_rope_bf16_devwin(
    ctx: &Ctx,
    q: *mut bf16,
    k: *mut bf16,
    q_weight: *const bf16,
    k_weight: *const bf16,
    positions: *const i32,
    win: *const u32,
    n_max: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    eps: f32,
) -> Result<(), Refusal> {
    let total_heads = num_q_heads + num_kv_heads;
    if n_max <= 0 {
        return Err(Refusal::Empty { what: "n_max" });
    }
    if total_heads <= 0 {
        return Err(Refusal::Empty { what: "num_q_heads + num_kv_heads" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &ROOT,
            inst::QK_RMSNORM_ROTATE_DEVWIN,
            fused_launch(n_max, total_heads),
            &[
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
            ],
        )
    }
}

/// `rope.cu:29` — `rope::qk_rmsnorm_mrope_bf16`.
///
/// # Safety
///
/// [`qk_rmsnorm_rope_bf16_devwin`]'s, without `win`, and `positions` must
/// address `num_tokens * 3` live `i32`s rather than `num_tokens`.
pub fn qk_rmsnorm_mrope_bf16(
    ctx: &Ctx,
    q: *mut bf16,
    k: *mut bf16,
    q_weight: *const bf16,
    k_weight: *const bf16,
    positions: *const i32,
    num_tokens: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    eps: f32,
    mrope_section_t: i32,
    mrope_section_h: i32,
    mrope_section_w: i32,
) -> Result<(), Refusal> {
    let total_heads = num_q_heads + num_kv_heads;
    if num_tokens <= 0 {
        return Err(Refusal::Empty { what: "num_tokens" });
    }
    if total_heads <= 0 {
        return Err(Refusal::Empty { what: "num_q_heads + num_kv_heads" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &ROOT,
            inst::QK_RMSNORM_ROTATE_MROPE,
            fused_launch(num_tokens, total_heads),
            &[
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
            ],
        )
    }
}

/// `rope.cu:200` — `rope::qk_rmsnorm_rope_bf16_rounded`.
///
/// # Safety
///
/// [`qk_rmsnorm_mrope_bf16`]'s. `k` and `k_weight` may be null together, and
/// the kernel reads the pair as "there is no k".
pub fn qk_rmsnorm_rope_bf16_rounded(
    ctx: &Ctx,
    q: *mut bf16,
    k: *mut bf16,
    q_weight: *const bf16,
    k_weight: *const bf16,
    positions: *const i32,
    num_tokens: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    eps: f32,
) -> Result<(), Refusal> {
    let total_heads = num_q_heads + num_kv_heads;
    if num_tokens <= 0 {
        return Err(Refusal::Empty { what: "num_tokens" });
    }
    if total_heads <= 0 {
        return Err(Refusal::Empty { what: "num_q_heads + num_kv_heads" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &ROOT,
            inst::QK_RMSNORM_ROTATE_ROUNDED,
            fused_launch(num_tokens, total_heads),
            &[
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
            ],
        )
    }
}

/// `rope.cu:226` — `rope::rope_yarn_bf16`.
///
/// # Safety
///
/// [`rope_bf16`]'s.
pub fn rope_yarn_bf16(
    ctx: &Ctx,
    q: *mut bf16,
    k: *mut bf16,
    positions: *const i32,
    num_tokens: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_position: i32,
) -> Result<(), Refusal> {
    let half = head_dim / 2;
    if half <= 0 {
        return Err(Refusal::Empty { what: "head_dim / 2" });
    }
    if num_tokens <= 0 {
        return Err(Refusal::Empty { what: "num_tokens" });
    }
    let total_heads = num_q_heads + num_kv_heads;
    let per_block = heads_per_block(half);
    #[allow(clippy::cast_precision_loss)]
    let orig_max_pos = original_max_position as f32;
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &ROOT,
            inst::ROTATE_YARN,
            rotate_launch(num_tokens, total_heads, per_block, 0),
            &[
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
            ],
        )
    }
}

/// `rope.cu:255` — `rope::rope_yarn_original_bf16` (OLMo-3, gpt-oss).
///
/// # Safety
///
/// [`rope_bf16`]'s.
pub fn rope_yarn_original_bf16(
    ctx: &Ctx,
    q: *mut bf16,
    k: *mut bf16,
    positions: *const i32,
    num_tokens: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    attention_factor: f32,
    original_max_position: i32,
    interleaved: bool,
) -> Result<(), Refusal> {
    let (low_dim, high_dim) =
        ramp_bounds(head_dim, theta, beta_fast, beta_slow, original_max_position);
    let half = head_dim / 2;
    if half <= 0 {
        return Err(Refusal::Empty { what: "head_dim / 2" });
    }
    if num_tokens <= 0 {
        return Err(Refusal::Empty { what: "num_tokens" });
    }
    let pairs = cache_pairs(half);
    let smem = pairs.unsigned_abs() * 8;
    let total_heads = num_q_heads + num_kv_heads;
    let per_block = heads_per_block(half);
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &ROOT,
            inst::ROTATE_YARN_ORIGINAL,
            rotate_launch(num_tokens, total_heads, per_block, smem),
            &[
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
            ],
        )
    }
}

/// `rope::rope_partial_bf16` — partial rotary over the first `rotary_dim`
///
/// # Safety
///
/// [`rope_bf16`]'s.
fn rope_partial<T>(
    ctx: &Ctx,
    instantiation: &'static str,
    q: *mut T,
    k: *mut T,
    positions: *const i32,
    position_delta: i32,
    num_tokens: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    rotary_dim: i32,
    theta: f32,
) -> Result<(), Refusal>
where
    *mut T: Abi,
{
    if num_tokens <= 0 {
        return Err(Refusal::Empty { what: "num_tokens" });
    }
    if rotary_dim <= 0 {
        return Err(Refusal::Empty { what: "rotary_dim" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &ROOT,
            instantiation,
            Launch::per_row(num_tokens.unsigned_abs(), ROTATE_BLOCK.unsigned_abs()),
            &[
                q.arg(),
                k.arg(),
                positions.arg(),
                position_delta.arg(),
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
/// A routine is a concrete `fn`: the generic above serves both element types
/// and neither the table nor `call()` can name a generic.
pub fn rope_partial_bf16(
    ctx: &Ctx,
    q: *mut bf16,
    k: *mut bf16,
    positions: *const i32,
    position_delta: i32,
    num_tokens: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    rotary_dim: i32,
    theta: f32,
) -> Result<(), Refusal> {
    rope_partial(
        ctx,
        inst::ROTATE_PARTIAL_BF16,
        q,
        k,
        positions,
        position_delta,
        num_tokens,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        theta,
    )
}

/// `rope::rope_partial_f16` — [`rope_partial`] over f16.
pub fn rope_partial_f16(
    ctx: &Ctx,
    q: *mut f16,
    k: *mut f16,
    positions: *const i32,
    position_delta: i32,
    num_tokens: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    rotary_dim: i32,
    theta: f32,
) -> Result<(), Refusal> {
    rope_partial(
        ctx,
        inst::ROTATE_PARTIAL_F16,
        q,
        k,
        positions,
        position_delta,
        num_tokens,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        theta,
    )
}

/// `rope.cu:348` — `rope::rope_partial_last_bf16` (deepseek-v4).
///
/// # Safety
///
/// [`rope_bf16`]'s.
pub fn rope_partial_last_bf16(
    ctx: &Ctx,
    q: *mut bf16,
    k: *mut bf16,
    positions: *const i32,
    num_tokens: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    rotary_dim: i32,
    theta: f32,
    inverse: bool,
    interleaved: bool,
    yarn_factor: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    yarn_original_max_position: i32,
) -> Result<(), Refusal> {
    let (low_dim, high_dim) = if yarn_factor > 1.0 && yarn_original_max_position > 0 {
        ramp_bounds(rotary_dim, theta, yarn_beta_fast, yarn_beta_slow, yarn_original_max_position)
    } else {
        (0.0, 0.0)
    };
    if num_tokens <= 0 {
        return Err(Refusal::Empty { what: "num_tokens" });
    }
    if rotary_dim <= 0 {
        return Err(Refusal::Empty { what: "rotary_dim" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &ROOT,
            inst::ROTATE_PARTIAL_LAST,
            Launch::per_row(num_tokens.unsigned_abs(), ROTATE_BLOCK.unsigned_abs()),
            &[
                q.arg(),
                k.arg(),
                positions.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                rotary_dim.arg(),
                theta.arg(),
                inverse.arg(),
                interleaved.arg(),
                yarn_factor.arg(),
                low_dim.arg(),
                high_dim.arg(),
            ],
        )
    }
}

/// This family's routines, and what a trace may say about each.
///
/// The argument lists are DERIVED from the `fn`s above -- `routine!` sees only
/// the identifier. What is stated here is what no signature carries: whether a
/// statement consumes its whole operand, and which operands must be given the
/// same address.
pub static ROUTINES: &[Routine] = &[
    routine!(rope_standard_table),
    routine!(rope_bf16, in_place = &[(0, 0), (1, 1)]),
    routine!(rope_write_kv_bf16, whole),
    routine!(qk_rmsnorm_rope_bf16, in_place = &[(0, 0), (1, 1)]),
    routine!(qk_rmsnorm_rope_bf16_devwin, whole, in_place = &[(0, 0), (1, 1)]),
    routine!(qk_rmsnorm_rope_bf16_rounded, in_place = &[(0, 0), (1, 1)]),
    routine!(qk_rmsnorm_mrope_bf16),
    routine!(rope_yarn_bf16),
    routine!(rope_yarn_original_bf16, in_place = &[(0, 0), (1, 1)]),
    routine!(rope_partial_bf16, in_place = &[(0, 0), (1, 1)]),
    routine!(rope_partial_f16, in_place = &[(0, 0), (1, 1)]),
    routine!(rope_partial_last_bf16),
];

/// `rope`, as a trace names it.
pub static FAMILY: Family = Family { namespace: "rope", routines: ROUTINES };

#![allow(clippy::too_many_arguments)]

use crate::jit::{Ctx, Family, Launch, Routine};
use crate::routine;
use crate::x::Abi;
use crate::x::abi::{bf16, f16};
use kernels::Refusal;

use core::ffi::c_void;
use core::ptr::NonNull;

/// `norm/rmsnorm.cuh` — twelve `__global__` templates, fifteen
pub mod rmsnorm {

    use crate::jit::Root;

    /// `norm/rmsnorm.cuh` — the root these routines compile a symbol out of.
    pub static ROOT: Root = Root::new(
        "norm/rmsnorm",
        include_str!("../../csrc/src/norm/rmsnorm.cuh"),
        "norm/rmsnorm.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// Absolute, because a routine body names the instantiation itself rather
    /// than a label some other table maps to one. The `<...>` arguments are
    /// what used to be a row's `elem`, and for the four vectorised rows the
    /// FIRST of them is the block width — `i32(256)` and `i32(512)` are
    /// compile-time template arguments, not extents.
    pub mod inst {
        /// `rmsnorm.cuh:220` — the scalar norm. ALSO the strided form: the two
        /// differ only in whether the host passes `hidden` for both strides,
        /// so they are one instantiation under two trace symbols.
        pub const RMSNORM: &str = "::pie_cuda_driver::kernels::norm::device::rmsnorm\
             <::pie_cuda_driver::kernels::device::bf16, 256>";
        /// `rmsnorm.cuh:236` — the same with `(1 + w)` folded instead of `w`.
        pub const RMSNORM_GEMMA: &str = "::pie_cuda_driver::kernels::norm::device::rmsnorm_gemma\
             <::pie_cuda_driver::kernels::device::bf16, 256>";
        /// `rmsnorm.cuh:262` — eight bf16 per thread, 256-wide, no fp16 copy.
        pub const RMSNORM_VEC8_256: &str = "::pie_cuda_driver::kernels::norm::device::rmsnorm_vec8\
             <::pie_cuda_driver::kernels::device::i32(256), false, false>";
        /// The same, ALSO writing the fp16 copy.
        pub const RMSNORM_VEC8_256_FP16: &str = "::pie_cuda_driver::kernels::norm::device::rmsnorm_vec8\
             <::pie_cuda_driver::kernels::device::i32(256), false, true>";
        /// `rmsnorm.cuh:262` at a 512-thread block — the arm the vectorised
        /// norm actually fires.
        pub const RMSNORM_VEC8_512: &str = "::pie_cuda_driver::kernels::norm::device::rmsnorm_vec8\
             <::pie_cuda_driver::kernels::device::i32(512), false, false>";
        /// The same, ALSO writing the fp16 copy.
        pub const RMSNORM_VEC8_512_FP16: &str = "::pie_cuda_driver::kernels::norm::device::rmsnorm_vec8\
             <::pie_cuda_driver::kernels::device::i32(512), false, true>";
        /// `rmsnorm.cuh:401` — the residual add and the NEXT block's pre-norm.
        pub const RESIDUAL_ADD_RMSNORM: &str = "::pie_cuda_driver::kernels::norm::device::residual_add_rmsnorm\
             <::pie_cuda_driver::kernels::device::bf16, 256>";
        /// `rmsnorm.cuh:492` — norm `x`, then add the result into the stream.
        pub const RMSNORM_RESIDUAL_ADD: &str = "::pie_cuda_driver::kernels::norm::device::rmsnorm_residual_add\
             <::pie_cuda_driver::kernels::device::bf16, 256>";
        /// `rmsnorm.cuh:548` — gemma-4's landing, vectorised, 512-wide.
        pub const RASR_VEC8_512: &str = "::pie_cuda_driver::kernels::norm::device::rmsnorm_rasr_vec8\
             <::pie_cuda_driver::kernels::device::i32(512)>";
        /// The same, 256-wide, for a row below [`super::super::RASR_VEC512_ABOVE`].
        pub const RASR_VEC8_256: &str = "::pie_cuda_driver::kernels::norm::device::rmsnorm_rasr_vec8\
             <::pie_cuda_driver::kernels::device::i32(256)>";
        /// `rmsnorm.cuh:631` — the same three passes, scalar.
        pub const RASR_SCALAR_512: &str = "::pie_cuda_driver::kernels::norm::device::rmsnorm_residual_add_scale_rmsnorm\
             <::pie_cuda_driver::kernels::device::bf16, 512>";
        /// `rmsnorm.cuh:686` — the weightless per-head norm, the V-norm.
        pub const RMSNORM_NO_SCALE: &str = "::pie_cuda_driver::kernels::norm::device::rmsnorm_no_scale\
             <::pie_cuda_driver::kernels::device::bf16, 256>";
        /// `rmsnorm.cuh:718` — norm gated by a second activation.
        pub const RMSNORM_GATED: &str = "::pie_cuda_driver::kernels::norm::device::rmsnorm_gated\
             <::pie_cuda_driver::kernels::device::bf16, 256>";
        /// `rmsnorm.cuh:763` — the same with an fp32 INPUT as well.
        pub const RMSNORM_GATED_F32_IN: &str = "::pie_cuda_driver::kernels::norm::device::rmsnorm_gated_f32_in\
             <::pie_cuda_driver::kernels::device::bf16, 256>";
    }
}

/// `norm/add_bias.cuh` — one `__device__` row body, two `__global__`s over
pub mod add_bias {

    use crate::jit::Root;

    /// `norm/add_bias.cuh` — the root this routine compiles its symbol out of.
    pub static ROOT: Root = Root::new(
        "norm/add_bias",
        include_str!("../../csrc/src/norm/add_bias.cuh"),
        "norm/add_bias.cuh",
    );

    /// The template-id NVRTC is handed, spelled as it is handed it.
    pub mod inst {
        /// `add_bias.cuh:82` — `out[row][i] += bias[i]`, contiguous rows.
        pub const ADD_BIAS: &str = "::pie_cuda_driver::kernels::norm::device::add_bias\
             <::pie_cuda_driver::kernels::device::bf16>";
    }
}

/// `norm/altup.cuh` — gemma-3n's rank-K residual, predict and correct.
pub mod altup {

    use crate::jit::Root;

    /// `norm/altup.cuh` — the root these routines compile a symbol out of.
    pub static ROOT: Root =
        Root::new("norm/altup", include_str!("../../csrc/src/norm/altup.cuh"), "norm/altup.cuh");

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    pub mod inst {
        /// `altup.cuh:77` — predict each of `K` streams as a coefficient
        /// combination of the others.
        pub const ALTUP_PREDICT: &str = "::pie_cuda_driver::kernels::norm::device::altup_predict\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `altup.cuh:106` — correct every stream from the one the layer ran.
        pub const ALTUP_CORRECT: &str = "::pie_cuda_driver::kernels::norm::device::altup_correct\
             <::pie_cuda_driver::kernels::device::bf16>";
    }
}

/// `norm/altup_aux.cuh` — the five auxiliaries AltUp needs around the two
pub mod altup_aux {

    use crate::jit::Root;

    /// `norm/altup_aux.cuh` — the root these routines compile a symbol out of.
    pub static ROOT: Root = Root::new(
        "norm/altup_aux",
        include_str!("../../csrc/src/norm/altup_aux.cuh"),
        "norm/altup_aux.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    pub mod inst {
        /// `altup_aux.cuh:91` — the RMS of each row, to fp32.
        pub const COMPUTE_RMS: &str = "::pie_cuda_driver::kernels::norm::device::compute_rms\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `altup_aux.cuh:111` — rescale each row to a target RMS, in place.
        pub const MAGNITUDE_RESCALE: &str = "::pie_cuda_driver::kernels::norm::device::magnitude_rescale\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `altup_aux.cuh:141` — the mean of `K` streams at each position.
        pub const MEAN_STREAMS: &str = "::pie_cuda_driver::kernels::norm::device::mean_streams\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `altup_aux.cuh:163` — unpack and TRANSPOSE a `[T, K, K]` bf16 block.
        pub const UNPACK_PREDICT_COEFS: &str = "::pie_cuda_driver::kernels::norm::device::unpack_predict_coefs\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `altup_aux.cuh:178` — unpack a `[T, K]` bf16 block to fp32.
        pub const UNPACK_CORRECT_COEFS: &str = "::pie_cuda_driver::kernels::norm::device::unpack_correct_coefs\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `altup_aux.cuh:189` — elementwise `tanh`, in place, over bf16.
        pub const TANH_BF16: &str = "::pie_cuda_driver::kernels::norm::device::tanh_inplace\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// The same, over f16.
        pub const TANH_F16: &str = "::pie_cuda_driver::kernels::norm::device::tanh_inplace\
             <::pie_cuda_driver::kernels::device::f16>";
    }
}

/// `norm/elementwise.cuh` — the residual add and the scalar multiply.
pub mod elementwise {

    use crate::jit::Root;

    /// `norm/elementwise.cuh` — the root these routines compile a symbol out
    /// of.
    pub static ROOT: Root = Root::new(
        "norm/elementwise",
        include_str!("../../csrc/src/norm/elementwise.cuh"),
        "norm/elementwise.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    pub mod inst {
        /// `elementwise.cuh:56` — `y += x`, accumulated in fp32, over bf16.
        pub const RESIDUAL_ADD_BF16: &str = "::pie_cuda_driver::kernels::norm::device::residual_add\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// The same, over f16.
        pub const RESIDUAL_ADD_F16: &str = "::pie_cuda_driver::kernels::norm::device::residual_add\
             <::pie_cuda_driver::kernels::device::f16>";
        /// `elementwise.cuh:74` — `x *= s`, with `s` ROUNDED TO `T` FIRST.
        pub const SCALAR_MUL_BF16: &str = "::pie_cuda_driver::kernels::norm::device::scalar_mul\
             <::pie_cuda_driver::kernels::device::bf16>";
    }
}

/// `norm/dsv4_hc.cuh` — deepseek-v4's hyper-connections, the attention sink
pub mod dsv4_hc {

    use crate::jit::Root;

    /// `norm/dsv4_hc.cuh` — the root these routines compile a symbol out of.
    pub static ROOT: Root = Root::new(
        "norm/dsv4_hc",
        include_str!("../../csrc/src/norm/dsv4_hc.cuh"),
        "norm/dsv4_hc.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    pub mod inst {
        /// `dsv4_hc.cuh:103` — split the mix matrix and Sinkhorn-normalise it.
        pub const HC_PRE_POSTPROCESS: &str = "::pie_cuda_driver::kernels::norm::device::hc_pre_postprocess\
             <::pie_cuda_driver::kernels::device::bf16, 256>";
        /// `dsv4_hc.cuh:239` — scatter the layer's output back across the `M`
        /// streams.
        pub const HC_POST: &str = "::pie_cuda_driver::kernels::norm::device::hc_post\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `dsv4_hc.cuh:285` — the same collapse without the Sinkhorn.
        pub const HC_HEAD_POSTPROCESS: &str = "::pie_cuda_driver::kernels::norm::device::hc_head_postprocess\
             <::pie_cuda_driver::kernels::device::bf16, 256>";
        /// `dsv4_hc.cuh:327` — the degenerate mixer: broadcast one stream.
        pub const HC_EXPAND: &str = "::pie_cuda_driver::kernels::norm::device::hc_expand\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `dsv4_hc.cuh:358` — RMS-normalise `[N, dim]` bf16 into fp32.
        pub const HC_RMSNORM_TO_F32: &str = "::pie_cuda_driver::kernels::norm::device::hc_rmsnorm_to_f32\
             <::pie_cuda_driver::kernels::device::bf16, 256>";
        /// `dsv4_hc.cuh:406` — fold an attention sink logit into the output.
        pub const ATTN_SINK_CORRECTION: &str = "::pie_cuda_driver::kernels::norm::device::attn_sink_correction\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `dsv4_hc.cuh:431` — RMS-normalise each attention head in place.
        pub const PER_HEAD_RMSNORM: &str = "::pie_cuda_driver::kernels::norm::device::per_head_rmsnorm\
             <::pie_cuda_driver::kernels::device::bf16>";
    }
}

/// `rmsnorm.cu:85`, `dsv4_hc.cu:18`, `elementwise.cuh:12`, `add_bias.cuh:12` —
const BLOCK: u32 = 256;

/// `rmsnorm.cu:88` — `constexpr int VBLOCK = 512;`
const VBLOCK: u32 = 512;

/// `runtime/launch.rs:584` — the warp width, for the two kernels that share
const WARP: u32 = 32;

/// `runtime/launch.rs:581` — the largest block CUDA will launch.
const MAX_BLOCK: u32 = 1024;

/// `runtime/launch.rs:743` — one float per warp of a [`BLOCK`]-wide block.
const RMS_SMEM: u32 = (BLOCK / WARP) * 4;

/// `runtime/launch.rs:727` — `altup.cu:18-19` and `:32-33`'s block width.
const ALTUP_BLOCK: u32 = 128;

/// AltUp's epsilon, which is the ALGORITHM's and not the model's.
pub const ALTUP_EPS: f32 = 1e-5;

/// The width above which the vectorised fused norm prefers a 512-thread
pub const RASR_VEC512_ABOVE: i32 = 2560;

/// `dsv4_hc.cuh:91` — `constexpr int MAX_HC_MULT = 8;`
pub const MAX_HC_MULT: i32 = 8;

/// One block per row, [`BLOCK`] wide, **nothing shared**.
#[must_use]
const fn per_row(rows: i32) -> Launch {
    Launch::per_row(rows.unsigned_abs(), BLOCK)
}

/// [`per_row`] with the warp-reduction scratch the two `altup_aux` kernels
#[must_use]
const fn per_row_reducing(rows: i32) -> Launch {
    Launch::per_row(rows.unsigned_abs(), BLOCK).smem(RMS_SMEM)
}

/// Flat pointwise over `n` elements, [`BLOCK`] per block, rounded up.
#[must_use]
const fn elementwise(n: i32) -> Launch {
    Launch::flat(n.unsigned_abs(), BLOCK)
}

/// Flat pointwise over `n` elements given as a 64-bit count, saturating.
#[must_use]
fn elementwise_wide(n: i64) -> Launch {
    let blocks = (n + i64::from(BLOCK) - 1) / i64::from(BLOCK);
    Launch::grid([u32::try_from(blocks).unwrap_or(u32::MAX), 1, 1], [BLOCK, 1, 1])
}

/// Pointwise with the row on its own grid axis.
#[must_use]
const fn elementwise_rows(rows: i32, width: i32) -> Launch {
    Launch::grid([rows.unsigned_abs(), width.unsigned_abs().div_ceil(BLOCK), 1], [BLOCK, 1, 1])
}

/// One block per row, as wide as the row rounded up to a warp and capped at
#[must_use]
const fn route_rows(rows: i32, width: i32) -> Launch {
    let warps = width.unsigned_abs().div_ceil(WARP);
    let warps = if warps == 0 { 1 } else { warps };
    let block = warps.saturating_mul(WARP);
    let block = if block > MAX_BLOCK { MAX_BLOCK } else { block };
    Launch::grid([rows.unsigned_abs(), 1, 1], [block, 1, 1])
}

/// One block per (row, head), [`BLOCK`] wide.
#[must_use]
const fn gated_rms(rows: i32, heads: i32) -> Launch {
    Launch::grid([rows.unsigned_abs(), heads.unsigned_abs(), 1], [BLOCK, 1, 1])
}

/// `dim3(T, K, ceil(H / 128))` at [`ALTUP_BLOCK`] threads.
#[must_use]
const fn altup_streams(rows: i32, streams: i32, hidden: i32) -> Launch {
    Launch::grid(
        [rows.unsigned_abs(), streams.unsigned_abs(), hidden.unsigned_abs().div_ceil(ALTUP_BLOCK)],
        [ALTUP_BLOCK, 1, 1],
    )
}

/// One block per row PER HEAD, [`BLOCK`] wide, nothing shared.
fn rows_per_head(rows: i32, width: i32, stated_head_dim: i32) -> Result<Launch, Refusal> {
    if stated_head_dim == 0 {
        return Ok(per_row(rows));
    }
    let (w, hd) = (width.unsigned_abs(), stated_head_dim.unsigned_abs());
    if w == 0 || !w.is_multiple_of(hd) {
        return Err(Refusal::Narrow {
            what: "a row that divides by head_dim",
            at: i64::from(width),
        });
    }
    let blocks = rows
        .unsigned_abs()
        .checked_mul(w / hd)
        .ok_or(Refusal::Narrow { what: "a row count that fits a grid", at: i64::from(rows) })?;
    Ok(Launch::per_row(blocks, BLOCK))
}

/// `((uintptr_t)p & 15u) == 0`, which is what `x::fire::aligned16` is.
///
/// Carried here rather than imported: `x::fire` is a `_cuda`-only module and
/// a routine body is feature-free.
#[must_use]
fn aligned16(p: *const c_void) -> bool {
    p.addr() & 15 == 0
}

/// `rmsnorm.cu:26` — `rmsnorm_vec8_ok`.
#[must_use]
fn vec8_ok(
    x: *const c_void,
    y: *const c_void,
    weight: *const c_void,
    hidden: i32,
    x_row_stride: i32,
    y_row_stride: i32,
) -> bool {
    hidden % 8 == 0
        && x_row_stride % 8 == 0
        && y_row_stride % 8 == 0
        && aligned16(x)
        && aligned16(y)
        && aligned16(weight)
}

/// `rmsnorm.cu:80` — `norm::rmsnorm_strided_bf16`, both arms.
///
/// # Safety
///
/// `x`, `weight` and `y` must address live device memory of the extents the
/// strides describe, and `ctx`'s stream must be live for the duration of the
/// launch, which is asynchronous — so that ends at the next synchronisation
/// and not at this call's return.
pub fn rmsnorm_strided_bf16(
    ctx: &Ctx,
    x: *const bf16,
    weight: *const bf16,
    y: *mut bf16,
    num_rows: i32,
    hidden: i32,
    x_row_stride: i32,
    y_row_stride: i32,
    eps: f32,
) -> Result<(), Refusal> {
    if num_rows <= 0 {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    if vec8_ok(x.cast(), y.cast_const().cast(), weight.cast(), hidden, x_row_stride, y_row_stride) {
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        return unsafe {
            ctx.launch(
                &rmsnorm::ROOT,
                rmsnorm::inst::RMSNORM_VEC8_512,
                Launch::per_row(num_rows.unsigned_abs(), VBLOCK),
                &[
                    x.arg(),
                    weight.arg(),
                    y.arg(),
                    None::<NonNull<f16>>.arg(),
                    hidden.arg(),
                    x_row_stride.arg(),
                    y_row_stride.arg(),
                    eps.arg(),
                ],
            )
        };
    }
    // SAFETY: as above.
    unsafe {
        ctx.launch(
            &rmsnorm::ROOT,
            rmsnorm::inst::RMSNORM,
            per_row(num_rows),
            &[
                x.arg(),
                weight.arg(),
                y.arg(),
                hidden.arg(),
                x_row_stride.arg(),
                y_row_stride.arg(),
                eps.arg(),
            ],
        )
    }
}

/// `rmsnorm.cu:38` — `norm::rmsnorm_bf16`, which is one call and nothing
///
/// # Safety
///
/// [`rmsnorm_strided_bf16`]'s, unchanged.
pub fn unstrided_bf16(
    ctx: &Ctx,
    x: *const bf16,
    weight: *const bf16,
    y: *mut bf16,
    num_rows: i32,
    hidden: i32,
    eps: f32,
) -> Result<(), Refusal> {
    rmsnorm_strided_bf16(ctx, x, weight, y, num_rows, hidden, hidden, hidden, eps)
}

/// `rmsnorm.cu:54` — `norm::rmsnorm_bf16_with_fp16`, all three arms.
///
/// # Safety
///
/// `x`, `weight` and `y` must address `num_rows * hidden` live bf16 elements;
/// `y_fp16`, when `Some`, `num_rows * hidden` live fp16 elements. `ctx`'s
/// stream must be live across every launch this makes.
pub fn rmsnorm_bf16_with_fp16(
    ctx: &Ctx,
    x: *const bf16,
    weight: *const bf16,
    y: *mut bf16,
    y_fp16: Option<NonNull<f16>>,
    num_rows: i32,
    hidden: i32,
    eps: f32,
) -> Result<(), Refusal> {
    if num_rows <= 0 {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    let Some(fp16) = y_fp16 else {
        return unstrided_bf16(ctx, x, weight, y, num_rows, hidden, eps);
    };
    if !vec8_ok(x.cast(), y.cast_const().cast(), weight.cast(), hidden, hidden, hidden) {
        let n = i64::from(num_rows) * i64::from(hidden);
        // `quant::bf16_to_fp16` sizes a 32-bit launch extent from the count
        // and panics above it. This count is a product of two `i32` that
        // nothing in the signature bounds, and the kernel it feeds is
        // grid-strided over a 64-bit count, so the ceiling belongs to the
        // caller of the cast and is refused rather than hit.
        if n > i64::from(u32::MAX) {
            return Err(Refusal::Wide {
                what: "the fp16 copy's element count, which the cast sizes a \
                       32-bit launch extent from",
                at: n,
                max: i64::from(u32::MAX),
            });
        }
        // The second launch reads what the first wrote, and the stream orders
        // them. A refused first launch is a refused pair.
        unstrided_bf16(ctx, x, weight, y, num_rows, hidden, eps)?;
        return crate::x::quant::bf16_to_fp16(
            ctx,
            y.cast_const(),
            fp16.as_ptr(),
            usize::try_from(n).unwrap_or(0),
        );
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &rmsnorm::ROOT,
            rmsnorm::inst::RMSNORM_VEC8_512_FP16,
            Launch::per_row(num_rows.unsigned_abs(), VBLOCK),
            &[
                x.arg(),
                weight.arg(),
                y.arg(),
                Some(fp16).arg(),
                hidden.arg(),
                hidden.arg(),
                hidden.arg(),
                eps.arg(),
            ],
        )
    }
}

/// The SEMANTIC `OpKind::Rmsnorm`'s launcher — `norm::rmsnorm_bf16`.
///
/// # Safety
///
/// [`rmsnorm_strided_bf16`]'s.
pub fn rmsnorm_bf16(
    ctx: &Ctx,
    x: *const bf16,
    weight: *const bf16,
    y: *mut bf16,
    rows: i32,
    width: i32,
    per_head_dim: i32,
    eps: f32,
) -> Result<(), Refusal> {
    let hidden = if per_head_dim == 0 { width } else { per_head_dim };
    let launch = rows_per_head(rows, width, per_head_dim)?;
    if launch.empty() {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &rmsnorm::ROOT,
            rmsnorm::inst::RMSNORM,
            launch,
            &[x.arg(), weight.arg(), y.arg(), hidden.arg(), hidden.arg(), hidden.arg(), eps.arg()],
        )
    }
}

/// gemma's `(1 + w)` fold — `norm::rmsnorm_gemma_bf16`.
///
/// # Safety
///
/// [`rmsnorm_strided_bf16`]'s.
pub fn rmsnorm_gemma_bf16(
    ctx: &Ctx,
    x: *const bf16,
    weight: *const bf16,
    y: *mut bf16,
    rows: i32,
    width: i32,
    per_head_dim: i32,
    eps: f32,
) -> Result<(), Refusal> {
    let hidden = if per_head_dim == 0 { width } else { per_head_dim };
    let launch = rows_per_head(rows, width, per_head_dim)?;
    if launch.empty() {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &rmsnorm::ROOT,
            rmsnorm::inst::RMSNORM_GEMMA,
            launch,
            &[x.arg(), weight.arg(), y.arg(), hidden.arg(), hidden.arg(), hidden.arg(), eps.arg()],
        )
    }
}

/// The weightless per-head norm — `norm::rmsnorm_no_scale_bf16`.
///
/// # Safety
///
/// `x` and `y` must address `rows * width` live bf16 elements, and `ctx`'s
/// stream must be live across the launch.
pub fn rmsnorm_no_scale_bf16(
    ctx: &Ctx,
    x: *const bf16,
    y: *mut bf16,
    rows: i32,
    width: i32,
    per_head_dim: i32,
    eps: f32,
) -> Result<(), Refusal> {
    let hidden = if per_head_dim == 0 { width } else { per_head_dim };
    let launch = rows_per_head(rows, width, per_head_dim)?;
    if launch.empty() {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &rmsnorm::ROOT,
            rmsnorm::inst::RMSNORM_NO_SCALE,
            launch,
            &[x.arg(), y.arg(), hidden.arg(), eps.arg()],
        )
    }
}

/// The gated norm — `norm::rmsnorm_gated_bf16`.
///
/// # Safety
///
/// `x`, `gate` and `y` must address `rows * width` live bf16 elements,
/// `weight` `hidden` live floats, and `ctx`'s stream must be live across the
/// launch.
pub fn rmsnorm_gated_bf16(
    ctx: &Ctx,
    x: *const bf16,
    gate: *const bf16,
    weight: *const f32,
    y: *mut bf16,
    rows: i32,
    width: i32,
    per_head_dim: i32,
    eps: f32,
) -> Result<(), Refusal> {
    let hidden = if per_head_dim == 0 { width } else { per_head_dim };
    let launch = rows_per_head(rows, width, per_head_dim)?;
    if launch.empty() {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &rmsnorm::ROOT,
            rmsnorm::inst::RMSNORM_GATED,
            launch,
            &[x.arg(), gate.arg(), weight.arg(), y.arg(), hidden.arg(), eps.arg()],
        )
    }
}

/// The gated norm with an fp32 INPUT — `norm::rmsnorm_gated_fp32_in_bf16`.
///
/// # Safety
///
/// `x` must address `rows * width` live floats, `gate` and `y` the same count
/// of bf16, `weight` `hidden` live floats, and `ctx`'s stream must be live
/// across the launch.
pub fn rmsnorm_gated_fp32_in_bf16(
    ctx: &Ctx,
    x: *const f32,
    gate: *const bf16,
    weight: *const f32,
    y: *mut bf16,
    rows: i32,
    width: i32,
    per_head_dim: i32,
    eps: f32,
) -> Result<(), Refusal> {
    let hidden = if per_head_dim == 0 { width } else { per_head_dim };
    let launch = rows_per_head(rows, width, per_head_dim)?;
    if launch.empty() {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &rmsnorm::ROOT,
            rmsnorm::inst::RMSNORM_GATED_F32_IN,
            launch,
            &[x.arg(), gate.arg(), weight.arg(), y.arg(), hidden.arg(), eps.arg()],
        )
    }
}

/// The residual add and the NEXT block's pre-norm, fused —
///
/// # Safety
///
/// `hidden`, `residual`, `norm_out` must address `num_rows * hidden_size`
/// live bf16 elements and `weight` `hidden_size` of them; `ctx`'s stream must
/// be live across the launch.
pub fn residual_add_rmsnorm_bf16(
    ctx: &Ctx,
    hidden: *mut bf16,
    residual: *const bf16,
    weight: *const bf16,
    norm_out: *mut bf16,
    num_rows: i32,
    hidden_size: i32,
    eps: f32,
) -> Result<(), Refusal> {
    if num_rows <= 0 {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &rmsnorm::ROOT,
            rmsnorm::inst::RESIDUAL_ADD_RMSNORM,
            per_row(num_rows),
            &[
                hidden.arg(),
                residual.arg(),
                weight.arg(),
                norm_out.arg(),
                hidden_size.arg(),
                eps.arg(),
            ],
        )
    }
}

/// Norm, then add into the residual stream in place —
///
/// # Safety
///
/// [`residual_add_rmsnorm_bf16`]'s.
pub fn rmsnorm_residual_add_bf16(
    ctx: &Ctx,
    x: *const bf16,
    weight: *const bf16,
    hidden: *mut bf16,
    num_rows: i32,
    hidden_size: i32,
    eps: f32,
) -> Result<(), Refusal> {
    if num_rows <= 0 {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &rmsnorm::ROOT,
            rmsnorm::inst::RMSNORM_RESIDUAL_ADD,
            per_row(num_rows),
            &[x.arg(), weight.arg(), hidden.arg(), hidden_size.arg(), eps.arg()],
        )
    }
}

/// `rmsnorm.cu:119` — `norm::rmsnorm_residual_add_scale_rmsnorm_bf16`, all
///
/// The three arms take the same six operands in the same order, so the choice
/// is a choice of instantiation and nothing else: vectorised at 512 threads
/// above [`RASR_VEC512_ABOVE`], vectorised at 256 below it, and the scalar
/// 512-wide kernel when any of the five pointers is not 16-byte aligned or
/// the row is not a whole number of eight-element vectors.
///
/// # Safety
///
/// `x`, `weight`, `hidden`, `next_weight` and `norm_out` must address live
/// device memory of `num_rows * hidden_size` (the two weights,
/// `hidden_size`) bf16 elements, and `ctx`'s stream must be live across the
/// launch.
pub fn rmsnorm_residual_add_scale_rmsnorm_bf16(
    ctx: &Ctx,
    x: *const bf16,
    weight: *const bf16,
    hidden: *mut bf16,
    scale: f32,
    next_weight: *const bf16,
    norm_out: *mut bf16,
    num_rows: i32,
    hidden_size: i32,
    eps: f32,
) -> Result<(), Refusal> {
    if num_rows <= 0 {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    let rows = num_rows.unsigned_abs();
    let vec_ok = hidden_size % 8 == 0
        && aligned16(x.cast())
        && aligned16(hidden.cast_const().cast())
        && aligned16(norm_out.cast_const().cast())
        && aligned16(weight.cast())
        && aligned16(next_weight.cast());
    let (instantiation, block) = if vec_ok {
        if hidden_size >= RASR_VEC512_ABOVE {
            (rmsnorm::inst::RASR_VEC8_512, VBLOCK)
        } else {
            (rmsnorm::inst::RASR_VEC8_256, BLOCK)
        }
    } else {
        (rmsnorm::inst::RASR_SCALAR_512, VBLOCK)
    };
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &rmsnorm::ROOT,
            instantiation,
            Launch::per_row(rows, block),
            &[
                x.arg(),
                weight.arg(),
                hidden.arg(),
                scale.arg(),
                next_weight.arg(),
                norm_out.arg(),
                hidden_size.arg(),
                eps.arg(),
            ],
        )
    }
}

/// `out[row][i] += bias[i]` — `norm::add_bias_bf16`.
///
/// # Safety
///
/// `out` must address `num_rows * dim` live bf16 elements, `bias` `dim` of
/// them, and `ctx`'s stream must be live across the launch.
pub fn add_bias_bf16(
    ctx: &Ctx,
    out: *mut bf16,
    bias: *const bf16,
    num_rows: i32,
    dim: i32,
) -> Result<(), Refusal> {
    if num_rows <= 0 {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    if dim <= 0 {
        return Err(Refusal::Empty { what: "the bias width" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &add_bias::ROOT,
            add_bias::inst::ADD_BIAS,
            route_rows(num_rows, dim),
            &[out.arg(), bias.arg(), dim.arg()],
        )
    }
}

/// gemma-3n's altup predict — `norm::altup_predict_bf16`.
///
/// # Safety
///
/// `streams` and `predictions` must address `k * t_len * h` live bf16
/// elements, `coefs` `t_len * k * k` live floats, and `ctx`'s stream must be
/// live across the launch.
pub fn altup_predict_bf16(
    ctx: &Ctx,
    streams: *const bf16,
    coefs: *const f32,
    predictions: *mut bf16,
    k: i32,
    t_len: i32,
    h: i32,
) -> Result<(), Refusal> {
    if t_len <= 0 {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    if k <= 0 {
        return Err(Refusal::Empty { what: "the altup stream count" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &altup::ROOT,
            altup::inst::ALTUP_PREDICT,
            altup_streams(t_len, k, h),
            &[streams.arg(), coefs.arg(), predictions.arg(), k.arg(), t_len.arg(), h.arg()],
        )
    }
}

/// gemma-3n's altup correct — `norm::altup_correct_bf16`.
///
/// # Safety
///
/// [`altup_predict_bf16`]'s, with `activated` addressing `t_len * h` live
/// bf16 elements and `corrected` `k * t_len * h`.
pub fn altup_correct_bf16(
    ctx: &Ctx,
    predictions: *const bf16,
    activated: *const bf16,
    correction_coefs_plus_one: *const f32,
    corrected: *mut bf16,
    k: i32,
    t_len: i32,
    h: i32,
    active_idx: i32,
) -> Result<(), Refusal> {
    if t_len <= 0 {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    if k <= 0 {
        return Err(Refusal::Empty { what: "the altup stream count" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &altup::ROOT,
            altup::inst::ALTUP_CORRECT,
            altup_streams(t_len, k, h),
            &[
                predictions.arg(),
                activated.arg(),
                correction_coefs_plus_one.arg(),
                corrected.arg(),
                k.arg(),
                t_len.arg(),
                h.arg(),
                active_idx.arg(),
            ],
        )
    }
}

/// The per-row RMS of the reference stream — `norm::compute_rms_bf16`.
///
/// # Safety
///
/// `reference` must address `rows * h` live bf16 elements, `out` `rows` live
/// floats, and `ctx`'s stream must be live across the launch.
pub fn compute_rms_bf16(
    ctx: &Ctx,
    reference: *const bf16,
    out: *mut f32,
    rows: i32,
    h: i32,
    eps: f32,
) -> Result<(), Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &altup_aux::ROOT,
            altup_aux::inst::COMPUTE_RMS,
            per_row_reducing(rows),
            &[reference.arg(), out.arg(), h.arg(), eps.arg()],
        )
    }
}

/// Rescale each row to a stated RMS, in place — `norm::magnitude_rescale_bf16`.
///
/// # Safety
///
/// `x` must address `rows * h` live bf16 elements, `target_rms` `rows` live
/// floats, and `ctx`'s stream must be live across the launch.
pub fn magnitude_rescale_bf16(
    ctx: &Ctx,
    x: *mut bf16,
    target_rms: *const f32,
    rows: i32,
    h: i32,
    eps: f32,
) -> Result<(), Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &altup_aux::ROOT,
            altup_aux::inst::MAGNITUDE_RESCALE,
            per_row_reducing(rows),
            &[x.arg(), target_rms.arg(), h.arg(), eps.arg()],
        )
    }
}

/// The mean over altup's `k` streams — `norm::mean_streams_bf16`.
///
/// # Safety
///
/// `streams` must address `k * t_stride * h` live bf16 elements, `out`
/// `rows * h` of them, and `ctx`'s stream must be live across the launch.
pub fn mean_streams_bf16(
    ctx: &Ctx,
    streams: *const bf16,
    out: *mut bf16,
    k: i32,
    rows: i32,
    h: i32,
) -> Result<(), Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    if k <= 0 {
        return Err(Refusal::Empty { what: "the altup stream count" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &altup_aux::ROOT,
            altup_aux::inst::MEAN_STREAMS,
            elementwise_rows(rows, h),
            &[streams.arg(), out.arg(), k.arg(), rows.arg(), h.arg()],
        )
    }
}

/// bf16 predict coefficients widened to `float` —
///
/// # Safety
///
/// `in_bf16` must address `rows * k * k` live bf16 elements, `out` the same
/// count of floats, and `ctx`'s stream must be live across the launch.
pub fn altup_unpack_predict_coefs(
    ctx: &Ctx,
    in_bf16: *const bf16,
    out: *mut f32,
    rows: i32,
    k: i32,
) -> Result<(), Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    if k <= 0 {
        return Err(Refusal::Empty { what: "the altup stream count" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &altup_aux::ROOT,
            altup_aux::inst::UNPACK_PREDICT_COEFS,
            route_rows(rows, k.saturating_mul(k)),
            &[in_bf16.arg(), out.arg(), k.arg()],
        )
    }
}

/// bf16 correct coefficients widened to `float` —
///
/// # Safety
///
/// `in_bf16` must address `rows * k` live bf16 elements, `out` the same count
/// of floats, and `ctx`'s stream must be live across the launch.
pub fn altup_unpack_correct_coefs(
    ctx: &Ctx,
    in_bf16: *const bf16,
    out: *mut f32,
    rows: i32,
    k: i32,
) -> Result<(), Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    if k <= 0 {
        return Err(Refusal::Empty { what: "the altup stream count" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &altup_aux::ROOT,
            altup_aux::inst::UNPACK_CORRECT_COEFS,
            route_rows(rows, k),
            &[in_bf16.arg(), out.arg(), k.arg()],
        )
    }
}

/// `tanh` in place over a bf16 slab — `norm::tanh_bf16`.
///
/// # Safety
///
/// `x` must address `n` live bf16 elements and `ctx`'s stream must be live
/// across the launch.
pub fn tanh_bf16(ctx: &Ctx, x: *mut bf16, n: i32) -> Result<(), Refusal> {
    if n <= 0 {
        return Err(Refusal::Empty { what: "the element count" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &altup_aux::ROOT,
            altup_aux::inst::TANH_BF16,
            elementwise(n),
            &[x.arg(), n.arg()],
        )
    }
}

/// [`tanh_bf16`] over fp16 — `norm::tanh_f16`.
///
/// # Safety
///
/// [`tanh_bf16`]'s, with `x` addressing fp16.
pub fn tanh_f16(ctx: &Ctx, x: *mut f16, n: i32) -> Result<(), Refusal> {
    if n <= 0 {
        return Err(Refusal::Empty { what: "the element count" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(&altup_aux::ROOT, altup_aux::inst::TANH_F16, elementwise(n), &[x.arg(), n.arg()])
    }
}

/// `y += x` — `norm::residual_add_bf16`.
///
/// # Safety
///
/// `y` and `x` must address `n` live bf16 elements and `ctx`'s stream must be
/// live across the launch.
pub fn residual_add_bf16(ctx: &Ctx, y: *mut bf16, x: *const bf16, n: usize) -> Result<(), Refusal> {
    if n == 0 {
        return Err(Refusal::Empty { what: "the element count" });
    }
    let launch = Launch::flat(u32::try_from(n).unwrap_or(u32::MAX), BLOCK);
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &elementwise::ROOT,
            elementwise::inst::RESIDUAL_ADD_BF16,
            launch,
            &[y.arg(), x.arg(), n.arg()],
        )
    }
}

/// [`residual_add_bf16`] over fp16 — `norm::residual_add_f16`.
///
/// # Safety
///
/// [`residual_add_bf16`]'s, with both pointers addressing fp16.
pub fn residual_add_f16(ctx: &Ctx, y: *mut f16, x: *const f16, n: usize) -> Result<(), Refusal> {
    if n == 0 {
        return Err(Refusal::Empty { what: "the element count" });
    }
    let launch = Launch::flat(u32::try_from(n).unwrap_or(u32::MAX), BLOCK);
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &elementwise::ROOT,
            elementwise::inst::RESIDUAL_ADD_F16,
            launch,
            &[y.arg(), x.arg(), n.arg()],
        )
    }
}

/// `x *= s` — `norm::scalar_mul_bf16`.
///
/// # Safety
///
/// `x` must address `n` live bf16 elements and `ctx`'s stream must be live
/// across the launch.
pub fn scalar_mul_bf16(ctx: &Ctx, x: *mut bf16, s: f32, n: usize) -> Result<(), Refusal> {
    if n == 0 {
        return Err(Refusal::Empty { what: "the element count" });
    }
    let launch = Launch::flat(u32::try_from(n).unwrap_or(u32::MAX), BLOCK);
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &elementwise::ROOT,
            elementwise::inst::SCALAR_MUL_BF16,
            launch,
            &[x.arg(), s.arg(), n.arg()],
        )
    }
}

/// `dsv4_hc.cu:22` — `norm::hc_pre_postprocess_bf16`.
///
/// # Safety
///
/// `residual` and `layer_input` must address `n * hc_mult * hidden_size` and
/// `n * hidden_size` live bf16 elements; `mixes`, `scale` and `base` the
/// slabs the layer carries; `post_mix` and `comb_mix` scratch of `n *
/// hc_mult` and `n * hc_mult * hc_mult` floats. `ctx`'s stream must be live
/// across the launch.
pub fn hc_pre_postprocess_bf16(
    ctx: &Ctx,
    mixes: *const f32,
    scale: *const f32,
    base: *const f32,
    residual: *const bf16,
    post_mix: *mut f32,
    comb_mix: *mut f32,
    layer_input: *mut bf16,
    n: i32,
    hc_mult: i32,
    hidden_size: i32,
    hc_eps: f32,
    hc_post_alpha: f32,
    sinkhorn_iters: i32,
) -> Result<(), Refusal> {
    if n <= 0 {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    hc_mult_ok(hc_mult)?;
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dsv4_hc::ROOT,
            dsv4_hc::inst::HC_PRE_POSTPROCESS,
            per_row(n),
            &[
                mixes.arg(),
                scale.arg(),
                base.arg(),
                residual.arg(),
                post_mix.arg(),
                comb_mix.arg(),
                layer_input.arg(),
                hc_mult.arg(),
                hidden_size.arg(),
                hc_eps.arg(),
                hc_post_alpha.arg(),
                sinkhorn_iters.arg(),
            ],
        )
    }
}

/// `dsv4_hc.cu:47` — `norm::hc_post_bf16`.
///
/// # Safety
///
/// [`hc_pre_postprocess_bf16`]'s, with `out_residual` addressing `n * hc_mult
/// * hidden_size` live bf16 elements.
pub fn hc_post_bf16(
    ctx: &Ctx,
    x: *const bf16,
    residual: *const bf16,
    post_mix: *const f32,
    comb_mix: *const f32,
    out_residual: *mut bf16,
    n: i32,
    hc_mult: i32,
    hidden_size: i32,
) -> Result<(), Refusal> {
    hc_mult_ok(hc_mult)?;
    let total = i64::from(n) * i64::from(hidden_size);
    if total <= 0 {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dsv4_hc::ROOT,
            dsv4_hc::inst::HC_POST,
            elementwise_wide(total),
            &[
                x.arg(),
                residual.arg(),
                post_mix.arg(),
                comb_mix.arg(),
                out_residual.arg(),
                n.arg(),
                hc_mult.arg(),
                hidden_size.arg(),
            ],
        )
    }
}

/// `dsv4_hc.cu:69` — `norm::hc_head_postprocess_bf16`.
///
/// # Safety
///
/// [`hc_pre_postprocess_bf16`]'s, with `out` addressing `n * hidden_size`
/// live bf16 elements.
pub fn hc_head_postprocess_bf16(
    ctx: &Ctx,
    mixes: *const f32,
    scale: *const f32,
    base: *const f32,
    residual: *const bf16,
    out: *mut bf16,
    n: i32,
    hc_mult: i32,
    hidden_size: i32,
    hc_eps: f32,
) -> Result<(), Refusal> {
    if n <= 0 {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    hc_mult_ok(hc_mult)?;
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dsv4_hc::ROOT,
            dsv4_hc::inst::HC_HEAD_POSTPROCESS,
            per_row(n),
            &[
                mixes.arg(),
                scale.arg(),
                base.arg(),
                residual.arg(),
                out.arg(),
                hc_mult.arg(),
                hidden_size.arg(),
                hc_eps.arg(),
            ],
        )
    }
}

/// `[n, hidden] -> [n, hc_mult, hidden]` — `norm::hc_expand_bf16`.
///
/// # Safety
///
/// `input` must address `n * hidden_size` live bf16 elements, `output`
/// `n * hc_mult * hidden_size` of them, and `ctx`'s stream must be live
/// across the launch.
pub fn hc_expand_bf16(
    ctx: &Ctx,
    input: *const bf16,
    output: *mut bf16,
    n: i32,
    hc_mult: i32,
    hidden_size: i32,
) -> Result<(), Refusal> {
    hc_mult_ok(hc_mult)?;
    let total = i64::from(n) * i64::from(hidden_size);
    if total <= 0 {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dsv4_hc::ROOT,
            dsv4_hc::inst::HC_EXPAND,
            elementwise_wide(total),
            &[input.arg(), output.arg(), n.arg(), hc_mult.arg(), hidden_size.arg()],
        )
    }
}

/// `dsv4_hc.cu:89` — `norm::hc_rmsnorm_to_f32`.
///
/// # Safety
///
/// `input` must address `n * dim` live bf16 elements, `output` `n * dim` live
/// floats, and `ctx`'s stream must be live across the launch.
pub fn hc_rmsnorm_to_f32(
    ctx: &Ctx,
    input: *const bf16,
    output: *mut f32,
    n: i32,
    dim: i32,
    eps: f32,
) -> Result<(), Refusal> {
    if n <= 0 {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dsv4_hc::ROOT,
            dsv4_hc::inst::HC_RMSNORM_TO_F32,
            per_row(n),
            &[input.arg(), output.arg(), dim.arg(), eps.arg()],
        )
    }
}

/// The attention sink's log-sum-exp correction —
///
/// # Safety
///
/// `out` must address `n * num_heads * head_dim` live bf16 elements, `lse`
/// and `sink` `n * num_heads` and `num_heads` live floats, and `ctx`'s stream
/// must be live across the launch.
pub fn attn_sink_correction_bf16(
    ctx: &Ctx,
    out: *mut bf16,
    lse: *const f32,
    sink: *const f32,
    n: i32,
    num_heads: i32,
    head_dim: i32,
) -> Result<(), Refusal> {
    if n <= 0 {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    if num_heads <= 0 {
        return Err(Refusal::Empty { what: "the head count" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dsv4_hc::ROOT,
            dsv4_hc::inst::ATTN_SINK_CORRECTION,
            gated_rms(n, num_heads),
            &[out.arg(), lse.arg(), sink.arg(), num_heads.arg(), head_dim.arg()],
        )
    }
}

/// QK-norm in place over a packed head axis —
///
/// # Safety
///
/// `q` must address `n * num_heads * head_dim` live bf16 elements and `ctx`'s
/// stream must be live across the launch.
pub fn per_head_rmsnorm_bf16(
    ctx: &Ctx,
    q: *mut bf16,
    n: i32,
    num_heads: i32,
    head_dim: i32,
    eps: f32,
) -> Result<(), Refusal> {
    if n <= 0 {
        return Err(Refusal::Empty { what: "num_rows" });
    }
    if num_heads <= 0 {
        return Err(Refusal::Empty { what: "the head count" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &dsv4_hc::ROOT,
            dsv4_hc::inst::PER_HEAD_RMSNORM,
            gated_rms(n, num_heads),
            &[q.arg(), head_dim.arg(), eps.arg()],
        )
    }
}

/// `hc_mult <= MAX_HC_MULT`, as a refusal.
fn hc_mult_ok(hc_mult: i32) -> Result<(), Refusal> {
    if hc_mult > MAX_HC_MULT {
        return Err(Refusal::Wide {
            what: "hc_mult, which `hc_post` unrolls into a register array",
            at: i64::from(hc_mult),
            max: i64::from(MAX_HC_MULT),
        });
    }
    Ok(())
}

/// This family's routines, and what a trace may say about each.
///
/// The argument lists are DERIVED from the `fn`s above -- `routine!` sees only
/// the identifier. What is stated here is what no signature carries: which
/// operands must be given the same address. Not one of the twenty-eight
/// contracts this family declares says `whole` or `depth_prefix_plan`, so
/// nothing here does either.
pub static ROUTINES: &[Routine] = &[
    routine!(rmsnorm_strided_bf16),
    routine!(unstrided_bf16),
    routine!(rmsnorm_bf16_with_fp16),
    routine!(rmsnorm_bf16),
    routine!(rmsnorm_gemma_bf16),
    routine!(rmsnorm_no_scale_bf16, in_place = &[(0, 0)]),
    routine!(rmsnorm_gated_bf16),
    routine!(rmsnorm_gated_fp32_in_bf16),
    routine!(residual_add_rmsnorm_bf16),
    routine!(rmsnorm_residual_add_bf16, in_place = &[(0, 1)]),
    routine!(rmsnorm_residual_add_scale_rmsnorm_bf16, in_place = &[(0, 1)]),
    routine!(add_bias_bf16, in_place = &[(0, 0)]),
    routine!(altup_predict_bf16),
    routine!(altup_correct_bf16),
    routine!(compute_rms_bf16),
    routine!(magnitude_rescale_bf16, in_place = &[(0, 0)]),
    routine!(mean_streams_bf16),
    routine!(altup_unpack_predict_coefs),
    routine!(altup_unpack_correct_coefs),
    routine!(tanh_bf16, in_place = &[(0, 0)]),
    routine!(tanh_f16),
    routine!(residual_add_bf16, in_place = &[(0, 0)]),
    routine!(residual_add_f16),
    routine!(scalar_mul_bf16, in_place = &[(0, 0)]),
    routine!(hc_pre_postprocess_bf16),
    routine!(hc_post_bf16),
    routine!(hc_head_postprocess_bf16),
    routine!(hc_expand_bf16),
    routine!(hc_rmsnorm_to_f32),
    routine!(attn_sink_correction_bf16, in_place = &[(0, 0)]),
    routine!(per_head_rmsnorm_bf16, in_place = &[(0, 0)]),
];

/// `norm`, as a trace names it.
pub static FAMILY: Family = Family { namespace: "norm", routines: ROUTINES };

#![allow(clippy::too_many_arguments)]

use crate::unit::Unit;
use crate::x::abi::{bf16, f16};
use crate::x::launch::Launch;

#[cfg(feature = "_cuda")]
use crate::x::contract::{Fired, Refusal};
#[cfg(feature = "_cuda")]
use core::ffi::c_void;
use core::ptr::NonNull;

/// `norm/rmsnorm.cuh` — twelve `__global__` templates, fifteen
pub mod rmsnorm {
    use super::{bf16, f16};
    use core::ptr::NonNull;

    unit! {
        /// The RMSNorm family proper: the scalar kernels whose launcher was
        unit RMSNORM = "norm/rmsnorm",
            text = include_str!("../../csrc/src/norm/rmsnorm.cuh"),
            file = "norm/rmsnorm.cuh";

        /// `rmsnorm.cuh:220` — `y = x * rsqrt(mean(x^2) + eps) * w`, with
        fn rmsnorm = "norm::device::rmsnorm" <T> (
            x: *const T,
            weight: *const T,
            y: *mut T,
            hidden: i32,
            x_row_stride: i32,
            y_row_stride: i32,
            eps: f32,
        ) where *const T, *mut T {
            "norm::rmsnorm_bf16" => where [T = bf16] "device::bf16, 256",
            "norm::rmsnorm_strided_bf16" => where [T = bf16] "device::bf16, 256",
        }

        /// `rmsnorm.cuh:236` — the same with `(1 + w)` folded instead of `w`.
        fn rmsnorm_gemma = "norm::device::rmsnorm_gemma" <T> (
            x: *const T,
            weight: *const T,
            y: *mut T,
            hidden: i32,
            x_row_stride: i32,
            y_row_stride: i32,
            eps: f32,
        ) where *const T, *mut T {
            "norm::rmsnorm_gemma_bf16" => where [T = bf16] "device::bf16, 256",
        }

        /// `rmsnorm.cuh:262` — eight contiguous bf16 per thread, one 16-byte
        fn rmsnorm_vec8 = "norm::device::rmsnorm_vec8" (
            x: *const bf16,
            weight: *const bf16,
            y: *mut bf16,
            y_fp16: Option<NonNull<f16>>,
            hidden: i32,
            x_row_stride: i32,
            y_row_stride: i32,
            eps: f32,
        ) {
            "norm::rmsnorm_strided_bf16#vec8" => "device::i32(256), false, false",
            "norm::rmsnorm_bf16_with_fp16#vec8" => "device::i32(256), false, true",
            "norm::rmsnorm_strided_bf16#vec8_512" => "device::i32(512), false, false",
            "norm::rmsnorm_bf16_with_fp16#vec8_512" => "device::i32(512), false, true",
        }

        /// `rmsnorm.cuh:401` — the residual add and the NEXT block's pre-norm,
        fn residual_add_rmsnorm = "norm::device::residual_add_rmsnorm" <T> (
            hidden: *mut T,
            residual: *const T,
            weight: *const T,
            norm_out: *mut T,
            hidden_size: i32,
            eps: f32,
        ) where *const T, *mut T {
            "norm::residual_add_rmsnorm_bf16" => where [T = bf16] "device::bf16, 256",
        }

        /// `rmsnorm.cuh:492` — norm `x`, then add the result into the
        fn rmsnorm_residual_add = "norm::device::rmsnorm_residual_add" <T> (
            x: *const T,
            weight: *const T,
            hidden: *mut T,
            hidden_size: i32,
            eps: f32,
        ) where *const T, *mut T {
            "norm::rmsnorm_residual_add_bf16" => where [T = bf16] "device::bf16, 256",
        }

        /// `rmsnorm.cuh:548` — gemma-4's landing: residual add, scale, and
        fn rmsnorm_rasr_vec8 = "norm::device::rmsnorm_rasr_vec8" (
            x: *const bf16,
            weight: *const bf16,
            hidden: *mut bf16,
            scale: f32,
            next_weight: *const bf16,
            norm_out: *mut bf16,
            hidden_size: i32,
            eps: f32,
        ) {
            "norm::rmsnorm_residual_add_scale_rmsnorm_bf16#vec8_512" => "device::i32(512)",
            "norm::rmsnorm_residual_add_scale_rmsnorm_bf16#vec8_256" => "device::i32(256)",
        }

        /// `rmsnorm.cuh:631` — the same three passes, scalar.
        fn rmsnorm_residual_add_scale_rmsnorm =
            "norm::device::rmsnorm_residual_add_scale_rmsnorm" <T> (
            x: *const T,
            weight: *const T,
            hidden: *mut T,
            scale: f32,
            next_weight: *const T,
            norm_out: *mut T,
            hidden_size: i32,
            eps: f32,
        ) where *const T, *mut T {
            "norm::rmsnorm_residual_add_scale_rmsnorm_bf16#scalar_512" =>
                where [T = bf16] "device::bf16, 512",
        }

        /// `rmsnorm.cuh:686` — the weightless per-head norm, the V-norm.
        fn rmsnorm_no_scale = "norm::device::rmsnorm_no_scale" <T> (
            x: *const T,
            y: *mut T,
            hidden: i32,
            eps: f32,
        ) where *const T, *mut T {
            "norm::rmsnorm_no_scale_bf16" => where [T = bf16] "device::bf16, 256",
        }

        /// `rmsnorm.cuh:718` — norm gated by a second activation, with an
        fn rmsnorm_gated = "norm::device::rmsnorm_gated" <T> (
            x: *const T,
            gate: *const T,
            weight: *const f32,
            y: *mut T,
            hidden: i32,
            eps: f32,
        ) where *const T, *mut T {
            "norm::rmsnorm_gated_bf16" => where [T = bf16] "device::bf16, 256",
        }

        /// `rmsnorm.cuh:763` — the same with an fp32 INPUT as well.
        fn rmsnorm_gated_f32_in = "norm::device::rmsnorm_gated_f32_in" <T> (
            x: *const f32,
            gate: *const T,
            weight: *const f32,
            y: *mut T,
            hidden: i32,
            eps: f32,
        ) where *const T, *mut T {
            "norm::rmsnorm_gated_fp32_in_bf16" => where [T = bf16] "device::bf16, 256",
        }
    }
}

/// `norm/add_bias.cuh` — one `__device__` row body, two `__global__`s over
pub mod add_bias {
    use super::bf16;

    unit! {
        /// Both launchers were the same three lines
        unit ADD_BIAS = "norm/add_bias",
            text = include_str!("../../csrc/src/norm/add_bias.cuh"),
            file = "norm/add_bias.cuh";

        /// `add_bias.cuh:82` — `out[row][i] += bias[i]`, contiguous rows.
        fn add_bias = "norm::device::add_bias" <T> (
            out: *mut T,
            bias: *const T,
            dim: i32,
        ) where *const T, *mut T {
            "norm::add_bias_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `norm/altup.cuh` — gemma-3n's rank-K residual, predict and correct.
pub mod altup {
    use super::bf16;

    unit! {
        /// Two `__global__` templates. The header records that these rows
        unit ALTUP = "norm/altup",
            text = include_str!("../../csrc/src/norm/altup.cuh"),
            file = "norm/altup.cuh";

        /// `altup.cuh:77` — predict each of `K` streams as a coefficient
        fn altup_predict = "norm::device::altup_predict" <T> (
            streams: *const T,
            coefs: *const f32,
            predictions: *mut T,
            k: i32,
            t_len: i32,
            h: i32,
        ) where *const T, *mut T {
            "norm::altup_predict_bf16" => where [T = bf16] "device::bf16",
        }

        /// `altup.cuh:106` — correct every stream from the one the layer
        fn altup_correct = "norm::device::altup_correct" <T> (
            predictions: *const T,
            activated: *const T,
            correction_coefs_plus_one: *const f32,
            corrected: *mut T,
            k: i32,
            t_len: i32,
            h: i32,
            active_idx: i32,
        ) where *const T, *mut T {
            "norm::altup_correct_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `norm/altup_aux.cuh` — the five auxiliaries AltUp needs around the two
pub mod altup_aux {
    use super::{bf16, f16};

    unit! {
        /// Six `__global__` templates and one include, and the header is
        unit ALTUP_AUX = "norm/altup_aux",
            text = include_str!("../../csrc/src/norm/altup_aux.cuh"),
            file = "norm/altup_aux.cuh";

        /// `altup_aux.cuh:91` — the RMS of each row, to fp32.
        fn compute_rms = "norm::device::compute_rms" <T> (
            reference: *const T,
            out: *mut f32,
            h: i32,
            eps: f32,
        ) where *const T {
            "norm::compute_rms_bf16" => where [T = bf16] "device::bf16",
        }

        /// `altup_aux.cuh:111` — rescale each row to a target RMS, in place.
        fn magnitude_rescale = "norm::device::magnitude_rescale" <T> (
            x: *mut T,
            target_rms: *const f32,
            h: i32,
            eps: f32,
        ) where *mut T {
            "norm::magnitude_rescale_bf16" => where [T = bf16] "device::bf16",
        }

        /// `altup_aux.cuh:141` — the mean of `K` streams at each position.
        fn mean_streams = "norm::device::mean_streams" <T> (
            streams: *const T,
            out: *mut T,
            k: i32,
            t_stride: i32,
            h: i32,
        ) where *const T, *mut T {
            "norm::mean_streams_bf16" => where [T = bf16] "device::bf16",
        }

        /// `altup_aux.cuh:163` — unpack and TRANSPOSE a `[T, K, K]` bf16
        fn unpack_predict_coefs = "norm::device::unpack_predict_coefs" <T> (
            in_bf16: *const T,
            out: *mut f32,
            k: i32,
        ) where *const T {
            "norm::altup_unpack_predict_coefs" => where [T = bf16] "device::bf16",
        }

        /// `altup_aux.cuh:178` — unpack a `[T, K]` bf16 block to fp32 and add
        fn unpack_correct_coefs = "norm::device::unpack_correct_coefs" <T> (
            in_bf16: *const T,
            out: *mut f32,
            k: i32,
        ) where *const T {
            "norm::altup_unpack_correct_coefs" => where [T = bf16] "device::bf16",
        }

        /// `altup_aux.cuh:189` — elementwise `tanh`, in place, over a flat
        fn tanh_inplace = "norm::device::tanh_inplace" <T> (
            x: *mut T,
            n: i32,
        ) where *mut T {
            "norm::tanh_bf16" => where [T = bf16] "device::bf16",
            "norm::tanh_f16" => where [T = f16] "device::f16",
        }
    }
}

/// `norm/elementwise.cuh` — the residual add and the scalar multiply.
pub mod elementwise {
    use super::{bf16, f16};

    unit! {
        /// Both launchers were the same four lines (`elementwise.cuh:9-13`):
        unit ELEMENTWISE = "norm/elementwise",
            text = include_str!("../../csrc/src/norm/elementwise.cuh"),
            file = "norm/elementwise.cuh";

        /// `elementwise.cuh:56` — `y += x`, elementwise, accumulated in fp32
        fn residual_add = "norm::device::residual_add" <T> (
            y: *mut T,
            x: *const T,
            n: usize,
        ) where *const T, *mut T {
            "norm::residual_add_bf16" => where [T = bf16] "device::bf16",
            "norm::residual_add_f16" => where [T = f16] "device::f16",
        }

        /// `elementwise.cuh:74` — `x *= s`, with `s` ROUNDED TO `T` FIRST.
        fn scalar_mul = "norm::device::scalar_mul" <T> (
            x: *mut T,
            s: f32,
            n: usize,
        ) where *mut T {
            "norm::scalar_mul_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `norm/dsv4_hc.cuh` — deepseek-v4's hyper-connections, the attention sink
pub mod dsv4_hc {
    use super::bf16;

    unit! {
        /// Seven `__global__` templates, all seven instantiated here.
        unit DSV4_HC = "norm/dsv4_hc",
            text = include_str!("../../csrc/src/norm/dsv4_hc.cuh"),
            file = "norm/dsv4_hc.cuh";

        /// `dsv4_hc.cuh:103` — split the mix matrix, Sinkhorn-normalise the
        fn hc_pre_postprocess = "norm::device::hc_pre_postprocess" <T> (
            mixes: *const f32,
            scale: *const f32,
            base: *const f32,
            residual: *const T,
            post_mix: *mut f32,
            comb_mix: *mut f32,
            layer_input: *mut T,
            m: i32,
            h: i32,
            hc_eps: f32,
            hc_post_alpha: f32,
            sinkhorn_iters: i32,
        ) where *const T, *mut T {
            "norm::hc_pre_postprocess_rows_bf16" => where [T = bf16] "device::bf16, 256",
        }

        /// `dsv4_hc.cuh:239` — scatter the layer's output back across the `M`
        fn hc_post = "norm::device::hc_post" <T> (
            x: *const T,
            residual: *const T,
            post_mix: *const f32,
            comb_mix: *const f32,
            out: *mut T,
            n: i32,
            m: i32,
            h: i32,
        ) where *const T, *mut T {
            "norm::hc_post_elems_bf16" => where [T = bf16] "device::bf16",
        }

        /// `dsv4_hc.cuh:285` — the same collapse without the Sinkhorn: a
        fn hc_head_postprocess = "norm::device::hc_head_postprocess" <T> (
            mixes: *const f32,
            scale: *const f32,
            base: *const f32,
            residual: *const T,
            out: *mut T,
            m: i32,
            h: i32,
            hc_eps: f32,
        ) where *const T, *mut T {
            "norm::hc_head_postprocess_rows_bf16" => where [T = bf16] "device::bf16, 256",
        }

        /// `dsv4_hc.cuh:327` — the degenerate mixer: broadcast one stream
        fn hc_expand = "norm::device::hc_expand" <T> (
            input: *const T,
            output: *mut T,
            n: i32,
            m: i32,
            h: i32,
        ) where *const T, *mut T {
            "norm::hc_expand_bf16" => where [T = bf16] "device::bf16",
        }

        /// `dsv4_hc.cuh:358` — RMS-normalise `[N, dim]` bf16 into `[N, dim]`
        fn hc_rmsnorm_to_f32 = "norm::device::hc_rmsnorm_to_f32" <T> (
            input: *const T,
            output: *mut f32,
            dim: i32,
            eps: f32,
        ) where *const T {
            "norm::hc_rmsnorm_to_f32_rows" => where [T = bf16] "device::bf16, 256",
        }

        /// `dsv4_hc.cuh:406` — fold an attention sink logit into an already
        fn attn_sink_correction = "norm::device::attn_sink_correction" <T> (
            out: *mut T,
            lse: *const f32,
            sink: *const f32,
            num_heads: i32,
            head_dim: i32,
        ) where *mut T {
            "norm::attn_sink_correction_bf16" => where [T = bf16] "device::bf16",
        }

        /// `dsv4_hc.cuh:431` — RMS-normalise each attention head of a
        fn per_head_rmsnorm = "norm::device::per_head_rmsnorm" <T> (
            q: *mut T,
            head_dim: i32,
            eps: f32,
        ) where *mut T {
            "norm::per_head_rmsnorm_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// The six roots this family compiles.
pub static UNITS: &[Unit] = &[
    rmsnorm::RMSNORM,
    add_bias::ADD_BIAS,
    altup::ALTUP,
    altup_aux::ALTUP_AUX,
    elementwise::ELEMENTWISE,
    dsv4_hc::DSV4_HC,
];

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
    Launch {
        grid: [u32::try_from(blocks).unwrap_or(u32::MAX), 1, 1],
        block: [BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
}

/// Pointwise with the row on its own grid axis.
#[must_use]
const fn elementwise_rows(rows: i32, width: i32) -> Launch {
    Launch {
        grid: [rows.unsigned_abs(), width.unsigned_abs().div_ceil(BLOCK), 1],
        block: [BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
}

/// One block per row, as wide as the row rounded up to a warp and capped at
#[must_use]
const fn route_rows(rows: i32, width: i32) -> Launch {
    let warps = width.unsigned_abs().div_ceil(WARP);
    let warps = if warps == 0 { 1 } else { warps };
    let block = warps.saturating_mul(WARP);
    let block = if block > MAX_BLOCK { MAX_BLOCK } else { block };
    Launch {
        grid: [rows.unsigned_abs(), 1, 1],
        block: [block, 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
}

/// One block per (row, head), [`BLOCK`] wide.
#[must_use]
const fn gated_rms(rows: i32, heads: i32) -> Launch {
    Launch {
        grid: [rows.unsigned_abs(), heads.unsigned_abs(), 1],
        block: [BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
}

/// `dim3(T, K, ceil(H / 128))` at [`ALTUP_BLOCK`] threads.
#[must_use]
const fn altup_streams(rows: i32, streams: i32, hidden: i32) -> Launch {
    Launch {
        grid: [
            rows.unsigned_abs(),
            streams.unsigned_abs(),
            hidden.unsigned_abs().div_ceil(ALTUP_BLOCK),
        ],
        block: [ALTUP_BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
}

/// One block per row PER HEAD, [`BLOCK`] wide, nothing shared.
#[cfg(feature = "_cuda")]
fn rows_per_head(rows: i32, width: i32, stated_head_dim: i32) -> Result<Launch, Refusal> {
    if stated_head_dim == 0 {
        return Ok(per_row(rows));
    }
    let (w, hd) = (width.unsigned_abs(), stated_head_dim.unsigned_abs());
    if w == 0 || !w.is_multiple_of(hd) {
        return Err(Refusal::Narrow { what: "a row that divides by head_dim", at: width });
    }
    let blocks = rows
        .unsigned_abs()
        .checked_mul(w / hd)
        .ok_or(Refusal::Narrow { what: "a row count that fits a grid", at: rows })?;
    Ok(Launch::per_row(blocks, BLOCK))
}

/// `rmsnorm.cu:26` — `rmsnorm_vec8_ok`.
#[cfg(feature = "_cuda")]
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
        && crate::x::fire::aligned16(x)
        && crate::x::fire::aligned16(y)
        && crate::x::fire::aligned16(weight)
}

/// `rmsnorm.cu:80` — `norm::rmsnorm_strided_bf16`, both arms.
///
/// # Safety
///
/// `x`, `weight` and `y` must address live device memory of the extents the
/// strides describe, and `stream` must be a live `cudaStream_t` — for the
/// duration of the launch, which is asynchronous, so that ends at the next
/// synchronisation and not at this call's return.
#[cfg(feature = "_cuda")]
pub unsafe fn strided_bf16(
    x: *const bf16,
    weight: *const bf16,
    y: *mut bf16,
    num_rows: i32,
    hidden: i32,
    x_row_stride: i32,
    y_row_stride: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if num_rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    if vec8_ok(
        x.cast(),
        y.cast_const().cast(),
        weight.cast(),
        hidden,
        x_row_stride,
        y_row_stride,
    ) {
        unsafe {
            rmsnorm::raw::rmsnorm_vec8(
                "norm::rmsnorm_strided_bf16#vec8_512",
                Launch::per_row(num_rows.unsigned_abs(), VBLOCK),
                x,
                weight,
                y,
                None,
                hidden,
                x_row_stride,
                y_row_stride,
                eps,
                stream,
            );
        }
        return Fired::Launched;
    }
    unsafe {
        rmsnorm::raw::rmsnorm(
            "norm::rmsnorm_strided_bf16",
            per_row(num_rows),
            x,
            weight,
            y,
            hidden,
            x_row_stride,
            y_row_stride,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// `rmsnorm.cu:38` — `norm::rmsnorm_bf16`, which is one call and nothing
///
/// # Safety
///
/// [`strided_bf16`]'s, unchanged.
#[cfg(feature = "_cuda")]
pub unsafe fn unstrided_bf16(
    x: *const bf16,
    weight: *const bf16,
    y: *mut bf16,
    num_rows: i32,
    hidden: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    // SAFETY: the caller's obligation, forwarded verbatim.
    unsafe { strided_bf16(x, weight, y, num_rows, hidden, hidden, hidden, eps, stream) }
}

/// `rmsnorm.cu:64` — `kernels::quant::bf16_to_fp16(y, y_fp16, n, stream)`.
///
/// # Safety
///
/// `src` must address `count` live bf16 elements, `dst` `count` live fp16
/// elements, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
unsafe fn cast_to_fp16(src: *const bf16, dst: *mut f16, count: i64, stream: *mut c_void) -> Fired {
    if count <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the element count" });
    }
    const BS: i64 = 256;
    const SLAB_GRID_MAX: i64 = 1024;
    let n_vec8 = count / 8;
    let units = if n_vec8 > 0 { n_vec8 } else { count };
    let blocks = ((units + BS - 1) / BS).clamp(1, SLAB_GRID_MAX);
    let launch = Launch {
        grid: [u32::try_from(blocks).unwrap_or(1024), 1, 1],
        block: [u32::try_from(BS).unwrap_or(256), 1, 1],
        smem: 0,
        smem_opt_in: false,
    };
    // SAFETY: the caller's obligation, above. The stub is `quant`'s and the
    unsafe {
        crate::x::quant::dequant_wna16::raw::bf16_to_narrow::<f16>(
            "quant::bf16_to_fp16",
            launch,
            src,
            dst,
            count,
            stream,
        );
    }
    Fired::Launched
}

/// `rmsnorm.cu:54` — `norm::rmsnorm_bf16_with_fp16`, all three arms.
///
/// # Safety
///
/// `x`, `weight` and `y` must address `num_rows * hidden` live bf16 elements;
/// `y_fp16`, when `Some`, `num_rows * hidden` live fp16 elements. `stream`
/// must be live across every launch this makes.
#[cfg(feature = "_cuda")]
pub unsafe fn rmsnorm_bf16_with_fp16(
    x: *const bf16,
    weight: *const bf16,
    y: *mut bf16,
    y_fp16: Option<NonNull<f16>>,
    num_rows: i32,
    hidden: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if num_rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    let Some(fp16) = y_fp16 else {
        // SAFETY: the caller's obligation, forwarded.
        return unsafe { unstrided_bf16(x, weight, y, num_rows, hidden, eps, stream) };
    };
    if !vec8_ok(x.cast(), y.cast_const().cast(), weight.cast(), hidden, hidden, hidden) {
        // SAFETY: as above, and the second launch reads what the first wrote
        unsafe {
            unstrided_bf16(x, weight, y, num_rows, hidden, eps, stream);
            cast_to_fp16(
                y.cast_const(),
                fp16.as_ptr(),
                i64::from(num_rows) * i64::from(hidden),
                stream,
            );
        }
        return Fired::Launched;
    }
    unsafe {
        rmsnorm::raw::rmsnorm_vec8(
            "norm::rmsnorm_bf16_with_fp16#vec8_512",
            Launch::per_row(num_rows.unsigned_abs(), VBLOCK),
            x,
            weight,
            y,
            Some(fp16),
            hidden,
            hidden,
            hidden,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// The SEMANTIC `OpKind::Rmsnorm`'s launcher — `norm::rmsnorm_bf16`.
///
/// # Safety
///
/// [`strided_bf16`]'s.
#[cfg(feature = "_cuda")]
pub unsafe fn rmsnorm_bf16(
    x: *const bf16,
    weight: *const bf16,
    y: *mut bf16,
    rows: i32,
    width: i32,
    per_head_dim: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    let hidden = if per_head_dim == 0 { width } else { per_head_dim };
    let launch = match rows_per_head(rows, width, per_head_dim) {
        Ok(l) => l,
        Err(r) => return Fired::Declined(r),
    };
    if launch.empty() {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    unsafe {
        rmsnorm::raw::rmsnorm(
            "norm::rmsnorm_bf16",
            launch,
            x,
            weight,
            y,
            hidden,
            hidden,
            hidden,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// gemma's `(1 + w)` fold — `norm::rmsnorm_gemma_bf16`.
///
/// # Safety
///
/// [`strided_bf16`]'s.
#[cfg(feature = "_cuda")]
pub unsafe fn rmsnorm_gemma_bf16(
    x: *const bf16,
    weight: *const bf16,
    y: *mut bf16,
    rows: i32,
    width: i32,
    per_head_dim: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    let hidden = if per_head_dim == 0 { width } else { per_head_dim };
    let launch = match rows_per_head(rows, width, per_head_dim) {
        Ok(l) => l,
        Err(r) => return Fired::Declined(r),
    };
    if launch.empty() {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    unsafe {
        rmsnorm::raw::rmsnorm_gemma(
            "norm::rmsnorm_gemma_bf16",
            launch,
            x,
            weight,
            y,
            hidden,
            hidden,
            hidden,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// The weightless per-head norm — `norm::rmsnorm_no_scale_bf16`.
///
/// # Safety
///
/// `x` and `y` must address `rows * width` live bf16 elements, and `stream`
/// must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn rmsnorm_no_scale_bf16(
    x: *const bf16,
    y: *mut bf16,
    rows: i32,
    width: i32,
    per_head_dim: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    let hidden = if per_head_dim == 0 { width } else { per_head_dim };
    let launch = match rows_per_head(rows, width, per_head_dim) {
        Ok(l) => l,
        Err(r) => return Fired::Declined(r),
    };
    if launch.empty() {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    unsafe {
        rmsnorm::raw::rmsnorm_no_scale("norm::rmsnorm_no_scale_bf16", launch, x, y, hidden, eps, stream);
    }
    Fired::Launched
}

/// The gated norm — `norm::rmsnorm_gated_bf16`.
///
/// # Safety
///
/// `x`, `gate` and `y` must address `rows * width` live bf16 elements,
/// `weight` `hidden` live floats, and `stream` must be live across the
/// launch.
#[cfg(feature = "_cuda")]
pub unsafe fn rmsnorm_gated_bf16(
    x: *const bf16,
    gate: *const bf16,
    weight: *const f32,
    y: *mut bf16,
    rows: i32,
    width: i32,
    per_head_dim: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    let hidden = if per_head_dim == 0 { width } else { per_head_dim };
    let launch = match rows_per_head(rows, width, per_head_dim) {
        Ok(l) => l,
        Err(r) => return Fired::Declined(r),
    };
    if launch.empty() {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    unsafe {
        rmsnorm::raw::rmsnorm_gated(
            "norm::rmsnorm_gated_bf16",
            launch,
            x,
            gate,
            weight,
            y,
            hidden,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// The gated norm with an fp32 INPUT — `norm::rmsnorm_gated_fp32_in_bf16`.
///
/// # Safety
///
/// `x` must address `rows * width` live floats, `gate` and `y` the same count
/// of bf16, `weight` `hidden` live floats, and `stream` must be live across
/// the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn rmsnorm_gated_fp32_in_bf16(
    x: *const f32,
    gate: *const bf16,
    weight: *const f32,
    y: *mut bf16,
    rows: i32,
    width: i32,
    per_head_dim: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    let hidden = if per_head_dim == 0 { width } else { per_head_dim };
    let launch = match rows_per_head(rows, width, per_head_dim) {
        Ok(l) => l,
        Err(r) => return Fired::Declined(r),
    };
    if launch.empty() {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    unsafe {
        rmsnorm::raw::rmsnorm_gated_f32_in(
            "norm::rmsnorm_gated_fp32_in_bf16",
            launch,
            x,
            gate,
            weight,
            y,
            hidden,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// The residual add and the NEXT block's pre-norm, fused —
///
/// # Safety
///
/// `hidden`, `residual`, `norm_out` must address `num_rows * hidden_size`
/// live bf16 elements and `weight` `hidden_size` of them; `stream` must be
/// live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn residual_add_rmsnorm_bf16(
    hidden: *mut bf16,
    residual: *const bf16,
    weight: *const bf16,
    norm_out: *mut bf16,
    num_rows: i32,
    hidden_size: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if num_rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    unsafe {
        rmsnorm::raw::residual_add_rmsnorm(
            "norm::residual_add_rmsnorm_bf16",
            per_row(num_rows),
            hidden,
            residual,
            weight,
            norm_out,
            hidden_size,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// Norm, then add into the residual stream in place —
///
/// # Safety
///
/// [`residual_add_rmsnorm_bf16`]'s.
#[cfg(feature = "_cuda")]
pub unsafe fn rmsnorm_residual_add_bf16(
    x: *const bf16,
    weight: *const bf16,
    hidden: *mut bf16,
    num_rows: i32,
    hidden_size: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if num_rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    unsafe {
        rmsnorm::raw::rmsnorm_residual_add(
            "norm::rmsnorm_residual_add_bf16",
            per_row(num_rows),
            x,
            weight,
            hidden,
            hidden_size,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// `rmsnorm.cu:119` — `norm::rmsnorm_residual_add_scale_rmsnorm_bf16`, all
///
/// # Safety
///
/// `x`, `weight`, `hidden`, `next_weight` and `norm_out` must address live
/// device memory of `num_rows * hidden_size` (the two weights,
/// `hidden_size`) bf16 elements, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn rmsnorm_residual_add_scale_rmsnorm_bf16(
    x: *const bf16,
    weight: *const bf16,
    hidden: *mut bf16,
    scale: f32,
    next_weight: *const bf16,
    norm_out: *mut bf16,
    num_rows: i32,
    hidden_size: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if num_rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    let rows = num_rows.unsigned_abs();
    let vec_ok = hidden_size % 8 == 0
        && crate::x::fire::aligned16(x.cast())
        && crate::x::fire::aligned16(hidden.cast_const().cast())
        && crate::x::fire::aligned16(norm_out.cast_const().cast())
        && crate::x::fire::aligned16(weight.cast())
        && crate::x::fire::aligned16(next_weight.cast());
    if vec_ok {
        let (symbol, block) = if hidden_size >= RASR_VEC512_ABOVE {
            ("norm::rmsnorm_residual_add_scale_rmsnorm_bf16#vec8_512", VBLOCK)
        } else {
            ("norm::rmsnorm_residual_add_scale_rmsnorm_bf16#vec8_256", BLOCK)
        };
        unsafe {
            rmsnorm::raw::rmsnorm_rasr_vec8(
                symbol,
                Launch::per_row(rows, block),
                x,
                weight,
                hidden,
                scale,
                next_weight,
                norm_out,
                hidden_size,
                eps,
                stream,
            );
        }
        return Fired::Launched;
    }
    unsafe {
        rmsnorm::raw::rmsnorm_residual_add_scale_rmsnorm(
            "norm::rmsnorm_residual_add_scale_rmsnorm_bf16#scalar_512",
            Launch::per_row(rows, VBLOCK),
            x,
            weight,
            hidden,
            scale,
            next_weight,
            norm_out,
            hidden_size,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// `out[row][i] += bias[i]` — `norm::add_bias_bf16`.
///
/// # Safety
///
/// `out` must address `num_rows * dim` live bf16 elements, `bias` `dim` of
/// them, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn add_bias_bf16(
    out: *mut bf16,
    bias: *const bf16,
    num_rows: i32,
    dim: i32,
    stream: *mut c_void,
) -> Fired {
    if num_rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    if dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the bias width" });
    }
    unsafe {
        add_bias::raw::add_bias(
            "norm::add_bias_bf16",
            route_rows(num_rows, dim),
            out,
            bias,
            dim,
            stream,
        );
    }
    Fired::Launched
}

/// gemma-3n's altup predict — `norm::altup_predict_bf16`.
///
/// # Safety
///
/// `streams` and `predictions` must address `k * t_len * h` live bf16
/// elements, `coefs` `t_len * k * k` live floats, and `stream` must be live
/// across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn altup_predict_bf16(
    streams: *const bf16,
    coefs: *const f32,
    predictions: *mut bf16,
    k: i32,
    t_len: i32,
    h: i32,
    stream: *mut c_void,
) -> Fired {
    if t_len <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    if k <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the altup stream count" });
    }
    unsafe {
        altup::raw::altup_predict(
            "norm::altup_predict_bf16",
            altup_streams(t_len, k, h),
            streams,
            coefs,
            predictions,
            k,
            t_len,
            h,
            stream,
        );
    }
    Fired::Launched
}

/// gemma-3n's altup correct — `norm::altup_correct_bf16`.
///
/// # Safety
///
/// [`altup_predict_bf16`]'s, with `activated` addressing `t_len * h` live
/// bf16 elements and `corrected` `k * t_len * h`.
#[cfg(feature = "_cuda")]
pub unsafe fn altup_correct_bf16(
    predictions: *const bf16,
    activated: *const bf16,
    correction_coefs_plus_one: *const f32,
    corrected: *mut bf16,
    k: i32,
    t_len: i32,
    h: i32,
    active_idx: i32,
    stream: *mut c_void,
) -> Fired {
    if t_len <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    if k <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the altup stream count" });
    }
    unsafe {
        altup::raw::altup_correct(
            "norm::altup_correct_bf16",
            altup_streams(t_len, k, h),
            predictions,
            activated,
            correction_coefs_plus_one,
            corrected,
            k,
            t_len,
            h,
            active_idx,
            stream,
        );
    }
    Fired::Launched
}

/// The per-row RMS of the reference stream — `norm::compute_rms_bf16`.
///
/// # Safety
///
/// `reference` must address `rows * h` live bf16 elements, `out` `rows` live
/// floats, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn compute_rms_bf16(
    reference: *const bf16,
    out: *mut f32,
    rows: i32,
    h: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    unsafe {
        altup_aux::raw::compute_rms(
            "norm::compute_rms_bf16",
            per_row_reducing(rows),
            reference,
            out,
            h,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// Rescale each row to a stated RMS, in place — `norm::magnitude_rescale_bf16`.
///
/// # Safety
///
/// `x` must address `rows * h` live bf16 elements, `target_rms` `rows` live
/// floats, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn magnitude_rescale_bf16(
    x: *mut bf16,
    target_rms: *const f32,
    rows: i32,
    h: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    unsafe {
        altup_aux::raw::magnitude_rescale(
            "norm::magnitude_rescale_bf16",
            per_row_reducing(rows),
            x,
            target_rms,
            h,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// The mean over altup's `k` streams — `norm::mean_streams_bf16`.
///
/// # Safety
///
/// `streams` must address `k * t_stride * h` live bf16 elements, `out`
/// `rows * h` of them, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn mean_streams_bf16(
    streams: *const bf16,
    out: *mut bf16,
    k: i32,
    rows: i32,
    h: i32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    if k <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the altup stream count" });
    }
    unsafe {
        altup_aux::raw::mean_streams(
            "norm::mean_streams_bf16",
            elementwise_rows(rows, h),
            streams,
            out,
            k,
            rows,
            h,
            stream,
        );
    }
    Fired::Launched
}

/// bf16 predict coefficients widened to `float` —
///
/// # Safety
///
/// `in_bf16` must address `rows * k * k` live bf16 elements, `out` the same
/// count of floats, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn altup_unpack_predict_coefs(
    in_bf16: *const bf16,
    out: *mut f32,
    rows: i32,
    k: i32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    if k <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the altup stream count" });
    }
    unsafe {
        altup_aux::raw::unpack_predict_coefs(
            "norm::altup_unpack_predict_coefs",
            route_rows(rows, k.saturating_mul(k)),
            in_bf16,
            out,
            k,
            stream,
        );
    }
    Fired::Launched
}

/// bf16 correct coefficients widened to `float` —
///
/// # Safety
///
/// `in_bf16` must address `rows * k` live bf16 elements, `out` the same count
/// of floats, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn altup_unpack_correct_coefs(
    in_bf16: *const bf16,
    out: *mut f32,
    rows: i32,
    k: i32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    if k <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the altup stream count" });
    }
    unsafe {
        altup_aux::raw::unpack_correct_coefs(
            "norm::altup_unpack_correct_coefs",
            route_rows(rows, k),
            in_bf16,
            out,
            k,
            stream,
        );
    }
    Fired::Launched
}

/// `tanh` in place over a bf16 slab — `norm::tanh_bf16`.
///
/// # Safety
///
/// `x` must address `n` live bf16 elements and `stream` must be live across
/// the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn tanh_bf16(x: *mut bf16, n: i32, stream: *mut c_void) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the element count" });
    }
    unsafe {
        altup_aux::raw::tanh_inplace("norm::tanh_bf16", elementwise(n), x, n, stream);
    }
    Fired::Launched
}

/// [`tanh_bf16`] over fp16 — `norm::tanh_f16`.
///
/// # Safety
///
/// [`tanh_bf16`]'s, with `x` addressing fp16.
#[cfg(feature = "_cuda")]
pub unsafe fn tanh_f16(x: *mut f16, n: i32, stream: *mut c_void) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the element count" });
    }
    unsafe {
        altup_aux::raw::tanh_inplace("norm::tanh_f16", elementwise(n), x, n, stream);
    }
    Fired::Launched
}

/// `y += x` — `norm::residual_add_bf16`.
///
/// # Safety
///
/// `y` and `x` must address `n` live bf16 elements and `stream` must be live
/// across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn residual_add_bf16(y: *mut bf16, x: *const bf16, n: usize, stream: *mut c_void) -> Fired {
    if n == 0 {
        return Fired::Declined(Refusal::Empty { what: "the element count" });
    }
    let launch = Launch::flat(u32::try_from(n).unwrap_or(u32::MAX), BLOCK);
    unsafe {
        elementwise::raw::residual_add("norm::residual_add_bf16", launch, y, x, n, stream);
    }
    Fired::Launched
}

/// [`residual_add_bf16`] over fp16 — `norm::residual_add_f16`.
///
/// # Safety
///
/// [`residual_add_bf16`]'s, with both pointers addressing fp16.
#[cfg(feature = "_cuda")]
pub unsafe fn residual_add_f16(y: *mut f16, x: *const f16, n: usize, stream: *mut c_void) -> Fired {
    if n == 0 {
        return Fired::Declined(Refusal::Empty { what: "the element count" });
    }
    let launch = Launch::flat(u32::try_from(n).unwrap_or(u32::MAX), BLOCK);
    unsafe {
        elementwise::raw::residual_add("norm::residual_add_f16", launch, y, x, n, stream);
    }
    Fired::Launched
}

/// `x *= s` — `norm::scalar_mul_bf16`.
///
/// # Safety
///
/// `x` must address `n` live bf16 elements and `stream` must be live across
/// the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn scalar_mul_bf16(x: *mut bf16, s: f32, n: usize, stream: *mut c_void) -> Fired {
    if n == 0 {
        return Fired::Declined(Refusal::Empty { what: "the element count" });
    }
    let launch = Launch::flat(u32::try_from(n).unwrap_or(u32::MAX), BLOCK);
    unsafe {
        elementwise::raw::scalar_mul("norm::scalar_mul_bf16", launch, x, s, n, stream);
    }
    Fired::Launched
}

/// `dsv4_hc.cu:22` — `norm::hc_pre_postprocess_bf16`.
///
/// # Safety
///
/// `residual` and `layer_input` must address `n * hc_mult * hidden_size` and
/// `n * hidden_size` live bf16 elements; `mixes`, `scale` and `base` the
/// slabs the layer carries; `post_mix` and `comb_mix` scratch of `n *
/// hc_mult` and `n * hc_mult * hc_mult` floats. `stream` must be live across
/// the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn hc_pre_postprocess_bf16(
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
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    if let Err(r) = hc_mult_ok(hc_mult) {
        return Fired::Declined(r);
    }
    unsafe {
        dsv4_hc::raw::hc_pre_postprocess(
            "norm::hc_pre_postprocess_rows_bf16",
            per_row(n),
            mixes,
            scale,
            base,
            residual,
            post_mix,
            comb_mix,
            layer_input,
            hc_mult,
            hidden_size,
            hc_eps,
            hc_post_alpha,
            sinkhorn_iters,
            stream,
        );
    }
    Fired::Launched
}

/// `dsv4_hc.cu:47` — `norm::hc_post_bf16`.
///
/// # Safety
///
/// [`hc_pre_postprocess_bf16`]'s, with `out_residual` addressing `n * hc_mult
/// * hidden_size` live bf16 elements.
#[cfg(feature = "_cuda")]
pub unsafe fn hc_post_bf16(
    x: *const bf16,
    residual: *const bf16,
    post_mix: *const f32,
    comb_mix: *const f32,
    out_residual: *mut bf16,
    n: i32,
    hc_mult: i32,
    hidden_size: i32,
    stream: *mut c_void,
) -> Fired {
    if let Err(r) = hc_mult_ok(hc_mult) {
        return Fired::Declined(r);
    }
    let total = i64::from(n) * i64::from(hidden_size);
    if total <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    unsafe {
        dsv4_hc::raw::hc_post(
            "norm::hc_post_elems_bf16",
            elementwise_wide(total),
            x,
            residual,
            post_mix,
            comb_mix,
            out_residual,
            n,
            hc_mult,
            hidden_size,
            stream,
        );
    }
    Fired::Launched
}

/// `dsv4_hc.cu:69` — `norm::hc_head_postprocess_bf16`.
///
/// # Safety
///
/// [`hc_pre_postprocess_bf16`]'s, with `out` addressing `n * hidden_size`
/// live bf16 elements.
#[cfg(feature = "_cuda")]
pub unsafe fn hc_head_postprocess_bf16(
    mixes: *const f32,
    scale: *const f32,
    base: *const f32,
    residual: *const bf16,
    out: *mut bf16,
    n: i32,
    hc_mult: i32,
    hidden_size: i32,
    hc_eps: f32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    if let Err(r) = hc_mult_ok(hc_mult) {
        return Fired::Declined(r);
    }
    unsafe {
        dsv4_hc::raw::hc_head_postprocess(
            "norm::hc_head_postprocess_rows_bf16",
            per_row(n),
            mixes,
            scale,
            base,
            residual,
            out,
            hc_mult,
            hidden_size,
            hc_eps,
            stream,
        );
    }
    Fired::Launched
}

/// `[n, hidden] -> [n, hc_mult, hidden]` — `norm::hc_expand_bf16`.
///
/// # Safety
///
/// `input` must address `n * hidden_size` live bf16 elements, `output`
/// `n * hc_mult * hidden_size` of them, and `stream` must be live across the
/// launch.
#[cfg(feature = "_cuda")]
pub unsafe fn hc_expand_bf16(
    input: *const bf16,
    output: *mut bf16,
    n: i32,
    hc_mult: i32,
    hidden_size: i32,
    stream: *mut c_void,
) -> Fired {
    if let Err(r) = hc_mult_ok(hc_mult) {
        return Fired::Declined(r);
    }
    let total = i64::from(n) * i64::from(hidden_size);
    if total <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    unsafe {
        dsv4_hc::raw::hc_expand(
            "norm::hc_expand_bf16",
            elementwise_wide(total),
            input,
            output,
            n,
            hc_mult,
            hidden_size,
            stream,
        );
    }
    Fired::Launched
}

/// `dsv4_hc.cu:89` — `norm::hc_rmsnorm_to_f32`.
///
/// # Safety
///
/// `input` must address `n * dim` live bf16 elements, `output` `n * dim` live
/// floats, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn hc_rmsnorm_to_f32(
    input: *const bf16,
    output: *mut f32,
    n: i32,
    dim: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    unsafe {
        dsv4_hc::raw::hc_rmsnorm_to_f32(
            "norm::hc_rmsnorm_to_f32_rows",
            per_row(n),
            input,
            output,
            dim,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// The attention sink's log-sum-exp correction —
///
/// # Safety
///
/// `out` must address `n * num_heads * head_dim` live bf16 elements, `lse`
/// and `sink` `n * num_heads` and `num_heads` live floats, and `stream` must
/// be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn attn_sink_correction_bf16(
    out: *mut bf16,
    lse: *const f32,
    sink: *const f32,
    n: i32,
    num_heads: i32,
    head_dim: i32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    if num_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the head count" });
    }
    unsafe {
        dsv4_hc::raw::attn_sink_correction(
            "norm::attn_sink_correction_bf16",
            gated_rms(n, num_heads),
            out,
            lse,
            sink,
            num_heads,
            head_dim,
            stream,
        );
    }
    Fired::Launched
}

/// QK-norm in place over a packed head axis —
///
/// # Safety
///
/// `q` must address `n * num_heads * head_dim` live bf16 elements and
/// `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn per_head_rmsnorm_bf16(
    q: *mut bf16,
    n: i32,
    num_heads: i32,
    head_dim: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_rows" });
    }
    if num_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the head count" });
    }
    unsafe {
        dsv4_hc::raw::per_head_rmsnorm(
            "norm::per_head_rmsnorm_bf16",
            gated_rms(n, num_heads),
            q,
            head_dim,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// `hc_mult <= MAX_HC_MULT`, as a refusal.
#[cfg(feature = "_cuda")]
fn hc_mult_ok(hc_mult: i32) -> Result<(), Refusal> {
    if hc_mult > MAX_HC_MULT {
        return Err(Refusal::Wide {
            what: "hc_mult, which `hc_post` unrolls into a register array",
            at: hc_mult,
            max: MAX_HC_MULT,
        });
    }
    Ok(())
}

contract! {
    /// The strided norm — the one `fire/rmsnorm.rs` vectorised.
    RMSNORM_STRIDED = "norm::rmsnorm_strided_bf16" as rmsnorm_strided

    /// The plain RMSNorm, one of the two `OpKind::Rmsnorm` fans to.
    RMSNORM = "norm::rmsnorm_bf16" as rmsnorm

    /// gemma's, folding `(1 + w)`.
    RMSNORM_GEMMA = "norm::rmsnorm_gemma_bf16" as rmsnorm_gemma

    /// The norm that also writes an fp16 copy.
    RMSNORM_WITH_FP16 = "norm::rmsnorm_bf16_with_fp16" as rmsnorm_with_fp16

    /// The weightless per-head norm.
    RMSNORM_NO_SCALE = "norm::rmsnorm_no_scale_bf16" as rmsnorm_no_scale {
        in_place: &[(0, 0)],
    }

    /// qwen3.5's gated norm in its own launch.
    RMSNORM_GATED_LAUNCH = "norm::rmsnorm_gated_bf16" as rmsnorm_gated_launch

    /// The gated norm reading an fp32 core output.
    RMSNORM_GATED_FP32_IN = "norm::rmsnorm_gated_fp32_in_bf16" as rmsnorm_gated_fp32_in

    /// Residual add and the next block's pre-norm, fused.
    RESIDUAL_ADD_RMSNORM = "norm::residual_add_rmsnorm_bf16" as residual_add_rmsnorm

    /// Norm, then add into the residual stream.
    NORM_RESIDUAL_ADD = "norm::rmsnorm_residual_add_bf16" as norm_residual_add {
        in_place: &[(0, 1)],
    }

    /// gemma-4's four-statements-in-one.
    NORM_RESIDUAL_SCALE_NORM = "norm::rmsnorm_residual_add_scale_rmsnorm_bf16"
        as norm_residual_scale_norm {
        in_place: &[(0, 1)],
    }

    /// `out[row][i] += bias[i]`.
    ADD_BIAS = "norm::add_bias_bf16" as add_bias {
        in_place: &[(0, 0)],
    }

    /// HC's RMSNorm into `float`.
    HC_RMSNORM_TO_F32 = "norm::hc_rmsnorm_to_f32" as hc_rmsnorm_to_f32

    /// Where a rank-K residual begins.
    HC_EXPAND = "norm::hc_expand_bf16" as hc_expand

    /// The per-token sinkhorn mixing matrix.
    HC_PRE = "norm::hc_pre_postprocess_bf16" as hc_pre

    /// The write-back half.
    HC_POST = "norm::hc_post_bf16" as hc_post

    /// The final collapse, for the LM head.
    HC_HEAD = "norm::hc_head_postprocess_bf16" as hc_head

    /// QK-norm where q lies.
    PER_HEAD_RMSNORM = "norm::per_head_rmsnorm_bf16" as per_head_rmsnorm {
        in_place: &[(0, 0)],
    }

    /// The attention sink's log-sum-exp correction.
    ATTN_SINK_CORRECTION = "norm::attn_sink_correction_bf16" as attn_sink_correction {
        in_place: &[(0, 0)],
    }

    /// AltUp's prediction step.
    ALTUP_PREDICT = "norm::altup_predict_bf16" as altup_predict

    /// AltUp's correction step.
    ALTUP_CORRECT = "norm::altup_correct_bf16" as altup_correct

    /// The `K*K` predict coefficients, bf16 to `float`.
    ALTUP_UNPACK_PREDICT_COEFS = "norm::altup_unpack_predict_coefs"
        as altup_unpack_predict_coefs

    /// The `K` correct coefficients, bf16 to `float`.
    ALTUP_UNPACK_CORRECT_COEFS = "norm::altup_unpack_correct_coefs"
        as altup_unpack_correct_coefs

    /// The mean over AltUp's streams.
    MEAN_STREAMS = "norm::mean_streams_bf16" as mean_streams

    /// The reference stream's per-row RMS.
    COMPUTE_RMS = "norm::compute_rms_bf16" as compute_rms

    /// The magnitude hold.
    MAGNITUDE_RESCALE = "norm::magnitude_rescale_bf16" as magnitude_rescale {
        in_place: &[(0, 0)],
    }

    /// `x *= s`.
    SCALAR_MUL = "norm::scalar_mul_bf16" as scalar_mul {
        in_place: &[(0, 0)],
    }

    /// `y += x`.
    RESIDUAL_ADD_CUDA = "norm::residual_add_bf16" as residual_add_cuda {
        in_place: &[(0, 0)],
    }

    /// `tanh` in place.
    TANH = "norm::tanh_bf16" as tanh {
        in_place: &[(0, 0)],
    }
}

#[cfg(feature = "_cuda")]
bind! {
    RMSNORM_STRIDED => { cx, stream => {
        unsafe {
            strided_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                cx.in_width(0)?,
                cx.out_width(0)?,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    RMSNORM => { none: "Cx has no query for the statement's per-head width. \
        The deleted row read Source::IfPresent(PerHeadDim, ..) on both num_rows \
        and hidden, because OpKind::RmsnormPerHead lowers to this same symbol \
        and norms rows x (width / head_dim) rows of head_dim where the plain \
        kind norms rows of width; without the query this fn would norm \
        gemma-4's q/k heads as one row each. Needs `Facts::per_head_dim() -> \
        Option<i32>` over LaunchSpec::per_head_dim (bind/mod.rs:1798), which \
        the driver already holds" },

    RMSNORM_GEMMA => { none: "Cx has no query for the statement's per-head \
        width, exactly as for norm::rmsnorm_bf16 — same operand contract, \
        different arithmetic. Needs `Facts::per_head_dim()`" },

    RMSNORM_WITH_FP16 => { none: "The deleted row stated no Source on any of \
        its eight operands, so there is nothing to read a binding from: it \
        described the launcher's C signature and never said where y_fp16, or \
        anything else, comes from. The host program above is complete and \
        proven; what is missing is a statement that names the fp16 copy. Needs \
        a lowering that produces two results, and then Source::Out(1)" },

    RMSNORM_NO_SCALE => { none: "Cx has no query for the statement's per-head \
        width; this is the V-norm and the per-head reading is the only one it \
        is ever fired with. Needs `Facts::per_head_dim()`" },

    RMSNORM_GATED_LAUNCH => { none: "Cx has no query for the statement's \
        per-head width. Needs `Facts::per_head_dim()`" },

    RMSNORM_GATED_FP32_IN => { none: "Cx has no query for the gated-delta-net \
        head width. The deleted row bound hidden from Source::Gdn(\"v_d\") and \
        families/norm.rs records the correction it wanted -- spec.per_head_dim \
        set from gdn.v_d where the statement is a gated norm -- so this needs \
        `Facts::per_head_dim()` and the driver-side assignment, not a new \
        Source" },

    RESIDUAL_ADD_RMSNORM => { cx, stream => {
        unsafe {
            residual_add_rmsnorm_bf16(
                cx.arg_in(0)?.cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    NORM_RESIDUAL_ADD => { cx, stream => {
        unsafe {
            rmsnorm_residual_add_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    NORM_RESIDUAL_SCALE_NORM => { none: "Cx has no query for the layer's \
        residual scale. Every other operand of gemma-4's fused landing is \
        available -- two weights, two results, the row count and the width -- \
        and the one that is not is Source::LayerScale, the per-layer constant \
        the binder reads off the model. Needs `Facts::layer_scale() -> \
        Option<f32>`. This is the family's most expensive refusal: the host \
        program above is the three-arm vectorised form measured at -38%, -49% \
        and -53% against the shipping scalar kernel" },

    ADD_BIAS => { cx, stream => {
        unsafe {
            add_bias_bf16(
                cx.arg_out(0)?.cast::<bf16>(),
                cx.weight_named(0)?.cast_const().cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                stream,
            )
        }
        .ok()
    }},

    HC_RMSNORM_TO_F32 => { cx, stream => {
        unsafe {
            hc_rmsnorm_to_f32(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<f32>(),
                cx.rows().count,
                cx.out_width(0)?,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    HC_EXPAND => { cx, stream => {
        let hidden = cx.in_width(0)?;
        if hidden <= 0 {
            return Err(Refusal::Empty { what: "the hidden width" });
        }
        unsafe {
            hc_expand_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)? / hidden,
                hidden,
                stream,
            )
        }
        .ok()
    }},

    HC_PRE => { none: "Cx has no query for the three float slabs a \
        hyper-connection layer carries -- mixes, scale and base -- nor for the \
        two scratch buffers this kernel hands to norm::hc_post_bf16, nor for \
        the model constants sinkhorn_iters and hc_post_alpha. The deleted row \
        stated no Source on any of its thirteen operands and that was the \
        honest spelling: a half-bound row generates exactly as much as an \
        unbound one while claiming bindings nobody checked. Needs a lowering \
        that states the slabs, which is a design question and not an accessor" },

    HC_POST => { cx, stream => {
        let hidden = cx.in_width(0)?;
        if hidden <= 0 {
            return Err(Refusal::Empty { what: "the hidden width" });
        }
        unsafe {
            hc_post_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.arg_in(2)?.cast_const().cast::<f32>(),
                cx.arg_in(3)?.cast_const().cast::<f32>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)? / hidden,
                hidden,
                stream,
            )
        }
        .ok()
    }},

    HC_HEAD => { none: "Cx has no query for the three float slabs a \
        hyper-connection layer carries -- mixes, scale and base -- or for \
        hc_eps. Same shape as norm::hc_pre_postprocess_bf16 and the same \
        answer" },

    PER_HEAD_RMSNORM => { cx, stream => {
        let head_dim = cx.head_dim()?;
        if head_dim <= 0 {
            return Err(Refusal::Narrow { what: "head_dim", at: head_dim });
        }
        unsafe {
            per_head_rmsnorm_bf16(
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)? / head_dim,
                head_dim,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    ATTN_SINK_CORRECTION => { cx, stream => {
        let head_dim = cx.head_dim()?;
        if head_dim <= 0 {
            return Err(Refusal::Narrow { what: "head_dim", at: head_dim });
        }
        unsafe {
            attn_sink_correction_bf16(
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.weight(0)?.cast_const().cast::<f32>(),
                cx.rows().count,
                cx.out_width(0)? / head_dim,
                head_dim,
                stream,
            )
        }
        .ok()
    }},

    ALTUP_PREDICT => { none: "Cx has no query for the AltUp stream count. \
        `streams` is [t, k*h] with the streams interleaved, so only the fire \
        knows how that row divides and the deleted row read \
        Source::Ctx(\"altup_streams\"); DispatchCtx::altup_streams \
        (bind/mod.rs:1244) is the accessor. Needs `Facts::altup_streams() -> \
        Option<i32>`" },

    ALTUP_CORRECT => { none: "Cx has no query for which AltUp stream was run \
        through the real layer. Every extent on this statement comes off its \
        own values -- k from input 2's width, h from input 1's -- and the one \
        that does not is active_idx, DispatchCtx::altup_active \
        (bind/mod.rs:1246). Needs `Facts::altup_active() -> Option<i32>`" },

    ALTUP_UNPACK_PREDICT_COEFS => { cx, stream => {
        unsafe {
            altup_unpack_predict_coefs(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<f32>(),
                cx.rows().count,
                isqrt_exact(cx.in_width(0)?),
                stream,
            )
        }
        .ok()
    }},

    ALTUP_UNPACK_CORRECT_COEFS => { cx, stream => {
        unsafe {
            altup_unpack_correct_coefs(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<f32>(),
                cx.rows().count,
                cx.in_width(0)?,
                stream,
            )
        }
        .ok()
    }},

    MEAN_STREAMS => { none: "Cx has no query for the AltUp stream count, and \
        here it is not an extent at all: the streams arrive interleaved and \
        only the fire knows how the row divides, which is why the deleted row \
        said CtxNonZero rather than Ctx -- declining is better than dividing \
        by zero. Needs `Facts::altup_streams()`" },

    COMPUTE_RMS => { cx, stream => {
        unsafe {
            compute_rms_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<f32>(),
                cx.rows().count,
                cx.in_width(0)?,
                ALTUP_EPS,
                stream,
            )
        }
        .ok()
    }},

    MAGNITUDE_RESCALE => { cx, stream => {
        unsafe {
            magnitude_rescale_bf16(
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.rows().count,
                cx.out_width(0)?,
                ALTUP_EPS,
                stream,
            )
        }
        .ok()
    }},

    SCALAR_MUL => { none: "Cx can read a stated scale but not a named one. \
        The deleted row said Source::Or(ParamF32(0), NamedScale): a statement \
        that carries the number binds today through Cx::param_f32(0), and one \
        that carries a NAME -- which is what gemma-3n and gemma-2 state -- has \
        nowhere to read it from. Binding only the first half would make this \
        symbol work for some models and refuse at fire for exactly the two the \
        deleted row named, which is worse than refusing at load. Needs \
        `Facts::named_scale() -> Option<f32>`" },

    RESIDUAL_ADD_CUDA => { cx, stream => {
        let n = usize::try_from(elements(cx)?).unwrap_or(0);
        unsafe {
            residual_add_bf16(
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                n,
                stream,
            )
        }
        .ok()
    }},

    TANH => { cx, stream => {
        unsafe { tanh_bf16(cx.arg_out(0)?.cast::<bf16>(), elements(cx)?, stream) }.ok()
    }},
}

/// `Source::OutElements(0)` — the region's rows times the result's width.
#[cfg(feature = "_cuda")]
fn elements(cx: &crate::x::Cx<'_>) -> Result<i32, Refusal> {
    Ok(cx.rows().count.saturating_mul(cx.out_width(0)?))
}

/// `Source::Isqrt` — the exact integer square root, or `0`.
#[cfg(feature = "_cuda")]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn isqrt_exact(n: i32) -> i32 {
    if n <= 0 {
        return 0;
    }
    let mut r = f64::from(n).sqrt() as i32;
    while r > 0 && r.saturating_mul(r) > n {
        r -= 1;
    }
    while (r + 1).saturating_mul(r + 1) <= n {
        r += 1;
    }
    if r.saturating_mul(r) == n {
        r
    } else {
        0
    }
}

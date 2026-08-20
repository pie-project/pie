//! The affine-quantised projections, and the codecs around them.
//!
//! This is 32 of the 99 rows and 304 of the 480 entrypoints, and the whole
//! argument of `.wiki/kernel-metal-refactor.md` §2 is visible here: `qmm_t` is
//! ONE template body in `quantized_qmm_t.wgsl` stamped over (group x bits x
//! row tile x column tile), and enumerating its 54 instantiations as 54 rows
//! would state the macro's job a second time by hand.
//!
//! The five `_wm_`/`_wn_` rows are the exception that proves it: they are
//! `host_name` lines typed out at `quantized_qmm_t.wgsl:2918-2966` rather
//! than stamped, so they are five kernels and get five rows.

use kernels_macros::routine;

use crate::routine::{Asks, Bind, Const, Ctx, Fire, In, Out, Tensor, bf16, elementwise, elementwise_rows, f16, keys};
use kernels::routine::Refusal;

/// `affine_qmm_t`, indexed by `qmm_point`.
static QMM_T: [&str; 54] = [
    "affine_qmm_t_bfloat16_gs_32_b_4_bm_16_bn_16",
    "affine_qmm_t_bfloat16_gs_32_b_4_bm_16_bn_32",
    "affine_qmm_t_bfloat16_gs_32_b_4_bm_16_bn_64",
    "affine_qmm_t_bfloat16_gs_32_b_4_bm_32_bn_16",
    "affine_qmm_t_bfloat16_gs_32_b_4_bm_32_bn_32",
    "affine_qmm_t_bfloat16_gs_32_b_4_bm_32_bn_64",
    "affine_qmm_t_bfloat16_gs_32_b_4_bm_64_bn_16",
    "affine_qmm_t_bfloat16_gs_32_b_4_bm_64_bn_32",
    "affine_qmm_t_bfloat16_gs_32_b_4_bm_64_bn_64",
    "affine_qmm_t_bfloat16_gs_32_b_8_bm_16_bn_16",
    "affine_qmm_t_bfloat16_gs_32_b_8_bm_16_bn_32",
    "affine_qmm_t_bfloat16_gs_32_b_8_bm_16_bn_64",
    "affine_qmm_t_bfloat16_gs_32_b_8_bm_32_bn_16",
    "affine_qmm_t_bfloat16_gs_32_b_8_bm_32_bn_32",
    "affine_qmm_t_bfloat16_gs_32_b_8_bm_32_bn_64",
    "affine_qmm_t_bfloat16_gs_32_b_8_bm_64_bn_16",
    "affine_qmm_t_bfloat16_gs_32_b_8_bm_64_bn_32",
    "affine_qmm_t_bfloat16_gs_32_b_8_bm_64_bn_64",
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_16_bn_16",
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_16_bn_64",
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_16",
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_64",
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_16",
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32",
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64",
    "affine_qmm_t_bfloat16_gs_64_b_8_bm_16_bn_16",
    "affine_qmm_t_bfloat16_gs_64_b_8_bm_16_bn_32",
    "affine_qmm_t_bfloat16_gs_64_b_8_bm_16_bn_64",
    "affine_qmm_t_bfloat16_gs_64_b_8_bm_32_bn_16",
    "affine_qmm_t_bfloat16_gs_64_b_8_bm_32_bn_32",
    "affine_qmm_t_bfloat16_gs_64_b_8_bm_32_bn_64",
    "affine_qmm_t_bfloat16_gs_64_b_8_bm_64_bn_16",
    "affine_qmm_t_bfloat16_gs_64_b_8_bm_64_bn_32",
    "affine_qmm_t_bfloat16_gs_64_b_8_bm_64_bn_64",
    "affine_qmm_t_bfloat16_gs_128_b_4_bm_16_bn_16",
    "affine_qmm_t_bfloat16_gs_128_b_4_bm_16_bn_32",
    "affine_qmm_t_bfloat16_gs_128_b_4_bm_16_bn_64",
    "affine_qmm_t_bfloat16_gs_128_b_4_bm_32_bn_16",
    "affine_qmm_t_bfloat16_gs_128_b_4_bm_32_bn_32",
    "affine_qmm_t_bfloat16_gs_128_b_4_bm_32_bn_64",
    "affine_qmm_t_bfloat16_gs_128_b_4_bm_64_bn_16",
    "affine_qmm_t_bfloat16_gs_128_b_4_bm_64_bn_32",
    "affine_qmm_t_bfloat16_gs_128_b_4_bm_64_bn_64",
    "affine_qmm_t_bfloat16_gs_128_b_8_bm_16_bn_16",
    "affine_qmm_t_bfloat16_gs_128_b_8_bm_16_bn_32",
    "affine_qmm_t_bfloat16_gs_128_b_8_bm_16_bn_64",
    "affine_qmm_t_bfloat16_gs_128_b_8_bm_32_bn_16",
    "affine_qmm_t_bfloat16_gs_128_b_8_bm_32_bn_32",
    "affine_qmm_t_bfloat16_gs_128_b_8_bm_32_bn_64",
    "affine_qmm_t_bfloat16_gs_128_b_8_bm_64_bn_16",
    "affine_qmm_t_bfloat16_gs_128_b_8_bm_64_bn_32",
    "affine_qmm_t_bfloat16_gs_128_b_8_bm_64_bn_64",
];

/// `affine_qmm_t_bias`, indexed by `qmm_point`.
static QMM_T_BIAS: [&str; 54] = [
    "affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_16_bn_16",
    "affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_16_bn_32",
    "affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_16_bn_64",
    "affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_32_bn_16",
    "affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_32_bn_32",
    "affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_32_bn_64",
    "affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_64_bn_16",
    "affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_64_bn_32",
    "affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_64_bn_64",
    "affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_16_bn_16",
    "affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_16_bn_32",
    "affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_16_bn_64",
    "affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_32_bn_16",
    "affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_32_bn_32",
    "affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_32_bn_64",
    "affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_64_bn_16",
    "affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_64_bn_32",
    "affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_64_bn_64",
    "affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_16_bn_16",
    "affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_16_bn_64",
    "affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_32_bn_16",
    "affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_32_bn_64",
    "affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_64_bn_16",
    "affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_64_bn_32",
    "affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_64_bn_64",
    "affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_16_bn_16",
    "affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_16_bn_32",
    "affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_16_bn_64",
    "affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_32_bn_16",
    "affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_32_bn_32",
    "affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_32_bn_64",
    "affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_64_bn_16",
    "affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_64_bn_32",
    "affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_64_bn_64",
    "affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_16_bn_16",
    "affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_16_bn_32",
    "affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_16_bn_64",
    "affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_32_bn_16",
    "affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_32_bn_32",
    "affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_32_bn_64",
    "affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_64_bn_16",
    "affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_64_bn_32",
    "affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_64_bn_64",
    "affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_16_bn_16",
    "affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_16_bn_32",
    "affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_16_bn_64",
    "affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_32_bn_16",
    "affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_32_bn_32",
    "affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_32_bn_64",
    "affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_64_bn_16",
    "affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_64_bn_32",
    "affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_64_bn_64",
];

/// `affine_qmm_t_residual`, indexed by `qmm_point`.
static QMM_T_RESIDUAL: [&str; 54] = [
    "affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_16_bn_16",
    "affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_16_bn_32",
    "affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_16_bn_64",
    "affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_32_bn_16",
    "affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_32_bn_32",
    "affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_32_bn_64",
    "affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_64_bn_16",
    "affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_64_bn_32",
    "affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_64_bn_64",
    "affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_16_bn_16",
    "affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_16_bn_32",
    "affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_16_bn_64",
    "affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_32_bn_16",
    "affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_32_bn_32",
    "affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_32_bn_64",
    "affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_64_bn_16",
    "affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_64_bn_32",
    "affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_64_bn_64",
    "affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_16_bn_16",
    "affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_16_bn_64",
    "affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_32_bn_16",
    "affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_32_bn_64",
    "affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_64_bn_16",
    "affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_64_bn_32",
    "affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_64_bn_64",
    "affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_16_bn_16",
    "affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_16_bn_32",
    "affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_16_bn_64",
    "affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_32_bn_16",
    "affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_32_bn_32",
    "affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_32_bn_64",
    "affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_64_bn_16",
    "affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_64_bn_32",
    "affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_64_bn_64",
    "affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_16_bn_16",
    "affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_16_bn_32",
    "affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_16_bn_64",
    "affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_32_bn_16",
    "affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_32_bn_32",
    "affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_32_bn_64",
    "affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_64_bn_16",
    "affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_64_bn_32",
    "affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_64_bn_64",
    "affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_16_bn_16",
    "affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_16_bn_32",
    "affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_16_bn_64",
    "affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_32_bn_16",
    "affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_32_bn_32",
    "affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_32_bn_64",
    "affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_64_bn_16",
    "affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_64_bn_32",
    "affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_64_bn_64",
];

/// `affine_qmm_t_fp16_precast`, indexed by `tile_point`. One group size and one bit width.
static QMM_T_FP16_PRECAST: [&str; 9] = [
    "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_16",
    "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_64",
    "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_16",
    "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_64",
    "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_16",
    "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32",
    "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_64",
];

/// `affine_qmm_t_bias_fp16_precast`, indexed by `tile_point`. One group size and one bit width.
static QMM_T_BIAS_FP16_PRECAST: [&str; 9] = [
    "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_16",
    "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_64",
    "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_16",
    "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_64",
    "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_16",
    "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32",
    "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_64",
];

/// `affine_qmm_t_residual_fp16_precast`, indexed by `tile_point`. One group size and one bit width.
static QMM_T_RESIDUAL_FP16_PRECAST: [&str; 9] = [
    "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_16",
    "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_64",
    "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_16",
    "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_64",
    "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_16",
    "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32",
    "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_64",
];

/// `affine_qmm_t_splitk`, indexed by `wide_point`. The column tile is 32 alone.
static QMM_T_SPLITK: [&str; 18] = [
    "affine_qmm_t_splitk_bfloat16_gs_32_b_4_bm_16_bn_32",
    "affine_qmm_t_splitk_bfloat16_gs_32_b_4_bm_32_bn_32",
    "affine_qmm_t_splitk_bfloat16_gs_32_b_4_bm_64_bn_32",
    "affine_qmm_t_splitk_bfloat16_gs_32_b_8_bm_16_bn_32",
    "affine_qmm_t_splitk_bfloat16_gs_32_b_8_bm_32_bn_32",
    "affine_qmm_t_splitk_bfloat16_gs_32_b_8_bm_64_bn_32",
    "affine_qmm_t_splitk_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_splitk_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_splitk_bfloat16_gs_64_b_4_bm_64_bn_32",
    "affine_qmm_t_splitk_bfloat16_gs_64_b_8_bm_16_bn_32",
    "affine_qmm_t_splitk_bfloat16_gs_64_b_8_bm_32_bn_32",
    "affine_qmm_t_splitk_bfloat16_gs_64_b_8_bm_64_bn_32",
    "affine_qmm_t_splitk_bfloat16_gs_128_b_4_bm_16_bn_32",
    "affine_qmm_t_splitk_bfloat16_gs_128_b_4_bm_32_bn_32",
    "affine_qmm_t_splitk_bfloat16_gs_128_b_4_bm_64_bn_32",
    "affine_qmm_t_splitk_bfloat16_gs_128_b_8_bm_16_bn_32",
    "affine_qmm_t_splitk_bfloat16_gs_128_b_8_bm_32_bn_32",
    "affine_qmm_t_splitk_bfloat16_gs_128_b_8_bm_64_bn_32",
];

/// `affine_qmm_t_splitk_f32`, indexed by `wide_point`. The column tile is 32 alone.
static QMM_T_SPLITK_F32: [&str; 18] = [
    "affine_qmm_t_splitk_f32_bfloat16_gs_32_b_4_bm_16_bn_32",
    "affine_qmm_t_splitk_f32_bfloat16_gs_32_b_4_bm_32_bn_32",
    "affine_qmm_t_splitk_f32_bfloat16_gs_32_b_4_bm_64_bn_32",
    "affine_qmm_t_splitk_f32_bfloat16_gs_32_b_8_bm_16_bn_32",
    "affine_qmm_t_splitk_f32_bfloat16_gs_32_b_8_bm_32_bn_32",
    "affine_qmm_t_splitk_f32_bfloat16_gs_32_b_8_bm_64_bn_32",
    "affine_qmm_t_splitk_f32_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_splitk_f32_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_splitk_f32_bfloat16_gs_64_b_4_bm_64_bn_32",
    "affine_qmm_t_splitk_f32_bfloat16_gs_64_b_8_bm_16_bn_32",
    "affine_qmm_t_splitk_f32_bfloat16_gs_64_b_8_bm_32_bn_32",
    "affine_qmm_t_splitk_f32_bfloat16_gs_64_b_8_bm_64_bn_32",
    "affine_qmm_t_splitk_f32_bfloat16_gs_128_b_4_bm_16_bn_32",
    "affine_qmm_t_splitk_f32_bfloat16_gs_128_b_4_bm_32_bn_32",
    "affine_qmm_t_splitk_f32_bfloat16_gs_128_b_4_bm_64_bn_32",
    "affine_qmm_t_splitk_f32_bfloat16_gs_128_b_8_bm_16_bn_32",
    "affine_qmm_t_splitk_f32_bfloat16_gs_128_b_8_bm_32_bn_32",
    "affine_qmm_t_splitk_f32_bfloat16_gs_128_b_8_bm_64_bn_32",
];

/// `affine_qmm_t_strided`, indexed by `wide_point`. The column tile is 32 alone.
static QMM_T_STRIDED: [&str; 18] = [
    "affine_qmm_t_strided_bfloat16_gs_32_b_4_bm_16_bn_32",
    "affine_qmm_t_strided_bfloat16_gs_32_b_4_bm_32_bn_32",
    "affine_qmm_t_strided_bfloat16_gs_32_b_4_bm_64_bn_32",
    "affine_qmm_t_strided_bfloat16_gs_32_b_8_bm_16_bn_32",
    "affine_qmm_t_strided_bfloat16_gs_32_b_8_bm_32_bn_32",
    "affine_qmm_t_strided_bfloat16_gs_32_b_8_bm_64_bn_32",
    "affine_qmm_t_strided_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_strided_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_strided_bfloat16_gs_64_b_4_bm_64_bn_32",
    "affine_qmm_t_strided_bfloat16_gs_64_b_8_bm_16_bn_32",
    "affine_qmm_t_strided_bfloat16_gs_64_b_8_bm_32_bn_32",
    "affine_qmm_t_strided_bfloat16_gs_64_b_8_bm_64_bn_32",
    "affine_qmm_t_strided_bfloat16_gs_128_b_4_bm_16_bn_32",
    "affine_qmm_t_strided_bfloat16_gs_128_b_4_bm_32_bn_32",
    "affine_qmm_t_strided_bfloat16_gs_128_b_4_bm_64_bn_32",
    "affine_qmm_t_strided_bfloat16_gs_128_b_8_bm_16_bn_32",
    "affine_qmm_t_strided_bfloat16_gs_128_b_8_bm_32_bn_32",
    "affine_qmm_t_strided_bfloat16_gs_128_b_8_bm_64_bn_32",
];

/// `affine_qmm_t_strided_residual`, indexed by `wide_point`. The column tile is 32 alone.
static QMM_T_STRIDED_RESIDUAL: [&str; 18] = [
    "affine_qmm_t_strided_residual_bfloat16_gs_32_b_4_bm_16_bn_32",
    "affine_qmm_t_strided_residual_bfloat16_gs_32_b_4_bm_32_bn_32",
    "affine_qmm_t_strided_residual_bfloat16_gs_32_b_4_bm_64_bn_32",
    "affine_qmm_t_strided_residual_bfloat16_gs_32_b_8_bm_16_bn_32",
    "affine_qmm_t_strided_residual_bfloat16_gs_32_b_8_bm_32_bn_32",
    "affine_qmm_t_strided_residual_bfloat16_gs_32_b_8_bm_64_bn_32",
    "affine_qmm_t_strided_residual_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_strided_residual_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_strided_residual_bfloat16_gs_64_b_4_bm_64_bn_32",
    "affine_qmm_t_strided_residual_bfloat16_gs_64_b_8_bm_16_bn_32",
    "affine_qmm_t_strided_residual_bfloat16_gs_64_b_8_bm_32_bn_32",
    "affine_qmm_t_strided_residual_bfloat16_gs_64_b_8_bm_64_bn_32",
    "affine_qmm_t_strided_residual_bfloat16_gs_128_b_4_bm_16_bn_32",
    "affine_qmm_t_strided_residual_bfloat16_gs_128_b_4_bm_32_bn_32",
    "affine_qmm_t_strided_residual_bfloat16_gs_128_b_4_bm_64_bn_32",
    "affine_qmm_t_strided_residual_bfloat16_gs_128_b_8_bm_16_bn_32",
    "affine_qmm_t_strided_residual_bfloat16_gs_128_b_8_bm_32_bn_32",
    "affine_qmm_t_strided_residual_bfloat16_gs_128_b_8_bm_64_bn_32",
];

/// `affine_qmm_t_splitk_fp16_precast`, indexed by `row_tile_point`.
static QMM_T_SPLITK_FP16_PRECAST: [&str; 3] = [
    "affine_qmm_t_splitk_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_splitk_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_splitk_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32",
];

/// `affine_qmm_t_splitk_fp16_precast_f32`, indexed by `row_tile_point`.
static QMM_T_SPLITK_FP16_PRECAST_F32: [&str; 3] = [
    "affine_qmm_t_splitk_fp16_precast_f32_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_splitk_fp16_precast_f32_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_splitk_fp16_precast_f32_bfloat16_gs_64_b_4_bm_64_bn_32",
];

/// `affine_qmm_t_strided_fp16_precast`, indexed by `row_tile_point`.
static QMM_T_STRIDED_FP16_PRECAST: [&str; 3] = [
    "affine_qmm_t_strided_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_strided_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_strided_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32",
];

/// `affine_qmm_t_strided_fp16_precast_residual`, indexed by `row_tile_point`.
static QMM_T_STRIDED_FP16_PRECAST_RESIDUAL: [&str; 3] = [
    "affine_qmm_t_strided_fp16_precast_residual_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_strided_fp16_precast_residual_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_strided_fp16_precast_residual_bfloat16_gs_64_b_4_bm_64_bn_32",
];

/// How many entrypoints a matvec table holds per row-count arm.
///
/// The tables below are TWO tables end to end -- the one-row body first, the
/// `PIE_MT4` four-row body second -- because the reader that says which
/// entrypoints this crate can fire only understands a lookup into a table of
/// literals. A pair of tables chosen by an `if` is the same fact and it cannot
/// see it.

/// `affine_qmv_fast`, indexed by `codec_point`.
static QMV_FAST: [&str; 6] = [
    "affine_qmv_fast_bfloat16_gs_32_b_4",
    "affine_qmv_fast_bfloat16_gs_32_b_8",
    "affine_qmv_fast_bfloat16_gs_64_b_4",
    "affine_qmv_fast_bfloat16_gs_64_b_8",
    "affine_qmv_fast_bfloat16_gs_128_b_4",
    "affine_qmv_fast_bfloat16_gs_128_b_8",
];

/// `affine_qmv_fast_residual`, indexed by `codec_point`.
static QMV_FAST_RESIDUAL: [&str; 6] = [
    "affine_qmv_fast_residual_bfloat16_gs_32_b_4",
    "affine_qmv_fast_residual_bfloat16_gs_32_b_8",
    "affine_qmv_fast_residual_bfloat16_gs_64_b_4",
    "affine_qmv_fast_residual_bfloat16_gs_64_b_8",
    "affine_qmv_fast_residual_bfloat16_gs_128_b_4",
    "affine_qmv_fast_residual_bfloat16_gs_128_b_8",
];

/// `affine_qmv_tail`, indexed by `bits_point`.
static QMV_TAIL: [&str; 2] = [
    "affine_qmv_tail_bfloat16_gs_64_b_4",
    "affine_qmv_tail_bfloat16_gs_64_b_8",
];

/// `affine_qmv_tail_bias`, indexed by `bits_point`.
static QMV_TAIL_BIAS: [&str; 2] = [
    "affine_qmv_tail_bias_bfloat16_gs_64_b_4",
    "affine_qmv_tail_bias_bfloat16_gs_64_b_8",
];

/// `affine_qmv_wide_strided`, indexed by `bits_point`.
static QMV_WIDE_STRIDED: [&str; 2] = [
    "affine_qmv_wide_strided_bfloat16_gs_64_b_4_v_4_kl_8",
    "affine_qmv_wide_strided_bfloat16_gs_64_b_8_v_4_kl_8",
];

/// Group sizes the affine tree is compiled for, in table order.
///
/// `PIE_GROUP` and `PIE_BITS` are a COORDINATE and not a label: g64/b8 and
/// g128/b4 pack to identical shapes, so a module chosen for the wrong pair
/// unpacks fluent nonsense rather than failing. That is why both are points of
/// an axis here instead of numbers a caller passes through.
const GROUPS: [i32; 3] = [32, 64, 128];

/// Bit widths, in table order.
const BIT_WIDTHS: [i32; 2] = [4, 8];

/// Tile edges, in table order, on both the row and the column axis.
const TILES: [i32; 3] = [16, 32, 64];

/// The column tile the wide forms are stamped at, and only that one.
///
/// `qmm_t_splitk`, `_strided` and their kin instantiate `_bn_32` alone -- 18
/// points where the plain form has 54 -- so the column tile is not a choice
/// the caller has and the grid reads it from here.
const WIDE_BN: i32 = 32;

/// Where a number sits on an axis, or a refusal naming the axis it is off.
fn point(points: &[i32], v: i32, what: &'static str) -> Result<usize, Refusal> {
    points.iter().position(|p| *p == v).ok_or(Refusal::Narrow {
        what,
        at: i64::from(v),
    })
}

/// The quantisation point: group size major, bit width minor.
fn codec_point(group: i32, bits: i32) -> Result<usize, Refusal> {
    Ok(point(&GROUPS, group, "the group size")? * BIT_WIDTHS.len()
        + point(&BIT_WIDTHS, bits, "the bit width")?)
}

/// The bit width alone, for the forms stamped at one group size.
fn bits_point(bits: i32) -> Result<usize, Refusal> {
    point(&BIT_WIDTHS, bits, "the bit width")
}

/// The tile point: row tile major, column tile minor.
fn tile_point(bm: i32, bn: i32) -> Result<usize, Refusal> {
    Ok(point(&TILES, bm, "the row tile")? * TILES.len() + point(&TILES, bn, "the column tile")?)
}

/// The row tile alone, for the forms stamped at `_bn_32`.
fn row_tile_point(bm: i32) -> Result<usize, Refusal> {
    point(&TILES, bm, "the row tile")
}

/// The full four-axis point of the tiled matmul.
fn qmm_point(group: i32, bits: i32, bm: i32, bn: i32) -> Result<usize, Refusal> {
    Ok(codec_point(group, bits)? * (TILES.len() * TILES.len()) + tile_point(bm, bn)?)
}

/// The three-axis point of the `_bn_32` forms.
fn wide_point(group: i32, bits: i32, bm: i32) -> Result<usize, Refusal> {
    Ok(codec_point(group, bits)? * TILES.len() + row_tile_point(bm)?)
}

/// The tiled matmul's rectangle, in THREADS.
///
/// `quant/qmm_t.slang` is `[numthreads(32, 2, 2)]` and both its `main`s read
/// `SV_GroupID` alone: `group.x` is the column tile, `group.y` the row tile
/// and `group.z` the split-K partition. So the group counts are
/// `[n/bn, m/bm, split_k]` and this multiplies each by its own local size,
/// because the driver divides them back out.
///
/// The row count is rounded UP to whole tiles and the overhang is real: no
/// entrypoint's push block carries `m`, so `write_out` cannot guard the row
/// axis and the contract is that the output allocation is a whole number of
/// `bm` rows. The column overhang IS guarded, by `n`, which the push block
/// does carry.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty extent or a non-positive tile, and
/// [`Refusal::Grid`] if a tile count times its local size leaves a `u32`.
fn qmm_grid(n: i32, bn: i32, m: i32, bm: i32, split_k: i32) -> Result<[u32; 3], Refusal> {
    let tiles = |extent: i32, tile: i32, what: &'static str| -> Result<u32, Refusal> {
        if extent <= 0 {
            return Err(Refusal::Empty { what });
        }
        if tile <= 0 {
            return Err(Refusal::Empty { what: "the tile" });
        }
        u32::try_from(extent)
            .map(|e| e.div_ceil(tile.unsigned_abs()))
            .map_err(|_| Refusal::Grid {
                what,
                at: i64::from(extent),
            })
    };
    if split_k <= 0 {
        return Err(Refusal::Empty {
            what: "the k split",
        });
    }
    let x = tiles(n, bn, "the column count")?;
    let y = tiles(m, bm, "the row count")?;
    let z = split_k.unsigned_abs();
    let lanes = |groups: u32, local: u32, what: &'static str| -> Result<u32, Refusal> {
        groups.checked_mul(local).ok_or(Refusal::Grid {
            what,
            at: i64::from(groups),
        })
    };
    Ok([
        lanes(x, 32, "the column tiles")?,
        lanes(y, 2, "the row tiles")?,
        lanes(z, 2, "the k splits")?,
    ])
}

/// The matvec's rectangle, in THREADS.
///
/// `quant/qmv.slang` is `[numthreads(PIE_LANES, 2, 1)]` with `PIE_LANES` at 64,
/// `group.x` is the batch vector and one workgroup covers eight output rows --
/// `out0 = group.y * 8 + ly * 4`, two lanes of four. So the group counts are
/// `[vecs, out/8, 1]`.
///
/// The 64 is the shader's `PIE_LANES` and nothing else. It was 32, inherited
/// from the Metal port's simdgroup width; widening it doubled the threads a
/// projection launches, which is what a matvec on this card is short of. If it
/// moves again it moves in both files at once.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty extent, [`Refusal::Grid`] on overflow.
/// The x extent every `quant/qmv.wgsl` form takes: four batch vectors to a
/// group.
///
/// It was the WIDE form's alone until `PIE_MT` gave the reducing form the same
/// shape. The reducing form's workgroup owns eight output columns and reads
/// their weights over the whole of K, so one activation row to a workgroup
/// read the whole packed matrix once per token -- 67 GiB for a 512-token
/// prefill of a 1B head, which is what the lm head's 617 ms was.
///
/// Rounded up, and left non-positive as it came, so that `qmv_grid` stays
/// the one place that refuses an empty batch.
fn quarters(m: i32) -> i32 {
    if m <= 0 {
        m
    } else {
        m / 4 + i32::from(m % 4 != 0)
    }
}

fn qmv_grid(vecs: i32, out_vec_size: i32) -> Result<[u32; 3], Refusal> {
    if vecs <= 0 {
        return Err(Refusal::Empty {
            what: "the vectors",
        });
    }
    if out_vec_size <= 0 {
        return Err(Refusal::Empty {
            what: "the output vector",
        });
    }
    // THIRTY-TWO, not vulkan's sixty-four. `quant/qmv.wgsl` is
    // `@workgroup_size(32, 2, 1)`; the Slang module is twice as wide on x.
    // `driver-wgpu::geometry`'s `Rule::Qmv` states `module.local.at(0) * rows`
    // for the same reason, and a `Fire` states LANES which the driver divides
    // by the module's own width -- so vulkan's constant asks for two
    // workgroups per vector where the shader reduces over one.
    let x = vecs.unsigned_abs().checked_mul(32).ok_or(Refusal::Grid {
        what: "the vectors",
        at: i64::from(vecs),
    })?;
    let y = out_vec_size
        .unsigned_abs()
        .div_ceil(8)
        .checked_mul(2)
        .ok_or(Refusal::Grid {
            what: "the output rows",
            at: i64::from(out_vec_size),
        })?;
    Ok([x, y, 1])
}

/// The batched projection: a `bm x bn` tile of the output per workgroup.
///
/// Five buffers -- the packed weight, its two dequant planes, the activation
/// and the result -- and a push block of `k` and `n`. `n` is both a scalar the
/// shader guards its column overhang with and the extent this body tiles, which
/// is why it appears once in each list.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a point the shader tree does not carry, and
/// whatever `qmm_grid` refuses.
#[routine]
pub fn qmm_t(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    group: Const<i32>,
    bits: Const<i32>,
    bm: Const<i32>,
    bn: Const<i32>) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    let m = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/qmm_t.wgsl", QMM_T[qmm_point(*group, *bits, *bm, *bn)?]).apply(qmm_grid(n, *bn, m, *bm, 1)?),
        &[w.arg(), scales.arg(), biases.arg(), x.arg(), y.arg(), k.arg(), n.arg()],
    )
}

/// The same tile, plus a per-COLUMN bias its epilogue adds.
///
/// `extra` binds at 5 and is indexed by the column alone -- one value per
/// output feature, not per element, which is what tells this apart from the
/// residual form that shares the binding.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a point the shader tree does not carry, and
/// whatever `qmm_grid` refuses.
#[routine]
pub fn qmm_t_bias(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    bias: Const<Tensor<bf16>>,
    group: Const<i32>,
    bits: Const<i32>,
    bm: Const<i32>,
    bn: Const<i32>) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    let m = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/qmm_t.wgsl", QMM_T_BIAS[qmm_point(*group, *bits, *bm, *bn)?]).apply(qmm_grid(n, *bn, m, *bm, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            bias.arg(),
            k.arg(),
            n.arg(),
        ],
    )
}

/// The same tile, plus a residual added ELEMENTWISE.
///
/// Same binding as the bias and a different index: `extra[row * stride + col]`
/// against the bias's `extra[col]`. Two rows exist rather than one flag because
/// the shader is two `#define`s, and passing a per-column plane to this form
/// reads a whole matrix out of a vector.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a point the shader tree does not carry, and
/// whatever `qmm_grid` refuses.
#[routine]
pub fn qmm_t_residual(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    residual: In<Tensor<bf16>>,
    group: Const<i32>,
    bits: Const<i32>,
    bm: Const<i32>,
    bn: Const<i32>) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    let m = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/qmm_t.wgsl", QMM_T_RESIDUAL[qmm_point(*group, *bits, *bm, *bn)?]).apply(qmm_grid(n, *bn, m, *bm, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
            residual.arg(),
        ],
    )
}

/// The tiled projection over an activation already cast to half.
///
/// The activation arrives ALREADY half: `half_in` binds at 7 and `x` at 3 is
/// compiled out, so slangc decorates no binding for it and this body must not
/// pass one. `cast_qmm_input` is what fills `half_in`.
///
/// One group size and one bit width -- gs64/b4 -- because that is all
/// `qmm_t.slang` instantiates the precast forms at.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a point the shader tree does not carry, and
/// whatever `qmm_grid` refuses.
#[routine]
pub fn qmm_t_fp16_precast(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    half_in: In<Tensor<f16>>,
    bm: Const<i32>,
    bn: Const<i32>) -> Result<(), Refusal> {
    let k = half_in.width;
    let n = y.width;
    let m = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/qmm_t.wgsl", QMM_T_FP16_PRECAST[tile_point(*bm, *bn)?]).apply(qmm_grid(n, *bn, m, *bm, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            y.arg(),
            half_in.arg(),
            k.arg(),
            n.arg(),
        ],
    )
}

/// The precast tile with a per-column bias.
///
/// The activation arrives ALREADY half: `half_in` binds at 7 and `x` at 3 is
/// compiled out, so slangc decorates no binding for it and this body must not
/// pass one. `cast_qmm_input` is what fills `half_in`.
///
/// One group size and one bit width -- gs64/b4 -- because that is all
/// `qmm_t.slang` instantiates the precast forms at.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a point the shader tree does not carry, and
/// whatever `qmm_grid` refuses.
#[routine]
pub fn qmm_t_bias_fp16_precast(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    bias: Const<Tensor<bf16>>,
    half_in: In<Tensor<f16>>,
    bm: Const<i32>,
    bn: Const<i32>) -> Result<(), Refusal> {
    let k = half_in.width;
    let n = y.width;
    let m = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/qmm_t.wgsl", QMM_T_BIAS_FP16_PRECAST[tile_point(*bm, *bn)?]).apply(qmm_grid(n, *bn, m, *bm, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            y.arg(),
            bias.arg(),
            half_in.arg(),
            k.arg(),
            n.arg(),
        ],
    )
}

/// The precast tile with an elementwise residual.
///
/// The activation arrives ALREADY half: `half_in` binds at 7 and `x` at 3 is
/// compiled out, so slangc decorates no binding for it and this body must not
/// pass one. `cast_qmm_input` is what fills `half_in`.
///
/// One group size and one bit width -- gs64/b4 -- because that is all
/// `qmm_t.slang` instantiates the precast forms at.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a point the shader tree does not carry, and
/// whatever `qmm_grid` refuses.
#[routine]
pub fn qmm_t_residual_fp16_precast(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    // `half_in` FIRST: it is the statement's input 0 and the residual its
    // input 1, whatever order the shader's buffer table wants them in.
    half_in: In<Tensor<f16>>,
    residual: In<Tensor<bf16>>,
    bm: Const<i32>,
    bn: Const<i32>) -> Result<(), Refusal> {
    let k = residual.width;
    let n = y.width;
    let m = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/qmm_t.wgsl", QMM_T_RESIDUAL_FP16_PRECAST[tile_point(*bm, *bn)?]).apply(qmm_grid(n, *bn, m, *bm, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            y.arg(),
            residual.arg(),
            half_in.arg(),
            k.arg(),
            n.arg(),
        ],
    )
}

/// The tiled projection split along k, accumulating into bf16 planes.
///
/// `group.z` is the k partition, so the z extent is the split count and each
/// partition writes its own plane at `group_z * split_k_partition_stride`.
/// `qmm_splitk_reduce` sums them.
///
/// `y` at binding 4 is compiled out -- the result goes to `out` at 6 -- so this
/// body passes five buffers where the dense form's five are a different five.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a point the shader tree does not carry, and
/// whatever `qmm_grid` refuses.
#[routine]
pub fn qmm_t_splitk(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    group: Const<i32>,
    bits: Const<i32>,
    bm: Const<i32>) -> Result<(), Refusal> {
    let k = x.width;
    let n = out.width;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<2>` named.
    // This body forwards `ctx.params()` as a STRUCT, so the run is the
    // shader's layout and no `Const` mark can name a word inside it; the
    // migration made this `keys::RowStride`, which no driver answers.
    let row_stride = ctx.param(2)?;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<3>` named.
    // This body forwards `ctx.params()` as a STRUCT, so the run is the
    // shader's layout and no `Const` mark can name a word inside it; the
    // migration made this `keys::KPartitionSize`, which no driver answers.
    let k_partition_size = ctx.param(3)?;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<4>` named.
    // This body forwards `ctx.params()` as a STRUCT, so the run is the
    // shader's layout and no `Const` mark can name a word inside it; the
    // migration made this `keys::SplitKPartitionStride`, which no driver answers.
    let split_k_partition_stride = ctx.param(4)?;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<5>` named.
    // This body forwards `ctx.params()` as a STRUCT, so the run is the
    // shader's layout and no `Const` mark can name a word inside it; the
    // migration made this `keys::SplitK`, which no driver answers.
    let split_k = ctx.param(5)?;
    let m = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/qmm_t.wgsl", QMM_T_SPLITK[wide_point(*group, *bits, *bm)?]).apply(qmm_grid(n, WIDE_BN, m, *bm, split_k)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            out.arg(),
            k.arg(),
            n.arg(),
            row_stride.arg(),
            k_partition_size.arg(),
            split_k_partition_stride.arg(),
            split_k.arg(),
        ],
    )
}

/// The same split, accumulating into f32 planes.
///
/// `group.z` is the k partition, so the z extent is the split count and each
/// partition writes its own plane at `group_z * split_k_partition_stride`.
/// `qmm_splitk_reduce` sums them.
///
/// `y` at binding 4 is compiled out -- the result goes to `out` at 6 -- so this
/// body passes five buffers where the dense form's five are a different five.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a point the shader tree does not carry, and
/// whatever `qmm_grid` refuses.
#[routine]
pub fn qmm_t_splitk_f32(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    group: Const<i32>,
    bits: Const<i32>,
    bm: Const<i32>) -> Result<(), Refusal> {
    let k = x.width;
    let n = out.width;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<2>` named.
    // This body forwards `ctx.params()` as a STRUCT, so the run is the
    // shader's layout and no `Const` mark can name a word inside it; the
    // migration made this `keys::RowStride`, which no driver answers.
    let row_stride = ctx.param(2)?;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<3>` named.
    // This body forwards `ctx.params()` as a STRUCT, so the run is the
    // shader's layout and no `Const` mark can name a word inside it; the
    // migration made this `keys::KPartitionSize`, which no driver answers.
    let k_partition_size = ctx.param(3)?;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<4>` named.
    // This body forwards `ctx.params()` as a STRUCT, so the run is the
    // shader's layout and no `Const` mark can name a word inside it; the
    // migration made this `keys::SplitKPartitionStride`, which no driver answers.
    let split_k_partition_stride = ctx.param(4)?;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<5>` named.
    // This body forwards `ctx.params()` as a STRUCT, so the run is the
    // shader's layout and no `Const` mark can name a word inside it; the
    // migration made this `keys::SplitK`, which no driver answers.
    let split_k = ctx.param(5)?;
    let m = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/qmm_t.wgsl", QMM_T_SPLITK_F32[wide_point(*group, *bits, *bm)?]).apply(qmm_grid(n, WIDE_BN, m, *bm, split_k)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            out.arg(),
            k.arg(),
            n.arg(),
            row_stride.arg(),
            k_partition_size.arg(),
            split_k_partition_stride.arg(),
            split_k.arg(),
        ],
    )
}

/// The split-k projection over a precast activation.
///
/// `group.z` is the k partition, so the z extent is the split count and each
/// partition writes its own plane at `group_z * split_k_partition_stride`.
/// `qmm_splitk_reduce` sums them.
///
/// `y` at binding 4 is compiled out -- the result goes to `out` at 6 -- so this
/// body passes five buffers where the dense form's five are a different five.
///
/// The activation arrives ALREADY half: `half_in` binds at 7 and `x` at 3 is
/// compiled out, so slangc decorates no binding for it and this body must not
/// pass one. `cast_qmm_input` is what fills `half_in`.
///
/// One group size and one bit width -- gs64/b4 -- because that is all
/// `qmm_t.slang` instantiates the precast forms at.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a point the shader tree does not carry, and
/// whatever `qmm_grid` refuses.
#[routine]
pub fn qmm_t_splitk_fp16_precast(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    half_in: In<Tensor<f16>>,
    bm: Const<i32>) -> Result<(), Refusal> {
    let k = half_in.width;
    let n = out.width;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<2>` named.
    // This body forwards `ctx.params()` as a STRUCT, so the run is the
    // shader's layout and no `Const` mark can name a word inside it; the
    // migration made this `keys::RowStride`, which no driver answers.
    let row_stride = ctx.param(2)?;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<3>` named.
    // This body forwards `ctx.params()` as a STRUCT, so the run is the
    // shader's layout and no `Const` mark can name a word inside it; the
    // migration made this `keys::KPartitionSize`, which no driver answers.
    let k_partition_size = ctx.param(3)?;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<4>` named.
    // This body forwards `ctx.params()` as a STRUCT, so the run is the
    // shader's layout and no `Const` mark can name a word inside it; the
    // migration made this `keys::SplitKPartitionStride`, which no driver answers.
    let split_k_partition_stride = ctx.param(4)?;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<5>` named.
    // This body forwards `ctx.params()` as a STRUCT, so the run is the
    // shader's layout and no `Const` mark can name a word inside it; the
    // migration made this `keys::SplitK`, which no driver answers.
    let split_k = ctx.param(5)?;
    let m = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/qmm_t.wgsl", QMM_T_SPLITK_FP16_PRECAST[row_tile_point(*bm)?]).apply(qmm_grid(n, WIDE_BN, m, *bm, split_k)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            out.arg(),
            half_in.arg(),
            k.arg(),
            n.arg(),
            row_stride.arg(),
            k_partition_size.arg(),
            split_k_partition_stride.arg(),
            split_k.arg(),
        ],
    )
}

/// The same, accumulating into f32 planes.
///
/// `group.z` is the k partition, so the z extent is the split count and each
/// partition writes its own plane at `group_z * split_k_partition_stride`.
/// `qmm_splitk_reduce` sums them.
///
/// `y` at binding 4 is compiled out -- the result goes to `out` at 6 -- so this
/// body passes five buffers where the dense form's five are a different five.
///
/// The activation arrives ALREADY half: `half_in` binds at 7 and `x` at 3 is
/// compiled out, so slangc decorates no binding for it and this body must not
/// pass one. `cast_qmm_input` is what fills `half_in`.
///
/// One group size and one bit width -- gs64/b4 -- because that is all
/// `qmm_t.slang` instantiates the precast forms at.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a point the shader tree does not carry, and
/// whatever `qmm_grid` refuses.
#[routine]
pub fn qmm_t_splitk_fp16_precast_f32(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    half_in: In<Tensor<f16>>,
    bm: Const<i32>) -> Result<(), Refusal> {
    let k = half_in.width;
    let n = out.width;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<2>` named.
    // This body forwards `ctx.params()` as a STRUCT, so the run is the
    // shader's layout and no `Const` mark can name a word inside it; the
    // migration made this `keys::RowStride`, which no driver answers.
    let row_stride = ctx.param(2)?;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<3>` named.
    // This body forwards `ctx.params()` as a STRUCT, so the run is the
    // shader's layout and no `Const` mark can name a word inside it; the
    // migration made this `keys::KPartitionSize`, which no driver answers.
    let k_partition_size = ctx.param(3)?;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<4>` named.
    // This body forwards `ctx.params()` as a STRUCT, so the run is the
    // shader's layout and no `Const` mark can name a word inside it; the
    // migration made this `keys::SplitKPartitionStride`, which no driver answers.
    let split_k_partition_stride = ctx.param(4)?;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<5>` named.
    // This body forwards `ctx.params()` as a STRUCT, so the run is the
    // shader's layout and no `Const` mark can name a word inside it; the
    // migration made this `keys::SplitK`, which no driver answers.
    let split_k = ctx.param(5)?;
    let m = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/qmm_t.wgsl", QMM_T_SPLITK_FP16_PRECAST_F32[row_tile_point(*bm)?]).apply(qmm_grid(n, WIDE_BN, m, *bm, split_k)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            out.arg(),
            half_in.arg(),
            k.arg(),
            n.arg(),
            row_stride.arg(),
            k_partition_size.arg(),
            split_k_partition_stride.arg(),
            split_k.arg(),
        ],
    )
}

/// The tiled projection over rows that are not contiguous.
///
/// `row_stride` replaces both `k` and `n` as the addressing pitch: the input
/// and the output are windows into wider buffers, and `input_stride()` and
/// `output_stride()` both return it. Only `_bn_32` is instantiated.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a point the shader tree does not carry, and
/// whatever `qmm_grid` refuses.
#[routine]
pub fn qmm_t_strided(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    group: Const<i32>,
    bits: Const<i32>,
    bm: Const<i32>) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<2>` named.
    // The run is the shader's struct layout, so no `Const` mark can name a
    // word inside it, and `keys::RowStride` is answered by no driver.
    let row_stride = ctx.param(2)?;
    let m = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/qmm_t.wgsl", QMM_T_STRIDED[wide_point(*group, *bits, *bm)?]).apply(qmm_grid(n, WIDE_BN, m, *bm, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
            row_stride.arg(),
        ],
    )
}

/// The strided tile with an elementwise residual.
///
/// `row_stride` replaces both `k` and `n` as the addressing pitch: the input
/// and the output are windows into wider buffers, and `input_stride()` and
/// `output_stride()` both return it. Only `_bn_32` is instantiated.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a point the shader tree does not carry, and
/// whatever `qmm_grid` refuses.
#[routine]
pub fn qmm_t_strided_residual(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    residual: In<Tensor<bf16>>,
    group: Const<i32>,
    bits: Const<i32>,
    bm: Const<i32>) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<2>` named.
    // The run is the shader's struct layout, so no `Const` mark can name a
    // word inside it, and `keys::RowStride` is answered by no driver.
    let row_stride = ctx.param(2)?;
    let m = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/qmm_t.wgsl", QMM_T_STRIDED_RESIDUAL[wide_point(*group, *bits, *bm)?]).apply(qmm_grid(n, WIDE_BN, m, *bm, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            residual.arg(),
            k.arg(),
            n.arg(),
            row_stride.arg(),
        ],
    )
}

/// The strided tile over a precast activation.
///
/// `row_stride` replaces both `k` and `n` as the addressing pitch: the input
/// and the output are windows into wider buffers, and `input_stride()` and
/// `output_stride()` both return it. Only `_bn_32` is instantiated.
///
/// The activation arrives ALREADY half: `half_in` binds at 7 and `x` at 3 is
/// compiled out, so slangc decorates no binding for it and this body must not
/// pass one. `cast_qmm_input` is what fills `half_in`.
///
/// One group size and one bit width -- gs64/b4 -- because that is all
/// `qmm_t.slang` instantiates the precast forms at.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a point the shader tree does not carry, and
/// whatever `qmm_grid` refuses.
#[routine]
pub fn qmm_t_strided_fp16_precast(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    half_in: In<Tensor<f16>>,
    bm: Const<i32>) -> Result<(), Refusal> {
    let k = half_in.width;
    let n = y.width;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<2>` named.
    // The run is the shader's struct layout, so no `Const` mark can name a
    // word inside it, and `keys::RowStride` is answered by no driver.
    let row_stride = ctx.param(2)?;
    let m = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/qmm_t.wgsl", QMM_T_STRIDED_FP16_PRECAST[row_tile_point(*bm)?]).apply(qmm_grid(n, WIDE_BN, m, *bm, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            y.arg(),
            half_in.arg(),
            k.arg(),
            n.arg(),
            row_stride.arg(),
        ],
    )
}

/// The strided precast tile with an elementwise residual.
///
/// `row_stride` replaces both `k` and `n` as the addressing pitch: the input
/// and the output are windows into wider buffers, and `input_stride()` and
/// `output_stride()` both return it. Only `_bn_32` is instantiated.
///
/// The activation arrives ALREADY half: `half_in` binds at 7 and `x` at 3 is
/// compiled out, so slangc decorates no binding for it and this body must not
/// pass one. `cast_qmm_input` is what fills `half_in`.
///
/// One group size and one bit width -- gs64/b4 -- because that is all
/// `qmm_t.slang` instantiates the precast forms at.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a point the shader tree does not carry, and
/// whatever `qmm_grid` refuses.
#[routine]
pub fn qmm_t_strided_fp16_precast_residual(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    // As [`qmm_t_residual_fp16_precast`]: input 0 then input 1.
    half_in: In<Tensor<f16>>,
    residual: In<Tensor<bf16>>,
    bm: Const<i32>) -> Result<(), Refusal> {
    let k = residual.width;
    let n = y.width;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<2>` named.
    // The run is the shader's struct layout, so no `Const` mark can name a
    // word inside it, and `keys::RowStride` is answered by no driver.
    let row_stride = ctx.param(2)?;
    let m = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/qmm_t.wgsl", QMM_T_STRIDED_FP16_PRECAST_RESIDUAL[row_tile_point(*bm)?]).apply(qmm_grid(n, WIDE_BN, m, *bm, 1)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            y.arg(),
            residual.arg(),
            half_in.arg(),
            k.arg(),
            n.arg(),
            row_stride.arg(),
        ],
    )
}

/// Sum the split-k partial planes into the result, from bf16 partials.
///
/// Two buffers: the result at 4 and the partial planes at 8. Every other
/// binding of the file is compiled out under `PIE_REDUCE`, so the descriptor set
/// is nine wide and seven of it is holes -- which the LAYOUT keeps and the CALL
/// does not.
///
/// The push block is the split-k block whole, five words, because a push run
/// that is not exactly the pipeline's range is refused.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty rectangle.
#[routine]
pub fn qmm_splitk_reduce(
    ctx: &Ctx<'_>,
    y: Out<Tensor<bf16>>,
    partial: In<Tensor<bf16>>) -> Result<(), Refusal> {
    let k = partial.width;
    let n = y.width;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<2>` named.
    // This body forwards `ctx.params()` as a STRUCT, so the run is the
    // shader's layout and no `Const` mark can name a word inside it; the
    // migration made this `keys::RowStride`, which no driver answers.
    let row_stride = ctx.param(2)?;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<3>` named.
    // This body forwards `ctx.params()` as a STRUCT, so the run is the
    // shader's layout and no `Const` mark can name a word inside it; the
    // migration made this `keys::SplitKPartitionStride`, which no driver answers.
    let split_k_partition_stride = ctx.param(3)?;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<4>` named.
    // This body forwards `ctx.params()` as a STRUCT, so the run is the
    // shader's layout and no `Const` mark can name a word inside it; the
    // migration made this `keys::SplitK`, which no driver answers.
    let split_k = ctx.param(4)?;
    let m = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/qmm_t.wgsl", "qmm_splitk_reduce_bfloat16").apply(elementwise_rows(n, m)?),
        &[
            y.arg(),
            partial.arg(),
            k.arg(),
            n.arg(),
            row_stride.arg(),
            split_k_partition_stride.arg(),
            split_k.arg(),
        ],
    )
}

/// The same sum, from f32 partials.
///
/// Two buffers: the result at 4 and the partial planes at 8. Every other
/// binding of the file is compiled out under `PIE_REDUCE`, so the descriptor set
/// is nine wide and seven of it is holes -- which the LAYOUT keeps and the CALL
/// does not.
///
/// The push block is the split-k block whole, five words, because a push run
/// that is not exactly the pipeline's range is refused.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty rectangle.
#[routine]
pub fn qmm_splitk_reduce_f32(
    ctx: &Ctx<'_>,
    y: Out<Tensor<bf16>>,
    partial: In<Tensor<f32>>) -> Result<(), Refusal> {
    let k = partial.width;
    let n = y.width;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<2>` named.
    // This body forwards `ctx.params()` as a STRUCT, so the run is the
    // shader's layout and no `Const` mark can name a word inside it; the
    // migration made this `keys::RowStride`, which no driver answers.
    let row_stride = ctx.param(2)?;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<3>` named.
    // This body forwards `ctx.params()` as a STRUCT, so the run is the
    // shader's layout and no `Const` mark can name a word inside it; the
    // migration made this `keys::SplitKPartitionStride`, which no driver answers.
    let split_k_partition_stride = ctx.param(3)?;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<4>` named.
    // This body forwards `ctx.params()` as a STRUCT, so the run is the
    // shader's layout and no `Const` mark can name a word inside it; the
    // migration made this `keys::SplitK`, which no driver answers.
    let split_k = ctx.param(4)?;
    let m = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/qmm_t.wgsl", "qmm_splitk_reduce_f32_bfloat16").apply(elementwise_rows(n, m)?),
        &[
            y.arg(),
            partial.arg(),
            k.arg(),
            n.arg(),
            row_stride.arg(),
            split_k_partition_stride.arg(),
            split_k.arg(),
        ],
    )
}

/// Cast a bf16 activation to half, flat.
///
/// Two buffers, at 3 and 12, and a four-word push block of which each form
/// reads two. The block is declared unconditionally so both forms push all four.
///
/// The flat form recovers its index as `thread.x + thread.y * groups.x * 32`,
/// so a one-dimensional extent covers it and the `count` guard kills the rest
/// of the workgroup.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty count.
#[routine]
pub fn cast_qmm_input_bfloat16_to_float16(
    ctx: &Ctx<'_>,
    cast_in: In<Tensor<bf16>>,
    half_out: Out<Tensor<f16>>) -> Result<(), Refusal> {
    let k = cast_in.width;
    let n = half_out.width;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<2>` named.
    // The run is the shader's struct layout, so no `Const` mark can name a
    // word inside it, and `keys::RowStride` is answered by no driver.
    let row_stride = ctx.param(2)?;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<3>` named.
    // The run is the shader's struct layout, so no `Const` mark can name a
    // word inside it, and `keys::Count` is answered by no driver.
    let count = ctx.param(3)?;
    ctx.fire(
        Fire::at("quant/qmm_t.wgsl", "cast_qmm_input_bfloat16_to_float16").apply(elementwise(count, 1)?),
        &[
            cast_in.arg(),
            half_out.arg(),
            k.arg(),
            n.arg(),
            row_stride.arg(),
            count.arg(),
        ],
    )
}

/// Cast a bf16 activation to half, row by row.
///
/// Two buffers, at 3 and 12, and a four-word push block of which each form
/// reads two. The block is declared unconditionally so both forms push all four.
///
/// The strided form is two-dimensional -- `thread.x` is the column and
/// `thread.y` the row -- and steps both buffers by `row_stride`.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty rectangle.
#[routine]
pub fn cast_qmm_input_strided_bfloat16_to_float16(
    ctx: &Ctx<'_>,
    cast_in: In<Tensor<bf16>>,
    half_out: Out<Tensor<f16>>,
    // THE SOURCE'S ROW PITCH, WHICH THE STATEMENT CARRIES. It was
    // `Param<2, i32>` and the migration made it an ask no driver answers;
    // it is the activation's own stride, which the text knows and the fire
    // does not. `n` and `count` were asked for beside it and thrown away
    // (`let _ = (n, count)`) -- two slots the shared argument table keeps
    // between this and its packed twin, and neither is read here.
    row_stride: Const<i32>) -> Result<(), Refusal> {
    let k = cast_in.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/qmm_t.wgsl", "cast_qmm_input_strided_bfloat16_to_float16").apply(elementwise_rows(k, rows)?),
        &[cast_in.arg(), half_out.arg(), k.arg(), row_stride.arg()],
    )
}

/// The projection as a matvec: one warp per output row, eight rows a group.
///
/// This is the loudest misbinding the tables were grown for: the shader
/// declares its WEIGHTS first and the trace states them last, so a positional
/// bind put the activation where the packed weight belongs, on every projection
/// of every layer.
///
/// `group.x` is the batch vector, so a decode of one row runs one group wide.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a quantisation point the tree does not carry, and
/// whatever `qmv_grid` refuses.
#[routine]
pub fn qmv_fast(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    group: Const<i32>,
    bits: Const<i32>) -> Result<(), Refusal> {
    let in_vec_size = x.width;
    let out_vec_size = y.width;
    let vecs = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/qmv.wgsl", QMV_FAST[codec_point(*group, *bits)?]).apply(qmv_grid(quarters(vecs), out_vec_size)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            in_vec_size.arg(),
            out_vec_size.arg(),
        ],
    )
}

/// The same matvec with the block residual its epilogue folds.
///
/// `extra` binds at 5 and is indexed by the output row.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a quantisation point the tree does not carry, and
/// whatever `qmv_grid` refuses.
#[routine]
pub fn qmv_fast_residual(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    residual: In<Tensor<bf16>>,
    group: Const<i32>,
    bits: Const<i32>) -> Result<(), Refusal> {
    let in_vec_size = x.width;
    let out_vec_size = y.width;
    let vecs = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/qmv.wgsl", QMV_FAST_RESIDUAL[codec_point(*group, *bits)?]).apply(qmv_grid(quarters(vecs), out_vec_size)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            in_vec_size.arg(),
            out_vec_size.arg(),
            residual.arg(),
        ],
    )
}

/// The matvec's tail form, stamped at gs64 alone.
///
/// Same five bindings and same push block as [`qmv_fast`]; what differs is the
/// k-loop, which is why it is a separate entrypoint and not a point of an axis.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a quantisation point the tree does not carry, and
/// whatever `qmv_grid` refuses.
#[routine]
pub fn qmv_tail(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    bits: Const<i32>) -> Result<(), Refusal> {
    let in_vec_size = x.width;
    let out_vec_size = y.width;
    let vecs = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/qmv.wgsl", QMV_TAIL[bits_point(*bits)?]).apply(qmv_grid(quarters(vecs), out_vec_size)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            in_vec_size.arg(),
            out_vec_size.arg(),
        ],
    )
}

/// The tail form with a per-row bias at binding 5.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a quantisation point the tree does not carry, and
/// whatever `qmv_grid` refuses.
#[routine]
pub fn qmv_tail_bias(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    bias: Const<Tensor<bf16>>,
    bits: Const<i32>) -> Result<(), Refusal> {
    let in_vec_size = x.width;
    let out_vec_size = y.width;
    let vecs = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/qmv.wgsl", QMV_TAIL_BIAS[bits_point(*bits)?]).apply(qmv_grid(quarters(vecs), out_vec_size)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            bias.arg(),
            in_vec_size.arg(),
            out_vec_size.arg(),
        ],
    )
}

/// The matvec over four batch vectors per group, from a strided source.
///
/// `PIE_VEC` is 4 here, so `group.x` covers four vectors rather than one and
/// the x extent is the vector count divided by four. `row_stride` and `m` join
/// the push block; nothing else moves.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a quantisation point the tree does not carry, and
/// whatever `qmv_grid` refuses.
#[routine]
pub fn qmv_wide_strided(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    bits: Const<i32>) -> Result<(), Refusal> {
    let in_vec_size = x.width;
    let out_vec_size = y.width;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<2>` named.
    // The run is the shader's struct layout, so no `Const` mark can name a
    // word inside it, and `keys::RowStride` is answered by no driver.
    let row_stride = ctx.param(2)?;
    let m = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/qmv.wgsl", QMV_WIDE_STRIDED[bits_point(*bits)?]).apply(qmv_grid(quarters(m), out_vec_size)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            in_vec_size.arg(),
            out_vec_size.arg(),
            row_stride.arg(),
            m.arg(),
        ],
    )
}

/// `affine_qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4` -- a hand-typed warp shape.
///
/// The five `_wm_`/`_wn_` entrypoints are typed out in the shader rather
/// than stamped from the axis, so they are five kernels with five rows and
/// five routines. Their tile is fixed by the NAME, which is why this takes
/// no tile arguments and states the pair as constants.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty rectangle, and whatever `qmm_grid` refuses.
#[routine]
pub fn qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    let m = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/qmm_t.wgsl", "affine_qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4").apply(qmm_grid(n, 32, m, 128, 1)?),
        &[w.arg(), scales.arg(), biases.arg(), x.arg(), y.arg(), k.arg(), n.arg()],
    )
}

/// `affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2` -- a hand-typed warp shape.
///
/// The five `_wm_`/`_wn_` entrypoints are typed out in the shader rather
/// than stamped from the axis, so they are five kernels with five rows and
/// five routines. Their tile is fixed by the NAME, which is why this takes
/// no tile arguments and states the pair as constants.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty rectangle, and whatever `qmm_grid` refuses.
#[routine]
pub fn qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    let m = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/qmm_t.wgsl", "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2").apply(qmm_grid(n, 32, m, 32, 1)?),
        &[w.arg(), scales.arg(), biases.arg(), x.arg(), y.arg(), k.arg(), n.arg()],
    )
}

/// `affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2` -- a hand-typed warp shape.
///
/// The five `_wm_`/`_wn_` entrypoints are typed out in the shader rather
/// than stamped from the axis, so they are five kernels with five rows and
/// five routines. Their tile is fixed by the NAME, which is why this takes
/// no tile arguments and states the pair as constants.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty rectangle, and whatever `qmm_grid` refuses.
#[routine]
pub fn qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    let m = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/qmm_t.wgsl", "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2").apply(qmm_grid(n, 32, m, 64, 1)?),
        &[w.arg(), scales.arg(), biases.arg(), x.arg(), y.arg(), k.arg(), n.arg()],
    )
}

/// `affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1` -- a hand-typed warp shape.
///
/// The five `_wm_`/`_wn_` entrypoints are typed out in the shader rather
/// than stamped from the axis, so they are five kernels with five rows and
/// five routines. Their tile is fixed by the NAME, which is why this takes
/// no tile arguments and states the pair as constants.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty rectangle, and whatever `qmm_grid` refuses.
#[routine]
pub fn qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    let m = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/qmm_t.wgsl", "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1").apply(qmm_grid(n, 32, m, 64, 1)?),
        &[w.arg(), scales.arg(), biases.arg(), x.arg(), y.arg(), k.arg(), n.arg()],
    )
}

/// `affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4` -- a hand-typed warp shape.
///
/// The five `_wm_`/`_wn_` entrypoints are typed out in the shader rather
/// than stamped from the axis, so they are five kernels with five rows and
/// five routines. Their tile is fixed by the NAME, which is why this takes
/// no tile arguments and states the pair as constants.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty rectangle, and whatever `qmm_grid` refuses.
#[routine]
pub fn qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    let m = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/qmm_t.wgsl", "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4").apply(qmm_grid(n, 64, m, 64, 1)?),
        &[w.arg(), scales.arg(), biases.arg(), x.arg(), y.arg(), k.arg(), n.arg()],
    )
}

/// Quantise a bf16 plane to affine u4, group by group.
///
/// `[numthreads(1, 1, 1)]` deliberately: all three transcodes state no launch
/// rule, nothing in this workspace dispatches them, and the transcodes a model
/// needs happen host-side in `model-loader`. The rows exist for parity with
/// `kernels-metal`. So a wider workgroup would be a change no test could run.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty count.
#[routine]
pub fn encode_u4_bf16(
    ctx: &Ctx<'_>,
    input: In<Tensor<bf16>>,
    codes: Out<Tensor<u32>>,
    scales: Out<Tensor<bf16>>,
    biases: Out<Tensor<bf16>>,
    group_size: Const<i32>) -> Result<(), Refusal> {
    let groups = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/transcode.wgsl", "affine_encode_u4_bf16").apply(elementwise(groups, 1)?),
        &[
            input.arg(),
            codes.arg(),
            scales.arg(),
            biases.arg(),
            groups.arg(),
            group_size.arg(),
        ],
    )
}

/// The same encoder over an f32 source.
///
/// `[numthreads(1, 1, 1)]` deliberately: all three transcodes state no launch
/// rule, nothing in this workspace dispatches them, and the transcodes a model
/// needs happen host-side in `model-loader`. The rows exist for parity with
/// `kernels-metal`. So a wider workgroup would be a change no test could run.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty count.
#[routine]
pub fn encode_u4_f32(
    ctx: &Ctx<'_>,
    input: In<Tensor<bf16>>,
    codes: Out<Tensor<u32>>,
    scales: Out<Tensor<bf16>>,
    biases: Out<Tensor<bf16>>,
    group_size: Const<i32>) -> Result<(), Refusal> {
    let groups = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/transcode.wgsl", "affine_encode_u4_f32").apply(elementwise(groups, 1)?),
        &[
            input.arg(),
            codes.arg(),
            scales.arg(),
            biases.arg(),
            groups.arg(),
            group_size.arg(),
        ],
    )
}

/// Expand an MXFP4 payload and its exponents into bf16.
///
/// `[numthreads(1, 1, 1)]` deliberately: all three transcodes state no launch
/// rule, nothing in this workspace dispatches them, and the transcodes a model
/// needs happen host-side in `model-loader`. The rows exist for parity with
/// `kernels-metal`. So a wider workgroup would be a change no test could run.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty count.
#[routine]
pub fn mxfp4_dequant_bf16(
    ctx: &Ctx<'_>,
    payload: In<Tensor<u8>>,
    exponents: In<Tensor<u8>>,
    out: Out<Tensor<bf16>>,
    block_size: Const<i32>) -> Result<(), Refusal> {
    let blocks = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("quant/transcode.wgsl", "mxfp4_dequant_bf16").apply(elementwise(blocks, 1)?),
        &[
            payload.arg(),
            exponents.arg(),
            out.arg(),
            blocks.arg(),
            block_size.arg(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one grid that is NOT `kernels-vulkan`'s, pinned as numbers.
    ///
    /// `qmm_grid` transferred unchanged -- its `(32, 2, 2)` is exactly
    /// `quant/qmm_t.wgsl`'s `@workgroup_size` -- and `qmv_grid`'s x did not:
    /// the Slang module is sixty-four lanes wide and `quant/qmv.wgsl` is
    /// `@workgroup_size(32, 2, 1)`. `driver-wgpu::geometry`'s `Rule::Qmv`
    /// states `module.local.at(0) * rows`, which is what this was checked
    /// against.
    ///
    /// Vulkan's own family tests are NOT ported, for the reason `attn`'s
    /// were not: two of them would assert the sibling's lane count about this
    /// backend's shaders and pass while doing it.
    #[test]
    fn the_matvec_grid_is_thirty_two_lanes_a_vector_and_not_sixty_four() {
        assert_eq!(qmv_grid(1, 1024).expect("a real shape"), [32, 256, 1]);
        assert_eq!(qmv_grid(7, 1024).expect("a real shape"), [224, 256, 1]);
        // y rounds the output vector up to whole eights and doubles, which is
        // `dims.width.div_ceil(4)` said the other way round.
        assert_eq!(qmv_grid(1, 9).expect("a ragged output"), [32, 4, 1]);

        // And the GEMM's, which did transfer: (32, 2, 2) is the workgroup.
        assert_eq!(
            qmm_grid(1024, 32, 64, 32, 1).expect("whole tiles"),
            [1024, 4, 2]
        );
    }

    /// An empty extent is refused rather than dispatched.
    #[test]
    fn an_empty_extent_is_refused() {
        assert!(matches!(qmv_grid(0, 1024), Err(Refusal::Empty { .. })));
        assert!(matches!(qmv_grid(7, 0), Err(Refusal::Empty { .. })));
        assert!(matches!(
            qmm_grid(0, 32, 64, 32, 1),
            Err(Refusal::Empty { .. })
        ));
        assert!(matches!(
            qmm_grid(1024, 32, 64, 32, 0),
            Err(Refusal::Empty { .. })
        ));
    }

    /// A codec or tile the tree does not carry is refused by NAME.
    ///
    /// The bodies index literal spelling tables with these, so an unknown
    /// point must not reach the index.
    #[test]
    fn a_point_the_tree_does_not_carry_is_refused_by_name() {
        assert!(codec_point(48, 4).is_err());
        assert!(codec_point(64, 3).is_err());
        assert!(tile_point(24, 32).is_err());
        assert!(bits_point(16).is_err());
        // And the real points resolve inside their tables.
        assert!(qmm_point(64, 4, 32, 32).expect("a real point") < QMM_T.len());
        assert!(wide_point(128, 8, 64).expect("a real point") < QMM_T_SPLITK.len());
        assert!(codec_point(128, 8).expect("a real point") < QMV_FAST.len());
    }
}

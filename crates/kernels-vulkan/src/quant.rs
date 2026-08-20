//! The affine-quantised projections, and the codecs around them.
//!
//! This is 32 of the 99 rows and 304 of the 480 entrypoints, and the whole
//! argument for an axis is visible here: `qmm_t` is ONE body in
//! `quant/qmm_t.slang` compiled over (group x bits x row tile x column tile),
//! and enumerating its 54 instantiations as 54 rows would state the
//! instantiation matrix a second time by hand.
//!
//! The five `_wm_`/`_wn_` rows are the exception that proves it: on the Metal
//! side they are `host_name` lines typed out rather than stamped, so they are
//! five kernels and get five rows, and this table keeps that distinction
//! because it is a real one about what the bodies share.

use kernels_macros::routine;
use kernels::routine::Refusal;

use crate::routine::{Asks, Bind, Const, Ctx, Fire, In, Out, Tensor, bf16, elementwise, elementwise_rows, f16, keys};

/// `affine_qmm_t`, indexed by [`qmm_point`].
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

/// `affine_qmm_t_bias`, indexed by [`qmm_point`].
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

/// `affine_qmm_t_residual`, indexed by [`qmm_point`].
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

/// `affine_qmm_t_fp16_precast`, indexed by [`tile_point`]. One group size and one bit width.
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

/// `affine_qmm_t_bias_fp16_precast`, indexed by [`tile_point`]. One group size and one bit width.
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

/// `affine_qmm_t_residual_fp16_precast`, indexed by [`tile_point`]. One group size and one bit width.
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

/// `affine_qmm_t_splitk`, indexed by [`wide_point`]. The column tile is 32 alone.
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

/// `affine_qmm_t_splitk_f32`, indexed by [`wide_point`]. The column tile is 32 alone.
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

/// `affine_qmm_t_strided`, indexed by [`wide_point`]. The column tile is 32 alone.
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

/// `affine_qmm_t_strided_residual`, indexed by [`wide_point`]. The column tile is 32 alone.
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

/// `affine_qmm_t_splitk_fp16_precast`, indexed by [`row_tile_point`].
static QMM_T_SPLITK_FP16_PRECAST: [&str; 3] = [
    "affine_qmm_t_splitk_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_splitk_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_splitk_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32",
];

/// `affine_qmm_t_splitk_fp16_precast_f32`, indexed by [`row_tile_point`].
static QMM_T_SPLITK_FP16_PRECAST_F32: [&str; 3] = [
    "affine_qmm_t_splitk_fp16_precast_f32_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_splitk_fp16_precast_f32_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_splitk_fp16_precast_f32_bfloat16_gs_64_b_4_bm_64_bn_32",
];

/// `affine_qmm_t_strided_fp16_precast`, indexed by [`row_tile_point`].
static QMM_T_STRIDED_FP16_PRECAST: [&str; 3] = [
    "affine_qmm_t_strided_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_strided_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_strided_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32",
];

/// `affine_qmm_t_strided_fp16_precast_residual`, indexed by [`row_tile_point`].
static QMM_T_STRIDED_FP16_PRECAST_RESIDUAL: [&str; 3] = [
    "affine_qmm_t_strided_fp16_precast_residual_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_strided_fp16_precast_residual_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_strided_fp16_precast_residual_bfloat16_gs_64_b_4_bm_64_bn_32",
];

/// `affine_qmv_fast`, indexed by [`codec_point`].
static QMV_FAST: [&str; 6] = [
    "affine_qmv_fast_bfloat16_gs_32_b_4",
    "affine_qmv_fast_bfloat16_gs_32_b_8",
    "affine_qmv_fast_bfloat16_gs_64_b_4",
    "affine_qmv_fast_bfloat16_gs_64_b_8",
    "affine_qmv_fast_bfloat16_gs_128_b_4",
    "affine_qmv_fast_bfloat16_gs_128_b_8",
];

/// `affine_qmv_fast_residual`, indexed by [`codec_point`].
static QMV_FAST_RESIDUAL: [&str; 6] = [
    "affine_qmv_fast_residual_bfloat16_gs_32_b_4",
    "affine_qmv_fast_residual_bfloat16_gs_32_b_8",
    "affine_qmv_fast_residual_bfloat16_gs_64_b_4",
    "affine_qmv_fast_residual_bfloat16_gs_64_b_8",
    "affine_qmv_fast_residual_bfloat16_gs_128_b_4",
    "affine_qmv_fast_residual_bfloat16_gs_128_b_8",
];

/// `affine_qmv_tail`, indexed by [`bits_point`].
static QMV_TAIL: [&str; 2] = [
    "affine_qmv_tail_bfloat16_gs_64_b_4",
    "affine_qmv_tail_bfloat16_gs_64_b_8",
];

/// `affine_qmv_tail_bias`, indexed by [`bits_point`].
static QMV_TAIL_BIAS: [&str; 2] = [
    "affine_qmv_tail_bias_bfloat16_gs_64_b_4",
    "affine_qmv_tail_bias_bfloat16_gs_64_b_8",
];

/// `affine_qmv_wide_strided`, indexed by [`bits_point`].
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
/// The x extent of the wide forms: four batch vectors to a group.
///
/// Rounded up, and left non-positive as it came, so that [`qmv_grid`] stays
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
    let x = vecs.unsigned_abs().checked_mul(64).ok_or(Refusal::Grid {
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
/// whatever [`qmm_grid`] refuses.
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
        Fire::at(crate::routine::module_path(QMM_T[qmm_point(*group, *bits, *bm, *bn)?], ctx.best()), QMM_T[qmm_point(*group, *bits, *bm, *bn)?]).apply(qmm_grid(n, *bn, m, *bm, 1)?),
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
/// whatever [`qmm_grid`] refuses.
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
        Fire::at(crate::routine::module_path(QMM_T_BIAS[qmm_point(*group, *bits, *bm, *bn)?], ctx.best()), QMM_T_BIAS[qmm_point(*group, *bits, *bm, *bn)?]).apply(qmm_grid(n, *bn, m, *bm, 1)?),
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
/// whatever [`qmm_grid`] refuses.
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
        Fire::at(crate::routine::module_path(QMM_T_RESIDUAL[qmm_point(*group, *bits, *bm, *bn)?], ctx.best()), QMM_T_RESIDUAL[qmm_point(*group, *bits, *bm, *bn)?]).apply(qmm_grid(n, *bn, m, *bm, 1)?),
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
/// whatever [`qmm_grid`] refuses.
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
        Fire::at(crate::routine::module_path(QMM_T_FP16_PRECAST[tile_point(*bm, *bn)?], ctx.best()), QMM_T_FP16_PRECAST[tile_point(*bm, *bn)?]).apply(qmm_grid(n, *bn, m, *bm, 1)?),
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
/// whatever [`qmm_grid`] refuses.
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
        Fire::at(crate::routine::module_path(QMM_T_BIAS_FP16_PRECAST[tile_point(*bm, *bn)?], ctx.best()), QMM_T_BIAS_FP16_PRECAST[tile_point(*bm, *bn)?]).apply(qmm_grid(n, *bn, m, *bm, 1)?),
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
/// whatever [`qmm_grid`] refuses.
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
    // THE ACTIVATION'S WIDTH AND NOT THE RESIDUAL'S. `k` is the reduction
    // depth -- how far along a row of the weight the tile walks -- so it is
    // the width of what is being multiplied, and the residual is shaped like
    // the RESULT. The two are equal only where a projection is square, which
    // is why reading the wrong one survives every shape test and fails on a
    // real model: qwen3's attention output projection takes a 2048-wide q
    // and adds a 1024-wide residual, so this halved `k` and the tile summed
    // the first half of every row. Finite, varied and wrong.
    let k = half_in.width;
    let n = y.width;
    let m = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path(QMM_T_RESIDUAL_FP16_PRECAST[tile_point(*bm, *bn)?], ctx.best()), QMM_T_RESIDUAL_FP16_PRECAST[tile_point(*bm, *bn)?]).apply(qmm_grid(n, *bn, m, *bm, 1)?),
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
/// whatever [`qmm_grid`] refuses.
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
    // A SCALAR AND ZERO, NOT AN ABSENT BUFFER. `Push`, `PushReduce` and
    // `PushCast` in `quant/qmm_t.slang` declare these words unconditionally
    // so that every form pushes the pipeline layout's whole range, and the
    // form fired here reads none of them -- `input_stride()` and
    // `output_stride()` answer `k` and `n` unless `PIE_STRIDED` is on, and
    // the reduce and cast arms index by their own two. The words exist to
    // hold the places the scalars after them sit at, so any value does.
    //
    // They were `ctx.absent()`, which on this plane mints a null BUFFER: each
    // one landed among the OPERANDS instead of among the scalars, so the body
    // bound more buffers than the module decorates AND pushed a block short
    // by that many words, every later scalar reading its neighbour's. Both
    // halves are a refusal or silent garbage at the first real dispatch, and
    // no shipped text fires a split-K GEMM or the unstrided cast, which is
    // why neither was seen.
    let row_stride = 0i32;
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
        Fire::at(crate::routine::module_path(QMM_T_SPLITK[wide_point(*group, *bits, *bm)?], ctx.best()), QMM_T_SPLITK[wide_point(*group, *bits, *bm)?]).apply(qmm_grid(n, WIDE_BN, m, *bm, split_k)?),
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
/// whatever [`qmm_grid`] refuses.
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
    // A SCALAR AND ZERO, NOT AN ABSENT BUFFER. `Push`, `PushReduce` and
    // `PushCast` in `quant/qmm_t.slang` declare these words unconditionally
    // so that every form pushes the pipeline layout's whole range, and the
    // form fired here reads none of them -- `input_stride()` and
    // `output_stride()` answer `k` and `n` unless `PIE_STRIDED` is on, and
    // the reduce and cast arms index by their own two. The words exist to
    // hold the places the scalars after them sit at, so any value does.
    //
    // They were `ctx.absent()`, which on this plane mints a null BUFFER: each
    // one landed among the OPERANDS instead of among the scalars, so the body
    // bound more buffers than the module decorates AND pushed a block short
    // by that many words, every later scalar reading its neighbour's. Both
    // halves are a refusal or silent garbage at the first real dispatch, and
    // no shipped text fires a split-K GEMM or the unstrided cast, which is
    // why neither was seen.
    let row_stride = 0i32;
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
        Fire::at(crate::routine::module_path(QMM_T_SPLITK_F32[wide_point(*group, *bits, *bm)?], ctx.best()), QMM_T_SPLITK_F32[wide_point(*group, *bits, *bm)?]).apply(qmm_grid(n, WIDE_BN, m, *bm, split_k)?),
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
/// whatever [`qmm_grid`] refuses.
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
    // A SCALAR AND ZERO, NOT AN ABSENT BUFFER. `Push`, `PushReduce` and
    // `PushCast` in `quant/qmm_t.slang` declare these words unconditionally
    // so that every form pushes the pipeline layout's whole range, and the
    // form fired here reads none of them -- `input_stride()` and
    // `output_stride()` answer `k` and `n` unless `PIE_STRIDED` is on, and
    // the reduce and cast arms index by their own two. The words exist to
    // hold the places the scalars after them sit at, so any value does.
    //
    // They were `ctx.absent()`, which on this plane mints a null BUFFER: each
    // one landed among the OPERANDS instead of among the scalars, so the body
    // bound more buffers than the module decorates AND pushed a block short
    // by that many words, every later scalar reading its neighbour's. Both
    // halves are a refusal or silent garbage at the first real dispatch, and
    // no shipped text fires a split-K GEMM or the unstrided cast, which is
    // why neither was seen.
    let row_stride = 0i32;
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
        Fire::at(crate::routine::module_path(QMM_T_SPLITK_FP16_PRECAST[row_tile_point(*bm)?], ctx.best()), QMM_T_SPLITK_FP16_PRECAST[row_tile_point(*bm)?]).apply(qmm_grid(n, WIDE_BN, m, *bm, split_k)?),
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
/// whatever [`qmm_grid`] refuses.
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
    // A SCALAR AND ZERO, NOT AN ABSENT BUFFER. `Push`, `PushReduce` and
    // `PushCast` in `quant/qmm_t.slang` declare these words unconditionally
    // so that every form pushes the pipeline layout's whole range, and the
    // form fired here reads none of them -- `input_stride()` and
    // `output_stride()` answer `k` and `n` unless `PIE_STRIDED` is on, and
    // the reduce and cast arms index by their own two. The words exist to
    // hold the places the scalars after them sit at, so any value does.
    //
    // They were `ctx.absent()`, which on this plane mints a null BUFFER: each
    // one landed among the OPERANDS instead of among the scalars, so the body
    // bound more buffers than the module decorates AND pushed a block short
    // by that many words, every later scalar reading its neighbour's. Both
    // halves are a refusal or silent garbage at the first real dispatch, and
    // no shipped text fires a split-K GEMM or the unstrided cast, which is
    // why neither was seen.
    let row_stride = 0i32;
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
        Fire::at(crate::routine::module_path(QMM_T_SPLITK_FP16_PRECAST_F32[row_tile_point(*bm)?], ctx.best()), QMM_T_SPLITK_FP16_PRECAST_F32[row_tile_point(*bm)?]).apply(qmm_grid(n, WIDE_BN, m, *bm, split_k)?),
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
/// whatever [`qmm_grid`] refuses.
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
        Fire::at(crate::routine::module_path(QMM_T_STRIDED[wide_point(*group, *bits, *bm)?], ctx.best()), QMM_T_STRIDED[wide_point(*group, *bits, *bm)?]).apply(qmm_grid(n, WIDE_BN, m, *bm, 1)?),
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
/// whatever [`qmm_grid`] refuses.
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
        Fire::at(crate::routine::module_path(QMM_T_STRIDED_RESIDUAL[wide_point(*group, *bits, *bm)?], ctx.best()), QMM_T_STRIDED_RESIDUAL[wide_point(*group, *bits, *bm)?]).apply(qmm_grid(n, WIDE_BN, m, *bm, 1)?),
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
/// whatever [`qmm_grid`] refuses.
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
        Fire::at(crate::routine::module_path(QMM_T_STRIDED_FP16_PRECAST[row_tile_point(*bm)?], ctx.best()), QMM_T_STRIDED_FP16_PRECAST[row_tile_point(*bm)?]).apply(qmm_grid(n, WIDE_BN, m, *bm, 1)?),
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
/// whatever [`qmm_grid`] refuses.
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
    // The activation's width, for the reason [`qmm_t_residual_fp16_precast`]
    // states: `k` is the reduction depth and the residual is result-shaped.
    let k = half_in.width;
    let n = y.width;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<2>` named.
    // The run is the shader's struct layout, so no `Const` mark can name a
    // word inside it, and `keys::RowStride` is answered by no driver.
    let row_stride = ctx.param(2)?;
    let m = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path(QMM_T_STRIDED_FP16_PRECAST_RESIDUAL[row_tile_point(*bm)?], ctx.best()), QMM_T_STRIDED_FP16_PRECAST_RESIDUAL[row_tile_point(*bm)?]).apply(qmm_grid(n, WIDE_BN, m, *bm, 1)?),
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
    let k = 0i32;
    let n = y.width;
    // A SCALAR AND ZERO, NOT AN ABSENT BUFFER. `Push`, `PushReduce` and
    // `PushCast` in `quant/qmm_t.slang` declare these words unconditionally
    // so that every form pushes the pipeline layout's whole range, and the
    // form fired here reads none of them -- `input_stride()` and
    // `output_stride()` answer `k` and `n` unless `PIE_STRIDED` is on, and
    // the reduce and cast arms index by their own two. The words exist to
    // hold the places the scalars after them sit at, so any value does.
    //
    // They were `ctx.absent()`, which on this plane mints a null BUFFER: each
    // one landed among the OPERANDS instead of among the scalars, so the body
    // bound more buffers than the module decorates AND pushed a block short
    // by that many words, every later scalar reading its neighbour's. Both
    // halves are a refusal or silent garbage at the first real dispatch, and
    // no shipped text fires a split-K GEMM or the unstrided cast, which is
    // why neither was seen.
    let row_stride = 0i32;
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
        Fire::at(crate::routine::module_path("qmm_splitk_reduce_bfloat16", ctx.best()), "qmm_splitk_reduce_bfloat16").apply(elementwise_rows(n, m)?),
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
    let k = 0i32;
    let n = y.width;
    // A SCALAR AND ZERO, NOT AN ABSENT BUFFER. `Push`, `PushReduce` and
    // `PushCast` in `quant/qmm_t.slang` declare these words unconditionally
    // so that every form pushes the pipeline layout's whole range, and the
    // form fired here reads none of them -- `input_stride()` and
    // `output_stride()` answer `k` and `n` unless `PIE_STRIDED` is on, and
    // the reduce and cast arms index by their own two. The words exist to
    // hold the places the scalars after them sit at, so any value does.
    //
    // They were `ctx.absent()`, which on this plane mints a null BUFFER: each
    // one landed among the OPERANDS instead of among the scalars, so the body
    // bound more buffers than the module decorates AND pushed a block short
    // by that many words, every later scalar reading its neighbour's. Both
    // halves are a refusal or silent garbage at the first real dispatch, and
    // no shipped text fires a split-K GEMM or the unstrided cast, which is
    // why neither was seen.
    let row_stride = 0i32;
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
        Fire::at(crate::routine::module_path("qmm_splitk_reduce_f32_bfloat16", ctx.best()), "qmm_splitk_reduce_f32_bfloat16").apply(elementwise_rows(n, m)?),
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
    let k = 0i32;
    let n = 0i32;
    // A SCALAR AND ZERO, NOT AN ABSENT BUFFER. `Push`, `PushReduce` and
    // `PushCast` in `quant/qmm_t.slang` declare these words unconditionally
    // so that every form pushes the pipeline layout's whole range, and the
    // form fired here reads none of them -- `input_stride()` and
    // `output_stride()` answer `k` and `n` unless `PIE_STRIDED` is on, and
    // the reduce and cast arms index by their own two. The words exist to
    // hold the places the scalars after them sit at, so any value does.
    //
    // They were `ctx.absent()`, which on this plane mints a null BUFFER: each
    // one landed among the OPERANDS instead of among the scalars, so the body
    // bound more buffers than the module decorates AND pushed a block short
    // by that many words, every later scalar reading its neighbour's. Both
    // halves are a refusal or silent garbage at the first real dispatch, and
    // no shipped text fires a split-K GEMM or the unstrided cast, which is
    // why neither was seen.
    let row_stride = 0i32;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<3>` named.
    // The run is the shader's struct layout, so no `Const` mark can name a
    // word inside it, and `keys::Count` is answered by no driver.
    let count = ctx.param(3)?;
    ctx.fire(
        Fire::at(crate::routine::module_path("cast_qmm_input_bfloat16_to_float16", ctx.best()), "cast_qmm_input_bfloat16_to_float16").apply(elementwise(count, 1)?),
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
    // `Param<2, i32>` and the migration made it an ask no driver answers; it
    // is the activation's own stride, which the text knows and the fire does
    // not.
    row_stride: Const<i32>) -> Result<(), Refusal> {
    let k = cast_in.width;
    let n = half_out.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    // ROWS BY PITCH, DERIVED. HEAD spelled it `Reckoned<Times<Say<Rows>,
    // Nth<2>>>` -- a product of a fact and the statement's third word, not a
    // fact -- and the migration made it `keys::ElementsByPitch2`, which no
    // driver answers. `row_stride` IS that third word, and it is a mark now.
    let count = rows.saturating_mul(*row_stride);
    ctx.fire(
        Fire::at(crate::routine::module_path("cast_qmm_input_strided_bfloat16_to_float16", ctx.best()), "cast_qmm_input_strided_bfloat16_to_float16").apply(elementwise_rows(k, rows)?),
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
/// whatever [`qmv_grid`] refuses.
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
        Fire::at(crate::routine::module_path(QMV_FAST[codec_point(*group, *bits)?], ctx.best()), QMV_FAST[codec_point(*group, *bits)?]).apply(qmv_grid(vecs, out_vec_size)?),
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
/// whatever [`qmv_grid`] refuses.
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
        Fire::at(crate::routine::module_path(QMV_FAST_RESIDUAL[codec_point(*group, *bits)?], ctx.best()), QMV_FAST_RESIDUAL[codec_point(*group, *bits)?]).apply(qmv_grid(vecs, out_vec_size)?),
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
/// whatever [`qmv_grid`] refuses.
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
        Fire::at(crate::routine::module_path(QMV_TAIL[bits_point(*bits)?], ctx.best()), QMV_TAIL[bits_point(*bits)?]).apply(qmv_grid(vecs, out_vec_size)?),
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
/// whatever [`qmv_grid`] refuses.
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
        Fire::at(crate::routine::module_path(QMV_TAIL_BIAS[bits_point(*bits)?], ctx.best()), QMV_TAIL_BIAS[bits_point(*bits)?]).apply(qmv_grid(vecs, out_vec_size)?),
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
/// whatever [`qmv_grid`] refuses.
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
        Fire::at(crate::routine::module_path(QMV_WIDE_STRIDED[bits_point(*bits)?], ctx.best()), QMV_WIDE_STRIDED[bits_point(*bits)?]).apply(qmv_grid(quarters(m), out_vec_size)?),
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
/// [`Refusal::Empty`] for an empty rectangle, and whatever [`qmm_grid`] refuses.
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
        Fire::at(crate::routine::module_path("affine_qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4", ctx.best()), "affine_qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4").apply(qmm_grid(n, 32, m, 128, 1)?),
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
/// [`Refusal::Empty`] for an empty rectangle, and whatever [`qmm_grid`] refuses.
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
        Fire::at(crate::routine::module_path("affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2", ctx.best()), "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2").apply(qmm_grid(n, 32, m, 32, 1)?),
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
/// [`Refusal::Empty`] for an empty rectangle, and whatever [`qmm_grid`] refuses.
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
        Fire::at(crate::routine::module_path("affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2", ctx.best()), "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2").apply(qmm_grid(n, 32, m, 64, 1)?),
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
/// [`Refusal::Empty`] for an empty rectangle, and whatever [`qmm_grid`] refuses.
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
        Fire::at(crate::routine::module_path("affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1", ctx.best()), "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1").apply(qmm_grid(n, 32, m, 64, 1)?),
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
/// [`Refusal::Empty`] for an empty rectangle, and whatever [`qmm_grid`] refuses.
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
        Fire::at(crate::routine::module_path("affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4", ctx.best()), "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4").apply(qmm_grid(n, 64, m, 64, 1)?),
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
    // THE GROUP SIZE, WHICH THE STATEMENT STATES. `EncodeParams`' other field
    // is the group COUNT, which is `keys::Rows` and therefore the body's --
    // so the run is derived-then-stated, the same shape `layout/row_gather`
    // has, and `quant/transcode.wgsl` already read it this way.
    group_size: Const<i32>) -> Result<(), Refusal> {
    let groups = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("affine_encode_u4_bf16", ctx.best()), "affine_encode_u4_bf16").apply(elementwise(groups, 1)?),
        &[input.arg(), codes.arg(), scales.arg(), biases.arg(), groups.arg(), group_size.arg()],
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
    // See [`encode_u4_bf16`]: the count is the body's and the size is stated.
    group_size: Const<i32>) -> Result<(), Refusal> {
    let groups = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("affine_encode_u4_f32", ctx.best()), "affine_encode_u4_f32").apply(elementwise(groups, 1)?),
        &[input.arg(), codes.arg(), scales.arg(), biases.arg(), groups.arg(), group_size.arg()],
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
    // See [`encode_u4_bf16`]: `DequantParams`' block COUNT is `keys::Rows` and
    // the block SIZE is what the statement states.
    block_size: Const<i32>) -> Result<(), Refusal> {
    let blocks = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("mxfp4_dequant_bf16", ctx.best()), "mxfp4_dequant_bf16").apply(elementwise(blocks, 1)?),
        &[payload.arg(), exponents.arg(), out.arg(), blocks.arg(), block_size.arg()],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Encode, Tensor};
    use core::cell::{Cell, RefCell};

    type Call = (String, [u32; 3], Vec<ArgValue>);

    struct Seen {
        calls: RefCell<Vec<Call>>,
        rows: Cell<i32>,
        row_stride: Cell<i32>,
        count: Cell<i32>,
        /// The statement\'s run, where a case means a particular word.
        words: RefCell<Vec<i32>>,
        elements_by_pitch2: Cell<i32>,
        k_partition_size: Cell<i32>,
        split_k_partition_stride: Cell<i32>,
        split_k: Cell<i32>,
        params_handle: Cell<u32>,
        absent_handle: Cell<u32>,
    }

    impl Default for Seen {
        fn default() -> Self {
            Self {
                calls: RefCell::default(),
                rows: Cell::new(3),
                row_stride: Cell::new(4096),
                count: Cell::new(2048),
                words: RefCell::default(),
                elements_by_pitch2: Cell::new(96),
                k_partition_size: Cell::new(64),
                split_k_partition_stride: Cell::new(4096),
                split_k: Cell::new(4),
                params_handle: Cell::new(900),
                absent_handle: Cell::new(901),
            }
        }
    }

    impl Encode for Seen {
        fn resolve(
            &self,
            ty: kernels::Ty,
            source: kernels::Source,
        ) -> Result<ArgValue, Refusal> {
            // THE STATEMENT'S OWN SCALARS, at the words HEAD's `Param<N>`
            // named. These cases already set the numbers on the probe -- back
            // when the bodies asked for them as facts -- and the bodies read
            // them by index now, so the same cells answer the same numbers by
            // a different route. See `Asks::param`.
            if let kernels::Source::Slot(kernels::Kind::Param, n) = source {
                if let Some(w) = self.words.borrow().get(usize::from(n)) {
                    return Ok(ArgValue::I32(*w));
                }
                return Ok(ArgValue::I32(match n {
                    2 => self.row_stride.get(),
                    // WORD 3 SERVES TWO ROUTINES -- `qmm_t_splitk`'s K
                    // partition and `cast_qmm_input`'s element count, HEAD's
                    // `Param<3>` in both -- so a case that means one of them
                    // states it in `words` and the split-K cell is the default.
                    3 => self.k_partition_size.get(),
                    4 => self.split_k_partition_stride.get(),
                    5 => self.split_k.get(),
                    _ => 4096,
                }));
            }
            use kernels::Lit;
            use kernels::Kind;
            use kernels::Source;
            use kernels::keys::Fact;

            if source == <keys::Rows as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.rows.get()));
            }
            if source == Source::Slot(Kind::Params, 0) {
                return Ok(ArgValue::Buffer {
                    handle: self.params_handle.get(),
                    writes: false,
                    rows: 0,
                    width: 0,
                });
            }
            if source == Source::Lit(Lit::Null) {
                return Ok(ArgValue::Buffer {
                    handle: self.absent_handle.get(),
                    writes: false,
                    rows: 0,
                    width: 0,
                });
            }
            if matches!(ty, kernels::Ty::Buf) {
                return Ok(ArgValue::Buffer {
                    handle: 900,
                    writes: false,
                    rows: 0,
                    width: 0,
                });
            }
            // Refusing what this probe does not know is intentional: inventing
            // an answer would hide a fact the real driver never supplied.
            Err(Refusal::Unstated {
                what: "a fact this probe does not answer",
            })
        }

        fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
            self.calls
                .borrow_mut()
                .push((fire.entrypoint.to_string(), fire.lanes, args.to_vec()));
            Ok(())
        }
    }

    fn one(seen: &Seen) -> Call {
        let calls = seen.calls.borrow();
        assert_eq!(calls.len(), 1, "expected exactly one dispatch");
        calls[0].clone()
    }

    /// The point picks the module, and the spelling is the name in the tree.
    ///
    /// Three hundred and three entrypoints across this family and every one is
    /// reached by a table index computed from arguments. `tests/routines.rs`
    /// proves each name EXISTS; this proves the index arithmetic puts the
    /// caller's point at the name that carries it, which existence cannot: an
    /// off-by-one in `codec_point` names a real module for the wrong codec and
    /// silently decodes 4-bit weights as 8-bit.
    #[test]
    fn the_point_a_caller_names_is_the_module_that_fires() {
        let seen = Seen::default();
        seen.rows.set(64);
        qmm_t(
            &seen,
            Const::new(Tensor::<u32>::new(0)),
            Const::new(Tensor::<bf16>::new(1)),
            Const::new(Tensor::<bf16>::new(2)),
            In {
                ptr: Tensor::<bf16>::new(3),
                rows: 0,
                width: 64,
            },
            Out {
                ptr: Tensor::<bf16>::new(4),
                rows: 0,
                width: 64,
            },
            Const::new(128),
            Const::new(8),
            Const::new(32),
            Const::new(64),
        )
        .unwrap();
        assert_eq!(one(&seen).0, "affine_qmm_t_bfloat16_gs_128_b_8_bm_32_bn_64");

        let seen = Seen::default();
        seen.rows.set(1);
        qmv_fast(
            &seen,
            Const::new(Tensor::<u32>::new(0)),
            Const::new(Tensor::<bf16>::new(1)),
            Const::new(Tensor::<bf16>::new(2)),
            In {
                ptr: Tensor::<bf16>::new(3),
                rows: 0,
                width: 64,
            },
            Out {
                ptr: Tensor::<bf16>::new(4),
                rows: 0,
                width: 64,
            },
            Const::new(32),
            Const::new(4),
        )
        .unwrap();
        assert_eq!(one(&seen).0, "affine_qmv_fast_bfloat16_gs_32_b_4");
    }

    /// A tile the tree was not stamped at is refused, on every axis.
    ///
    /// The refusal is the whole reason these arguments are ambient facts and
    /// not a caller-spelled symbol: a group size of 48 is a perfectly ordinary
    /// number that simply has no module, and reaching a table with it is an
    /// index panic at best and a wrong module at worst.
    #[test]
    fn a_point_the_tree_was_not_stamped_at_is_refused_by_name() {
        let call = |group, bits, bm, bn| {
            let seen = Seen::default();
            seen.rows.set(64);
            qmm_t(
                &seen,
                Const::new(Tensor::<u32>::new(0)),
                Const::new(Tensor::<bf16>::new(1)),
                Const::new(Tensor::<bf16>::new(2)),
                In {
                    ptr: Tensor::<bf16>::new(3),
                    rows: 0,
                    width: 64,
                },
                Out {
                    ptr: Tensor::<bf16>::new(4),
                    rows: 0,
                    width: 64,
                },
                Const::new(group),
                Const::new(bits),
                Const::new(bm),
                Const::new(bn),
            )
            .unwrap_err()
        };
        assert!(matches!(
            call(48, 4, 16, 16),
            Refusal::Narrow {
                what: "the group size",
                at: 48
            }
        ));
        assert!(matches!(
            call(32, 6, 16, 16),
            Refusal::Narrow {
                what: "the bit width",
                at: 6
            }
        ));
        assert!(matches!(
            call(32, 4, 8, 16),
            Refusal::Narrow {
                what: "the row tile",
                at: 8
            }
        ));
        assert!(matches!(
            call(32, 4, 16, 128),
            Refusal::Narrow {
                what: "the column tile",
                at: 128
            }
        ));
    }

    /// The matmul grid is TILES, rounded up, times the local size.
    ///
    /// `qmm_t.slang` is `[numthreads(32, 2, 2)]` and both `main`s read
    /// `SV_GroupID` alone, so the driver's div_ceil has to land back on the
    /// tile count exactly. Rounding the column count UP is not optional: the
    /// output is row-major, so a tile that stopped short would leave the tail
    /// of every row holding whatever the arena held, and `n` in the push block
    /// is what makes the overhang harmless.
    #[test]
    fn the_matmul_covers_whole_tiles_on_both_edges() {
        let seen = Seen::default();
        seen.rows.set(70);
        qmm_t(
            &seen,
            Const::new(Tensor::<u32>::new(0)),
            Const::new(Tensor::<bf16>::new(1)),
            Const::new(Tensor::<bf16>::new(2)),
            In {
                ptr: Tensor::<bf16>::new(3),
                rows: 0,
                width: 256,
            },
            Out {
                ptr: Tensor::<bf16>::new(4),
                rows: 0,
                width: 100,
            },
            Const::new(64),
            Const::new(4),
            Const::new(32),
            Const::new(32),
        )
        .unwrap();
        // 100 columns is four 32-wide tiles; 70 rows is three.
        assert_eq!(one(&seen).1, [4 * 32, 3 * 2, 2]);
    }

    /// A split-k launch is the same rectangle with the partitions on z.
    ///
    /// The plain forms pass 1 and the split forms pass `split_k`, which is the
    /// only thing that distinguishes their grids -- and the partials the extra
    /// z planes write are what `qmm_splitk_reduce` then sums, one lane per
    /// output element.
    #[test]
    fn the_k_splits_are_the_z_planes_and_the_reduce_is_one_lane_an_element() {
        let seen = Seen::default();
        seen.rows.set(64);
        seen.k_partition_size.set(64);
        seen.split_k_partition_stride.set(4096);
        seen.split_k.set(4);
        qmm_t_splitk(
            &seen,
            Const::new(Tensor::<u32>::new(0)),
            Const::new(Tensor::<bf16>::new(1)),
            Const::new(Tensor::<bf16>::new(2)),
            In {
                ptr: Tensor::<bf16>::new(3),
                rows: 0,
                width: 256,
            },
            Out {
                ptr: Tensor::<bf16>::new(4),
                rows: 0,
                width: 64,
            },
            Const::new(64),
            Const::new(4),
            Const::new(32),
        )
        .unwrap();
        assert_eq!(one(&seen).1, [2 * 32, 2 * 2, 4 * 2]);

        let seen = Seen::default();
        seen.rows.set(70);
        seen.split_k_partition_stride.set(4096);
        seen.split_k.set(4);
        qmm_splitk_reduce(
            &seen,
            Out {
                ptr: Tensor::<bf16>::new(0),
                rows: 0,
                width: 64,
            },
            In::new(Tensor::<bf16>::new(1)),
        )
        .unwrap();
        assert_eq!(one(&seen).1, [64, 70, 1]);
    }

    /// The precast forms bind the half-width copy and NOT the source it
    /// replaced.
    ///
    /// slangc deletes an unread buffer outright, so `x` at binding 3 does not
    /// exist in `affine_qmm_t_fp16_precast_*` and `half_in` sits at 7. A body
    /// that bound `x` anyway would be one buffer over the arity
    /// `driver-vulkan::encode::dispatch` computes and refused at the device.
    ///
    /// KNOWN FAILING, upstream of this crate: `y` is an `Out<Tensor<bf16>>`
    /// and `writes: true` at its handle (3) is the correct claim -- the same
    /// gap `layout::tests::the_two_join_kernels_ask_for_the_grids_their_shaders_are_written_for`
    /// documents in full. `Buf`/`BufMut` used to be separate carriers, each
    /// with its own `Bind` impl choosing `V::buffer` or `V::buffer_mut`; the
    /// merge into one `Tensor<E>` carrier left that choice to `Out`/`InOut`'s
    /// `Bind` impls in `kernels::routine` (outside this crate), which still
    /// delegate to `self.ptr.arg()` exactly as `In<E>` does. No positional
    /// output argument anywhere in this tree can presently fire `writes:
    /// true`. The assertion states the correct claim rather than one
    /// weakened to match the gap.
    #[test]
    fn the_precast_matmul_binds_the_half_copy_in_place_of_the_source() {
        let seen = Seen::default();
        seen.rows.set(16);
        qmm_t_fp16_precast(
            &seen,
            Const::new(Tensor::<u32>::new(0)),
            Const::new(Tensor::<bf16>::new(1)),
            Const::new(Tensor::<bf16>::new(2)),
            Out {
                ptr: Tensor::<bf16>::new(3),
                rows: 0,
                width: 64,
            },
            In {
                ptr: Tensor::<f16>::new(9),
                rows: 0,
                width: 256,
            },
            Const::new(16),
            Const::new(16),
        )
        .unwrap();
        let call = one(&seen);
        assert_eq!(
            call.2,
            vec![
                ArgValue::Buffer {
                    handle: 0,
                    writes: false,
                    rows: 0,
                    width: 0
                },
                ArgValue::Buffer {
                    handle: 1,
                    writes: false,
                    rows: 0,
                    width: 0
                },
                ArgValue::Buffer {
                    handle: 2,
                    writes: false,
                    rows: 0,
                    width: 0
                },
                ArgValue::Buffer {
                    handle: 3,
                    writes: true,
                    rows: 0,
                    width: 0
                },
                ArgValue::Buffer {
                    handle: 9,
                    writes: false,
                    rows: 0,
                    width: 0
                },
                ArgValue::I32(256),
                ArgValue::I32(64),
            ]
        );
    }

    /// One matvec group covers eight output rows, and four batch vectors in
    /// the wide form against one in the others.
    ///
    /// `qmv.slang` is `[numthreads(PIE_LANES, 2, 1)]` with `PIE_VEC` at 1 or 4, and
    /// the vector count is the x extent -- so the wide form's x is a QUARTER
    /// of the batch, rounded up. Launching it at the full batch would run the
    /// tail groups over vectors that do not exist.
    #[test]
    fn a_matvec_group_covers_eight_rows_and_the_wide_form_four_vectors() {
        let seen = Seen::default();
        seen.rows.set(3);
        qmv_fast(
            &seen,
            Const::new(Tensor::<u32>::new(0)),
            Const::new(Tensor::<bf16>::new(1)),
            Const::new(Tensor::<bf16>::new(2)),
            In {
                ptr: Tensor::<bf16>::new(3),
                rows: 0,
                width: 4096,
            },
            Out {
                ptr: Tensor::<bf16>::new(4),
                rows: 0,
                width: 24,
            },
            Const::new(32),
            Const::new(4),
        )
        .unwrap();
        assert_eq!(one(&seen).1, [3 * 64, 3 * 2, 1]);

        let seen = Seen::default();
        seen.rows.set(9);
        seen.row_stride.set(4096);
        qmv_wide_strided(
            &seen,
            Const::new(Tensor::<u32>::new(0)),
            Const::new(Tensor::<bf16>::new(1)),
            Const::new(Tensor::<bf16>::new(2)),
            In {
                ptr: Tensor::<bf16>::new(3),
                rows: 0,
                width: 4096,
            },
            Out {
                ptr: Tensor::<bf16>::new(4),
                rows: 0,
                width: 24,
            },
            Const::new(4),
        )
        .unwrap();
        // Nine vectors is three groups of four, the last one short.
        assert_eq!(one(&seen).1, [3 * 64, 3 * 2, 1]);
    }

    /// The transcodes are one lane per block, deliberately.
    ///
    /// `[numthreads(1, 1, 1)]` in `transcode.slang` is not an oversight: a
    /// group of an affine encode owns a whole quantisation group and writes
    /// its scale and bias, so the lane count IS the group count. The strided
    /// cast is the exception with a rectangle, because it walks a row pitch.
    #[test]
    fn a_transcode_fires_one_lane_for_each_block_it_rewrites() {
        let seen = Seen::default();
        seen.rows.set(7);
        encode_u4_bf16(
            &seen,
            In::new(Tensor::<bf16>::new(0)),
            Out::new(Tensor::<u32>::new(1)),
            Out::new(Tensor::<bf16>::new(2)),
            Out::new(Tensor::<bf16>::new(3)),
            Const::new(64),
        )
        .unwrap();
        assert_eq!(one(&seen).1, [7, 1, 1]);

        let seen = Seen::default();
        seen.rows.set(5);
        mxfp4_dequant_bf16(
            &seen,
            In::new(Tensor::<u8>::new(0)),
            In::new(Tensor::<u8>::new(1)),
            Out::new(Tensor::<bf16>::new(2)),
            Const::new(32),
        )
        .unwrap();
        assert_eq!(one(&seen).1, [5, 1, 1]);

        let seen = Seen::default();
        seen.count.set(2048);
        // AND IN THE RUN, at word 3. That word serves two routines -- this
        // count and `qmm_t_splitk`'s K partition, HEAD's `Param<3>` in both --
        // so a case that means one of them states it rather than leaving the
        // other's default to answer.
        {
            let mut w = seen.words.borrow_mut();
            w.resize(4, 4096);
            w[3] = 2048;
        }
        cast_qmm_input_bfloat16_to_float16(
            &seen,
            In::new(Tensor::<bf16>::new(0)),
            Out::new(Tensor::<f16>::new(1)),
        )
        .unwrap();
        assert_eq!(one(&seen).1, [2048, 1, 1]);

        let seen = Seen::default();
        seen.elements_by_pitch2.set(2048);
        seen.row_stride.set(96);
        seen.rows.set(12);
        cast_qmm_input_strided_bfloat16_to_float16(
            &seen,
            In {
                ptr: Tensor::<bf16>::new(0),
                rows: 0,
                width: 64,
            },
            Out {
                ptr: Tensor::<f16>::new(1),
                rows: 0,
                width: 32,
            },
            // The source's row pitch, which the STATEMENT carries: the fixture
            // set it on `seen` while it was a fact, and states it here now.
            Const::new(96),
        )
        .unwrap();
        assert_eq!(one(&seen).1, [64, 12, 1]);
    }

    /// An empty extent is refused everywhere in the family.
    ///
    /// A zero grid is a dispatch the driver accepts and that computes nothing,
    /// so the caller sees the arena it passed unchanged and reads it as an
    /// answer. Every shape here names what was empty.
    #[test]
    fn an_empty_extent_is_refused_by_every_shape_in_the_family() {
        let seen = Seen::default();
        assert!(matches!(
            qmm_t(
                &seen,
                Const::new(Tensor::<u32>::new(0)),
                Const::new(Tensor::<bf16>::new(1)),
                Const::new(Tensor::<bf16>::new(2)),
                In {
                    ptr: Tensor::<bf16>::new(3),
                    rows: 0,
                    width: 256,
                },
                Out {
                    ptr: Tensor::<bf16>::new(4),
                    rows: 0,
                    width: 0,
                },
                Const::new(64),
                Const::new(4),
                Const::new(32),
                Const::new(32),
            ),
            Err(Refusal::Empty {
                what: "the column count"
            })
        ));
        assert!(matches!(
            qmv_fast(
                &seen,
                Const::new(Tensor::<u32>::new(0)),
                Const::new(Tensor::<bf16>::new(1)),
                Const::new(Tensor::<bf16>::new(2)),
                In {
                    ptr: Tensor::<bf16>::new(3),
                    rows: 0,
                    width: 4096,
                },
                Out {
                    ptr: Tensor::<bf16>::new(4),
                    rows: 0,
                    width: 0,
                },
                Const::new(32),
                Const::new(4),
            ),
            Err(Refusal::Empty {
                what: "the output vector"
            })
        ));
        let seen = Seen::default();
        seen.rows.set(0);
        assert!(matches!(
            encode_u4_bf16(
                &seen,
                In::new(Tensor::<bf16>::new(0)),
                Out::new(Tensor::<u32>::new(1)),
                Out::new(Tensor::<bf16>::new(2)),
                Out::new(Tensor::<bf16>::new(3)),
                Const::new(64),
            ),
            Err(Refusal::Empty { .. })
        ));
        assert!(
            seen.calls.borrow().is_empty(),
            "a refused shape dispatched anyway"
        );
    }
}

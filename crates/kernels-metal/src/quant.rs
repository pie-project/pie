//! The affine-quantised projections, and the codecs around them.
//!
//! This is 32 of the 99 rows and 304 of the 480 entrypoints, and the whole
//! argument of `.wiki/kernel-metal-refactor.md` §2 is visible here: `qmm_t` is
//! ONE template body in `quantized_qmm_t.metal` stamped over (group x bits x
//! row tile x column tile), and enumerating its 54 instantiations as 54 rows
//! would state the macro's job a second time by hand.
//!
//! The five `_wm_`/`_wn_` rows are the exception that proves it: they are
//! `host_name` lines typed out at `quantized_qmm_t.metal:2918-2966` rather
//! than stamped, so they are five kernels and get five rows.

use kernels::{Axis, KernelSig, kernel};

use crate::axes::*;

pub static KERNELS: &[KernelSig] = &[
    kernel!(cast_qmm_input_bfloat16_to_float16 "cast_qmm_input_bfloat16_to_float16"), // quantized_qmm_t.metal
    kernel!(cast_qmm_input_strided_bfloat16_to_float16 "cast_qmm_input_strided_bfloat16_to_float16"), // quantized_qmm_t.metal
    kernel!(encode_u4_bf16 "affine_encode_u4_bf16"), // transcode.metal
    kernel!(encode_u4_f32 "affine_encode_u4_f32"),   // transcode.metal
    kernel!(mxfp4_dequant_bf16 "mxfp4_dequant_bf16"), // transcode.metal
    // 1 in quantized_qmm_t.metal
    kernel!(qmm_splitk_reduce "qmm_splitk_reduce", axes = &[BF16]),
    // 1 in quantized_qmm_t.metal
    kernel!(qmm_splitk_reduce_f32 "qmm_splitk_reduce_f32", axes = &[BF16]),
    // 54 in quantized_qmm_t.metal
    kernel!(qmm_t "affine_qmm_t", file = Some("quant/qmm_t.metal"), launch = kernels::LaunchRule::Qmm, axes = &[BF16, GROUP, BITS, TILE_M, TILE_N]),
    // `affine_qmm_t_aligned` has no row: it is the TEMPLATE the two below are
    // stamped from, and the census counted it as an entrypoint until the
    // template guard learned that a parameter list wraps.
    kernel!(qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4 "affine_qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4"), // quantized_qmm_t.metal
    kernel!(qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2 "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2"), // quantized_qmm_t.metal
    kernel!(qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2 "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2"), // quantized_qmm_t.metal
    kernel!(qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1 "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1"), // quantized_qmm_t.metal
    kernel!(qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4 "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4"), // quantized_qmm_t.metal
    // 54 in quantized_qmm_t.metal
    kernel!(qmm_t_bias "affine_qmm_t_bias", axes = &[BF16, GROUP, BITS, TILE_M, TILE_N]),
    // 9 in quantized_qmm_t.metal
    kernel!(qmm_t_bias_fp16_precast "affine_qmm_t_bias_fp16_precast",
        axes = &[BF16, GROUP_64, BITS_4, TILE_M, TILE_N]),
    // 9 in quantized_qmm_t.metal
    kernel!(qmm_t_fp16_precast "affine_qmm_t_fp16_precast",
        axes = &[BF16, GROUP_64, BITS_4, TILE_M, TILE_N]),
    // 54 in quantized_qmm_t.metal
    kernel!(qmm_t_residual "affine_qmm_t_residual", file = Some("quant/qmm_t.metal"), launch = kernels::LaunchRule::Qmm, axes = &[BF16, GROUP, BITS, TILE_M, TILE_N]),
    // 9 in quantized_qmm_t.metal
    kernel!(qmm_t_residual_fp16_precast "affine_qmm_t_residual_fp16_precast",
        axes = &[BF16, GROUP_64, BITS_4, TILE_M, TILE_N]),
    // 18 in quantized_qmm_t.metal
    kernel!(qmm_t_splitk "affine_qmm_t_splitk", axes = &[BF16, GROUP, BITS, TILE_M, TILE_N_32]),
    // 18 in quantized_qmm_t.metal
    kernel!(qmm_t_splitk_f32 "affine_qmm_t_splitk_f32",
        axes = &[BF16, GROUP, BITS, TILE_M, TILE_N_32]),
    // 3 in quantized_qmm_t.metal
    kernel!(qmm_t_splitk_fp16_precast "affine_qmm_t_splitk_fp16_precast",
        axes = &[BF16, GROUP_64, BITS_4, TILE_M, TILE_N_32]),
    // 3 in quantized_qmm_t.metal
    kernel!(qmm_t_splitk_fp16_precast_f32 "affine_qmm_t_splitk_fp16_precast_f32",
        axes = &[BF16, GROUP_64, BITS_4, TILE_M, TILE_N_32]),
    // 18 in quantized_qmm_t.metal
    kernel!(qmm_t_strided "affine_qmm_t_strided", axes = &[BF16, GROUP, BITS, TILE_M, TILE_N_32]),
    // 3 in quantized_qmm_t.metal
    kernel!(qmm_t_strided_fp16_precast "affine_qmm_t_strided_fp16_precast",
        axes = &[BF16, GROUP_64, BITS_4, TILE_M, TILE_N_32]),
    // 3 in quantized_qmm_t.metal
    kernel!(qmm_t_strided_fp16_precast_residual "affine_qmm_t_strided_fp16_precast_residual",
        axes = &[BF16, GROUP_64, BITS_4, TILE_M, TILE_N_32]),
    // 18 in quantized_qmm_t.metal
    kernel!(qmm_t_strided_residual "affine_qmm_t_strided_residual",
        axes = &[BF16, GROUP, BITS, TILE_M, TILE_N_32]),
    // 6 in quantized_qmv.metal
    kernel!(qmv_fast "affine_qmv_fast", file = Some("quant/qmv.metal"), launch = kernels::LaunchRule::Qmv, axes = &[BF16, GROUP, BITS]),
    // 6 in quantized_qmv.metal
    kernel!(qmv_fast_residual "affine_qmv_fast_residual", file = Some("quant/qmv.metal"), launch = kernels::LaunchRule::Qmv, axes = &[BF16, GROUP, BITS]),
    // 2 in quantized_qmv.metal
    kernel!(qmv_tail "affine_qmv_tail", axes = &[BF16, GROUP_64, BITS]),
    // 2 in quantized_qmv.metal
    kernel!(qmv_tail_bias "affine_qmv_tail_bias", axes = &[BF16, GROUP_64, BITS]),
    // 2 in quantized_qmm_t.metal
    kernel!(qmv_wide_strided "affine_qmv_wide_strided",
        axes = &[BF16, GROUP_64, BITS, Axis { what: "value dim", points: &["_v_4"] }, Axis { what: "k-loop unroll", points: &["_kl_8"] }]),
];

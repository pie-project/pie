//! The affine-quantised projections, and the codecs around them.
//!
//! This is 32 of the 99 rows and 304 of the 480 entrypoints, and the whole
//! argument for an axis is visible here: `qmm_t` is ONE body in
//! `quant/qmm_t.comp` compiled over (group x bits x row tile x column tile),
//! and enumerating its 54 instantiations as 54 rows would state the
//! instantiation matrix a second time by hand.
//!
//! The five `_wm_`/`_wn_` rows are the exception that proves it: on the Metal
//! side they are `host_name` lines typed out rather than stamped, so they are
//! five kernels and get five rows, and this table keeps that distinction
//! because it is a real one about what the bodies share.

use kernels::{Axis, KernelSig, kernel};

use crate::axes::*;

pub static KERNELS: &[KernelSig] = &[
    kernel!(cast_qmm_input_bfloat16_to_float16 "cast_qmm_input_bfloat16_to_float16"), // quant/qmm_t.comp
    kernel!(cast_qmm_input_strided_bfloat16_to_float16 "cast_qmm_input_strided_bfloat16_to_float16"), // quant/qmm_t.comp
    kernel!(encode_u4_bf16 "affine_encode_u4_bf16"), // quant/transcode.comp
    kernel!(encode_u4_f32 "affine_encode_u4_f32"),   // quant/transcode.comp
    kernel!(mxfp4_dequant_bf16 "mxfp4_dequant_bf16"), // quant/transcode.comp
    // 1 in quant/qmm_t.comp
    kernel!(qmm_splitk_reduce "qmm_splitk_reduce", axes = &[BF16]),
    // 1 in quant/qmm_t.comp
    kernel!(qmm_splitk_reduce_f32 "qmm_splitk_reduce_f32", axes = &[BF16]),
    // 54 in quant/qmm_t.comp
    // The batched projection, and its operand order is the GEMV's: weights,
    // then the activation, then the result, then the two extents. The template
    // is `affine_qmm_t_aligned`, so a reader diffing this against the shader
    // looks for that name.
    kernel!(qmm_t "affine_qmm_t", file = Some("quant/qmm_t.comp"), launch = kernels::LaunchRule::Qmm,
        operands = kernels::operands![
            w: Buf <- kernels::Source::Weight(0),
            scales: Buf <- kernels::Source::Weight(1),
            biases: Buf <- kernels::Source::Weight(2),
            x: Buf <- kernels::Source::In(0),
            y: BufMut <- kernels::Source::Out(0),
            k: I32 <- kernels::Source::Param(0),
            n: I32 <- kernels::Source::Param(1),
        ],
        axes = &[BF16, GROUP, BITS, TILE_M, TILE_N]),
    // `affine_qmm_t_aligned` has no row: it is the TEMPLATE the two below are
    // stamped from, and the census counted it as an entrypoint until the
    // template guard learned that a parameter list wraps.
    kernel!(qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4 "affine_qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4"), // quant/qmm_t.comp
    kernel!(qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2 "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2"), // quant/qmm_t.comp
    kernel!(qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2 "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2"), // quant/qmm_t.comp
    kernel!(qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1 "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1"), // quant/qmm_t.comp
    kernel!(qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4 "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4"), // quant/qmm_t.comp
    // 54 in quant/qmm_t.comp
    kernel!(qmm_t_bias "affine_qmm_t_bias", axes = &[BF16, GROUP, BITS, TILE_M, TILE_N]),
    // 9 in quant/qmm_t.comp
    kernel!(qmm_t_bias_fp16_precast "affine_qmm_t_bias_fp16_precast",
        axes = &[BF16, GROUP_64, BITS_4, TILE_M, TILE_N]),
    // 9 in quant/qmm_t.comp
    kernel!(qmm_t_fp16_precast "affine_qmm_t_fp16_precast",
        axes = &[BF16, GROUP_64, BITS_4, TILE_M, TILE_N]),
    // 54 in quant/qmm_t.comp
    kernel!(qmm_t_residual "affine_qmm_t_residual", file = Some("quant/qmm_t.comp"), launch = kernels::LaunchRule::Qmm,
        operands = kernels::operands![
            w: Buf <- kernels::Source::Weight(0),
            scales: Buf <- kernels::Source::Weight(1),
            biases: Buf <- kernels::Source::Weight(2),
            x: Buf <- kernels::Source::In(0),
            y: BufMut <- kernels::Source::Out(0),
            k: I32 <- kernels::Source::Param(0),
            n: I32 <- kernels::Source::Param(1),
            residual: Buf <- kernels::Source::In(1),
        ],
        axes = &[BF16, GROUP, BITS, TILE_M, TILE_N]),
    // 9 in quant/qmm_t.comp
    kernel!(qmm_t_residual_fp16_precast "affine_qmm_t_residual_fp16_precast",
        axes = &[BF16, GROUP_64, BITS_4, TILE_M, TILE_N]),
    // 18 in quant/qmm_t.comp
    kernel!(qmm_t_splitk "affine_qmm_t_splitk", axes = &[BF16, GROUP, BITS, TILE_M, TILE_N_32]),
    // 18 in quant/qmm_t.comp
    kernel!(qmm_t_splitk_f32 "affine_qmm_t_splitk_f32",
        axes = &[BF16, GROUP, BITS, TILE_M, TILE_N_32]),
    // 3 in quant/qmm_t.comp
    kernel!(qmm_t_splitk_fp16_precast "affine_qmm_t_splitk_fp16_precast",
        axes = &[BF16, GROUP_64, BITS_4, TILE_M, TILE_N_32]),
    // 3 in quant/qmm_t.comp
    kernel!(qmm_t_splitk_fp16_precast_f32 "affine_qmm_t_splitk_fp16_precast_f32",
        axes = &[BF16, GROUP_64, BITS_4, TILE_M, TILE_N_32]),
    // 18 in quant/qmm_t.comp
    kernel!(qmm_t_strided "affine_qmm_t_strided", axes = &[BF16, GROUP, BITS, TILE_M, TILE_N_32]),
    // 3 in quant/qmm_t.comp
    kernel!(qmm_t_strided_fp16_precast "affine_qmm_t_strided_fp16_precast",
        axes = &[BF16, GROUP_64, BITS_4, TILE_M, TILE_N_32]),
    // 3 in quant/qmm_t.comp
    kernel!(qmm_t_strided_fp16_precast_residual "affine_qmm_t_strided_fp16_precast_residual",
        axes = &[BF16, GROUP_64, BITS_4, TILE_M, TILE_N_32]),
    // 18 in quant/qmm_t.comp
    kernel!(qmm_t_strided_residual "affine_qmm_t_strided_residual",
        axes = &[BF16, GROUP, BITS, TILE_M, TILE_N_32]),
    // 6 in quant/qmv.comp
    // The loudest misbinding, and the reason the rows grew operands at all:
    // this declares its WEIGHTS FIRST and the trace states them last, so
    // positional binding put the activation where the packed weight belongs.
    // Every projection of every layer.
    kernel!(qmv_fast "affine_qmv_fast", file = Some("quant/qmv.comp"), launch = kernels::LaunchRule::Qmv,
        operands = kernels::operands![
            w: Buf <- kernels::Source::Weight(0),
            scales: Buf <- kernels::Source::Weight(1),
            biases: Buf <- kernels::Source::Weight(2),
            x: Buf <- kernels::Source::In(0),
            y: BufMut <- kernels::Source::Out(0),
            in_vec_size: I32 <- kernels::Source::Param(0),
            out_vec_size: I32 <- kernels::Source::Param(1),
        ],
        axes = &[BF16, GROUP, BITS]),
    // 6 in quant/qmv.comp
    // The same, plus the block residual its epilogue folds — which the trace
    // states as a second INPUT and the kernel takes at the very end.
    kernel!(qmv_fast_residual "affine_qmv_fast_residual", file = Some("quant/qmv.comp"), launch = kernels::LaunchRule::Qmv,
        operands = kernels::operands![
            w: Buf <- kernels::Source::Weight(0),
            scales: Buf <- kernels::Source::Weight(1),
            biases: Buf <- kernels::Source::Weight(2),
            x: Buf <- kernels::Source::In(0),
            y: BufMut <- kernels::Source::Out(0),
            in_vec_size: I32 <- kernels::Source::Param(0),
            out_vec_size: I32 <- kernels::Source::Param(1),
            residual: Buf <- kernels::Source::In(1),
        ],
        axes = &[BF16, GROUP, BITS]),
    // 2 in quant/qmv.comp
    kernel!(qmv_tail "affine_qmv_tail", axes = &[BF16, GROUP_64, BITS]),
    // 2 in quant/qmv.comp
    kernel!(qmv_tail_bias "affine_qmv_tail_bias", axes = &[BF16, GROUP_64, BITS]),
    // 2 in quant/qmm_t.comp
    kernel!(qmv_wide_strided "affine_qmv_wide_strided",
        axes = &[BF16, GROUP_64, BITS, Axis { what: "value dim", points: &["_v_4"] }, Axis { what: "k-loop unroll", points: &["_kl_8"] }]),
];

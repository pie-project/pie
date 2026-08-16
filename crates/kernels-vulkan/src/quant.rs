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

#![allow(clippy::too_many_arguments)]

use kernels::routine::Refusal;

use crate::routine::{keys, Ask, Bind, Block, Buf, BufMut, Ctx, Fire, Param, Routine};
use crate::routine::{InSlot, Nth, OutSlot, Reckoned, Say, Times, Weight};

/// The entrypoints this family's crossed routines spell, now that their
/// rows are gone. See [`crate::RETIRED`].
pub static ENTRYPOINTS: &[&str] = &[
    "cast_qmm_input_bfloat16_to_float16",
    "cast_qmm_input_strided_bfloat16_to_float16",
    "affine_encode_u4_bf16",
    "affine_encode_u4_f32",
    "mxfp4_dequant_bf16",
    "qmm_splitk_reduce_bfloat16",
    "qmm_splitk_reduce_f32_bfloat16",
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
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4",
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2",
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2",
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1",
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4",
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
    "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_16",
    "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_64",
    "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_16",
    "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_64",
    "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_16",
    "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32",
    "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_64",
    "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_16",
    "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_64",
    "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_16",
    "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_64",
    "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_16",
    "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32",
    "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_64",
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
    "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_16",
    "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_64",
    "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_16",
    "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_64",
    "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_16",
    "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32",
    "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_64",
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
    "affine_qmm_t_splitk_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_splitk_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_splitk_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32",
    "affine_qmm_t_splitk_fp16_precast_f32_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_splitk_fp16_precast_f32_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_splitk_fp16_precast_f32_bfloat16_gs_64_b_4_bm_64_bn_32",
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
    "affine_qmm_t_strided_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_strided_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_strided_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32",
    "affine_qmm_t_strided_fp16_precast_residual_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_strided_fp16_precast_residual_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_strided_fp16_precast_residual_bfloat16_gs_64_b_4_bm_64_bn_32",
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
    "affine_qmv_fast_bfloat16_gs_32_b_4",
    "affine_qmv_fast_bfloat16_gs_32_b_8",
    "affine_qmv_fast_bfloat16_gs_64_b_4",
    "affine_qmv_fast_bfloat16_gs_64_b_8",
    "affine_qmv_fast_bfloat16_gs_128_b_4",
    "affine_qmv_fast_bfloat16_gs_128_b_8",
    "affine_qmv_fast_residual_bfloat16_gs_32_b_4",
    "affine_qmv_fast_residual_bfloat16_gs_32_b_8",
    "affine_qmv_fast_residual_bfloat16_gs_64_b_4",
    "affine_qmv_fast_residual_bfloat16_gs_64_b_8",
    "affine_qmv_fast_residual_bfloat16_gs_128_b_4",
    "affine_qmv_fast_residual_bfloat16_gs_128_b_8",
    "affine_qmv_tail_bfloat16_gs_64_b_4",
    "affine_qmv_tail_bfloat16_gs_64_b_8",
    "affine_qmv_tail_bias_bfloat16_gs_64_b_4",
    "affine_qmv_tail_bias_bfloat16_gs_64_b_8",
    "affine_qmv_wide_strided_bfloat16_gs_64_b_4_v_4_kl_8",
    "affine_qmv_wide_strided_bfloat16_gs_64_b_8_v_4_kl_8",
];

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
pub fn qmm_t(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    x: InSlot<0, Buf>,
    y: OutSlot<0, BufMut>,
    k: Param<0, i32>,
    n: Param<1, i32>,
    group: Ask<keys::QuantGroup, i32>,
    bits: Ask<keys::QuantBits, i32>,
    bm: Ask<keys::TileM, i32>,
    bn: Ask<keys::TileN, i32>,
    m: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: QMM_T[qmm_point(*group, *bits, *bm, *bn)?],
            lanes: qmm_grid(*n, *bn, *m, *bm, 1)?,
        },
        &[w.v(), scales.v(), biases.v(), x.v(), y.v(), k.v(), n.v()],
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
pub fn qmm_t_bias(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    x: InSlot<0, Buf>,
    y: OutSlot<0, BufMut>,
    bias: Weight<3, Buf>,
    k: Param<0, i32>,
    n: Param<1, i32>,
    group: Ask<keys::QuantGroup, i32>,
    bits: Ask<keys::QuantBits, i32>,
    bm: Ask<keys::TileM, i32>,
    bn: Ask<keys::TileN, i32>,
    m: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: QMM_T_BIAS[qmm_point(*group, *bits, *bm, *bn)?],
            lanes: qmm_grid(*n, *bn, *m, *bm, 1)?,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            x.v(),
            y.v(),
            bias.v(),
            k.v(),
            n.v(),
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
pub fn qmm_t_residual(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    x: InSlot<0, Buf>,
    y: OutSlot<0, BufMut>,
    k: Param<0, i32>,
    n: Param<1, i32>,
    residual: InSlot<1, Buf>,
    group: Ask<keys::QuantGroup, i32>,
    bits: Ask<keys::QuantBits, i32>,
    bm: Ask<keys::TileM, i32>,
    bn: Ask<keys::TileN, i32>,
    m: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: QMM_T_RESIDUAL[qmm_point(*group, *bits, *bm, *bn)?],
            lanes: qmm_grid(*n, *bn, *m, *bm, 1)?,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            x.v(),
            y.v(),
            k.v(),
            n.v(),
            residual.v(),
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
pub fn qmm_t_fp16_precast(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    y: OutSlot<0, BufMut>,
    half_in: InSlot<0, Buf>,
    k: Param<0, i32>,
    n: Param<1, i32>,
    bm: Ask<keys::TileM, i32>,
    bn: Ask<keys::TileN, i32>,
    m: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: QMM_T_FP16_PRECAST[tile_point(*bm, *bn)?],
            lanes: qmm_grid(*n, *bn, *m, *bm, 1)?,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            y.v(),
            half_in.v(),
            k.v(),
            n.v(),
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
pub fn qmm_t_bias_fp16_precast(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    y: OutSlot<0, BufMut>,
    bias: Weight<3, Buf>,
    half_in: InSlot<0, Buf>,
    k: Param<0, i32>,
    n: Param<1, i32>,
    bm: Ask<keys::TileM, i32>,
    bn: Ask<keys::TileN, i32>,
    m: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: QMM_T_BIAS_FP16_PRECAST[tile_point(*bm, *bn)?],
            lanes: qmm_grid(*n, *bn, *m, *bm, 1)?,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            y.v(),
            bias.v(),
            half_in.v(),
            k.v(),
            n.v(),
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
pub fn qmm_t_residual_fp16_precast(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    y: OutSlot<0, BufMut>,
    residual: InSlot<1, Buf>,
    half_in: InSlot<0, Buf>,
    k: Param<0, i32>,
    n: Param<1, i32>,
    bm: Ask<keys::TileM, i32>,
    bn: Ask<keys::TileN, i32>,
    m: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: QMM_T_RESIDUAL_FP16_PRECAST[tile_point(*bm, *bn)?],
            lanes: qmm_grid(*n, *bn, *m, *bm, 1)?,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            y.v(),
            residual.v(),
            half_in.v(),
            k.v(),
            n.v(),
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
pub fn qmm_t_splitk(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    x: InSlot<0, Buf>,
    out: OutSlot<0, BufMut>,
    k: Param<0, i32>,
    n: Param<1, i32>,
    row_stride: i32,
    k_partition_size: Param<3, i32>,
    split_k_partition_stride: Param<4, i32>,
    split_k: Param<5, i32>,
    group: Ask<keys::QuantGroup, i32>,
    bits: Ask<keys::QuantBits, i32>,
    bm: Ask<keys::TileM, i32>,
    m: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: QMM_T_SPLITK[wide_point(*group, *bits, *bm)?],
            lanes: qmm_grid(*n, WIDE_BN, *m, *bm, *split_k)?,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            x.v(),
            out.v(),
            k.v(),
            n.v(),
            row_stride.v(),
            k_partition_size.v(),
            split_k_partition_stride.v(),
            split_k.v(),
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
pub fn qmm_t_splitk_f32(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    x: InSlot<0, Buf>,
    out: OutSlot<0, BufMut>,
    k: Param<0, i32>,
    n: Param<1, i32>,
    row_stride: i32,
    k_partition_size: Param<3, i32>,
    split_k_partition_stride: Param<4, i32>,
    split_k: Param<5, i32>,
    group: Ask<keys::QuantGroup, i32>,
    bits: Ask<keys::QuantBits, i32>,
    bm: Ask<keys::TileM, i32>,
    m: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: QMM_T_SPLITK_F32[wide_point(*group, *bits, *bm)?],
            lanes: qmm_grid(*n, WIDE_BN, *m, *bm, *split_k)?,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            x.v(),
            out.v(),
            k.v(),
            n.v(),
            row_stride.v(),
            k_partition_size.v(),
            split_k_partition_stride.v(),
            split_k.v(),
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
pub fn qmm_t_splitk_fp16_precast(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    out: OutSlot<0, BufMut>,
    half_in: InSlot<0, Buf>,
    k: Param<0, i32>,
    n: Param<1, i32>,
    row_stride: i32,
    k_partition_size: Param<3, i32>,
    split_k_partition_stride: Param<4, i32>,
    split_k: Param<5, i32>,
    bm: Ask<keys::TileM, i32>,
    m: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: QMM_T_SPLITK_FP16_PRECAST[row_tile_point(*bm)?],
            lanes: qmm_grid(*n, WIDE_BN, *m, *bm, *split_k)?,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            out.v(),
            half_in.v(),
            k.v(),
            n.v(),
            row_stride.v(),
            k_partition_size.v(),
            split_k_partition_stride.v(),
            split_k.v(),
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
pub fn qmm_t_splitk_fp16_precast_f32(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    out: OutSlot<0, BufMut>,
    half_in: InSlot<0, Buf>,
    k: Param<0, i32>,
    n: Param<1, i32>,
    row_stride: i32,
    k_partition_size: Param<3, i32>,
    split_k_partition_stride: Param<4, i32>,
    split_k: Param<5, i32>,
    bm: Ask<keys::TileM, i32>,
    m: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: QMM_T_SPLITK_FP16_PRECAST_F32[row_tile_point(*bm)?],
            lanes: qmm_grid(*n, WIDE_BN, *m, *bm, *split_k)?,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            out.v(),
            half_in.v(),
            k.v(),
            n.v(),
            row_stride.v(),
            k_partition_size.v(),
            split_k_partition_stride.v(),
            split_k.v(),
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
pub fn qmm_t_strided(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    x: InSlot<0, Buf>,
    y: OutSlot<0, BufMut>,
    k: Param<0, i32>,
    n: Param<1, i32>,
    row_stride: Param<2, i32>,
    group: Ask<keys::QuantGroup, i32>,
    bits: Ask<keys::QuantBits, i32>,
    bm: Ask<keys::TileM, i32>,
    m: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: QMM_T_STRIDED[wide_point(*group, *bits, *bm)?],
            lanes: qmm_grid(*n, WIDE_BN, *m, *bm, 1)?,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            x.v(),
            y.v(),
            k.v(),
            n.v(),
            row_stride.v(),
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
pub fn qmm_t_strided_residual(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    x: InSlot<0, Buf>,
    y: OutSlot<0, BufMut>,
    residual: InSlot<1, Buf>,
    k: Param<0, i32>,
    n: Param<1, i32>,
    row_stride: Param<2, i32>,
    group: Ask<keys::QuantGroup, i32>,
    bits: Ask<keys::QuantBits, i32>,
    bm: Ask<keys::TileM, i32>,
    m: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: QMM_T_STRIDED_RESIDUAL[wide_point(*group, *bits, *bm)?],
            lanes: qmm_grid(*n, WIDE_BN, *m, *bm, 1)?,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            x.v(),
            y.v(),
            residual.v(),
            k.v(),
            n.v(),
            row_stride.v(),
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
pub fn qmm_t_strided_fp16_precast(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    y: OutSlot<0, BufMut>,
    half_in: InSlot<0, Buf>,
    k: Param<0, i32>,
    n: Param<1, i32>,
    row_stride: Param<2, i32>,
    bm: Ask<keys::TileM, i32>,
    m: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: QMM_T_STRIDED_FP16_PRECAST[row_tile_point(*bm)?],
            lanes: qmm_grid(*n, WIDE_BN, *m, *bm, 1)?,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            y.v(),
            half_in.v(),
            k.v(),
            n.v(),
            row_stride.v(),
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
pub fn qmm_t_strided_fp16_precast_residual(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    y: OutSlot<0, BufMut>,
    residual: InSlot<1, Buf>,
    half_in: InSlot<0, Buf>,
    k: Param<0, i32>,
    n: Param<1, i32>,
    row_stride: Param<2, i32>,
    bm: Ask<keys::TileM, i32>,
    m: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: QMM_T_STRIDED_FP16_PRECAST_RESIDUAL[row_tile_point(*bm)?],
            lanes: qmm_grid(*n, WIDE_BN, *m, *bm, 1)?,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            y.v(),
            residual.v(),
            half_in.v(),
            k.v(),
            n.v(),
            row_stride.v(),
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
pub fn qmm_splitk_reduce(
    ctx: &Ctx<'_>,
    y: OutSlot<0, BufMut>,
    partial: InSlot<0, Buf>,
    k: i32,
    n: Param<1, i32>,
    row_stride: i32,
    split_k_partition_stride: Param<3, i32>,
    split_k: Param<4, i32>,
    m: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "qmm_splitk_reduce_bfloat16",
            lanes: crate::routine::elementwise_rows(*n, *m)?,
        },
        &[
            y.v(),
            partial.v(),
            k.v(),
            n.v(),
            row_stride.v(),
            split_k_partition_stride.v(),
            split_k.v(),
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
pub fn qmm_splitk_reduce_f32(
    ctx: &Ctx<'_>,
    y: OutSlot<0, BufMut>,
    partial: InSlot<0, Buf>,
    k: i32,
    n: Param<1, i32>,
    row_stride: i32,
    split_k_partition_stride: Param<3, i32>,
    split_k: Param<4, i32>,
    m: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "qmm_splitk_reduce_f32_bfloat16",
            lanes: crate::routine::elementwise_rows(*n, *m)?,
        },
        &[
            y.v(),
            partial.v(),
            k.v(),
            n.v(),
            row_stride.v(),
            split_k_partition_stride.v(),
            split_k.v(),
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
pub fn cast_qmm_input_bfloat16_to_float16(
    ctx: &Ctx<'_>,
    cast_in: InSlot<0, Buf>,
    half_out: OutSlot<0, BufMut>,
    k: i32,
    n: i32,
    row_stride: i32,
    count: Param<3, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "cast_qmm_input_bfloat16_to_float16",
            lanes: crate::routine::elementwise(*count, 1)?,
        },
        &[
            cast_in.v(),
            half_out.v(),
            k.v(),
            n.v(),
            row_stride.v(),
            count.v(),
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
pub fn cast_qmm_input_strided_bfloat16_to_float16(
    ctx: &Ctx<'_>,
    cast_in: InSlot<0, Buf>,
    half_out: OutSlot<0, BufMut>,
    k: Param<0, i32>,
    n: Param<1, i32>,
    row_stride: Param<2, i32>,
    // UNREAD by this walk, and filled anyway with the number it names.
    //
    // The block is shared with the packed cast, which reads `n` and `count`
    // and walks a flat range; this one reads `k` and `row_stride` and walks a
    // rectangle. A push field that says `count` and holds a zero is a trap
    // for whoever adds a `count` read to the strided walk later, so the
    // column states the product the name promises: how many elements this
    // rectangle spans, rows by pitch.
    count: Reckoned<Times<Say<keys::Rows>, Nth<2>>, i32>,
    rows: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "cast_qmm_input_strided_bfloat16_to_float16",
            lanes: crate::routine::elementwise_rows(*k, *rows)?,
        },
        &[
            cast_in.v(),
            half_out.v(),
            k.v(),
            n.v(),
            row_stride.v(),
            count.v(),
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
pub fn qmv_fast(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    x: InSlot<0, Buf>,
    y: OutSlot<0, BufMut>,
    in_vec_size: Param<0, i32>,
    out_vec_size: Param<1, i32>,
    group: Ask<keys::QuantGroup, i32>,
    bits: Ask<keys::QuantBits, i32>,
    vecs: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: QMV_FAST[codec_point(*group, *bits)?],
            lanes: qmv_grid(*vecs, *out_vec_size)?,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            x.v(),
            y.v(),
            in_vec_size.v(),
            out_vec_size.v(),
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
pub fn qmv_fast_residual(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    x: InSlot<0, Buf>,
    y: OutSlot<0, BufMut>,
    in_vec_size: Param<0, i32>,
    out_vec_size: Param<1, i32>,
    residual: InSlot<1, Buf>,
    group: Ask<keys::QuantGroup, i32>,
    bits: Ask<keys::QuantBits, i32>,
    vecs: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: QMV_FAST_RESIDUAL[codec_point(*group, *bits)?],
            lanes: qmv_grid(*vecs, *out_vec_size)?,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            x.v(),
            y.v(),
            in_vec_size.v(),
            out_vec_size.v(),
            residual.v(),
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
pub fn qmv_tail(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    x: InSlot<0, Buf>,
    y: OutSlot<0, BufMut>,
    in_vec_size: Param<0, i32>,
    out_vec_size: Param<1, i32>,
    bits: Ask<keys::QuantBits, i32>,
    vecs: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: QMV_TAIL[bits_point(*bits)?],
            lanes: qmv_grid(*vecs, *out_vec_size)?,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            x.v(),
            y.v(),
            in_vec_size.v(),
            out_vec_size.v(),
        ],
    )
}

/// The tail form with a per-row bias at binding 5.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a quantisation point the tree does not carry, and
/// whatever [`qmv_grid`] refuses.
pub fn qmv_tail_bias(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    x: InSlot<0, Buf>,
    y: OutSlot<0, BufMut>,
    bias: Weight<3, Buf>,
    in_vec_size: Param<0, i32>,
    out_vec_size: Param<1, i32>,
    bits: Ask<keys::QuantBits, i32>,
    vecs: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: QMV_TAIL_BIAS[bits_point(*bits)?],
            lanes: qmv_grid(*vecs, *out_vec_size)?,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            x.v(),
            y.v(),
            bias.v(),
            in_vec_size.v(),
            out_vec_size.v(),
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
pub fn qmv_wide_strided(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    x: InSlot<0, Buf>,
    y: OutSlot<0, BufMut>,
    in_vec_size: Param<0, i32>,
    out_vec_size: Param<1, i32>,
    row_stride: Param<2, i32>,
    m: Ask<keys::Rows, i32>,
    bits: Ask<keys::QuantBits, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: QMV_WIDE_STRIDED[bits_point(*bits)?],
            lanes: qmv_grid(quarters(*m), *out_vec_size)?,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            x.v(),
            y.v(),
            in_vec_size.v(),
            out_vec_size.v(),
            row_stride.v(),
            m.v(),
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
pub fn qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    x: InSlot<0, Buf>,
    y: OutSlot<0, BufMut>,
    k: Param<0, i32>,
    n: Param<1, i32>,
    m: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "affine_qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4",
            lanes: qmm_grid(*n, 32, *m, 128, 1)?,
        },
        &[w.v(), scales.v(), biases.v(), x.v(), y.v(), k.v(), n.v()],
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
pub fn qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    x: InSlot<0, Buf>,
    y: OutSlot<0, BufMut>,
    k: Param<0, i32>,
    n: Param<1, i32>,
    m: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2",
            lanes: qmm_grid(*n, 32, *m, 32, 1)?,
        },
        &[w.v(), scales.v(), biases.v(), x.v(), y.v(), k.v(), n.v()],
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
pub fn qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    x: InSlot<0, Buf>,
    y: OutSlot<0, BufMut>,
    k: Param<0, i32>,
    n: Param<1, i32>,
    m: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2",
            lanes: qmm_grid(*n, 32, *m, 64, 1)?,
        },
        &[w.v(), scales.v(), biases.v(), x.v(), y.v(), k.v(), n.v()],
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
pub fn qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    x: InSlot<0, Buf>,
    y: OutSlot<0, BufMut>,
    k: Param<0, i32>,
    n: Param<1, i32>,
    m: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1",
            lanes: qmm_grid(*n, 32, *m, 64, 1)?,
        },
        &[w.v(), scales.v(), biases.v(), x.v(), y.v(), k.v(), n.v()],
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
pub fn qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    x: InSlot<0, Buf>,
    y: OutSlot<0, BufMut>,
    k: Param<0, i32>,
    n: Param<1, i32>,
    m: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4",
            lanes: qmm_grid(*n, 64, *m, 64, 1)?,
        },
        &[w.v(), scales.v(), biases.v(), x.v(), y.v(), k.v(), n.v()],
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
pub fn encode_u4_bf16(
    ctx: &Ctx<'_>,
    input: InSlot<0, Buf>,
    codes: OutSlot<0, BufMut>,
    scales: OutSlot<1, BufMut>,
    biases: OutSlot<2, BufMut>,
    params: Block<Buf>,
    groups: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "affine_encode_u4_bf16",
            lanes: crate::routine::elementwise(*groups, 1)?,
        },
        &[input.v(), codes.v(), scales.v(), biases.v(), params.v()],
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
pub fn encode_u4_f32(
    ctx: &Ctx<'_>,
    input: InSlot<0, Buf>,
    codes: OutSlot<0, BufMut>,
    scales: OutSlot<1, BufMut>,
    biases: OutSlot<2, BufMut>,
    params: Block<Buf>,
    groups: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "affine_encode_u4_f32",
            lanes: crate::routine::elementwise(*groups, 1)?,
        },
        &[input.v(), codes.v(), scales.v(), biases.v(), params.v()],
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
pub fn mxfp4_dequant_bf16(
    ctx: &Ctx<'_>,
    payload: InSlot<0, Buf>,
    exponents: InSlot<1, Buf>,
    out: OutSlot<0, BufMut>,
    params: Block<Buf>,
    blocks: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "mxfp4_dequant_bf16",
            lanes: crate::routine::elementwise(*blocks, 1)?,
        },
        &[payload.v(), exponents.v(), out.v(), params.v()],
    )
}

/// The crossed rows of this family.
pub static ROUTINES: &[Routine] = &[
    crate::routine!(qmm_t),
    crate::routine!(qmm_t_bias),
    crate::routine!(qmm_t_residual),
    crate::routine!(qmm_t_fp16_precast),
    crate::routine!(qmm_t_bias_fp16_precast),
    crate::routine!(qmm_t_residual_fp16_precast),
    crate::routine!(qmm_t_splitk),
    crate::routine!(qmm_t_splitk_f32),
    crate::routine!(qmm_t_splitk_fp16_precast),
    crate::routine!(qmm_t_splitk_fp16_precast_f32),
    crate::routine!(qmm_t_strided),
    crate::routine!(qmm_t_strided_residual),
    crate::routine!(qmm_t_strided_fp16_precast),
    crate::routine!(qmm_t_strided_fp16_precast_residual),
    crate::routine!(qmm_splitk_reduce),
    crate::routine!(qmm_splitk_reduce_f32),
    crate::routine!(cast_qmm_input_bfloat16_to_float16),
    crate::routine!(cast_qmm_input_strided_bfloat16_to_float16),
    crate::routine!(qmv_fast),
    crate::routine!(qmv_fast_residual),
    crate::routine!(qmv_tail),
    crate::routine!(qmv_tail_bias),
    crate::routine!(qmv_wide_strided),
    crate::routine!(qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4),
    crate::routine!(qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2),
    crate::routine!(qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2),
    crate::routine!(qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1),
    crate::routine!(qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4),
    crate::routine!(encode_u4_bf16),
    crate::routine!(encode_u4_f32),
    crate::routine!(mxfp4_dequant_bf16),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Encode};
    use core::cell::RefCell;

    type Call = (String, [u32; 3], Vec<ArgValue>);

    #[derive(Default)]
    struct Seen(RefCell<Vec<Call>>);

    impl Encode for Seen {
        fn dispatch(&self, fire: Fire<'_>, args: &[ArgValue]) -> Result<(), Refusal> {
            self.0
                .borrow_mut()
                .push((fire.entrypoint.to_string(), fire.lanes, args.to_vec()));
            Ok(())
        }
    }

    fn one(seen: &Seen) -> Call {
        let calls = seen.0.borrow();
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
        qmm_t(
            &seen,
            Weight::new(Buf(0)),
            Weight::new(Buf(1)),
            Weight::new(Buf(2)),
            InSlot::new(Buf(3)),
            OutSlot::new(BufMut(4)),
            Param::new(64),
            Param::new(64),
            Ask::new(128),
            Ask::new(8),
            Ask::new(32),
            Ask::new(64),
            Ask::new(32),
        )
        .unwrap();
        assert_eq!(one(&seen).0, "affine_qmm_t_bfloat16_gs_128_b_8_bm_32_bn_64");

        let seen = Seen::default();
        qmv_fast(
            &seen,
            Weight::new(Buf(0)),
            Weight::new(Buf(1)),
            Weight::new(Buf(2)),
            InSlot::new(Buf(3)),
            OutSlot::new(BufMut(4)),
            Param::new(64),
            Param::new(64),
            Ask::new(32),
            Ask::new(4),
            Ask::new(1),
        )
        .unwrap();
        assert_eq!(one(&seen).0, "affine_qmv_fast_bfloat16_gs_32_b_4");
    }

    /// A tile the tree was not stamped at is refused, on every axis.
    ///
    /// The refusal is the whole reason these arguments are `Env` and not
    /// pushed words: a group size of 48 is a perfectly ordinary number that
    /// simply has no module, and reaching a table with it is an index panic at
    /// best and a wrong module at worst.
    #[test]
    fn a_point_the_tree_was_not_stamped_at_is_refused_by_name() {
        let call = |group, bits, bm, bn| {
            let seen = Seen::default();
            qmm_t(
                &seen,
                Weight::new(Buf(0)),
                Weight::new(Buf(1)),
                Weight::new(Buf(2)),
                InSlot::new(Buf(3)),
                OutSlot::new(BufMut(4)),
                Param::new(64),
                Param::new(64),
                Ask::new(group),
                Ask::new(bits),
                Ask::new(bm),
                Ask::new(bn),
                Ask::new(32),
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
        qmm_t(
            &seen,
            Weight::new(Buf(0)),
            Weight::new(Buf(1)),
            Weight::new(Buf(2)),
            InSlot::new(Buf(3)),
            OutSlot::new(BufMut(4)),
            Param::new(256),
            Param::new(100),
            Ask::new(64),
            Ask::new(4),
            Ask::new(32),
            Ask::new(32),
            Ask::new(70),
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
        qmm_t_splitk(
            &seen,
            Weight::new(Buf(0)),
            Weight::new(Buf(1)),
            Weight::new(Buf(2)),
            InSlot::new(Buf(3)),
            OutSlot::new(BufMut(4)),
            Param::new(256),
            Param::new(64),
            64,
            Param::new(64),
            Param::new(4096),
            Param::new(4),
            Ask::new(64),
            Ask::new(4),
            Ask::new(32),
            Ask::new(64),
        )
        .unwrap();
        assert_eq!(one(&seen).1, [2 * 32, 2 * 2, 4 * 2]);

        let seen = Seen::default();
        qmm_splitk_reduce(&seen, OutSlot::new(BufMut(0)), InSlot::new(Buf(1)), 256, Param::new(64), 64, Param::new(4096), Param::new(4), Ask::new(70)).unwrap();
        assert_eq!(one(&seen).1, [64, 70, 1]);
    }

    /// The precast forms bind the half-width copy and NOT the source it
    /// replaced.
    ///
    /// slangc deletes an unread buffer outright, so `x` at binding 3 does not
    /// exist in `affine_qmm_t_fp16_precast_*` and `half_in` sits at 7. A body
    /// that bound `x` anyway would be one buffer over the arity
    /// `driver-vulkan::encode::dispatch` computes and refused at the device.
    #[test]
    fn the_precast_matmul_binds_the_half_copy_in_place_of_the_source() {
        let seen = Seen::default();
        qmm_t_fp16_precast(
            &seen,
            Weight::new(Buf(0)),
            Weight::new(Buf(1)),
            Weight::new(Buf(2)),
            OutSlot::new(BufMut(3)),
            InSlot::new(Buf(9)),
            Param::new(256),
            Param::new(64),
            Ask::new(16),
            Ask::new(16),
            Ask::new(16),
        )
        .unwrap();
        let call = one(&seen);
        assert_eq!(
            call.2,
            vec![
                ArgValue::Buffer {
                    handle: 0,
                    writes: false
                },
                ArgValue::Buffer {
                    handle: 1,
                    writes: false
                },
                ArgValue::Buffer {
                    handle: 2,
                    writes: false
                },
                ArgValue::Buffer {
                    handle: 3,
                    writes: true
                },
                ArgValue::Buffer {
                    handle: 9,
                    writes: false
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
        qmv_fast(
            &seen,
            Weight::new(Buf(0)),
            Weight::new(Buf(1)),
            Weight::new(Buf(2)),
            InSlot::new(Buf(3)),
            OutSlot::new(BufMut(4)),
            Param::new(4096),
            Param::new(24),
            Ask::new(32),
            Ask::new(4),
            Ask::new(3),
        )
        .unwrap();
        assert_eq!(one(&seen).1, [3 * 64, 3 * 2, 1]);

        let seen = Seen::default();
        qmv_wide_strided(
            &seen,
            Weight::new(Buf(0)),
            Weight::new(Buf(1)),
            Weight::new(Buf(2)),
            InSlot::new(Buf(3)),
            OutSlot::new(BufMut(4)),
            Param::new(4096),
            Param::new(24),
            Param::new(4096),
            Ask::new(9),
            Ask::new(4),
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
        encode_u4_bf16(
            &seen,
            InSlot::new(Buf(0)),
            OutSlot::new(BufMut(1)),
            OutSlot::new(BufMut(2)),
            OutSlot::new(BufMut(3)),
            Block::new(Buf(4)),
            Ask::new(7),
        )
        .unwrap();
        assert_eq!(one(&seen).1, [7, 1, 1]);

        let seen = Seen::default();
        mxfp4_dequant_bf16(&seen, InSlot::new(Buf(0)), InSlot::new(Buf(1)), OutSlot::new(BufMut(2)), Block::new(Buf(3)), Ask::new(5)).unwrap();
        assert_eq!(one(&seen).1, [5, 1, 1]);

        let seen = Seen::default();
        cast_qmm_input_bfloat16_to_float16(&seen, InSlot::new(Buf(0)), OutSlot::new(BufMut(1)), 64, 32, 64, Param::new(2048)).unwrap();
        assert_eq!(one(&seen).1, [2048, 1, 1]);

        let seen = Seen::default();
        cast_qmm_input_strided_bfloat16_to_float16(
            &seen,
            InSlot::new(Buf(0)),
            OutSlot::new(BufMut(1)),
            Param::new(64),
            Param::new(32),
            Param::new(96),
            Reckoned::new(2048),
            Ask::new(12),
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
                Weight::new(Buf(0)),
                Weight::new(Buf(1)),
                Weight::new(Buf(2)),
                InSlot::new(Buf(3)),
                OutSlot::new(BufMut(4)),
                Param::new(256),
                Param::new(0),
                Ask::new(64),
                Ask::new(4),
                Ask::new(32),
                Ask::new(32),
                Ask::new(70),
            ),
            Err(Refusal::Empty {
                what: "the column count"
            })
        ));
        assert!(matches!(
            qmv_fast(
                &seen,
                Weight::new(Buf(0)),
                Weight::new(Buf(1)),
                Weight::new(Buf(2)),
                InSlot::new(Buf(3)),
                OutSlot::new(BufMut(4)),
                Param::new(4096),
                Param::new(0),
                Ask::new(32),
                Ask::new(4),
                Ask::new(3),
            ),
            Err(Refusal::Empty {
                what: "the output vector"
            })
        ));
        assert!(matches!(
            encode_u4_bf16(
                &seen,
                InSlot::new(Buf(0)),
                OutSlot::new(BufMut(1)),
                OutSlot::new(BufMut(2)),
                OutSlot::new(BufMut(3)),
                Block::new(Buf(4)),
                Ask::new(0)
            ),
            Err(Refusal::Empty { .. })
        ));
        assert!(
            seen.0.borrow().is_empty(),
            "a refused shape dispatched anyway"
        );
    }
}

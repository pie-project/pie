//! The affine-quantised projections, and the codecs around them.
//!
//! This is 32 of the 99 rows and 304 of the 480 entrypoints, and the whole
//! argument of `.wiki/kernel-metal-refactor.md` §2 is visible here: `qmm_t` is
//! ONE template body in `quant/qmm_t.metal` stamped over (group x bits x
//! row tile x column tile), and enumerating its 54 instantiations as 54 rows
//! would state the macro's job a second time by hand.
//!
//! The five `_wm_`/`_wn_` rows are the exception that proves it: they are
//! `host_name` lines typed out at `quant/qmm_t.metal:1356-1408` rather
//! than stamped, so they are five kernels and get five rows.


use kernels::Grid;
use kernels_macros::routine;
use kernels::routine::Refusal;

use crate::routine::{Asks, Bind, Const, Ctx, Fire, In, Out, Tensor, bf16, elementwise, elementwise_rows, f16, keys};

/// The shaders this family's routines reach: `(file, entrypoint)`, one pair
/// per instantiated name.
///
/// A row's `axes` GENERATED these names and its `file` column said where they
/// live. Retiring the row moved who NAMES them, not what exists -- the shader
/// is still compiled and still dispatched -- so the pairs are stated here and
/// [`crate::entrypoints`] reads them back. The FILE rides along because Metal
/// compiles from `(path, entry name)` at run time, and `device_kernels.rs`
/// builds every one of them against a real device; a name without its file
/// would leave that sweep nothing to open. See [`crate::RETIRED`].
pub static ENTRYPOINTS: &[(&str, &str)] = &[
    ("quant/transcode.metal", "affine_encode_u4_bf16"),
    ("quant/transcode.metal", "affine_encode_u4_f32"),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_128_b_4_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_128_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_128_b_4_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_128_b_4_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_128_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_128_b_4_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_128_b_4_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_128_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_128_b_4_bm_64_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_128_b_8_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_128_b_8_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_128_b_8_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_128_b_8_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_128_b_8_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_128_b_8_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_128_b_8_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_128_b_8_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_128_b_8_bm_64_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_32_b_4_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_32_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_32_b_4_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_32_b_4_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_32_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_32_b_4_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_32_b_4_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_32_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_32_b_4_bm_64_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_32_b_8_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_32_b_8_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_32_b_8_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_32_b_8_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_32_b_8_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_32_b_8_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_32_b_8_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_32_b_8_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_32_b_8_bm_64_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_64_b_4_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_64_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_64_b_4_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_64_b_8_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_64_b_8_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_64_b_8_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_64_b_8_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_64_b_8_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_64_b_8_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_64_b_8_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_64_b_8_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_64_b_8_bm_64_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_128_b_4_bm_64_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_128_b_8_bm_64_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_32_b_4_bm_64_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_32_b_8_bm_64_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_64_b_4_bm_64_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_bfloat16_gs_64_b_8_bm_64_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_128_b_4_bm_64_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_128_b_8_bm_64_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_32_b_4_bm_64_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_32_b_8_bm_64_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_64_b_4_bm_64_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_bfloat16_gs_64_b_8_bm_64_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_bfloat16_gs_128_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_bfloat16_gs_128_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_bfloat16_gs_128_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_bfloat16_gs_128_b_8_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_bfloat16_gs_128_b_8_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_bfloat16_gs_128_b_8_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_bfloat16_gs_32_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_bfloat16_gs_32_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_bfloat16_gs_32_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_bfloat16_gs_32_b_8_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_bfloat16_gs_32_b_8_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_bfloat16_gs_32_b_8_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_bfloat16_gs_64_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_bfloat16_gs_64_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_bfloat16_gs_64_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_bfloat16_gs_64_b_8_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_bfloat16_gs_64_b_8_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_bfloat16_gs_64_b_8_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_f32_bfloat16_gs_128_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_f32_bfloat16_gs_128_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_f32_bfloat16_gs_128_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_f32_bfloat16_gs_128_b_8_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_f32_bfloat16_gs_128_b_8_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_f32_bfloat16_gs_128_b_8_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_f32_bfloat16_gs_32_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_f32_bfloat16_gs_32_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_f32_bfloat16_gs_32_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_f32_bfloat16_gs_32_b_8_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_f32_bfloat16_gs_32_b_8_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_f32_bfloat16_gs_32_b_8_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_f32_bfloat16_gs_64_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_f32_bfloat16_gs_64_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_f32_bfloat16_gs_64_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_f32_bfloat16_gs_64_b_8_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_f32_bfloat16_gs_64_b_8_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_f32_bfloat16_gs_64_b_8_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_fp16_precast_f32_bfloat16_gs_64_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_fp16_precast_f32_bfloat16_gs_64_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_splitk_fp16_precast_f32_bfloat16_gs_64_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_bfloat16_gs_128_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_bfloat16_gs_128_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_bfloat16_gs_128_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_bfloat16_gs_128_b_8_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_bfloat16_gs_128_b_8_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_bfloat16_gs_128_b_8_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_bfloat16_gs_32_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_bfloat16_gs_32_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_bfloat16_gs_32_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_bfloat16_gs_32_b_8_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_bfloat16_gs_32_b_8_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_bfloat16_gs_32_b_8_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_bfloat16_gs_64_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_bfloat16_gs_64_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_bfloat16_gs_64_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_bfloat16_gs_64_b_8_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_bfloat16_gs_64_b_8_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_bfloat16_gs_64_b_8_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_fp16_precast_residual_bfloat16_gs_64_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_fp16_precast_residual_bfloat16_gs_64_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_fp16_precast_residual_bfloat16_gs_64_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_residual_bfloat16_gs_128_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_residual_bfloat16_gs_128_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_residual_bfloat16_gs_128_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_residual_bfloat16_gs_128_b_8_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_residual_bfloat16_gs_128_b_8_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_residual_bfloat16_gs_128_b_8_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_residual_bfloat16_gs_32_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_residual_bfloat16_gs_32_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_residual_bfloat16_gs_32_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_residual_bfloat16_gs_32_b_8_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_residual_bfloat16_gs_32_b_8_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_residual_bfloat16_gs_32_b_8_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_residual_bfloat16_gs_64_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_residual_bfloat16_gs_64_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_residual_bfloat16_gs_64_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_residual_bfloat16_gs_64_b_8_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_residual_bfloat16_gs_64_b_8_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_strided_residual_bfloat16_gs_64_b_8_bm_64_bn_32",
    ),
    ("quant/qmv.metal", "affine_qmv_fast_bfloat16_gs_128_b_4"),
    ("quant/qmv.metal", "affine_qmv_fast_bfloat16_gs_128_b_8"),
    ("quant/qmv.metal", "affine_qmv_fast_bfloat16_gs_32_b_4"),
    ("quant/qmv.metal", "affine_qmv_fast_bfloat16_gs_32_b_8"),
    ("quant/qmv.metal", "affine_qmv_fast_bfloat16_gs_64_b_4"),
    ("quant/qmv.metal", "affine_qmv_fast_bfloat16_gs_64_b_8"),
    (
        "quant/qmv.metal",
        "affine_qmv_fast_residual_bfloat16_gs_128_b_4",
    ),
    (
        "quant/qmv.metal",
        "affine_qmv_fast_residual_bfloat16_gs_128_b_8",
    ),
    (
        "quant/qmv.metal",
        "affine_qmv_fast_residual_bfloat16_gs_32_b_4",
    ),
    (
        "quant/qmv.metal",
        "affine_qmv_fast_residual_bfloat16_gs_32_b_8",
    ),
    (
        "quant/qmv.metal",
        "affine_qmv_fast_residual_bfloat16_gs_64_b_4",
    ),
    (
        "quant/qmv.metal",
        "affine_qmv_fast_residual_bfloat16_gs_64_b_8",
    ),
    ("quant/qmv.metal", "affine_qmv_tail_bfloat16_gs_64_b_4"),
    ("quant/qmv.metal", "affine_qmv_tail_bfloat16_gs_64_b_8"),
    ("quant/qmv.metal", "affine_qmv_tail_bias_bfloat16_gs_64_b_4"),
    ("quant/qmv.metal", "affine_qmv_tail_bias_bfloat16_gs_64_b_8"),
    (
        "quant/qmm_t.metal",
        "affine_qmv_wide_strided_bfloat16_gs_64_b_4_v_4_kl_8",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmv_wide_strided_bfloat16_gs_64_b_8_v_4_kl_8",
    ),
    ("quant/qmm_t.metal", "cast_qmm_input_bfloat16_to_float16"),
    (
        "quant/qmm_t.metal",
        "cast_qmm_input_strided_bfloat16_to_float16",
    ),
    ("quant/transcode.metal", "mxfp4_dequant_bf16"),
    ("quant/qmm_t.metal", "qmm_splitk_reduce_bfloat16"),
    ("quant/qmm_t.metal", "qmm_splitk_reduce_f32_bfloat16"),
];
/// `affine_qmm_t`, indexed by [`qmm_point`].
pub static QMM_T: [&str; 54] = [
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
pub static QMM_T_BIAS: [&str; 54] = [
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
pub static QMM_T_RESIDUAL: [&str; 54] = [
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
pub static QMM_T_FP16_PRECAST: [&str; 9] = [
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
pub static QMM_T_BIAS_FP16_PRECAST: [&str; 9] = [
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
pub static QMM_T_RESIDUAL_FP16_PRECAST: [&str; 9] = [
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
pub static QMM_T_SPLITK: [&str; 18] = [
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
pub static QMM_T_SPLITK_F32: [&str; 18] = [
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
pub static QMM_T_STRIDED: [&str; 18] = [
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
pub static QMM_T_STRIDED_RESIDUAL: [&str; 18] = [
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
pub static QMM_T_SPLITK_FP16_PRECAST: [&str; 3] = [
    "affine_qmm_t_splitk_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_splitk_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_splitk_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32",
];

/// `affine_qmm_t_splitk_fp16_precast_f32`, indexed by [`row_tile_point`].
pub static QMM_T_SPLITK_FP16_PRECAST_F32: [&str; 3] = [
    "affine_qmm_t_splitk_fp16_precast_f32_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_splitk_fp16_precast_f32_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_splitk_fp16_precast_f32_bfloat16_gs_64_b_4_bm_64_bn_32",
];

/// `affine_qmm_t_strided_fp16_precast`, indexed by [`row_tile_point`].
pub static QMM_T_STRIDED_FP16_PRECAST: [&str; 3] = [
    "affine_qmm_t_strided_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_strided_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_strided_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32",
];

/// `affine_qmm_t_strided_fp16_precast_residual`, indexed by [`row_tile_point`].
pub static QMM_T_STRIDED_FP16_PRECAST_RESIDUAL: [&str; 3] = [
    "affine_qmm_t_strided_fp16_precast_residual_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_strided_fp16_precast_residual_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_strided_fp16_precast_residual_bfloat16_gs_64_b_4_bm_64_bn_32",
];

/// `affine_qmv_fast`, indexed by [`codec_point`].
pub static QMV_FAST: [&str; 6] = [
    "affine_qmv_fast_bfloat16_gs_32_b_4",
    "affine_qmv_fast_bfloat16_gs_32_b_8",
    "affine_qmv_fast_bfloat16_gs_64_b_4",
    "affine_qmv_fast_bfloat16_gs_64_b_8",
    "affine_qmv_fast_bfloat16_gs_128_b_4",
    "affine_qmv_fast_bfloat16_gs_128_b_8",
];

/// `affine_qmv_fast_residual`, indexed by [`codec_point`].
pub static QMV_FAST_RESIDUAL: [&str; 6] = [
    "affine_qmv_fast_residual_bfloat16_gs_32_b_4",
    "affine_qmv_fast_residual_bfloat16_gs_32_b_8",
    "affine_qmv_fast_residual_bfloat16_gs_64_b_4",
    "affine_qmv_fast_residual_bfloat16_gs_64_b_8",
    "affine_qmv_fast_residual_bfloat16_gs_128_b_4",
    "affine_qmv_fast_residual_bfloat16_gs_128_b_8",
];

/// `affine_qmv_tail`, indexed by [`bits_point`].
pub static QMV_TAIL: [&str; 2] = [
    "affine_qmv_tail_bfloat16_gs_64_b_4",
    "affine_qmv_tail_bfloat16_gs_64_b_8",
];

/// `affine_qmv_tail_bias`, indexed by [`bits_point`].
pub static QMV_TAIL_BIAS: [&str; 2] = [
    "affine_qmv_tail_bias_bfloat16_gs_64_b_4",
    "affine_qmv_tail_bias_bfloat16_gs_64_b_8",
];

/// `affine_qmv_wide_strided`, indexed by [`bits_point`].
pub static QMV_WIDE_STRIDED: [&str; 2] = [
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

/// The tiled matmul's rectangle, in THREADS, and its threadgroup.
///
/// `quant/qmm_t.metal` is a `[32, 2, 2]` threadgroup whose bodies read
/// `threadgroup_position_in_grid` alone: x is the column tile, y the row
/// tile, z the split-K partition. So the group counts are `[n/bn, m/bm,
/// split_k]` and each is multiplied by its own local size, because a Metal
/// dispatch states THREADS.
///
/// # The row axis divides exactly
///
/// No entrypoint's argument list carries `m`, so `write_out` cannot guard the
/// row axis: a grid rounded up past the row count writes the overhang into
/// whatever follows the output. The driver's `Rule::Qmm` refused a row count
/// its tile did not divide for that reason, and this keeps the refusal --
/// `Refusal::Misaligned` -- rather than rounding up and trusting the
/// allocation to be a whole number of tiles.
///
/// # The column axis divides too, and this said otherwise
///
/// This read "the COLUMN overhang is guarded, by `n`, which every one of them
/// does carry", and `n` guards the epilogue's STORE. It does not guard the
/// weight tile the MMA loop loads: `qmm_t.metal`'s own header states `M % BM
/// == 0, N % BN == 0 and K % BK == 0` as the condition under which the driver
/// may select the kernel at all, because `load_unsafe` is the only path the
/// hot loop takes and it reads a whole tile whether or not one exists.
///
/// qwen3.6's `in_proj_a` is 48 wide and the deployment's tile is `bn = 32`.
/// The second column tile read sixteen rows past the end of a 4-bit weight
/// and the projection came back partly NaN, from row 32 of 64, differently on
/// each fire. A guard on the store cannot undo a load.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty extent or a non-positive tile,
/// [`Refusal::Misaligned`] for a row or column count off the tile, and
/// [`Refusal::Grid`] if a tile count times its local size leaves a `u32`.
fn qmm_grid(n: i32, bn: i32, m: i32, bm: i32, split_k: i32) -> Result<[u32; 3], Refusal> {
    if n <= 0 {
        return Err(Refusal::Empty {
            what: "the column count",
        });
    }
    if m <= 0 {
        return Err(Refusal::Empty {
            what: "the row count",
        });
    }
    if bn <= 0 || bm <= 0 {
        return Err(Refusal::Empty { what: "the tile" });
    }
    if split_k <= 0 {
        return Err(Refusal::Empty {
            what: "the k split",
        });
    }
    if m % bm != 0 {
        return Err(Refusal::Misaligned {
            what: "the row count, which the tile must divide because no \
                   entrypoint takes m and the shader reads it from the grid",
        });
    }
    if n % bn != 0 {
        return Err(Refusal::Misaligned {
            what: "the column count, which the tile must divide: `qmm_t.metal` \
                   states `M % BM == 0, N % BN == 0 and K % BK == 0` as the \
                   condition under which the driver may select it at all, and \
                   `load_unsafe` is the only path its hot loop takes",
        });
    }
    let lanes = |groups: u32, local: u32, what: &'static str| -> Result<u32, Refusal> {
        groups.checked_mul(local).ok_or(Refusal::Grid {
            what,
            at: i64::from(groups),
        })
    };
    Ok([
        lanes(
            n.unsigned_abs().div_ceil(bn.unsigned_abs()),
            32,
            "the column tiles",
        )?,
        lanes(m.unsigned_abs() / bm.unsigned_abs(), 2, "the row tiles")?,
        lanes(split_k.unsigned_abs(), 2, "the k splits")?,
    ])
}

/// The matvec's rectangle, in THREADS.
///
/// `quant/qmv.metal` is a `[32, 2, 1]` threadgroup, x is the batch vector and
/// one threadgroup covers eight output rows -- `out0 = group.y * 8 + ly * 4`,
/// two lanes of four -- so the group counts are `[vecs, out/8, 1]` and the y
/// axis is stated as `out/4` threads.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty extent, [`Refusal::Grid`] on overflow.
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
    let x = vecs.unsigned_abs().checked_mul(32).ok_or(Refusal::Grid {
        what: "the vectors",
        at: i64::from(vecs),
    })?;
    Ok([x, out_vec_size.unsigned_abs().div_ceil(4), 1])
}

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

/// The threadgroup every tiled matmul in `quant/qmm_t.metal` declares.
const QMM_GROUP: [u32; 3] = [32, 2, 2];

/// The threadgroup every matvec in `quant/qmv.metal` declares.
const QMV_GROUP: [u32; 3] = [32, 2, 1];

/// The threadgroup the elementwise codecs and the split-K reduction take.
const GROUP_X: u32 = 256;

const QMM_FILE: &str = "quant/qmm_t.metal";

const QMV_FILE: &str = "quant/qmv.metal";

const TRANSCODE_FILE: &str = "quant/transcode.metal";

/// The batched projection: a `bm x bn` tile of the output per threadgroup.
///
/// The operand order is the GEMV's -- weights, then the activation, then the
/// result, then the two extents -- and it is the order the SHADER declares,
/// not the order a trace states them in. Binding a trace's order here put the
/// activation where the packed weight belongs, on every projection of every
/// layer.
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
        Fire::at(QMM_FILE, QMM_T[qmm_point(*group, *bits, *bm, *bn)?]).apply(Grid::of(qmm_grid(n, *bn, m, *bm, 1)?, QMM_GROUP)),
        &[w.arg(), scales.arg(), biases.arg(), x.arg(), y.arg(), k.arg(), n.arg()],
    )
}

/// The same tile, plus a per-COLUMN bias its epilogue adds.
///
/// `bias` binds at 5 and is indexed by the column alone -- one value per
/// output feature, not per element, which is what tells this apart from the
/// residual form that shares the slot.
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
        Fire::at(QMM_FILE, QMM_T_BIAS[qmm_point(*group, *bits, *bm, *bn)?]).apply(Grid::of(qmm_grid(n, *bn, m, *bm, 1)?, QMM_GROUP)),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
            bias.arg(),
        ],
    )
}

/// The same tile, plus a residual added ELEMENTWISE.
///
/// The residual comes LAST here and the bias came at 5, which is not a
/// tidiness difference: they are two `#define`s of one body, indexed
/// `extra[row * stride + col]` against `extra[col]`. Handing this form a
/// per-column plane reads a whole matrix out of a vector.
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
        Fire::at(QMM_FILE, QMM_T_RESIDUAL[qmm_point(*group, *bits, *bm, *bn)?]).apply(Grid::of(qmm_grid(n, *bn, m, *bm, 1)?, QMM_GROUP)),
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

/// The tile over an activation ALREADY cast to half, which is what
/// `cast_qmm_input_bfloat16_to_float16` writes.
///
/// No `x`: the half plane replaces it and binds after the result rather than
/// before, so the argument table is not the plain form's with a substitution.
/// Compiled at g64/b4 alone, so neither the group size nor the bit width is a
/// point a caller picks.
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
        Fire::at(QMM_FILE, QMM_T_FP16_PRECAST[tile_point(*bm, *bn)?]).apply(Grid::of(qmm_grid(n, *bn, m, *bm, 1)?, QMM_GROUP)),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            w.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
            w.arg(),
            w.arg(),
            w.arg(),
            w.arg(),
            w.arg(),
            half_in.arg(),
        ],
    )
}

/// [`qmm_t_fp16_precast`] with the per-column bias.
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
        Fire::at(QMM_FILE, QMM_T_BIAS_FP16_PRECAST[tile_point(*bm, *bn)?]).apply(Grid::of(qmm_grid(n, *bn, m, *bm, 1)?, QMM_GROUP)),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            w.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
            bias.arg(),
            w.arg(),
            w.arg(),
            w.arg(),
            w.arg(),
            half_in.arg(),
        ],
    )
}

/// [`qmm_t_fp16_precast`] with the elementwise residual.
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
    // THE CONTRACTION IS THE ACTIVATION'S WIDTH, AND IT IS NOT THE RESIDUAL'S.
    //
    // This read `residual.width`, and a residual is the LAYER's stream: as
    // wide as the hidden size, which is the width of this projection's OUTPUT
    // and has nothing to do with the width of its input. The two coincide on
    // every stack whose fused-residual projection is square -- qwen3.5's
    // o_proj is 2048 into 2048, and that is the deployment this kernel was
    // measured on -- so the wrong operand answered right and stayed.
    //
    // gpt-oss's o_proj is 4096 into 2880. The GEMM was told `K = 2880`, which
    // is both a truncated contraction and, because `qmm_t_loaded_impl` starts
    // each row of the activation at `x + y_row * k_len`, the wrong ROW STRIDE:
    // every row but the first read from the middle of an earlier one. It only
    // ever fired on a prefill long enough to take the batched arm, so the
    // decode that every short prompt is stayed correct and the long prompt
    // came back generic.
    let k = half_in.width;
    let n = y.width;
    let m = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(QMM_FILE, QMM_T_RESIDUAL_FP16_PRECAST[tile_point(*bm, *bn)?]).apply(Grid::of(qmm_grid(n, *bn, m, *bm, 1)?, QMM_GROUP)),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            w.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
            residual.arg(),
            w.arg(),
            w.arg(),
            w.arg(),
            w.arg(),
            half_in.arg(),
        ],
    )
}

/// The tile with the K axis PARTITIONED across grid.z, writing one partial
/// sum per partition.
///
/// `split_k` is both an argument and the grid's z extent, and both are
/// needed: the shader indexes its partition from the position and strides
/// into the partial buffer by the count. [`qmm_splitk_reduce`] is what turns
/// the partials into the result.
///
/// Stamped at `_bn_32` alone -- 18 points where the plain form has 54 -- so
/// the column tile is not a choice and the grid reads it from [`WIDE_BN`].
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
        Fire::at(QMM_FILE, QMM_T_SPLITK[wide_point(*group, *bits, *bm)?]).apply(Grid::of(qmm_grid(n, WIDE_BN, m, *bm, split_k)?, QMM_GROUP)),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            w.arg(),
            k.arg(),
            n.arg(),
            w.arg(),
            out.arg(),
            k_partition_size.arg(),
            split_k_partition_stride.arg(),
        ],
    )
}

/// [`qmm_t_splitk`] accumulating its partials in f32.
///
/// A separate entrypoint and not a flag, because the partial buffer's element
/// type is the difference: pairing this with `qmm_splitk_reduce` rather than
/// `qmm_splitk_reduce_f32` reads f32 partials as bf16.
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
        Fire::at(QMM_FILE, QMM_T_SPLITK_F32[wide_point(*group, *bits, *bm)?]).apply(Grid::of(qmm_grid(n, WIDE_BN, m, *bm, split_k)?, QMM_GROUP)),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            w.arg(),
            k.arg(),
            n.arg(),
            w.arg(),
            out.arg(),
            k_partition_size.arg(),
            split_k_partition_stride.arg(),
        ],
    )
}

/// [`qmm_t_splitk`] over a pre-cast half activation.
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
        Fire::at(QMM_FILE, QMM_T_SPLITK_FP16_PRECAST[row_tile_point(*bm)?]).apply(Grid::of(qmm_grid(n, WIDE_BN, m, *bm, split_k)?, QMM_GROUP)),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            w.arg(),
            w.arg(),
            k.arg(),
            n.arg(),
            w.arg(),
            out.arg(),
            k_partition_size.arg(),
            split_k_partition_stride.arg(),
            w.arg(),
            half_in.arg(),
        ],
    )
}

/// [`qmm_t_splitk_f32`] over a pre-cast half activation.
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
        Fire::at(QMM_FILE, QMM_T_SPLITK_FP16_PRECAST_F32[row_tile_point(*bm)?]).apply(Grid::of(qmm_grid(n, WIDE_BN, m, *bm, split_k)?, QMM_GROUP)),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            w.arg(),
            w.arg(),
            k.arg(),
            n.arg(),
            w.arg(),
            out.arg(),
            k_partition_size.arg(),
            split_k_partition_stride.arg(),
            w.arg(),
            half_in.arg(),
        ],
    )
}

/// The tile over an output whose rows are not packed.
///
/// `row_stride` is the OUTPUT's pitch. A fused projection writes its share of
/// a wider tensor, and the column count `n` stays the share rather than the
/// pitch -- they are two numbers and the shader wants both.
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
        Fire::at(QMM_FILE, QMM_T_STRIDED[wide_point(*group, *bits, *bm)?]).apply(Grid::of(qmm_grid(n, WIDE_BN, m, *bm, 1)?, QMM_GROUP)),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
            w.arg(),
            row_stride.arg(),
        ],
    )
}

/// [`qmm_t_strided`] with the elementwise residual, which binds at 5 here
/// and last in [`qmm_t_residual`].
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
        Fire::at(QMM_FILE, QMM_T_STRIDED_RESIDUAL[wide_point(*group, *bits, *bm)?]).apply(Grid::of(qmm_grid(n, WIDE_BN, m, *bm, 1)?, QMM_GROUP)),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
            residual.arg(),
            row_stride.arg(),
        ],
    )
}

/// [`qmm_t_strided`] over a pre-cast half activation.
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
        Fire::at(QMM_FILE, QMM_T_STRIDED_FP16_PRECAST[row_tile_point(*bm)?]).apply(Grid::of(qmm_grid(n, WIDE_BN, m, *bm, 1)?, QMM_GROUP)),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            w.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
            w.arg(),
            row_stride.arg(),
            w.arg(),
            w.arg(),
            w.arg(),
            half_in.arg(),
        ],
    )
}

/// [`qmm_t_strided_fp16_precast`] with the elementwise residual.
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
    // The same wrong operand [`qmm_t_residual_fp16_precast`] carried, and it
    // is stated at length there.
    let k = half_in.width;
    let n = y.width;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<2>` named.
    // The run is the shader's struct layout, so no `Const` mark can name a
    // word inside it, and `keys::RowStride` is answered by no driver.
    let row_stride = ctx.param(2)?;
    let m = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(QMM_FILE, QMM_T_STRIDED_FP16_PRECAST_RESIDUAL[row_tile_point(*bm)?]).apply(Grid::of(qmm_grid(n, WIDE_BN, m, *bm, 1)?, QMM_GROUP)),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            w.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
            residual.arg(),
            row_stride.arg(),
            w.arg(),
            w.arg(),
            w.arg(),
            half_in.arg(),
        ],
    )
}

/// The sum over a split-K matmul's partials, one output element per lane.
///
/// It reads bf16 partials. The pairing with the matmul that wrote them is
/// not checked by anything: two entrypoints exist because the element type
/// differs, and crossing them reinterprets the buffer.
///
/// # Errors
///
/// Whatever [`crate::routine::elementwise_rows`] refuses.
#[routine]
pub fn qmm_splitk_reduce(
    ctx: &Ctx<'_>,
    y: Out<Tensor<bf16>>,
    partial: In<Tensor<bf16>>) -> Result<(), Refusal> {
    let n = y.width;
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
        Fire::at(QMM_FILE, "qmm_splitk_reduce_bfloat16").apply(Grid::of(elementwise_rows(n, m)?, [GROUP_X, 1, 1])),
        &[
            partial.arg(),
            partial.arg(),
            partial.arg(),
            partial.arg(),
            y.arg(),
            partial.arg(),
            n.arg(),
            partial.arg(),
            partial.arg(),
            partial.arg(),
            split_k_partition_stride.arg(),
            split_k.arg(),
        ],
    )
}

/// The sum over a split-K matmul's partials, one output element per lane.
///
/// It reads f32 partials. The pairing with the matmul that wrote them is
/// not checked by anything: two entrypoints exist because the element type
/// differs, and crossing them reinterprets the buffer.
///
/// # Errors
///
/// Whatever [`crate::routine::elementwise_rows`] refuses.
#[routine]
pub fn qmm_splitk_reduce_f32(
    ctx: &Ctx<'_>,
    y: Out<Tensor<bf16>>,
    partial: In<Tensor<f32>>) -> Result<(), Refusal> {
    let n = y.width;
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
        Fire::at(QMM_FILE, "qmm_splitk_reduce_f32_bfloat16").apply(Grid::of(elementwise_rows(n, m)?, [GROUP_X, 1, 1])),
        &[
            partial.arg(),
            partial.arg(),
            partial.arg(),
            partial.arg(),
            y.arg(),
            partial.arg(),
            n.arg(),
            partial.arg(),
            partial.arg(),
            partial.arg(),
            split_k_partition_stride.arg(),
            split_k.arg(),
        ],
    )
}

/// The activation cast to half, ahead of a `_fp16_precast` tile.
///
/// # The ten pad slots
///
/// This kernel is encoded into `affine_qmm_t`'s argument table and numbers
/// its three operands there rather than from zero: `x` at **3**, `y` at
/// **12** and `count` at **13** (`quant/qmm_t.metal:379`). It shares the
/// table so that a precast GEMM and the cast that feeds it can be encoded
/// against one ordinal, which is the same reason [`crate::moe::qmm_t_routed`]
/// numbers `tile_expert` at 12.
///
/// A routine's argument list is POSITIONAL -- the index in the list is the
/// index in the table -- so reaching 13 means fourteen entries, and the ten
/// the shader declares nothing at still have to hold an address. `pad` is
/// taken once and bound at each.
///
/// It read `x` from slot 0 and wrote `y` to slot 1 before this, so the body
/// took a scalar where the activation belongs and wrote through an index
/// nothing bound -- which on this driver is whatever the previous step left
/// at that address. `k`, `n` and `row_stride` went with the old list: the
/// body reads none of them, and the doc that said they were "carried so this
/// shares its argument list with the strided form" was describing a list that
/// matched neither shader.
///
/// # Errors
///
/// Whatever [`crate::routine::elementwise`] refuses.
#[routine]
pub fn cast_qmm_input_bfloat16_to_float16(
    ctx: &Ctx<'_>,
    cast_in: In<Tensor<bf16>>,
    half_out: Out<Tensor<f16>>) -> Result<(), Refusal> {
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<3>` named.
    // The run is the shader's struct layout, so no `Const` mark can name a
    // word inside it, and `keys::Count` is answered by no driver.
    let count = ctx.param(3)?;
    ctx.fire(
        Fire::at(QMM_FILE, "cast_qmm_input_bfloat16_to_float16").apply(Grid::of(elementwise(count, 1)?, [GROUP_X, 1, 1])),
        &[
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            half_out.arg(),
            count.arg(),
        ],
    )
}

/// The same cast over rows that are not packed, `k` wide and `rows` deep.
///
/// The same shared argument table as [`cast_qmm_input_bfloat16_to_float16`]
/// and a different four slots of it: `x` at **3**, `K` at **5**,
/// `row_stride` at **8** and `y` at **12** (`quant/qmm_t.metal:1004`). The
/// two casts held one argument list between them and it was the ordinal
/// numbering of NEITHER; they are the table's two shapes now, which is why
/// this no longer builds its list by pushing onto the other's.
///
/// # Errors
///
/// Whatever [`crate::routine::elementwise_rows`] refuses.
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
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(QMM_FILE, "cast_qmm_input_strided_bfloat16_to_float16").apply(Grid::of(elementwise_rows(k, rows)?, [GROUP_X, 1, 1])),
        &[
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            k.arg(),
            cast_in.arg(),
            cast_in.arg(),
            row_stride.arg(),
            cast_in.arg(),
            cast_in.arg(),
            cast_in.arg(),
            half_out.arg(),
        ],
    )
}

/// The decode projection: a quantised matvec over up to a few batch vectors.
///
/// The loudest misbinding this port had, and the reason the rows grew an
/// operand list at all: this declares its WEIGHTS FIRST and the trace states
/// them last, so positional binding put the activation where the packed
/// weight belongs. Every projection of every layer.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a point the shader tree does not carry, and
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
        Fire::at(QMV_FILE, QMV_FAST[codec_point(*group, *bits)?]).apply(Grid::of(qmv_grid(vecs, out_vec_size)?, QMV_GROUP)),
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

/// [`qmv_fast`] with the block residual its epilogue folds, which the trace
/// states as a second input and the kernel takes at the very end.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a point the shader tree does not carry, and
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
        Fire::at(QMV_FILE, QMV_FAST_RESIDUAL[codec_point(*group, *bits)?]).apply(Grid::of(qmv_grid(vecs, out_vec_size)?, QMV_GROUP)),
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

/// The matvec for an output whose row count the fast form's eight does not
/// divide.
///
/// Compiled at g64 alone, so the group size is not a point here even though
/// it is one for [`qmv_fast`].
///
/// # Errors
///
/// [`Refusal::Narrow`] for a point the shader tree does not carry, and
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
        Fire::at(QMV_FILE, QMV_TAIL[bits_point(*bits)?]).apply(Grid::of(qmv_grid(vecs, out_vec_size)?, QMV_GROUP)),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            in_vec_size.arg(),
            out_vec_size.arg(),
            w.arg(),
            w.arg(),
            w.arg(),
            w.arg(),
            w.arg(),
        ],
    )
}

/// [`qmv_tail`] with a per-column bias.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a point the shader tree does not carry, and
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
        Fire::at(QMV_FILE, QMV_TAIL_BIAS[bits_point(*bits)?]).apply(Grid::of(qmv_grid(vecs, out_vec_size)?, QMV_GROUP)),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            in_vec_size.arg(),
            out_vec_size.arg(),
            bias.arg(),
            w.arg(),
            w.arg(),
            w.arg(),
            w.arg(),
        ],
    )
}

/// The matvec that takes FOUR batch vectors to a threadgroup, over an
/// unpacked output.
///
/// `m` is an argument and the grid's x extent is `m / 4` rounded up, which is
/// why [`quarters`] exists: the two are different numbers and the kernel
/// needs both, the grid to know how many groups and the argument to guard the
/// last one.
///
/// It lives in `quant/qmm_t.metal` and not with the other matvecs.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a point the shader tree does not carry, and
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
        Fire::at(QMM_FILE, QMV_WIDE_STRIDED[bits_point(*bits)?]).apply(Grid::of(qmv_grid(quarters(m), out_vec_size)?, QMV_GROUP)),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            in_vec_size.arg(),
            out_vec_size.arg(),
            w.arg(),
            row_stride.arg(),
            m.arg(),
        ],
    )
}

/// The `wm`/`wn` tile at `bm_128`, `bn_32`.
///
/// One of five `host_name` lines typed out in the shader rather than stamped
/// from the instantiation macro, which is why it is its own routine with its
/// tile fixed rather than a point of [`qmm_t`]'s axis. `affine_qmm_t_aligned`
/// is the template they share and it has no routine: it is not an entrypoint.
///
/// # Errors
///
/// Whatever [`qmm_grid`] refuses.
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
        Fire::at(QMM_FILE, "affine_qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4").apply(Grid::of(qmm_grid(n, 32, m, 128, 1)?, QMM_GROUP)),
        &[w.arg(), scales.arg(), biases.arg(), x.arg(), y.arg(), k.arg(), n.arg()],
    )
}

/// The `wm`/`wn` tile at `bm_32`, `bn_32`.
///
/// One of five `host_name` lines typed out in the shader rather than stamped
/// from the instantiation macro, which is why it is its own routine with its
/// tile fixed rather than a point of [`qmm_t`]'s axis. `affine_qmm_t_aligned`
/// is the template they share and it has no routine: it is not an entrypoint.
///
/// # Errors
///
/// Whatever [`qmm_grid`] refuses.
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
        Fire::at(QMM_FILE, "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2").apply(Grid::of(qmm_grid(n, 32, m, 32, 1)?, QMM_GROUP)),
        &[w.arg(), scales.arg(), biases.arg(), x.arg(), y.arg(), k.arg(), n.arg()],
    )
}

/// The `wm`/`wn` tile at `bm_64`, `bn_32`.
///
/// One of five `host_name` lines typed out in the shader rather than stamped
/// from the instantiation macro, which is why it is its own routine with its
/// tile fixed rather than a point of [`qmm_t`]'s axis. `affine_qmm_t_aligned`
/// is the template they share and it has no routine: it is not an entrypoint.
///
/// # Errors
///
/// Whatever [`qmm_grid`] refuses.
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
        Fire::at(QMM_FILE, "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2").apply(Grid::of(qmm_grid(n, 32, m, 64, 1)?, QMM_GROUP)),
        &[w.arg(), scales.arg(), biases.arg(), x.arg(), y.arg(), k.arg(), n.arg()],
    )
}

/// The `wm`/`wn` tile at `bm_64`, `bn_32`.
///
/// One of five `host_name` lines typed out in the shader rather than stamped
/// from the instantiation macro, which is why it is its own routine with its
/// tile fixed rather than a point of [`qmm_t`]'s axis. `affine_qmm_t_aligned`
/// is the template they share and it has no routine: it is not an entrypoint.
///
/// # Errors
///
/// Whatever [`qmm_grid`] refuses.
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
        Fire::at(QMM_FILE, "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1").apply(Grid::of(qmm_grid(n, 32, m, 64, 1)?, QMM_GROUP)),
        &[w.arg(), scales.arg(), biases.arg(), x.arg(), y.arg(), k.arg(), n.arg()],
    )
}

/// The `wm`/`wn` tile at `bm_64`, `bn_64`.
///
/// One of five `host_name` lines typed out in the shader rather than stamped
/// from the instantiation macro, which is why it is its own routine with its
/// tile fixed rather than a point of [`qmm_t`]'s axis. `affine_qmm_t_aligned`
/// is the template they share and it has no routine: it is not an entrypoint.
///
/// # Errors
///
/// Whatever [`qmm_grid`] refuses.
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
        Fire::at(QMM_FILE, "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4").apply(Grid::of(qmm_grid(n, 64, m, 64, 1)?, QMM_GROUP)),
        &[w.arg(), scales.arg(), biases.arg(), x.arg(), y.arg(), k.arg(), n.arg()],
    )
}

/// A bf16 tensor quantised to affine u4, one group per lane.
///
/// Three results and their order is load-bearing: the codes, then the scale
/// plane, then the bias plane, which is the order every dequantising kernel
/// in this family binds them in.
///
/// # Errors
///
/// Whatever [`crate::routine::elementwise`] refuses.
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
        Fire::at(TRANSCODE_FILE, "affine_encode_u4_bf16").apply(Grid::of(elementwise(groups, 1)?, [GROUP_X, 1, 1])),
        &[input.arg(), codes.arg(), scales.arg(), biases.arg(), groups.arg(), group_size.arg()],
    )
}

/// A f32 tensor quantised to affine u4, one group per lane.
///
/// Three results and their order is load-bearing: the codes, then the scale
/// plane, then the bias plane, which is the order every dequantising kernel
/// in this family binds them in.
///
/// # Errors
///
/// Whatever [`crate::routine::elementwise`] refuses.
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
        Fire::at(TRANSCODE_FILE, "affine_encode_u4_f32").apply(Grid::of(elementwise(groups, 1)?, [GROUP_X, 1, 1])),
        &[input.arg(), codes.arg(), scales.arg(), biases.arg(), groups.arg(), group_size.arg()],
    )
}

/// An MXFP4 block expanded to bf16, one block per lane.
///
/// Two planes in and not three: MXFP4 has a shared exponent and no zero
/// point, so there is no bias to read.
///
/// # Errors
///
/// Whatever [`crate::routine::elementwise`] refuses.
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
        Fire::at(TRANSCODE_FILE, "mxfp4_dequant_bf16").apply(Grid::of(elementwise(blocks, 1)?, [GROUP_X, 1, 1])),
        &[payload.arg(), exponents.arg(), out.arg(), blocks.arg(), block_size.arg()],
    )
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Const, Encode, Tensor};
    use core::cell::{Cell, RefCell};

    /// One recorded dispatch: the fire, and the argument list.
    type Call = (Fire, Vec<ArgValue>);

    /// An `Encode` that remembers what it was asked to do, and answers the
    /// facts this file's TESTED bodies ask for: the staged scalar block
    /// every routine but the tiled/vector matmuls takes with `ctx.params()`,
    /// and the five scalars the tiled and vector forms ask under their own
    /// names -- `Rows` (every one of them, as a row count, a vector count or
    /// a group count), `RowStride` (the strided forms), and the split-K
    /// triple `qmm_t_splitk` alone reads.
    struct Seen {
        calls: RefCell<Vec<Call>>,
        params_handle: Cell<u32>,
        /// THE STATEMENT\'S SCALAR RUN, for a body that reads a word by
        /// index. Empty means "4096 at every slot", which is a plausible
        /// stride for the rows these tests build; a case that means a
        /// particular tiling or split count sets its own.
        words: RefCell<Vec<i32>>,
        rows: Cell<i32>,
        row_stride: Cell<i32>,
        k_partition_size: Cell<i32>,
        split_k_partition_stride: Cell<i32>,
        split_k: Cell<i32>,
    }

    impl Default for Seen {
        fn default() -> Self {
            Self {
                calls: RefCell::default(),
                params_handle: Cell::new(800),
                words: RefCell::default(),
                rows: Cell::new(4),
                row_stride: Cell::new(8192),
                k_partition_size: Cell::new(512),
                split_k_partition_stride: Cell::new(65536),
                split_k: Cell::new(8),
            }
        }
    }

    impl Encode for Seen {
        fn resolve(&self, _ty: kernels::Ty, source: kernels::Source) -> Result<ArgValue, Refusal> {
            use kernels::keys::Fact;
            // THE STATEMENT'S OWN SCALARS, which a body reads by index when its
            // params run is a struct and no `Const` mark can name a word inside
            // it -- see `Asks::param`. The probe answers a number that is
            // plausible for every reader: a stride wide enough for the rows
            // these tests build, and a positive tiling.
            if let kernels::Source::Slot(kernels::Kind::Param, n) = source {
                return Ok(ArgValue::I32(
                    self.words.borrow().get(usize::from(n)).copied().unwrap_or(4096),
                ));
            }
            if source == kernels::Source::Slot(kernels::Kind::Params, 0) {
                return Ok(ArgValue::Buffer(self.params_handle.get()));
            }
            if source == <keys::Rows as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.rows.get()));
            }
            // Anything else is refused: a probe that invented an answer to a
            // fact it does not know would let a body pass under test while
            // the same fact went unanswered on a real driver.
            Err(Refusal::Unstated { what: "a fact this probe does not answer" })
        }

        fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
            self.calls.borrow_mut().push((fire, args.to_vec()));
            Ok(())
        }
    }

    /// An extent the tile does not divide is refused, on EITHER axis.
    ///
    /// No entrypoint in `quant/qmm_t.metal` takes `m`, so nothing at all
    /// guards the row axis: a grid rounded up past the row count writes the
    /// overhang into whatever follows the output allocation. The driver's
    /// `Rule::Qmm` refused that and the routine keeps the refusal.
    ///
    /// The column half asserted the opposite -- "the COLUMN overhang is
    /// guarded by n, which every entrypoint takes" -- and `n` guards the
    /// epilogue's STORE, not the weight tile the MMA loop loads. The file's
    /// own header states `M % BM == 0, N % BN == 0 and K % BK == 0` as the
    /// condition under which the driver may select it at all. qwen3.6's
    /// `in_proj_a` is 48 wide against a `bn` of 32; the second column tile
    /// read past the end of the weight and the projection came back NaN from
    /// row 32 of 64, in a different place on each fire.
    #[test]
    fn an_extent_the_tile_does_not_divide_is_refused() {
        assert!(
            matches!(
                qmm_grid(4096, 32, 100, 32, 1),
                Err(Refusal::Misaligned { .. })
            ),
            "a hundred rows is three tiles of thirty-two and a remainder"
        );
        assert_eq!(qmm_grid(4096, 32, 96, 32, 1), Ok([4096, 6, 2]));
        assert!(
            matches!(
                qmm_grid(4095, 32, 96, 32, 1),
                Err(Refusal::Misaligned { .. })
            ),
            "and a column count off the tile is the same refusal"
        );
        assert!(
            matches!(qmm_grid(48, 32, 64, 32, 1), Err(Refusal::Misaligned { .. })),
            "qwen3.6's `in_proj_a`, which is what found this"
        );
    }

    /// The four-axis point is group major, then bits, then the row tile, then
    /// the column tile.
    ///
    /// The order is the instantiation macro's and not a choice: getting it
    /// wrong picks a real entrypoint for the wrong coordinate. g64/b8 and
    /// g128/b4 pack to identical shapes, so the wrong one of that pair
    /// unpacks fluent nonsense rather than failing.
    #[test]
    fn the_quantisation_point_is_group_major() {
        assert_eq!(qmm_point(32, 4, 16, 16), Ok(0));
        assert_eq!(QMM_T[0], "affine_qmm_t_bfloat16_gs_32_b_4_bm_16_bn_16");
        assert_eq!(qmm_point(32, 4, 16, 32), Ok(1));
        assert_eq!(QMM_T[1], "affine_qmm_t_bfloat16_gs_32_b_4_bm_16_bn_32");
        assert_eq!(qmm_point(32, 8, 16, 16), Ok(9));
        assert_eq!(QMM_T[9], "affine_qmm_t_bfloat16_gs_32_b_8_bm_16_bn_16");
        assert_eq!(qmm_point(64, 4, 16, 16), Ok(18));
        assert_eq!(QMM_T[18], "affine_qmm_t_bfloat16_gs_64_b_4_bm_16_bn_16");
        assert_eq!(qmm_point(128, 8, 64, 64), Ok(53));
        assert_eq!(QMM_T[53], "affine_qmm_t_bfloat16_gs_128_b_8_bm_64_bn_64");
        assert!(qmm_point(16, 4, 16, 16).is_err(), "no gs_16 is compiled");
        assert!(qmm_point(64, 6, 16, 16).is_err(), "nor a six-bit code");
        assert!(
            qmm_point(64, 4, 128, 16).is_err(),
            "nor a 128-row tile here"
        );
    }

    /// The `_bn_32` forms index a three-axis table and read their column tile
    /// from [`WIDE_BN`].
    ///
    /// Eighteen points where the plain form has fifty-four, because the
    /// column tile is not a choice the caller has -- passing one would let a
    /// `bn` reach the grid that no entrypoint was stamped at.
    #[test]
    fn the_wide_forms_take_their_column_tile_from_the_shader() {
        assert_eq!(QMM_T_SPLITK.len(), 18);
        assert_eq!(wide_point(32, 4, 16), Ok(0));
        assert_eq!(
            QMM_T_SPLITK[0],
            "affine_qmm_t_splitk_bfloat16_gs_32_b_4_bm_16_bn_32"
        );
        assert_eq!(wide_point(128, 8, 64), Ok(17));
        let seen = Seen::default();
        seen.rows.set(32);
        qmm_t_strided(
            &seen,
            Const::new(Tensor::<u32>::new(1)),
            Const::new(Tensor::<bf16>::new(2)),
            Const::new(Tensor::<bf16>::new(3)),
            In::new(Tensor::<bf16>::new(5)),
            Out { ptr: Tensor::<bf16>::new(6), rows: 0, width: 2048 },
            Const::new(64),
            Const::new(4),
            Const::new(32))
        .expect("a launch");
        let calls = seen.calls.borrow();
        let (fire, _) = &calls[0];
        assert_eq!(
            fire.lanes[0],
            32 * (2048 / WIDE_BN as u32),
            "the column axis is tiled at thirty-two whatever the caller thinks"
        );
    }

    /// The split-K matmul states its partition count on the GRID, and the
    /// argument table has a hole where a caller might expect it.
    ///
    /// It used to say "twice: once as the grid's z extent and once as an
    /// argument", and asserted the count at `args[10]`. The shader says
    /// otherwise. `affine_qmm_t_splitk` declares `w`, `scales`, `biases` and
    /// `x` at buffers 0-3, `y` at 8, `K` and `N` at 5 and 6, and
    /// `k_partition_size` and `split_k_partition_stride` at 9 and 10 --
    /// buffers 4 and 7 are declared by nothing, which is what the two `pad`
    /// arguments hold open, and NO slot carries `split_k`. A threadgroup
    /// learns which partition it is from its z position and strides the
    /// partial buffer by `split_k_partition_stride`; the count itself it
    /// never needs.
    ///
    /// So the grid half is the whole claim, and `args[10]` is pinned to the
    /// stride it actually carries -- which is the assertion that would have
    /// caught the table shifting under the count in the first place.
    #[test]
    fn a_split_k_matmul_states_its_partitions_on_the_grid() {
        let seen = Seen::default();
        seen.rows.set(32);
        // THE RUN THIS CASE IS ABOUT, at the words HEAD's `Param<N>` named:
        // a K partition of 16, a partial stride of 65536 and eight splits.
        // The body reads them by index because its params run is the shader's
        // struct -- see `Asks::param`.
        {
            let mut w = seen.words.borrow_mut();
            w.resize(6, 4096);
            w[3] = 16;
            w[4] = 65_536;
            w[5] = 8;
        }
        qmm_t_splitk(
            &seen,
            Const::new(Tensor::<u32>::new(1)),
            Const::new(Tensor::<bf16>::new(2)),
            Const::new(Tensor::<bf16>::new(3)),
            In::new(Tensor::<bf16>::new(5)),
            Out { ptr: Tensor::<bf16>::new(6), rows: 0, width: 32 },
            Const::new(64),
            Const::new(4),
            Const::new(32))
        .expect("a launch");
        let calls = seen.calls.borrow();
        let (fire, args) = &calls[0];
        assert_eq!(fire.lanes[2], 8 * QMM_GROUP[2], "eight partitions of k");
        assert_eq!(
            args[10],
            65536.arg(),
            "slot 10 is the partial buffer's stride, which is what the \
             shader declares there"
        );
        assert_eq!(args.len(), 11);
    }

    /// The matvec binds its WEIGHTS first.
    ///
    /// The trace states them last. Binding a trace's order here put the
    /// activation where the packed weight belongs -- on every projection of
    /// every layer -- and the fix is that this list is the shader's, not the
    /// trace's.
    #[test]
    fn a_matvec_binds_the_weight_planes_ahead_of_the_activation() {
        let seen = Seen::default();
        seen.rows.set(1);
        qmv_fast(
            &seen,
            Const::new(Tensor::<u32>::new(10)),
            Const::new(Tensor::<bf16>::new(11)),
            Const::new(Tensor::<bf16>::new(12)),
            In::new(Tensor::<bf16>::new(13)),
            Out { ptr: Tensor::<bf16>::new(14), rows: 0, width: 4096 },
            Const::new(64),
            Const::new(4))
        .expect("a launch");
        let calls = seen.calls.borrow();
        let (fire, args) = &calls[0];
        assert_eq!(args[0], Tensor::<bf16>::new(10).arg(), "the packed weight");
        assert_eq!(args[3], Tensor::<bf16>::new(13).arg(), "then the activation");
        // BOUND AS THE MARK BINDS IT. `Tensor::arg()` is the READ form; this
        // operand is an `Out`, and an `Out` reaches for `BindMut` so the
        // driver can tell which buffers a dispatch writes.
        assert_eq!(args[4], Out::<Tensor<bf16>>::new(Tensor::<bf16>::new(14)).arg(), "then the result");
        assert_eq!(fire.entrypoint, "affine_qmv_fast_bfloat16_gs_64_b_4");
        assert_eq!(fire.lanes, [32, 1024, 1], "one group per eight output rows");
    }

    /// A fused-residual GEMM contracts over its ACTIVATION and not over the
    /// residual it adds.
    ///
    /// Both precast residual forms read `residual.width`. A residual is the
    /// layer's stream, so it is as wide as this projection's OUTPUT -- which
    /// is the same number as its input only where the projection is square.
    /// qwen3.5's o_proj is 2048 into 2048 and this measured right there for
    /// as long as it existed.
    ///
    /// gpt-oss's o_proj is 4096 into 2880, so the GEMM was handed `K = 2880`:
    /// a contraction a third short, and -- because `qmm_t_loaded_impl` starts
    /// each row of the activation at `x + y_row * k_len` -- a row stride that
    /// had every row but the first reading out of the middle of an earlier
    /// one. Only a prefill long enough to take the batched arm ever fired it,
    /// so short prompts stayed fluent and long ones came back generic.
    #[test]
    fn a_fused_residual_gemm_contracts_over_its_activation() {
        let seen = Seen::default();
        seen.rows.set(64);
        let half_in = || In { ptr: Tensor::<f16>::new(20), rows: 64, width: 4096 };
        let residual = || In { ptr: Tensor::<bf16>::new(21), rows: 64, width: 2880 };
        let y = || Out { ptr: Tensor::<bf16>::new(22), rows: 64, width: 2880 };
        qmm_t_residual_fp16_precast(
            &seen,
            Const::new(Tensor::<u32>::new(10)),
            Const::new(Tensor::<bf16>::new(11)),
            Const::new(Tensor::<bf16>::new(12)),
            y(),
            half_in(),
            residual(),
            Const::new(32),
            Const::new(32))
        .expect("a launch");
        qmm_t_strided_fp16_precast_residual(
            &seen,
            Const::new(Tensor::<u32>::new(10)),
            Const::new(Tensor::<bf16>::new(11)),
            Const::new(Tensor::<bf16>::new(12)),
            y(),
            half_in(),
            residual(),
            Const::new(32))
        .expect("a launch");
        let calls = seen.calls.borrow();
        assert_eq!(calls.len(), 2, "both precast residual forms fired");
        for (fire, args) in calls.iter() {
            assert_eq!(
                args[5],
                ArgValue::I32(4096),
                "`{}` contracts over the activation's 4096, not the residual's 2880",
                fire.entrypoint
            );
            assert_eq!(
                args[6],
                ArgValue::I32(2880),
                "`{}` writes the output's 2880",
                fire.entrypoint
            );
        }
    }

    /// The wide matvec's grid counts groups of FOUR batch vectors and its
    /// argument counts the vectors.
    ///
    /// The two are different numbers whenever the batch does not divide by
    /// four, and both are read: the grid to know how many groups to run, `m`
    /// to guard the last one's tail.
    #[test]
    fn the_wide_matvec_counts_groups_of_four_and_states_the_vectors() {
        assert_eq!(quarters(8), 2);
        assert_eq!(quarters(9), 3, "rounded up");
        assert_eq!(
            quarters(0),
            0,
            "and left alone so qmv_grid does the refusing"
        );
        let seen = Seen::default();
        // The stride this case asserts, at the word the body reads it from.
        seen.words.borrow_mut().extend([4096, 4096, 8192]);
        seen.rows.set(9);
        qmv_wide_strided(
            &seen,
            Const::new(Tensor::<u32>::new(1)),
            Const::new(Tensor::<bf16>::new(2)),
            Const::new(Tensor::<bf16>::new(3)),
            In::new(Tensor::<bf16>::new(5)),
            Out { ptr: Tensor::<bf16>::new(6), rows: 0, width: 4096 },
            Const::new(4))
        .expect("a launch");
        let calls = seen.calls.borrow();
        let (fire, args) = &calls[0];
        assert_eq!(
            fire.lanes[0],
            3 * 32,
            "nine vectors is three groups of four"
        );
        // Slot 9, not 8. `affine_qmv_wide_strided` declares `row_stride` at
        // buffer 8 and `M` at 9, with 7 declared by nothing -- which is the
        // `pad` this call now passes. The count is still stated; the table
        // grew a hole in front of it.
        assert_eq!(args[8], 8192.arg(), "the row stride the shader reads at 8");
        assert_eq!(args[9], 9.arg(), "and the kernel is told there are nine");
        assert_eq!(
            fire.file, QMM_FILE,
            "it lives with the tiles, not the other matvecs"
        );
    }

    /// The affine encoder's scale and bias planes are RESULTS.
    ///
    /// They were typed as inputs when this family was first written, and the
    /// cross-backend gate caught it: an encoder that cannot write the two
    /// planes it exists to produce. The order is the one every dequantising
    /// kernel here reads them back in.
    #[test]
    fn the_affine_encoder_writes_three_planes() {
        let seen = Seen::default();
        seen.rows.set(1024);
        encode_u4_bf16(
            &seen,
            In::new(Tensor::<bf16>::new(1)),
            Out::new(Tensor::<u32>::new(2)),
            Out::new(Tensor::<bf16>::new(3)),
            Out::new(Tensor::<bf16>::new(4)),
            Const::new(64))
        .expect("a launch");
        let calls = seen.calls.borrow();
        let (fire, args) = &calls[0];
        // All three are `Out`s -- the test's own name says so -- so all three
        // bind through `BindMut`. `Tensor::arg()` would be the read form.
        assert_eq!(args[1], Out::<Tensor<u32>>::new(Tensor::<u32>::new(2)).arg(), "the codes");
        assert_eq!(args[2], Out::<Tensor<bf16>>::new(Tensor::<bf16>::new(3)).arg(), "then the scales");
        assert_eq!(args[3], Out::<Tensor<bf16>>::new(Tensor::<bf16>::new(4)).arg(), "then the biases");
        assert_eq!(fire.lanes, [1024, 1, 1], "one group per lane");
    }
}

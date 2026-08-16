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

#![allow(clippy::too_many_arguments)]

use kernels::routine::Refusal;

use crate::routine::{keys, Ask, Bind, Block, Buf, BufMut, Ctx, Env, Fire, Param, Routine};
use crate::routine::{InSlot, OutSlot, Weight};

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
            file: QMM_FILE,
            lanes: qmm_grid(*n, *bn, *m, *bm, 1)?,
            group: QMM_GROUP,
        },
        &[w.v(), scales.v(), biases.v(), x.v(), y.v(), k.v(), n.v()],
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
            file: QMM_FILE,
            lanes: qmm_grid(*n, *bn, *m, *bm, 1)?,
            group: QMM_GROUP,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            x.v(),
            y.v(),
            k.v(),
            n.v(),
            bias.v(),
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
            file: QMM_FILE,
            lanes: qmm_grid(*n, *bn, *m, *bm, 1)?,
            group: QMM_GROUP,
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
pub fn qmm_t_fp16_precast(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    pad: Weight<0, Env<Buf>>,
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
            file: QMM_FILE,
            lanes: qmm_grid(*n, *bn, *m, *bm, 1)?,
            group: QMM_GROUP,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            pad.v(),
            y.v(),
            k.v(),
            n.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            half_in.v(),
        ],
    )
}

/// [`qmm_t_fp16_precast`] with the per-column bias.
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
    pad: Weight<0, Env<Buf>>,
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
            file: QMM_FILE,
            lanes: qmm_grid(*n, *bn, *m, *bm, 1)?,
            group: QMM_GROUP,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            pad.v(),
            y.v(),
            k.v(),
            n.v(),
            bias.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            half_in.v(),
        ],
    )
}

/// [`qmm_t_fp16_precast`] with the elementwise residual.
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
    pad: Weight<0, Env<Buf>>,
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
            file: QMM_FILE,
            lanes: qmm_grid(*n, *bn, *m, *bm, 1)?,
            group: QMM_GROUP,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            pad.v(),
            y.v(),
            k.v(),
            n.v(),
            residual.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            half_in.v(),
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
pub fn qmm_t_splitk(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    pad: Weight<0, Buf>,
    x: InSlot<0, Buf>,
    out: OutSlot<0, BufMut>,
    k: Param<0, i32>,
    n: Param<1, i32>,
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
            file: QMM_FILE,
            lanes: qmm_grid(*n, WIDE_BN, *m, *bm, *split_k)?,
            group: QMM_GROUP,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            x.v(),
            pad.v(),
            k.v(),
            n.v(),
            pad.v(),
            out.v(),
            k_partition_size.v(),
            split_k_partition_stride.v(),
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
pub fn qmm_t_splitk_f32(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    pad: Weight<0, Buf>,
    x: InSlot<0, Buf>,
    out: OutSlot<0, BufMut>,
    k: Param<0, i32>,
    n: Param<1, i32>,
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
            file: QMM_FILE,
            lanes: qmm_grid(*n, WIDE_BN, *m, *bm, *split_k)?,
            group: QMM_GROUP,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            x.v(),
            pad.v(),
            k.v(),
            n.v(),
            pad.v(),
            out.v(),
            k_partition_size.v(),
            split_k_partition_stride.v(),
        ],
    )
}

/// [`qmm_t_splitk`] over a pre-cast half activation.
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
    pad: Weight<0, Env<Buf>>,
    out: OutSlot<0, BufMut>,
    half_in: InSlot<0, Buf>,
    k: Param<0, i32>,
    n: Param<1, i32>,
    k_partition_size: Param<3, i32>,
    split_k_partition_stride: Param<4, i32>,
    split_k: Param<5, i32>,
    bm: Ask<keys::TileM, i32>,
    m: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: QMM_T_SPLITK_FP16_PRECAST[row_tile_point(*bm)?],
            file: QMM_FILE,
            lanes: qmm_grid(*n, WIDE_BN, *m, *bm, *split_k)?,
            group: QMM_GROUP,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            pad.v(),
            pad.v(),
            k.v(),
            n.v(),
            pad.v(),
            out.v(),
            k_partition_size.v(),
            split_k_partition_stride.v(),
            pad.v(),
            half_in.v(),
        ],
    )
}

/// [`qmm_t_splitk_f32`] over a pre-cast half activation.
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
    pad: Weight<0, Env<Buf>>,
    out: OutSlot<0, BufMut>,
    half_in: InSlot<0, Buf>,
    k: Param<0, i32>,
    n: Param<1, i32>,
    k_partition_size: Param<3, i32>,
    split_k_partition_stride: Param<4, i32>,
    split_k: Param<5, i32>,
    bm: Ask<keys::TileM, i32>,
    m: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: QMM_T_SPLITK_FP16_PRECAST_F32[row_tile_point(*bm)?],
            file: QMM_FILE,
            lanes: qmm_grid(*n, WIDE_BN, *m, *bm, *split_k)?,
            group: QMM_GROUP,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            pad.v(),
            pad.v(),
            k.v(),
            n.v(),
            pad.v(),
            out.v(),
            k_partition_size.v(),
            split_k_partition_stride.v(),
            pad.v(),
            half_in.v(),
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
pub fn qmm_t_strided(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    pad: Weight<0, Buf>,
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
            file: QMM_FILE,
            lanes: qmm_grid(*n, WIDE_BN, *m, *bm, 1)?,
            group: QMM_GROUP,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            x.v(),
            y.v(),
            k.v(),
            n.v(),
            pad.v(),
            row_stride.v(),
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
            file: QMM_FILE,
            lanes: qmm_grid(*n, WIDE_BN, *m, *bm, 1)?,
            group: QMM_GROUP,
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
            row_stride.v(),
        ],
    )
}

/// [`qmm_t_strided`] over a pre-cast half activation.
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
    pad: Weight<0, Env<Buf>>,
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
            file: QMM_FILE,
            lanes: qmm_grid(*n, WIDE_BN, *m, *bm, 1)?,
            group: QMM_GROUP,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            pad.v(),
            y.v(),
            k.v(),
            n.v(),
            pad.v(),
            row_stride.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            half_in.v(),
        ],
    )
}

/// [`qmm_t_strided_fp16_precast`] with the elementwise residual.
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
    pad: Weight<0, Env<Buf>>,
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
            file: QMM_FILE,
            lanes: qmm_grid(*n, WIDE_BN, *m, *bm, 1)?,
            group: QMM_GROUP,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            pad.v(),
            y.v(),
            k.v(),
            n.v(),
            residual.v(),
            row_stride.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            half_in.v(),
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
pub fn qmm_splitk_reduce(
    ctx: &Ctx<'_>,
    y: OutSlot<0, BufMut>,
    partial: InSlot<0, Buf>,
    pad: InSlot<0, Buf>,
    n: Param<1, i32>,
    split_k_partition_stride: Param<3, i32>,
    split_k: Param<4, i32>,
    m: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "qmm_splitk_reduce_bfloat16",
            file: QMM_FILE,
            lanes: crate::routine::elementwise_rows(*n, *m)?,
            group: [GROUP_X, 1, 1],
        },
        &[
            pad.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            y.v(),
            pad.v(),
            n.v(),
            pad.v(),
            partial.v(),
            pad.v(),
            split_k_partition_stride.v(),
            split_k.v(),
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
pub fn qmm_splitk_reduce_f32(
    ctx: &Ctx<'_>,
    y: OutSlot<0, BufMut>,
    partial: InSlot<0, Buf>,
    pad: InSlot<0, Buf>,
    n: Param<1, i32>,
    split_k_partition_stride: Param<3, i32>,
    split_k: Param<4, i32>,
    m: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "qmm_splitk_reduce_f32_bfloat16",
            file: QMM_FILE,
            lanes: crate::routine::elementwise_rows(*n, *m)?,
            group: [GROUP_X, 1, 1],
        },
        &[
            pad.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            y.v(),
            pad.v(),
            n.v(),
            pad.v(),
            partial.v(),
            pad.v(),
            split_k_partition_stride.v(),
            split_k.v(),
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
pub fn cast_qmm_input_bfloat16_to_float16(
    ctx: &Ctx<'_>,
    pad: InSlot<0, Buf>,
    cast_in: InSlot<0, Buf>,
    half_out: OutSlot<0, BufMut>,
    count: Param<3, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "cast_qmm_input_bfloat16_to_float16",
            file: QMM_FILE,
            lanes: crate::routine::elementwise(*count, 1)?,
            group: [GROUP_X, 1, 1],
        },
        &[
            pad.v(),
            pad.v(),
            pad.v(),
            cast_in.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            half_out.v(),
            count.v(),
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
pub fn cast_qmm_input_strided_bfloat16_to_float16(
    ctx: &Ctx<'_>,
    pad: InSlot<0, Env<Buf>>,
    cast_in: InSlot<0, Buf>,
    half_out: OutSlot<0, BufMut>,
    k: Param<0, i32>,
    row_stride: Param<2, i32>,
    rows: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "cast_qmm_input_strided_bfloat16_to_float16",
            file: QMM_FILE,
            lanes: crate::routine::elementwise_rows(*k, *rows)?,
            group: [GROUP_X, 1, 1],
        },
        &[
            pad.v(),
            pad.v(),
            pad.v(),
            cast_in.v(),
            pad.v(),
            k.v(),
            pad.v(),
            pad.v(),
            row_stride.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            half_out.v(),
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
            file: QMV_FILE,
            lanes: qmv_grid(*vecs, *out_vec_size)?,
            group: QMV_GROUP,
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

/// [`qmv_fast`] with the block residual its epilogue folds, which the trace
/// states as a second input and the kernel takes at the very end.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a point the shader tree does not carry, and
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
            file: QMV_FILE,
            lanes: qmv_grid(*vecs, *out_vec_size)?,
            group: QMV_GROUP,
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
pub fn qmv_tail(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    pad: Weight<0, Buf>,
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
            file: QMV_FILE,
            lanes: qmv_grid(*vecs, *out_vec_size)?,
            group: QMV_GROUP,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            x.v(),
            y.v(),
            in_vec_size.v(),
            out_vec_size.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            pad.v(),
        ],
    )
}

/// [`qmv_tail`] with a per-column bias.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a point the shader tree does not carry, and
/// whatever [`qmv_grid`] refuses.
pub fn qmv_tail_bias(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    pad: Weight<0, Buf>,
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
            file: QMV_FILE,
            lanes: qmv_grid(*vecs, *out_vec_size)?,
            group: QMV_GROUP,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            x.v(),
            y.v(),
            in_vec_size.v(),
            out_vec_size.v(),
            bias.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            pad.v(),
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
pub fn qmv_wide_strided(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    pad: Weight<0, Buf>,
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
            file: QMM_FILE,
            lanes: qmv_grid(quarters(*m), *out_vec_size)?,
            group: QMV_GROUP,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            x.v(),
            y.v(),
            in_vec_size.v(),
            out_vec_size.v(),
            pad.v(),
            row_stride.v(),
            m.v(),
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
            file: QMM_FILE,
            lanes: qmm_grid(*n, 32, *m, 128, 1)?,
            group: QMM_GROUP,
        },
        &[w.v(), scales.v(), biases.v(), x.v(), y.v(), k.v(), n.v()],
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
            file: QMM_FILE,
            lanes: qmm_grid(*n, 32, *m, 32, 1)?,
            group: QMM_GROUP,
        },
        &[w.v(), scales.v(), biases.v(), x.v(), y.v(), k.v(), n.v()],
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
            file: QMM_FILE,
            lanes: qmm_grid(*n, 32, *m, 64, 1)?,
            group: QMM_GROUP,
        },
        &[w.v(), scales.v(), biases.v(), x.v(), y.v(), k.v(), n.v()],
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
            file: QMM_FILE,
            lanes: qmm_grid(*n, 32, *m, 64, 1)?,
            group: QMM_GROUP,
        },
        &[w.v(), scales.v(), biases.v(), x.v(), y.v(), k.v(), n.v()],
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
            file: QMM_FILE,
            lanes: qmm_grid(*n, 64, *m, 64, 1)?,
            group: QMM_GROUP,
        },
        &[w.v(), scales.v(), biases.v(), x.v(), y.v(), k.v(), n.v()],
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
            file: TRANSCODE_FILE,
            lanes: crate::routine::elementwise(*groups, 1)?,
            group: [GROUP_X, 1, 1],
        },
        &[input.v(), codes.v(), scales.v(), biases.v(), params.v()],
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
            file: TRANSCODE_FILE,
            lanes: crate::routine::elementwise(*groups, 1)?,
            group: [GROUP_X, 1, 1],
        },
        &[input.v(), codes.v(), scales.v(), biases.v(), params.v()],
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
            file: TRANSCODE_FILE,
            lanes: crate::routine::elementwise(*blocks, 1)?,
            group: [GROUP_X, 1, 1],
        },
        &[payload.v(), exponents.v(), out.v(), params.v()],
    )
}

/// The family, in the order the rows above state it.
pub static ROUTINES: &[Routine] = &[
    crate::routine!(cast_qmm_input_bfloat16_to_float16),
    crate::routine!(cast_qmm_input_strided_bfloat16_to_float16),
    crate::routine!(encode_u4_bf16),
    crate::routine!(encode_u4_f32),
    crate::routine!(mxfp4_dequant_bf16),
    crate::routine!(qmm_splitk_reduce),
    crate::routine!(qmm_splitk_reduce_f32),
    crate::routine!(qmm_t),
    crate::routine!(qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4),
    crate::routine!(qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2),
    crate::routine!(qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2),
    crate::routine!(qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1),
    crate::routine!(qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4),
    crate::routine!(qmm_t_bias),
    crate::routine!(qmm_t_bias_fp16_precast),
    crate::routine!(qmm_t_fp16_precast),
    crate::routine!(qmm_t_residual),
    crate::routine!(qmm_t_residual_fp16_precast),
    crate::routine!(qmm_t_splitk),
    crate::routine!(qmm_t_splitk_f32),
    crate::routine!(qmm_t_splitk_fp16_precast),
    crate::routine!(qmm_t_splitk_fp16_precast_f32),
    crate::routine!(qmm_t_strided),
    crate::routine!(qmm_t_strided_fp16_precast),
    crate::routine!(qmm_t_strided_fp16_precast_residual),
    crate::routine!(qmm_t_strided_residual),
    crate::routine!(qmv_fast),
    crate::routine!(qmv_fast_residual),
    crate::routine!(qmv_tail),
    crate::routine!(qmv_tail_bias),
    crate::routine!(qmv_wide_strided),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Encode};
    use core::cell::RefCell;

    /// One recorded dispatch: the fire, and the argument list.
    type Call = (Fire, Vec<ArgValue>);

    /// An `Encode` that remembers what it was asked to do.
    #[derive(Default)]
    struct Seen(RefCell<Vec<Call>>);

    impl Encode for Seen {
        fn dispatch(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
            self.0.borrow_mut().push((fire, args.to_vec()));
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
        qmm_t_strided(
            &seen,
            Weight::new(Buf(1)),
            Weight::new(Buf(2)),
            Weight::new(Buf(3)),
            Weight::new(Buf(4)),
            InSlot::new(Buf(5)),
            OutSlot::new(BufMut(6)),
            Param::new(4096),
            Param::new(2048),
            Param::new(8192),
            Ask::new(64),
            Ask::new(4),
            Ask::new(32),
            Ask::new(64),
        )
        .expect("a launch");
        let calls = seen.0.borrow();
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
        qmm_t_splitk(
            &seen,
            Weight::new(Buf(1)),
            Weight::new(Buf(2)),
            Weight::new(Buf(3)),
            Weight::new(Buf(4)),
            InSlot::new(Buf(5)),
            OutSlot::new(BufMut(6)),
            Param::new(4096),
            Param::new(2048),
            Param::new(512),
            Param::new(65536),
            Param::new(8),
            Ask::new(64),
            Ask::new(4),
            Ask::new(32),
            Ask::new(64),
        )
        .expect("a launch");
        let calls = seen.0.borrow();
        let (fire, args) = &calls[0];
        assert_eq!(fire.lanes[2], 8 * QMM_GROUP[2], "eight partitions of k");
        assert_eq!(
            args[10],
            65536.v(),
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
        qmv_fast(
            &seen,
            Weight::new(Buf(10)),
            Weight::new(Buf(11)),
            Weight::new(Buf(12)),
            InSlot::new(Buf(13)),
            OutSlot::new(BufMut(14)),
            Param::new(2048),
            Param::new(4096),
            Ask::new(64),
            Ask::new(4),
            Ask::new(1),
        )
        .expect("a launch");
        let calls = seen.0.borrow();
        let (fire, args) = &calls[0];
        assert_eq!(args[0], Buf(10).v(), "the packed weight");
        assert_eq!(args[3], Buf(13).v(), "then the activation");
        assert_eq!(args[4], BufMut(14).v(), "then the result");
        assert_eq!(fire.entrypoint, "affine_qmv_fast_bfloat16_gs_64_b_4");
        assert_eq!(fire.lanes, [32, 1024, 1], "one group per eight output rows");
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
        qmv_wide_strided(
            &seen,
            Weight::new(Buf(1)),
            Weight::new(Buf(2)),
            Weight::new(Buf(3)),
            Weight::new(Buf(4)),
            InSlot::new(Buf(5)),
            OutSlot::new(BufMut(6)),
            Param::new(2048),
            Param::new(4096),
            Param::new(8192),
            Ask::new(9),
            Ask::new(4),
        )
        .expect("a launch");
        let calls = seen.0.borrow();
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
        assert_eq!(args[8], 8192.v(), "the row stride the shader reads at 8");
        assert_eq!(args[9], 9.v(), "and the kernel is told there are nine");
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
        encode_u4_bf16(
            &seen,
            InSlot::new(Buf(1)),
            OutSlot::new(BufMut(2)),
            OutSlot::new(BufMut(3)),
            OutSlot::new(BufMut(4)),
            Block::new(Buf(5)),
            Ask::new(1024),
        )
        .expect("a launch");
        let calls = seen.0.borrow();
        let (fire, args) = &calls[0];
        assert_eq!(args[1], BufMut(2).v(), "the codes");
        assert_eq!(args[2], BufMut(3).v(), "then the scales");
        assert_eq!(args[3], BufMut(4).v(), "then the biases");
        assert_eq!(fire.lanes, [1024, 1, 1], "one group per lane");
    }
}

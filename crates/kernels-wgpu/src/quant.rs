#![allow(clippy::too_many_arguments)]
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

use kernels::KernelSig;

/// EMPTY: this family's rows have been RETIRED.
///
/// `refactor-bigplan.md` §7 Stage 3. The three transcode encoders were the
/// last to go and they could not go until their BODIES were fixed: each took a
/// `_params: Buf` it had no way to use and forwarded no scalars at all, so the
/// `@group(1)` block its shader reads would have arrived empty and the loop
/// over groups would have run zero times and reported success.
pub static KERNELS: &[KernelSig] = &[];
/// The entrypoints of this family's routines whose ROWS have been RETIRED.
///
/// `refactor-bigplan.md` §7 Stage 3. Not every kernel here has crossed its
/// arm — this family still states rows for the ones that have not — so this
/// is the retired SUBSET rather than the whole family, and
/// `a_retired_familys_stated_entrypoints_are_what_its_bodies_fire` compares
/// it against the bodies that fire them.
///
/// See [`crate::sample::ENTRYPOINTS`] for why a retired row's entrypoints
/// have to be stated at all.
pub static ENTRYPOINTS: &[&str] = &[
    "affine_encode_u4_bf16",
    "affine_encode_u4_f32",
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
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4",
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_16_bn_16",
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_16_bn_64",
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_16",
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2",
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_64",
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_16",
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32",
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2",
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1",
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64",
    "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4",
    "affine_qmm_t_bfloat16_gs_64_b_8_bm_16_bn_16",
    "affine_qmm_t_bfloat16_gs_64_b_8_bm_16_bn_32",
    "affine_qmm_t_bfloat16_gs_64_b_8_bm_16_bn_64",
    "affine_qmm_t_bfloat16_gs_64_b_8_bm_32_bn_16",
    "affine_qmm_t_bfloat16_gs_64_b_8_bm_32_bn_32",
    "affine_qmm_t_bfloat16_gs_64_b_8_bm_32_bn_64",
    "affine_qmm_t_bfloat16_gs_64_b_8_bm_64_bn_16",
    "affine_qmm_t_bfloat16_gs_64_b_8_bm_64_bn_32",
    "affine_qmm_t_bfloat16_gs_64_b_8_bm_64_bn_64",
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
    "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_16",
    "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_64",
    "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_16",
    "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_64",
    "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_16",
    "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32",
    "affine_qmm_t_residual_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_64",
    "affine_qmm_t_splitk_bfloat16_gs_128_b_4_bm_16_bn_32",
    "affine_qmm_t_splitk_bfloat16_gs_128_b_4_bm_32_bn_32",
    "affine_qmm_t_splitk_bfloat16_gs_128_b_4_bm_64_bn_32",
    "affine_qmm_t_splitk_bfloat16_gs_128_b_8_bm_16_bn_32",
    "affine_qmm_t_splitk_bfloat16_gs_128_b_8_bm_32_bn_32",
    "affine_qmm_t_splitk_bfloat16_gs_128_b_8_bm_64_bn_32",
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
    "affine_qmm_t_splitk_f32_bfloat16_gs_128_b_4_bm_16_bn_32",
    "affine_qmm_t_splitk_f32_bfloat16_gs_128_b_4_bm_32_bn_32",
    "affine_qmm_t_splitk_f32_bfloat16_gs_128_b_4_bm_64_bn_32",
    "affine_qmm_t_splitk_f32_bfloat16_gs_128_b_8_bm_16_bn_32",
    "affine_qmm_t_splitk_f32_bfloat16_gs_128_b_8_bm_32_bn_32",
    "affine_qmm_t_splitk_f32_bfloat16_gs_128_b_8_bm_64_bn_32",
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
    "affine_qmm_t_splitk_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_splitk_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_splitk_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32",
    "affine_qmm_t_splitk_fp16_precast_f32_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_splitk_fp16_precast_f32_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_splitk_fp16_precast_f32_bfloat16_gs_64_b_4_bm_64_bn_32",
    "affine_qmm_t_strided_bfloat16_gs_128_b_4_bm_16_bn_32",
    "affine_qmm_t_strided_bfloat16_gs_128_b_4_bm_32_bn_32",
    "affine_qmm_t_strided_bfloat16_gs_128_b_4_bm_64_bn_32",
    "affine_qmm_t_strided_bfloat16_gs_128_b_8_bm_16_bn_32",
    "affine_qmm_t_strided_bfloat16_gs_128_b_8_bm_32_bn_32",
    "affine_qmm_t_strided_bfloat16_gs_128_b_8_bm_64_bn_32",
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
    "affine_qmm_t_strided_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_strided_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_strided_fp16_precast_bfloat16_gs_64_b_4_bm_64_bn_32",
    "affine_qmm_t_strided_fp16_precast_residual_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_strided_fp16_precast_residual_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_strided_fp16_precast_residual_bfloat16_gs_64_b_4_bm_64_bn_32",
    "affine_qmm_t_strided_residual_bfloat16_gs_128_b_4_bm_16_bn_32",
    "affine_qmm_t_strided_residual_bfloat16_gs_128_b_4_bm_32_bn_32",
    "affine_qmm_t_strided_residual_bfloat16_gs_128_b_4_bm_64_bn_32",
    "affine_qmm_t_strided_residual_bfloat16_gs_128_b_8_bm_16_bn_32",
    "affine_qmm_t_strided_residual_bfloat16_gs_128_b_8_bm_32_bn_32",
    "affine_qmm_t_strided_residual_bfloat16_gs_128_b_8_bm_64_bn_32",
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
    "affine_qmv_fast_bfloat16_gs_128_b_4",
    "affine_qmv_fast_bfloat16_gs_128_b_8",
    "affine_qmv_fast_bfloat16_gs_32_b_4",
    "affine_qmv_fast_bfloat16_gs_32_b_8",
    "affine_qmv_fast_bfloat16_gs_64_b_4",
    "affine_qmv_fast_bfloat16_gs_64_b_8",
    "affine_qmv_fast_residual_bfloat16_gs_128_b_4",
    "affine_qmv_fast_residual_bfloat16_gs_128_b_8",
    "affine_qmv_fast_residual_bfloat16_gs_32_b_4",
    "affine_qmv_fast_residual_bfloat16_gs_32_b_8",
    "affine_qmv_fast_residual_bfloat16_gs_64_b_4",
    "affine_qmv_fast_residual_bfloat16_gs_64_b_8",
    "affine_qmv_tail_bfloat16_gs_64_b_4",
    "affine_qmv_tail_bfloat16_gs_64_b_8",
    "affine_qmv_tail_bias_bfloat16_gs_64_b_4",
    "affine_qmv_tail_bias_bfloat16_gs_64_b_8",
    "affine_qmv_wide_strided_bfloat16_gs_64_b_4_v_4_kl_8",
    "affine_qmv_wide_strided_bfloat16_gs_64_b_8_v_4_kl_8",
    "cast_qmm_input_bfloat16_to_float16",
    "cast_qmm_input_strided_bfloat16_to_float16",
    "qmm_splitk_reduce_bfloat16",
    "qmm_splitk_reduce_f32_bfloat16",
    "mxfp4_dequant_bf16",
];

use crate::routine::{Bind, Buf, BufMut, Ctx, Env, Fire, Routine};
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
/// The x extent of the wide forms: four batch vectors to a group.
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
pub fn qmm_t(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    x: Buf,
    y: BufMut,
    k: i32,
    n: i32,
    group: Env<i32>,
    bits: Env<i32>,
    bm: Env<i32>,
    bn: Env<i32>,
    m: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmm_t.wgsl",
            entrypoint: QMM_T[qmm_point(*group, *bits, *bm, *bn)?],
            lanes: qmm_grid(n, *bn, *m, *bm, 1)?,
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
/// whatever `qmm_grid` refuses.
pub fn qmm_t_bias(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    x: Buf,
    y: BufMut,
    bias: Buf,
    k: i32,
    n: i32,
    group: Env<i32>,
    bits: Env<i32>,
    bm: Env<i32>,
    bn: Env<i32>,
    m: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmm_t.wgsl",
            entrypoint: QMM_T_BIAS[qmm_point(*group, *bits, *bm, *bn)?],
            lanes: qmm_grid(n, *bn, *m, *bm, 1)?,
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
/// whatever `qmm_grid` refuses.
pub fn qmm_t_residual(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    x: Buf,
    y: BufMut,
    k: i32,
    n: i32,
    residual: Buf,
    group: Env<i32>,
    bits: Env<i32>,
    bm: Env<i32>,
    bn: Env<i32>,
    m: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmm_t.wgsl",
            entrypoint: QMM_T_RESIDUAL[qmm_point(*group, *bits, *bm, *bn)?],
            lanes: qmm_grid(n, *bn, *m, *bm, 1)?,
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
/// whatever `qmm_grid` refuses.
pub fn qmm_t_fp16_precast(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    y: BufMut,
    half_in: Buf,
    k: i32,
    n: i32,
    bm: Env<i32>,
    bn: Env<i32>,
    m: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmm_t.wgsl",
            entrypoint: QMM_T_FP16_PRECAST[tile_point(*bm, *bn)?],
            lanes: qmm_grid(n, *bn, *m, *bm, 1)?,
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
/// whatever `qmm_grid` refuses.
pub fn qmm_t_bias_fp16_precast(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    y: BufMut,
    bias: Buf,
    half_in: Buf,
    k: i32,
    n: i32,
    bm: Env<i32>,
    bn: Env<i32>,
    m: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmm_t.wgsl",
            entrypoint: QMM_T_BIAS_FP16_PRECAST[tile_point(*bm, *bn)?],
            lanes: qmm_grid(n, *bn, *m, *bm, 1)?,
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
/// whatever `qmm_grid` refuses.
pub fn qmm_t_residual_fp16_precast(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    y: BufMut,
    residual: Buf,
    half_in: Buf,
    k: i32,
    n: i32,
    bm: Env<i32>,
    bn: Env<i32>,
    m: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmm_t.wgsl",
            entrypoint: QMM_T_RESIDUAL_FP16_PRECAST[tile_point(*bm, *bn)?],
            lanes: qmm_grid(n, *bn, *m, *bm, 1)?,
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
/// whatever `qmm_grid` refuses.
pub fn qmm_t_splitk(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    x: Buf,
    out: BufMut,
    k: i32,
    n: i32,
    row_stride: i32,
    k_partition_size: i32,
    split_k_partition_stride: i32,
    split_k: i32,
    group: Env<i32>,
    bits: Env<i32>,
    bm: Env<i32>,
    m: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmm_t.wgsl",
            entrypoint: QMM_T_SPLITK[wide_point(*group, *bits, *bm)?],
            lanes: qmm_grid(n, WIDE_BN, *m, *bm, split_k)?,
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
/// whatever `qmm_grid` refuses.
pub fn qmm_t_splitk_f32(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    x: Buf,
    out: BufMut,
    k: i32,
    n: i32,
    row_stride: i32,
    k_partition_size: i32,
    split_k_partition_stride: i32,
    split_k: i32,
    group: Env<i32>,
    bits: Env<i32>,
    bm: Env<i32>,
    m: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmm_t.wgsl",
            entrypoint: QMM_T_SPLITK_F32[wide_point(*group, *bits, *bm)?],
            lanes: qmm_grid(n, WIDE_BN, *m, *bm, split_k)?,
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
/// whatever `qmm_grid` refuses.
pub fn qmm_t_splitk_fp16_precast(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    out: BufMut,
    half_in: Buf,
    k: i32,
    n: i32,
    row_stride: i32,
    k_partition_size: i32,
    split_k_partition_stride: i32,
    split_k: i32,
    bm: Env<i32>,
    m: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmm_t.wgsl",
            entrypoint: QMM_T_SPLITK_FP16_PRECAST[row_tile_point(*bm)?],
            lanes: qmm_grid(n, WIDE_BN, *m, *bm, split_k)?,
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
/// whatever `qmm_grid` refuses.
pub fn qmm_t_splitk_fp16_precast_f32(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    out: BufMut,
    half_in: Buf,
    k: i32,
    n: i32,
    row_stride: i32,
    k_partition_size: i32,
    split_k_partition_stride: i32,
    split_k: i32,
    bm: Env<i32>,
    m: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmm_t.wgsl",
            entrypoint: QMM_T_SPLITK_FP16_PRECAST_F32[row_tile_point(*bm)?],
            lanes: qmm_grid(n, WIDE_BN, *m, *bm, split_k)?,
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
/// whatever `qmm_grid` refuses.
pub fn qmm_t_strided(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    x: Buf,
    y: BufMut,
    k: i32,
    n: i32,
    row_stride: i32,
    group: Env<i32>,
    bits: Env<i32>,
    bm: Env<i32>,
    m: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmm_t.wgsl",
            entrypoint: QMM_T_STRIDED[wide_point(*group, *bits, *bm)?],
            lanes: qmm_grid(n, WIDE_BN, *m, *bm, 1)?,
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
/// whatever `qmm_grid` refuses.
pub fn qmm_t_strided_residual(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    x: Buf,
    y: BufMut,
    residual: Buf,
    k: i32,
    n: i32,
    row_stride: i32,
    group: Env<i32>,
    bits: Env<i32>,
    bm: Env<i32>,
    m: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmm_t.wgsl",
            entrypoint: QMM_T_STRIDED_RESIDUAL[wide_point(*group, *bits, *bm)?],
            lanes: qmm_grid(n, WIDE_BN, *m, *bm, 1)?,
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
/// whatever `qmm_grid` refuses.
pub fn qmm_t_strided_fp16_precast(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    y: BufMut,
    half_in: Buf,
    k: i32,
    n: i32,
    row_stride: i32,
    bm: Env<i32>,
    m: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmm_t.wgsl",
            entrypoint: QMM_T_STRIDED_FP16_PRECAST[row_tile_point(*bm)?],
            lanes: qmm_grid(n, WIDE_BN, *m, *bm, 1)?,
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
/// whatever `qmm_grid` refuses.
pub fn qmm_t_strided_fp16_precast_residual(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    y: BufMut,
    residual: Buf,
    half_in: Buf,
    k: i32,
    n: i32,
    row_stride: i32,
    bm: Env<i32>,
    m: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmm_t.wgsl",
            entrypoint: QMM_T_STRIDED_FP16_PRECAST_RESIDUAL[row_tile_point(*bm)?],
            lanes: qmm_grid(n, WIDE_BN, *m, *bm, 1)?,
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
    y: BufMut,
    partial: Buf,
    k: i32,
    n: i32,
    row_stride: i32,
    split_k_partition_stride: i32,
    split_k: i32,
    m: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmm_t.wgsl",
            entrypoint: "qmm_splitk_reduce_bfloat16",
            lanes: kernels::shader::elementwise_rows(n, *m)?,
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
    y: BufMut,
    partial: Buf,
    k: i32,
    n: i32,
    row_stride: i32,
    split_k_partition_stride: i32,
    split_k: i32,
    m: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmm_t.wgsl",
            entrypoint: "qmm_splitk_reduce_f32_bfloat16",
            lanes: kernels::shader::elementwise_rows(n, *m)?,
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
    cast_in: Buf,
    half_out: BufMut,
    k: i32,
    n: i32,
    row_stride: i32,
    count: i32,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmm_t.wgsl",
            entrypoint: "cast_qmm_input_bfloat16_to_float16",
            lanes: kernels::shader::elementwise(count, 1)?,
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
    cast_in: Buf,
    half_out: BufMut,
    k: i32,
    n: i32,
    row_stride: i32,
    count: i32,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmm_t.wgsl",
            entrypoint: "cast_qmm_input_strided_bfloat16_to_float16",
            lanes: kernels::shader::elementwise_rows(k, *rows)?,
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
/// whatever `qmv_grid` refuses.
pub fn qmv_fast(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    x: Buf,
    y: BufMut,
    in_vec_size: i32,
    out_vec_size: i32,
    group: Env<i32>,
    bits: Env<i32>,
    vecs: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmv.wgsl",
            entrypoint: QMV_FAST[codec_point(*group, *bits)?],
            lanes: qmv_grid(*vecs, out_vec_size)?,
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
/// whatever `qmv_grid` refuses.
pub fn qmv_fast_residual(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    x: Buf,
    y: BufMut,
    in_vec_size: i32,
    out_vec_size: i32,
    residual: Buf,
    group: Env<i32>,
    bits: Env<i32>,
    vecs: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmv.wgsl",
            entrypoint: QMV_FAST_RESIDUAL[codec_point(*group, *bits)?],
            lanes: qmv_grid(*vecs, out_vec_size)?,
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
/// whatever `qmv_grid` refuses.
pub fn qmv_tail(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    x: Buf,
    y: BufMut,
    in_vec_size: i32,
    out_vec_size: i32,
    bits: Env<i32>,
    vecs: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmv.wgsl",
            entrypoint: QMV_TAIL[bits_point(*bits)?],
            lanes: qmv_grid(*vecs, out_vec_size)?,
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
/// whatever `qmv_grid` refuses.
pub fn qmv_tail_bias(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    x: Buf,
    y: BufMut,
    bias: Buf,
    in_vec_size: i32,
    out_vec_size: i32,
    bits: Env<i32>,
    vecs: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmv.wgsl",
            entrypoint: QMV_TAIL_BIAS[bits_point(*bits)?],
            lanes: qmv_grid(*vecs, out_vec_size)?,
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
/// whatever `qmv_grid` refuses.
pub fn qmv_wide_strided(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    x: Buf,
    y: BufMut,
    in_vec_size: i32,
    out_vec_size: i32,
    row_stride: i32,
    m: i32,
    bits: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmv.wgsl",
            entrypoint: QMV_WIDE_STRIDED[bits_point(*bits)?],
            lanes: qmv_grid(quarters(m), out_vec_size)?,
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
/// [`Refusal::Empty`] for an empty rectangle, and whatever `qmm_grid` refuses.
pub fn qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    x: Buf,
    y: BufMut,
    k: i32,
    n: i32,
    m: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmm_t.wgsl",
            entrypoint: "affine_qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4",
            lanes: qmm_grid(n, 32, *m, 128, 1)?,
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
/// [`Refusal::Empty`] for an empty rectangle, and whatever `qmm_grid` refuses.
pub fn qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    x: Buf,
    y: BufMut,
    k: i32,
    n: i32,
    m: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmm_t.wgsl",
            entrypoint: "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2",
            lanes: qmm_grid(n, 32, *m, 32, 1)?,
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
/// [`Refusal::Empty`] for an empty rectangle, and whatever `qmm_grid` refuses.
pub fn qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    x: Buf,
    y: BufMut,
    k: i32,
    n: i32,
    m: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmm_t.wgsl",
            entrypoint: "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2",
            lanes: qmm_grid(n, 32, *m, 64, 1)?,
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
/// [`Refusal::Empty`] for an empty rectangle, and whatever `qmm_grid` refuses.
pub fn qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    x: Buf,
    y: BufMut,
    k: i32,
    n: i32,
    m: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmm_t.wgsl",
            entrypoint: "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1",
            lanes: qmm_grid(n, 32, *m, 64, 1)?,
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
/// [`Refusal::Empty`] for an empty rectangle, and whatever `qmm_grid` refuses.
pub fn qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    x: Buf,
    y: BufMut,
    k: i32,
    n: i32,
    m: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/qmm_t.wgsl",
            entrypoint: "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4",
            lanes: qmm_grid(n, 64, *m, 64, 1)?,
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
    input: Buf,
    codes: BufMut,
    scales: BufMut,
    biases: BufMut,
    // FORWARDED AS SCALARS, not as a buffer.
    //
    // `transcode.wgsl` puts this pair in a `@group(1) @binding(0)` UNIFORM —
    // Metal spells it `constant DequantParams&` and Vulkan copied that into a
    // storage binding, and neither shape is this one — so there is no
    // `@group(0)` slot to forward and the block is built from the scalars a
    // body passes. The signature took a `_params: Buf` it could not use and
    // passed NOTHING, so the block arrived empty and the shader read zeros:
    // zero groups is a loop that runs no iterations and reports success.
    // A TRACE scalar and not an `Env`, though it also fixes the grid.
    //
    // `transcode.wgsl` bounds its own loop with it (`if (block >= params.groups)
    // { return; }`), because `dispatch_workgroups` rounds up to a whole
    // workgroup and the buffer's length is not the bound. An `Env` argument
    // computes the grid and is NOT forwarded — that is what
    // `a_body_passes_the_arguments_its_signature_takes_in_order` enforces — so
    // a value the SHADER reads has to be stated.
    groups: i32,
    group_size: i32,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/transcode.wgsl",
            entrypoint: "affine_encode_u4_bf16",
            lanes: kernels::shader::elementwise(groups, 1)?,
        },
        &[
            input.v(),
            codes.v(),
            scales.v(),
            biases.v(),
            groups.v(),
            group_size.v(),
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
pub fn encode_u4_f32(
    ctx: &Ctx<'_>,
    input: Buf,
    codes: BufMut,
    scales: BufMut,
    biases: BufMut,
    // FORWARDED AS SCALARS, not as a buffer.
    //
    // `transcode.wgsl` puts this pair in a `@group(1) @binding(0)` UNIFORM —
    // Metal spells it `constant DequantParams&` and Vulkan copied that into a
    // storage binding, and neither shape is this one — so there is no
    // `@group(0)` slot to forward and the block is built from the scalars a
    // body passes. The signature took a `_params: Buf` it could not use and
    // passed NOTHING, so the block arrived empty and the shader read zeros:
    // zero groups is a loop that runs no iterations and reports success.
    //
    // A TRACE scalar, for the reason `encode_u4_bf16` states above.
    groups: i32,
    group_size: i32,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/transcode.wgsl",
            entrypoint: "affine_encode_u4_f32",
            lanes: kernels::shader::elementwise(groups, 1)?,
        },
        &[
            input.v(),
            codes.v(),
            scales.v(),
            biases.v(),
            groups.v(),
            group_size.v(),
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
pub fn mxfp4_dequant_bf16(
    ctx: &Ctx<'_>,
    payload: Buf,
    exponents: Buf,
    out: BufMut,
    // FORWARDED AS SCALARS, not as a buffer.
    //
    // `transcode.wgsl` puts this pair in a `@group(1) @binding(0)` UNIFORM —
    // Metal spells it `constant DequantParams&` and Vulkan copied that into a
    // storage binding, and neither shape is this one — so there is no
    // `@group(0)` slot to forward and the block is built from the scalars a
    // body passes. The signature took a `_params: Buf` it could not use and
    // passed NOTHING, so the block arrived empty and the shader read zeros:
    // zero groups is a loop that runs no iterations and reports success.
    // A TRACE scalar and not an `Env`, though it also fixes the grid.
    //
    // `transcode.wgsl` bounds its own loop with it (`if (block >= params.blocks)
    // { return; }`), because `dispatch_workgroups` rounds up to a whole
    // workgroup and the buffer's length is not the bound. An `Env` argument
    // computes the grid and is NOT forwarded — that is what
    // `a_body_passes_the_arguments_its_signature_takes_in_order` enforces — so
    // a value the SHADER reads has to be stated.
    blocks: i32,
    block_size: i32,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "quant/transcode.wgsl",
            entrypoint: "mxfp4_dequant_bf16",
            lanes: kernels::shader::elementwise(blocks, 1)?,
        },
        &[
            payload.v(),
            exponents.v(),
            out.v(),
            blocks.v(),
            block_size.v(),
        ],
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

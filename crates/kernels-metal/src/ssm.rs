//! Gated DeltaNet: the recurrent state kernels and their prep.
//!
//! `gdn` is an algorithm and not a model, so it takes no model qualifier --
//! the same call the CUDA table makes for `delta_attn_kda` and `indexer_dsa`.

use kernels::{Axis, KernelSig, kernel};

use crate::axes::*;

pub static KERNELS: &[KernelSig] = &[
    // 1 in gdn_core.metal
    kernel!(gdn_core "gdn_core", axes = &[BF16]),
    // 1 in gdn_prep.metal
    kernel!(gdn_core_recurrent "gdn_core_recurrent", axes = &[BF16]),
    // 9 in gdn_prep.metal
    kernel!(gdn_core_recurrent_prefill "gdn_core_recurrent_prefill",
        axes = &[Axis { what: "the shapes this kernel is compiled for", points: &["_bfloat16_l_16_v_1", "_bfloat16_l_16_v_2", "_bfloat16_l_16_v_4", "_bfloat16_l_32_v_2", "_bfloat16_l_32_v_4", "_bfloat16_l_32_v_8", "_bfloat16_l_4_v_1", "_bfloat16_l_8_v_1", "_bfloat16_l_8_v_2"] }]),
    // 1 in gdn_prep.metal
    kernel!(gdn_core_recurrent_slotted "gdn_core_recurrent_slotted", axes = &[BF16]),
    // 1 in gdn_core.metal
    kernel!(gdn_core_slotted "gdn_core_slotted", axes = &[BF16]),
    // 1 in gdn_prep.metal
    kernel!(gdn_prep "gdn_prep", axes = &[BF16]),
    // 1 in gdn_prep.metal
    kernel!(gdn_prep_prefill "gdn_prep_prefill", axes = &[BF16]),
    // 1 in gdn_prep.metal
    kernel!(gdn_prep_slotted "gdn_prep_slotted", axes = &[BF16]),
];

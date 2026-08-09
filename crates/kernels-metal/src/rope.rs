//! Rotary embeddings.
//!
//! Four spellings of the schedule (`neox`, `freqs`, `prop`, and the strided
//! form), each in a decode and a multi-batch shape. The `freqs` pair reads a
//! host-computed table, which is what llama-3.1's wavelength ramp needs.

use kernels::{KernelSig, kernel};

use crate::axes::*;

pub static KERNELS: &[KernelSig] = &[
    // 1 in rope.metal
    kernel!(neox_decode "neox_decode", file = Some("rope/neox.metal"), launch = kernels::LaunchRule::Rope, axes = &[BF16]),
    // 1 in rope.metal
    kernel!(neox_freqs_decode "neox_freqs_decode", axes = &[BF16]),
    // 1 in rope.metal
    kernel!(neox_freqs_mb "neox_freqs_mb", axes = &[BF16]),
    // 1 in rope.metal
    kernel!(neox_mb "neox_mb", file = Some("rope/neox.metal"), launch = kernels::LaunchRule::Rope, axes = &[BF16]),
    // 1 in rope.metal
    kernel!(neox_prop_decode "neox_prop_decode", axes = &[BF16]),
    // 1 in rope.metal
    kernel!(neox_prop_mb "neox_prop_mb", axes = &[BF16]),
    // 1 in rope.metal
    kernel!(neox_strided "neox_strided", axes = &[BF16]),
];

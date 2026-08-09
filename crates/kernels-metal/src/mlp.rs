//! The dense FFN activations.
//!
//! `gptoss_swiglu` is one of the three kernels in this crate that earn a model
//! name: it bakes gpt-oss's asymmetric clamp, its `alpha` and its `(up + 1)`
//! term, and its own first line says so.

use kernels::{KernelSig, kernel};

use crate::axes::*;

pub static KERNELS: &[KernelSig] = &[
    // 1 in geglu_tanh.metal
    kernel!(geglu_tanh "geglu_tanh", axes = &[BF16]),
    // 1 in geglu_tanh.metal
    kernel!(geglu_tanh_strided "geglu_tanh_strided", axes = &[BF16]),
    // 1 in gptoss.metal
    kernel!(gptoss_swiglu "gptoss_swiglu", axes = &[BF16]),
    // 1 in silu_mul.metal
    kernel!(silu_mul "silu_mul", file = Some("norm/residual_add.metal"), launch = kernels::LaunchRule::Elementwise, axes = &[BF16]),
    // 1 in silu_mul.metal
    kernel!(silu_mul_strided "silu_mul_strided", axes = &[BF16]),
];

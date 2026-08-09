//! RMSNorm and its neighbours: the residual add, the vector norm, the
//! scalar multiply.
//!
//! Every row here is one entrypoint. The dtype axis has a single point and
//! carries no information today; it is declared rather than baked into the
//! symbol so that a second activation dtype is a point, not eleven new names.

use kernels::{KernelSig, kernel};

use crate::axes::*;

pub static KERNELS: &[KernelSig] = &[
    // 1 in gated_rms.metal
    kernel!(gated_rms "gated_rms", axes = &[BF16]),
    // 1 in gated_rms.metal
    kernel!(gated_rms_strided "gated_rms_strided", axes = &[BF16]),
    // 1 in layer_scalar.metal
    kernel!(layer_scalar_mul "layer_scalar_mul", axes = &[BF16]),
    // 1 in residual_add.metal
    kernel!(residual_add "residual_add", axes = &[BF16]),
    // 1 in residual_add.metal
    kernel!(residual_add_strided "residual_add_strided", axes = &[BF16]),
    // 1 in rms_norm.metal
    kernel!(rms_residual "rms_residual", axes = &[BF16]),
    // 1 in rms_norm.metal
    kernel!(rms_residual_scaled "rms_residual_scaled", axes = &[BF16]),
    // 1 in rms_norm.metal
    kernel!(rms_single_row "rms_single_row", file = Some("norm/vector.metal"), launch = kernels::LaunchRule::Rms, axes = &[BF16]),
    // 1 in rms_norm.metal
    kernel!(rms_strided_head_row "rms_strided_head_row", axes = &[BF16]),
    // 1 in rms_norm.metal
    kernel!(rms_strided_row "rms_strided_row", axes = &[BF16]),
    // 1 in vnorm.metal
    kernel!(vnorm_single_row "vnorm_single_row", axes = &[BF16]),
];

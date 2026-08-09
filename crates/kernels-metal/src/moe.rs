//! Routing, and every projection that selects an expert.
//!
//! Filed by what the kernel DOES rather than by the file it sits in:
//! `affine_qmm_t_routed` lives in `quantized_qmm_t.metal` beside its dense
//! twin, but a routed matmul reads an expert slot and is only reachable from
//! a mixture. This is the caller-set rule `.wiki/kernel-refactor.md` §7 uses
//! to settle the same question on the CUDA side.
//!
//! Declaring the axes is what surfaced the one real coverage gap here, and
//! then closed it: `qmv_routed` was compiled for ONE affine format where the
//! dense `qmv_fast` had six, so a Qwen3-MoE or routed gemma-4 at any other
//! format had no pipeline at all. The five missing instantiations are in
//! `quantized_qmv.metal` now, with the evidence for widening rather than
//! refusing. `.wiki/kernel-metal-refactor.md` §9 records it.

use kernels::{KernelSig, kernel};

use crate::axes::*;

pub static KERNELS: &[KernelSig] = &[
    kernel!(combine_sorted "combine_sorted"), // moe_route.metal
    kernel!(route_gather "route_gather"),     // moe_route.metal
    kernel!(route_sort "route_sort"),         // moe_route.metal
    // 9 in quantized_qmm_t.metal
    kernel!(mxfp4_qmm_t_routed_bias "mxfp4_qmm_t_routed_bias", axes = &[BF16, TILE_M, TILE_N]),
    // 1 in quantized_qmv.metal
    kernel!(mxfp4_qmv_routed_bias "mxfp4_qmv_routed_bias", axes = &[BF16, GROUP_32, BITS_4]),
    // 54 in quantized_qmm_t.metal
    kernel!(qmm_t_routed "affine_qmm_t_routed", axes = &[BF16, GROUP, BITS, TILE_M, TILE_N]),
    // 9 in quantized_qmm_t.metal
    kernel!(qmm_t_routed_fp16 "affine_qmm_t_routed_fp16",
        axes = &[BF16, GROUP_64, BITS_4, TILE_M, TILE_N]),
    // 1 in quantized_qmv.metal
    // ONE affine format, and that is the kernel's design rather than a gap:
    // `AffineQ::group_size` is a constant, so a second group point would name
    // an instantiation that dequantises at 64 whatever it claims. A routed
    // checkpoint at another group is meant to fail by name when its pipeline
    // is built -- which `entrypoint()` now does at the call instead of in the
    // Metal compiler.
    kernel!(qmv_routed "affine_qmv_routed", axes = &[BF16, GROUP_64, BITS_4]),
    // 1 in quantized_qmv.metal
    kernel!(qmv_routed_bias "affine_qmv_routed_bias", axes = &[BF16, GROUP_64, BITS_4]),
    // 1 in moe_route.metal
    kernel!(router_topk "router_topk", axes = &[BF16]),
    // 1 in moe_route.metal
    kernel!(router_topk_scaled "router_topk_scaled", axes = &[BF16]),
    kernel!(shared_expert_combine "shared_expert_combine"), // moe_route.metal
    kernel!(shared_expert_combine_strided "shared_expert_combine_strided"), // moe_route.metal
];

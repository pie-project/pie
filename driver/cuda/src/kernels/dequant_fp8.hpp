#pragma once

// FP8 (E4M3FN) → bf16 dequantization with a per-tensor scale.
// Used by Mistral-Small-3.1 / mistral3 — its checkpoint stores
// quantized projection weights as FP8 plus a scalar `weight_scale_inv`
// (the *reciprocal* of the rescaling factor; bf16 = fp8 * scale_inv).
//
// We dequantize once at load time. Memory cost is `out_bytes - in_bytes`
// (≈ 2× the FP8 footprint). For small models that's acceptable; for
// 24B+ checkpoints the right move is fused FP8 GEMM via cuBLAS — see
// the note in the implementation.

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_dequant_fp8_e4m3_to_bf16(
    const std::uint8_t* fp8_in,    // [n] fp8 bits (e4m3fn) — stored as raw bytes
    void*               bf16_out,  // [n] bf16
    float               scale,     // weight_scale_inv (multiplicative)
    std::size_t         n,
    cudaStream_t        stream);

/// Per-channel variant: dequant a `[rows, cols]` row-major fp8 weight to
/// bf16 with one scale per row (weight_scale_inv applied along axis 0).
/// Used by the sm<89 fallback path in `gemm_act_x_w` when the weight
/// has `QuantMeta::PerChannel`.
void launch_dequant_fp8_e4m3_to_bf16_per_channel(
    const std::uint8_t* fp8_in,         // [rows, cols] fp8 bytes
    void*               bf16_out,       // [rows, cols] bf16
    const float*        scale_inv_dev,  // [rows] fp32 device scales
    int                 rows,
    int                 cols,
    cudaStream_t        stream);

}  // namespace pie_cuda_driver::kernels

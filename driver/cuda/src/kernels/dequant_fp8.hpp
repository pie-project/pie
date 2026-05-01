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

}  // namespace pie_cuda_driver::kernels

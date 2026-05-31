#pragma once

// FP8 (E4M3FN) → bf16 dequantization with a per-tensor scale. Launcher
// declaration; the kernel body lives in dequant_fp8.cu, lifted verbatim from
// driver/cuda/src/kernels/dequant_fp8.cu (the base scalar-scale variant). The
// per-channel and per-group variants are lifted later as the load paths that
// need them land.
//
// Used by Mistral-Small-3.1 / mistral3 — its checkpoint stores quantized
// projection weights as FP8 plus a scalar `weight_scale_inv` (the *reciprocal*
// of the rescaling factor; bf16 = fp8 * scale_inv). We dequantize once at load
// time.

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_device::kernels {

void dequant_fp8_e4m3_to_bf16(
    const std::uint8_t* fp8_in,    // [n] fp8 bits (e4m3fn) — stored as raw bytes
    void*               bf16_out,  // [n] bf16
    float               scale,     // weight_scale_inv (multiplicative)
    std::size_t         n,
    cudaStream_t        stream);

}  // namespace pie_cuda_device::kernels

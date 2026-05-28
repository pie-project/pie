#pragma once

// Runtime BF16 → MXFP4 (E2M1 + E8M0 block scales) quantization, used by the
// runtime_quant=fp4 path for GLM-5.1 routed experts. Output layout matches the
// MXFP4 contract consumed by `launch_dequant_mxfp4_to_bf16`:
//
//   * `W_bf16`      — input  `[rows, cols]` bf16 (cols % 32 == 0).
//   * `W_packed`    — output `[rows, cols/2]` uint8 (low nibble = element 2k,
//                     high nibble = element 2k+1).
//   * `W_scale_e8m0`— output `[rows, cols/32]` uint8. Each byte encodes
//                     2^(byte - 127) for the corresponding 32-element block.
//
// FP4 codepoints (E2M1, signed) – standard OCP MX spec:
//   {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}.
//
// The kernel processes one row per block, finds per-32-element absmax,
// rounds the block scale to the nearest E8M0 exponent, then maps each
// element to the closest FP4 codepoint.

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void quantize_bf16_to_mxfp4_e2m1_per_block(
    const void*    W_bf16,        // [rows, cols] bf16
    std::uint8_t*  W_packed,      // [rows, cols/2]
    std::uint8_t*  W_scale_e8m0,  // [rows, cols/32]
    int            rows,
    int            cols,          // must be a multiple of 32
    cudaStream_t   stream);

}  // namespace pie_cuda_driver::kernels

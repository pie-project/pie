#pragma once

// MXFP4 (E2M1 + E8M0 block scales) → bf16 dequantization. Used by
// GPT-OSS — its expert weights ship as packed uint8 (two 4-bit values
// per byte) plus per-32-element fp8-E8M0 block scales.
//
// Layout:
//   * `packed`      — uint8 [out_dim, in_dim/2]; low nibble = element 2k,
//                     high nibble = element 2k+1.
//   * `block_scale` — uint8 [out_dim, in_dim/32]; single E8M0 byte per
//                     32-element row block. E8M0 is a "bias-127 power-of-2",
//                     i.e. `scale = 2^(byte - 127)`.
//   * `out`         — bf16 [out_dim, in_dim].
//
// FP4 codepoints (E2M1, signed): 8 positive + 8 negative values, stored
// in a 16-entry LUT inside the kernel. Standard OCP MX spec values:
//   {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_dequant_mxfp4_to_bf16(
    const std::uint8_t* packed,        // [out_dim, in_dim/2]
    const std::uint8_t* block_scale,   // [out_dim, in_dim/32]
    void*               out,           // [out_dim, in_dim] bf16
    int                 out_dim,
    int                 in_dim,        // must be a multiple of 32
    cudaStream_t        stream);

}  // namespace pie_cuda_driver::kernels

#pragma once

// `split_gate_up_bf16`, split out of `attn/split_packed.hpp`.
//
// Its table row was always `layout`'s while its C++ sat among that header's
// launchers, so symbol and row disagreed -- and that disagreement was the
// marker for this split. It is self-contained: the kernel reaches for nothing
// else in the file it left. The MoE GEMVs in `quant/dequant_fp4.cu` and
// `quant/dequant_wna16.cu` look like the same case and are NOT -- they share
// that file's unpack helpers, so their rows stay `moe` with their C++ in
// `quant`, and that is correct rather than pending.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels::layout {

// `packed` is row-major [N, 2*inter]; outputs are row-major [N, inter].
void split_gate_up_bf16(
    const void* packed,
    void* gate_out, void* up_out,
    int n_tokens, int inter,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels::layout

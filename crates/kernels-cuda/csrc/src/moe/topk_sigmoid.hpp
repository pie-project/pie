#pragma once

// `topk_sigmoid_bf16`, split out of `attn/kimi_mla.hpp`.
//
// Its table row was always `moe`'s while its C++ sat among that header's
// launchers, so symbol and row disagreed -- and that disagreement was the
// marker for this split. It is self-contained: the kernel reaches for nothing
// else in the file it left. The MoE GEMVs in `quant/dequant_fp4.cu` and
// `quant/dequant_wna16.cu` look like the same case and are NOT -- they share
// that file's unpack helpers, so their rows stay `moe` with their C++ in
// `quant`, and that is correct rather than pending.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels::moe {

void topk_sigmoid_bf16(
    const void* logits,
    std::int32_t* topk_idx,
    float* topk_w,
    const float* correction_bias,
    int tokens,
    int num_experts,
    int top_k,
    bool renormalize,
    float routed_scaling_factor,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels::moe

#pragma once

// Attention-sink post-pass for GPT-OSS. Each layer learns a per-head sink
// scalar `sinks[h]`; the softmax denominator is extended by `exp(sinks[h])`
// (equivalent to a "virtual" KV slot with logit = sinks[h] and value = 0).
//
// Mathematically: with `lse = log Σ_kv exp(z_kv)` (returned by flashinfer
// when `params.lse != nullptr`), the corrected output is
//
//     o_sink[t, h, :] = o[t, h, :] · σ(lse[t, h] − sinks[h])
//
// This kernel applies the rescale in-place on `o` so subsequent ops
// (o_proj GEMM, residual add) see the corrected activations.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

// In-place: o[t, h, d] *= sigmoid(lse[t, h] - sinks[h])
//
// Layouts:
//   `o`     bf16 [N, H_q * D]    — packed (h_q × d contiguous per token)
//   `lse`   fp32 [N, H_q]
//   `sinks` bf16 [H_q]
//
// One block per (token, head); threads stride along the head dim.
void launch_attention_sink_rescale_bf16(
    void*        o,
    const float* lse,
    const void*  sinks,
    int N,
    int num_q_heads,
    int head_dim,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels

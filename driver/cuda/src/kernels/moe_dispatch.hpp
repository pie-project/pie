#pragma once

// Per-expert scatter helper for sparse-MoE forward.
//
// `scatter_add_expert`: Multiply the expert's `[Ne, hidden]` output by
//                       per-row routing weights and accumulate into the
//                       full `[N, hidden]` MoE output buffer.
//
// Routing decisions live on host (built from a D2H copy of `topk_idx`)
// since they are cheap O(N*K) bookkeeping. The corresponding "gather
// expert input rows" step reuses the existing `launch_gather_bf16_rows`.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

// `out[dst_idx[i]] += src[i] * row_weights[i]` for i ∈ [0, num_routed).
// Per-row RMW; safe because the per-expert calls are sequential — there
// are no concurrent writers to a given row across expert iterations.
void launch_scatter_add_weighted_bf16(
    void* out,                          // [N, hidden] bf16, in-place
    const void* src,                    // [num_routed, hidden] bf16
    const std::int32_t* dst_idx,        // [num_routed] i32
    const float* row_weights,           // [num_routed] fp32
    int num_routed,
    int hidden,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels

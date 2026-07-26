#pragma once

// Device-side math shared between the standalone Quest kernels
// (`kernels/envelope.cu`, bit-parity tested against
// `pie_sampling_ir::eval::envelope_dot_reference`) and the fused PTIR runtime's
// `envelope_dot` second-party region (`pipeline/generated/fused_runtime.cuh`).
//
// The two callers differ only in how they ADDRESS pages: the standalone kernel
// walks one request's dense envelope block, while the fused runtime resolves a
// physical page id out of the fire's KV page table first. The arithmetic they
// apply once a page is resolved must be identical, or the golden test stops
// covering the code that actually runs — which is precisely the failure mode
// this repo already paid for once with `top_k`, where the optimized kernel was
// not the dispatched one. So the arithmetic lives here and neither caller
// re-derives it.
//
// This header is CUDA-only. `kernels/envelope.hpp` is included by host
// translation units and must stay free of `__device__` code.

#include <cuda_runtime.h>

namespace pie_cuda_driver {
namespace kernels {

// One thread's partial of the Quest criticality upper bound for a single
// (page, kv_head):
//
//     Σ_{qh ∈ GQA group of kv_head} Σ_d max(q[qh,d]·kmin[d], q[qh,d]·kmax[d])
//
// `env_base` is the offset of this (page, kv_head)'s `head_dim` run inside the
// envelope tensors, i.e. `(page * num_kv_heads + kv_head) * head_dim`. Callers
// sum the returned partials across the block.
__device__ __forceinline__ float envelope_dot_thread_partial(
    const float* __restrict__ q,
    const float* __restrict__ env_min,
    const float* __restrict__ env_max,
    long env_base,
    int qh_base,
    int group,
    int head_dim,
    int thread,
    int nthreads) {
    float local = 0.f;
    const int terms = group * head_dim;
    for (int i = thread; i < terms; i += nthreads) {
        const int g = i / head_dim;
        const int d = i - g * head_dim;
        const float qd = q[static_cast<long>(qh_base + g) * head_dim + d];
        const float lo = qd * env_min[env_base + d];
        const float hi = qd * env_max[env_base + d];
        local += (lo > hi) ? lo : hi;
    }
    return local;
}

}  // namespace kernels
}  // namespace pie_cuda_driver

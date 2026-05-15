#pragma once

// Split a fused matmul output into separately-packed buffers.
//
// The fused QKV / gate-up matmuls write a row-major `[N, A + B (+ C)]`
// tensor where columns [0,A) are the first output, [A,A+B) the second,
// etc. Downstream kernels (rope, kv_paged, swiglu, …) want each output
// in its own packed `[N, A]` / `[N, B]` buffer so they can use the
// existing addressing.
//
// One pass over packed memory; pure copy, no compute.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

// `packed` is row-major [N, q_dim + 2*kv_dim]; outputs are row-major
// [N, q_dim] / [N, kv_dim] / [N, kv_dim]. Buffers must not overlap with
// `packed`.
void launch_split_qkv_bf16(
    const void* packed,
    void* q_out, void* k_out, void* v_out,
    int n_tokens, int q_dim, int kv_dim,
    cudaStream_t stream);

// `packed` is row-major [N, 2*inter]; outputs are row-major [N, inter].
void launch_split_gate_up_bf16(
    const void* packed,
    void* gate_out, void* up_out,
    int n_tokens, int inter,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels

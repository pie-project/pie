#pragma once

// Split a fused matmul output into separately-packed buffers.
//
// The fused QKV matmul writes a row-major `[N, q_dim + 2*kv_dim]` tensor
// where columns [0,q_dim) are Q, the next kv_dim are K, the last kv_dim
// are V. Downstream kernels (rope, kv_paged, …) want each output in its
// own packed `[N, dim]` buffer so they can use the existing addressing.
//
// One pass over packed memory; pure copy, no compute. That sentence is
// now true of everything in this file — the three kernels that normalise,
// rotate and write the paged cache moved to `attn/qkv_fused.hpp`, which
// is the fused alternative to (this split → rope → kv_paged).
//
// The gate-up form of the same split is `layout/split_gate_up.hpp`; it
// left for the family its table row already named.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels::attn {

// `packed` is row-major [N, q_dim + 2*kv_dim]; outputs are row-major
// [N, q_dim] / [N, kv_dim] / [N, kv_dim]. Buffers must not overlap with
// `packed`.
// Peel device-window variant: {start, len} in device memory, full-grid
// launch with early-out, base pointers.
void split_qkv_bf16_devwin(
    const void* packed,
    void* q_out, void* k_out, void* v_out,
    const std::uint32_t* win_d,
    int n_max, int q_dim, int kv_dim,
    cudaStream_t stream);

void split_qkv_bf16(
    const void* packed,
    void* q_out, void* k_out, void* v_out,
    int n_tokens, int q_dim, int kv_dim,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels::attn

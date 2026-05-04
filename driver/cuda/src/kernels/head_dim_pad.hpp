#pragma once

// Per-head zero-pad / strip on the last axis. Used to reach a
// flashinfer-supported HEAD_DIM (∈ {64, 128, 256, 512}) when the model
// ships with an unaligned head_dim — currently only Phi-3-mini at 96.
//
// Layout convention (row-major, contiguous):
//
//     packed [num_tokens, num_heads, head_dim]
//     padded [num_tokens, num_heads, head_dim_padded]
//
// `pad`: copies `head_dim` values per (token, head) block from `packed`
//        into `padded`, then writes zeros into the trailing
//        `head_dim_padded - head_dim` columns. Required for any axis
//        the attention kernel will read as part of `Q` / `K` / `V` —
//        the zero pad keeps `qe·ke = q[:d]·k[:d]` (so the dot product
//        is unchanged) and doesn't contribute to the V-weighted sum.
//
// `strip`: inverse — copies `head_dim` values back, dropping the
//          padding columns. Used after attention to feed a packed
//          `[N, num_heads * head_dim]` buffer into the o_proj GEMM.

#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_pad_head_dim_bf16(
    const void* packed,        // [N, num_heads, head_dim]
    void*       padded,        // [N, num_heads, head_dim_padded]
    int         num_tokens,
    int         num_heads,
    int         head_dim,
    int         head_dim_padded,
    cudaStream_t stream);

void launch_strip_head_dim_bf16(
    const void* padded,        // [N, num_heads, head_dim_padded]
    void*       packed,        // [N, num_heads, head_dim]
    int         num_tokens,
    int         num_heads,
    int         head_dim,
    int         head_dim_padded,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels

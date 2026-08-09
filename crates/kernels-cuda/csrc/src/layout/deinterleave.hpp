#pragma once

// Helpers used by GPT-OSS bind to split fused gate/up weights and
// biases. HF stores the GPT-OSS expert MLP as a single
//
//     gate_up_proj : [E, hidden, 2 * intermediate]
//
// where for each expert `gate = gate_up[..., ::2]` and `up =
// gate_up[..., 1::2]` — interleaved along the last axis. After we
// dequantize the MXFP4 storage (which is laid out as `[E, 2*I, H]`
// post-flatten), each expert's bf16 fused weight has shape `[2*I, H]`
// where ROW j corresponds to output column j of HF's view. Even rows
// are gate features, odd rows are up features.
//
// The two helpers below copy out the gate/up halves into separately
// owned bf16 tensors so the rest of the runtime can treat them like
// normal Mixtral expert weights (`x @ w_gate^T → [N, I]`, same for
// up) without paying for runtime stride-2 indexing.

#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels::layout {

// Split a fused [2*I, H] bf16 matrix into two [I, H] outputs by row
// parity. `gate_out[i, :] = fused[2*i, :]`, `up_out[i, :] =
// fused[2*i + 1, :]`.
void deinterleave_rows_bf16(
    const void* fused,    // bf16 [2*I, H]
    void*       gate_out, // bf16 [I, H]
    void*       up_out,   // bf16 [I, H]
    int         I,
    int         H,
    cudaStream_t stream);

// Split a fused [2*I] bf16 vector into two [I] outputs by parity.
// Used for the per-expert gate_up_proj_bias.
void deinterleave_vec_bf16(
    const void* fused,    // bf16 [2*I]
    void*       gate_out, // bf16 [I]
    void*       up_out,   // bf16 [I]
    int         I,
    cudaStream_t stream);

// Qwen3.5 full-attention q_proj produces a fused `[N, num_heads,
// 2*head_dim]` tensor where each head's first `head_dim` channels are
// the query and the next `head_dim` are the per-token output gate.
// Split into separate `[N, Hq]` query/gate tensors (Hq = num_heads *
// head_dim).
//
//     q[n, h*d + i]    = packed[n, h*2*d + i]
//     gate[n, h*d + i] = packed[n, h*2*d + d + i]
void split_q_gate_bf16(
    const void* packed,    // bf16 [N, num_heads * 2 * head_dim]
    void*       q_out,     // bf16 [N, num_heads * head_dim]
    void*       gate_out,  // bf16 [N, num_heads * head_dim]
    int N, int num_heads, int head_dim,
    cudaStream_t stream);

// Concatenate two bf16 row-major matrices with the same row count:
// `out[n] = [left[n], right[n]]`.
void concat_bf16_rows(
    const void* left,   // bf16 [N, left_dim]
    const void* right,  // bf16 [N, right_dim]
    void*       out,    // bf16 [N, left_dim + right_dim]
    int N, int left_dim, int right_dim,
    cudaStream_t stream);

// Inverse of `concat_bf16_rows`:
//   src: [N, left_dim + right_dim] -> left [N, left_dim], right [N, right_dim]

// Split just Qwen GDN's tiny fused b/a projection:
//   ba: [N, 2 * v_h] -> b [N, v_h], a [N, v_h]

// Repeat-interleave KV heads to query heads for the one-token attention
// case used by MTP. `out[n, qh] = in[n, qh / repeat]`.

}  // namespace pie_cuda_driver::kernels::layout

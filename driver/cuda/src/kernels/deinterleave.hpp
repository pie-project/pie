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

namespace pie_cuda_driver::kernels {

// Split a fused [2*I, H] bf16 matrix into two [I, H] outputs by row
// parity. `gate_out[i, :] = fused[2*i, :]`, `up_out[i, :] =
// fused[2*i + 1, :]`.
void launch_deinterleave_rows_bf16(
    const void* fused,    // bf16 [2*I, H]
    void*       gate_out, // bf16 [I, H]
    void*       up_out,   // bf16 [I, H]
    int         I,
    int         H,
    cudaStream_t stream);

// Split a fused [2*I] bf16 vector into two [I] outputs by parity.
// Used for the per-expert gate_up_proj_bias.
void launch_deinterleave_vec_bf16(
    const void* fused,    // bf16 [2*I]
    void*       gate_out, // bf16 [I]
    void*       up_out,   // bf16 [I]
    int         I,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels

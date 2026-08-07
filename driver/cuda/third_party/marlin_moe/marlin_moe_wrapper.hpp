#pragma once

// pie's torch-free entry point to the expert-indexed Marlin MoE GEMM.
//
// The difference from `../marlin/marlin_wrapper.hpp` is the whole point: that
// one solves a single (M, N, K) problem, so a MoE layer has to call it once per
// expert and pays a launch plus a routing round trip for each. This one takes
// the block-sorted routing metadata and covers every expert in ONE launch,
// which is the design both vLLM (Marlin, MXFP4) and SGLang (Triton, bf16) use
// on A100.
//
// The metadata is the `moe_align_block_size` contract:
//   * `sorted_token_ids[i]`  route ids grouped by expert and padded so each
//                            expert occupies a whole number of `block_size`
//                            blocks. Padding entries hold a sentinel
//                            (>= num_routes) that the kernel skips.
//   * `expert_ids[b]`        the expert owning M-block `b`.
//   * `num_tokens_past_padded[0]` total padded row count; the kernel iterates
//                            blocks up to `ceil(this / block_size)`.
// `launch_moe_align_decode` in `kernels/moe_dispatch.hpp` already produces
// exactly this.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace pie_cuda_driver::marlin_moe {

/// MXFP4 (FE2M1 weights, E8M0 group scales) x bf16 activations -> bf16, with
/// per-M-block expert selection.
///
/// `w_mxfp4_packed` and `scales_e8m0` are the whole expert stack: expert `e`
/// starts at `e * prob_n * prob_k / 8` bytes into the weights (the kernel
/// derives this itself from `prob_n`/`prob_k`), so the caller must pass a
/// tightly packed `[num_experts, prob_n, prob_k]` slab.
///
/// `out_bf16` is indexed by ROUTE (`token * top_k + k`), matching what the
/// per-route path this replaces produced, so the downstream weighted sum is
/// unchanged. Pass `topk_weights` with `mul_topk_weights=true` to fold the
/// routing weight into the epilogue instead.
void launch_mxfp4_moe_gemm_w4a16_bf16(
    const void*          act_bf16,          // [num_tokens, K] bf16
    const void*          w_mxfp4_packed,    // [E, N, K] marlin-repacked FE2M1
    const void*          scales_e8m0,       // [E, K/32, N] E8M0 bytes
    const void*          bias_bf16,         // [E, N] bf16, or nullptr
    void*                out_bf16,          // [num_routes, N] bf16
    void*                reduce_scratch,    // fp32 split-K scratch, or nullptr
    void*                workspace,         // device scratch (see below)
    const std::int32_t*  sorted_token_ids,
    const std::int32_t*  expert_ids,
    const std::int32_t*  num_tokens_past_padded,
    const float*         topk_weights,      // or nullptr
    int                  block_size,        // moe_align block size
    int                  num_experts,
    int                  top_k,
    bool                 mul_topk_weights,
    int                  prob_m,            // total routes (num_tokens*top_k)
    int                  prob_n,
    int                  prob_k,
    cudaStream_t         stream);

/// Device workspace bytes the MoE launcher needs. Conservative upper bound;
/// allocate once at engine init like the single-problem path does.
std::size_t marlin_moe_workspace_bytes(int prob_n, int block_size);

}  // namespace pie_cuda_driver::marlin_moe

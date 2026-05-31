#pragma once

// MoE token-dispatch core: the routing scatter/gather around the per-expert
// GEMMs. Lifted VERBATIM from driver/cuda/src/kernels/moe_dispatch.{hpp,cu},
// changing only the namespace (pie_cuda_driver -> pie_cuda_device) and
// dropping the `launch_` prefix.
//
// This covers two halves of sparse-MoE dispatch:
//   (a) PERMUTE/SCATTER the N tokens into expert-grouped order given topk_idx
//       (`moe_align_decode`, `moe_bucket_exact`, `gather_moe_aligned_inputs_bf16`)
//   (b) GATHER/COMBINE expert outputs back, weighted by topk_w
//       (`scatter_add_weighted_bf16`, `*_weighted_sum_*`,
//        `reorder_moe_aligned_output_bf16`)
//
// DROPPED (require cuBLAS grouped/batched-GEMM plumbing or are the GEMM
// itself, not the dispatch core): build_moe_ptrs_decode_bf16,
// build_moe_ptrs_decode_batched_bf16, build_dual_bf16_gemm_ptrs,
// build_moe_ptrs_aligned_bf16 (cuBLAS A/B/C pointer-array builders), and
// moe_gate_up_decode_wmma_bf16 / moe_down_decode_wmma_bf16 (the per-expert
// WMMA GEMM, not scatter/gather).

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_device::kernels {

// `out[dst_idx[i]] += src[i] * row_weights[i]` for i ∈ [0, num_routed).
// Per-row RMW; safe because the per-expert calls are sequential — there
// are no concurrent writers to a given row across expert iterations.
void scatter_add_weighted_bf16(
    void* out,                          // [N, hidden] bf16, in-place
    const void* src,                    // [num_routed, hidden] bf16
    const std::int32_t* dst_idx,        // [num_routed] i32
    const float* row_weights,           // [num_routed] fp32
    int num_routed,
    int hidden,
    cudaStream_t stream);

// Scalar-weighted in-place add: `out[i] += weight * src[i]` for
// i ∈ [0, n). Decode-time MoE fast-path: when only one token is being
// processed, every per-expert contribution writes to the same single
// destination row, so the indexed scatter degenerates to a flat
// fma-on-row. Saves an expert_idx D2H copy + the gather kernel.
void scalar_weighted_add_bf16(
    void*       out,    // bf16, in-place
    const void* src,    // bf16
    float       weight,
    int n,
    cudaStream_t stream);

// Batched version of the above for the N=1 MoE decode path: writes
// `out[h] = Σ_k weights[k] * src[k * stride_h + h]` for h ∈ [0, hidden).
// One block per `hidden`-tile, threads cooperate on the K-way sum.
//
//     src     : [batch, hidden] bf16  (batch outputs of the per-expert down_proj)
//     weights : [batch] fp32          (top-K routing weights)
//     out     : [hidden] bf16          (zeroed by caller)
//
// `batch` is small (=top_k=8 on Qwen3.6-MoE), `hidden` is the model's
// hidden_size. Single launch replaces top_k separate scalar-weighted
// adds.
void batched_weighted_sum_bf16(
    void*       out,
    const void* src,
    const float* weights,
    int batch,
    int hidden,
    cudaStream_t stream);

// Batched decode version: writes
// `out[n, h] = sum_k weights[n, k] * src[n, k, h]`.
//
//     src     : [num_tokens * top_k, hidden] bf16
//     weights : [num_tokens * top_k] fp32
//     out     : [num_tokens, hidden] bf16
void token_batched_weighted_sum_bf16(
    void*       out,
    const void* src,
    const float* weights,
    int num_tokens,
    int top_k,
    int hidden,
    cudaStream_t stream);

void token_batched_weighted_sum_add_bf16(
    void*       out,
    const void* src,
    const float* weights,
    int num_tokens,
    int top_k,
    int hidden,
    cudaStream_t stream);

// Fused combine for aligned MoE output. `route_to_aligned_row[route]`
// maps the original route id (`token * top_k + k`) to its row in
// `aligned_out`; accumulation still proceeds in top-k order for each token.
void token_batched_weighted_sum_aligned_bf16(
    void* out,
    const void* aligned_out,
    const float* weights,
    const std::int32_t* route_to_aligned_row,
    int num_tokens,
    int top_k,
    int hidden,
    cudaStream_t stream);

// vLLM/SGL-style decode alignment. Sorts route ids [0, num_routes) by expert
// into fixed-size blocks; padded entries are filled with sentinel num_routes.
// `expert_ids[b]` is the expert for block b or -1 for inactive padding blocks.
void moe_align_decode(
    const std::int32_t* topk_idx,
    std::int32_t* sorted_route_ids,
    std::int32_t* expert_ids,
    // Optional output: maps original route id (`token * top_k + k`) to the
    // padded sorted-row position. Pass nullptr if the caller does not need
    // this inverse map. Used by the fused `_aligned_bf16` combine kernel.
    std::int32_t* route_to_aligned_row,
    int num_routes,
    int num_experts,
    int block_size,
    int max_blocks,
    cudaStream_t stream);

// `moe_bucket_exact` is the same expert-bucketing setup as
// `moe_align_decode`, this does not pad to fixed-size expert blocks.
// It writes sorted route ids, the inverse route->sorted-row map, and exact
// per-expert counts. The host may copy only `counts_out[num_experts]` to build
// cuBLAS grouped shapes while route metadata stays on device.
void moe_bucket_exact(
    const std::int32_t* topk_idx,
    std::int32_t* sorted_route_ids,
    std::int32_t* route_to_sorted_row,
    std::int32_t* counts_out,
    int num_routes,
    int num_experts,
    cudaStream_t stream);

// Gather `norm_x[route / top_k]` into aligned rows. Sentinel route ids become
// zero rows so padded expert blocks are harmless GEMM work.
void gather_moe_aligned_inputs_bf16(
    const void* norm_x,
    const std::int32_t* sorted_route_ids,
    void* aligned_in,
    int num_routes,
    int aligned_rows,
    int top_k,
    int hidden,
    cudaStream_t stream);

// Undo the expert-block permutation after down_proj: copies aligned rows back
// to route order [num_tokens * top_k, hidden]. Sentinel rows are ignored.
void reorder_moe_aligned_output_bf16(
    const void* aligned_out,
    const std::int32_t* sorted_route_ids,
    void* route_out,
    int num_routes,
    int aligned_rows,
    int hidden,
    cudaStream_t stream);

}  // namespace pie_cuda_device::kernels

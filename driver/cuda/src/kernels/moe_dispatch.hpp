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

#include <cstdlib>
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

// Scalar-weighted in-place add: `out[i] += weight * src[i]` for
// i ∈ [0, n). Decode-time MoE fast-path: when only one token is being
// processed, every per-expert contribution writes to the same single
// destination row, so the indexed scatter degenerates to a flat
// fma-on-row. Saves an expert_idx D2H copy + the gather kernel.
void launch_scalar_weighted_add_bf16(
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
void launch_batched_weighted_sum_bf16(
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
void launch_token_batched_weighted_sum_bf16(
    void*       out,
    const void* src,
    const float* weights,
    int num_tokens,
    int top_k,
    int hidden,
    cudaStream_t stream);

void launch_token_batched_weighted_sum_add_bf16(
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
void launch_token_batched_weighted_sum_aligned_bf16(
    void* out,
    const void* aligned_out,
    const float* weights,
    const std::int32_t* route_to_aligned_row,
    int num_tokens,
    int top_k,
    int hidden,
    cudaStream_t stream);

// On-device construction of the per-expert cuBLAS pointer arrays for the
// N=1 MoE decode path. Replaces the host-side build_routing + 6
// cudaMemcpyAsync's that the original implementation used; produces the
// same arrays that `cublasGemmBatchedEx` would have read after the H2D
// stage, but does so with a single in-stream kernel — no D2H sync, no
// host work in the captured region, so the graph capture path can fire.
//
//     topk_idx        : [top_k] i32
//     topk_w          : [top_k] fp32
//     gate_up_base    : [E, 2*I_moe, H] bf16  fused weight tensor
//     down_base       : [E, H, I_moe]   bf16
//     norm_x          : [H]              bf16  (single token)
//     expert_gate_up  : [top_k, 2*I_moe] bf16  scratch — slots assigned
//     expert_act      : [top_k, I_moe]   bf16  scratch
//     expert_out      : [top_k, H]       bf16  scratch
//     a_gu / b_gu / c_gu : device arrays of `top_k` pointers each (cuBLAS A/B/C for gate_up)
//     a_dn / b_dn / c_dn : same for down_proj
//     weights_out     : [top_k] fp32 — `topk_w` copied through
//     stride_gu       : 2 * I_moe * H (bf16 elements per expert in gate_up_proj)
//     stride_dn       : H * I_moe       (bf16 elements per expert in down_proj)
void launch_build_moe_ptrs_decode_bf16(
    const std::int32_t* topk_idx,
    const float*        topk_w,
    const void*         gate_up_base,
    const void*         down_base,
    const void*         norm_x,
    void*               expert_gate_up,
    void*               expert_act,
    void*               expert_out,
    const void**        a_gu_ptrs,
    const void**        b_gu_ptrs,
    void**              c_gu_ptrs,
    const void**        a_dn_ptrs,
    const void**        b_dn_ptrs,
    void**              c_dn_ptrs,
    float*              weights_out,
    int top_k,
    int H, int I_moe,
    cudaStream_t stream);

// Multi-token decode equivalent of `launch_build_moe_ptrs_decode_bf16`.
// Produces `num_tokens * top_k` pointer triples, treating each routed
// token/expert pair as one M=1 batched-GEMM item.
void launch_build_moe_ptrs_decode_batched_bf16(
    const std::int32_t* topk_idx,
    const float*        topk_w,
    const void*         gate_up_base,
    const void*         down_base,
    const void*         norm_x,
    void*               expert_gate_up,
    void*               expert_act,
    void*               expert_out,
    const void**        a_gu_ptrs,
    const void**        b_gu_ptrs,
    void**              c_gu_ptrs,
    const void**        a_dn_ptrs,
    const void**        b_dn_ptrs,
    void**              c_dn_ptrs,
    float*              weights_out,
    int num_tokens,
    int top_k,
    int H, int I_moe,
    cudaStream_t stream);

// Bandwidth-optimal decode GEMVs for the sparse MoE hot path. At M=1 the
// routed GEMMs have no weight reuse, so they are pure streaming reads;
// one warp per output row beats both the tensor-core kernels below and
// `cublasGemmBatchedEx`, whose tiling assumes an M worth filling.
void launch_moe_gate_up_decode_gemv_bf16(
    const std::int32_t* topk_idx,
    const void* norm_x,
    const void* gate_up_base,
    void* expert_gate_up,
    int num_tokens,
    int top_k,
    int H,
    int I_moe,
    cudaStream_t stream);

void launch_moe_down_decode_gemv_bf16(
    const std::int32_t* topk_idx,
    const void* expert_act,
    const void* down_base,
    void* expert_out,
    int num_tokens,
    int top_k,
    int H,
    int I_moe,
    cudaStream_t stream);

// Tensor-core decode kernels for the sparse MoE hot path. Each routed
// token/expert pair is treated as a 1-row GEMM, but computed with BF16 WMMA
// tiles to avoid the overhead of many tiny cuBLAS batched GEMMs.
void launch_moe_gate_up_decode_wmma_bf16(
    const std::int32_t* topk_idx,
    const void* norm_x,
    const void* gate_up_base,
    void* expert_gate_up,
    int num_tokens,
    int top_k,
    int H,
    int I_moe,
    cudaStream_t stream);

void launch_moe_down_decode_wmma_bf16(
    const std::int32_t* topk_idx,
    const void* expert_act,
    const void* down_base,
    void* expert_out,
    int num_tokens,
    int top_k,
    int H,
    int I_moe,
    cudaStream_t stream);

// Builds a two-entry pointer table for running two same-shaped BF16 GEMMs as
// one cublasGemmBatchedEx call:
//   out0 = act @ w0^T
//   out1 = act @ w1^T
void launch_build_dual_bf16_gemm_ptrs(
    const void* act,
    const void* w0,
    const void* w1,
    void* out0,
    void* out1,
    const void** act_ptrs,
    const void** w_ptrs,
    void** out_ptrs,
    cudaStream_t stream);

// vLLM/SGL-style decode alignment. Sorts route ids [0, num_routes) by expert
// into fixed-size blocks; padded entries are filled with sentinel num_routes.
// `expert_ids[b]` is the expert for block b or -1 for inactive padding blocks.

// Block size for the aligned MoE path above. Every expert's routes are padded
// up to a multiple of this, so the useful value tracks how many rows an expert
// actually receives (routes / experts). A full 384-expert checkpoint gets ~3
// rows per expert at batch 128 and needs a small block or it pads 3 rows up to
// 64; a reduced expert bank gets ~128 and wants fat blocks so the batched GEMM
// has a usable M dimension. Measured on kimi26-mini at batch 128, moe_prefill:
// 16 -> 1.184 ms, 32 -> 0.811, 64 -> 0.746, 128 -> 0.796 -- it turns back up at
// 128 because eight blocks no longer fill the GPU, hence the cap.
//
// Callers pick this per forward from that batch's route count, so scratch must
// be sized for both extremes: kMoeAlignedBlockMin yields the most blocks, the
// value returned here for the largest batch yields the most padded rows.
inline constexpr int kMoeAlignedBlockMin = 16;
inline constexpr int kMoeAlignedBlockMax = 64;

inline int moe_aligned_block(int routes, int num_experts) {
    static const int forced = [] {
        return 0;
        const int parsed = 0;
        return (parsed >= 8 && parsed <= 256 && (parsed & (parsed - 1)) == 0)
                   ? parsed
                   : 0;
    }();
    if (forced != 0) return forced;
    if (num_experts <= 0) return kMoeAlignedBlockMin;
    const int per_expert = routes / num_experts;
    int block = kMoeAlignedBlockMin;
    while (block * 2 <= kMoeAlignedBlockMax && block * 2 <= per_expert) {
        block *= 2;
    }
    return block;
}

void launch_moe_align_decode(
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

// `launch_moe_bucket_exact` is the same expert-bucketing setup as
// `launch_moe_align_decode`, this does not pad to fixed-size expert blocks.
// It writes sorted route ids, the inverse route->sorted-row map, and exact
// per-expert counts. The host may copy only `counts_out[num_experts]` to build
// cuBLAS grouped shapes while route metadata stays on device.
void launch_moe_bucket_exact(
    const std::int32_t* topk_idx,
    std::int32_t* sorted_route_ids,
    std::int32_t* route_to_sorted_row,
    std::int32_t* counts_out,
    int num_routes,
    int num_experts,
    cudaStream_t stream);

// Gather `norm_x[route / top_k]` into aligned rows. Sentinel route ids become
// zero rows so padded expert blocks are harmless GEMM work.
//
// Rows at or after `shared_row_begin` are the shared expert's region: it is
// appended to the routed blocks as one more expert so its two projections
// ride inside the routed batched GEMM instead of running as their own
// 4-thread-block GEMMs. Those rows map straight to token `row -
// shared_row_begin` (every token visits the shared expert exactly once) and
// so are not described by `sorted_route_ids` at all. Pass
// `shared_row_begin < 0` to disable.
void launch_gather_moe_aligned_inputs_bf16(
    const void* norm_x,
    const std::int32_t* sorted_route_ids,
    void* aligned_in,
    int num_routes,
    int aligned_rows,
    int top_k,
    int hidden,
    int shared_row_begin,
    int num_tokens,
    cudaStream_t stream);

void launch_build_moe_ptrs_aligned_bf16(
    const std::int32_t* expert_ids,
    const void* gate_up_base,
    const void* down_base,
    const void* aligned_in,
    void* aligned_gate_up,
    void* aligned_act,
    void* aligned_out,
    const void** a_gu_ptrs,
    const void** b_gu_ptrs,
    void** c_gu_ptrs,
    const void** a_dn_ptrs,
    const void** b_dn_ptrs,
    void** c_dn_ptrs,
    int max_blocks,
    int block_size,
    int H,
    int I_moe,
    // Blocks at or after `routed_blocks` take the shared expert's weights
    // instead of indexing the routed stack. Pass `routed_blocks == max_blocks`
    // (or null bases) to disable.
    int routed_blocks,
    const void* shared_gate_up_base,
    const void* shared_down_base,
    cudaStream_t stream);

// Undo the expert-block permutation after down_proj: copies aligned rows back
// to route order [num_tokens * top_k, hidden]. Sentinel rows are ignored.
// Rows at or after `shared_row_begin` are the folded shared expert and go to
// `shared_out[token]` instead, still unscaled — the caller applies the
// sigmoid scalar gate exactly as it did for the standalone path.
void launch_reorder_moe_aligned_output_bf16(
    const void* aligned_out,
    const std::int32_t* sorted_route_ids,
    void* route_out,
    int num_routes,
    int aligned_rows,
    int hidden,
    int shared_row_begin,
    int num_tokens,
    void* shared_out,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels

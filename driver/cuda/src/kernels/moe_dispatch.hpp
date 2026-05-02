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

}  // namespace pie_cuda_driver::kernels

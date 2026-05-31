#pragma once

// Sparse (token-dispatched) MoE MLP sublayer (bf16) — the perf-real routed path
// that replaces the dense per-expert loop of `moe_mlp_block_bf16` (moe_mlp.cuh)
// with dispatch-scatter -> grouped GEMM -> combine. SAME signature and SAME
// numerical semantics as the dense block, which is the ORACLE: on identical
// inputs the two produce the same output up to bf16 / summation-order noise.
//
//   logits = hidden @ router^T                                  # [T, E]
//   (idx, w) = topk_softmax(logits)                             # [T, K] each
//   --- dispatch: permute the T*K routed rows into expert-contiguous order ---
//   for each expert e (its contiguous block of routed rows):
//       gate_up = x_e @ Wgu_e^T            # [M_e, 2I]   (grouped GEMM)
//       mlp     = silu(gate) * up          # [M_e, I]    (chunked swiglu)
//       down    = mlp @ Wdown_e^T          # [M_e, H]    (grouped GEMM)
//   --- combine: scatter routed rows back to tokens, weighted by topk_w ---
//   out[t] = Σ_k w[t,k] · down_{route(t,k)}                     # [T, H]
//
// Unlike the dense block (which runs ALL E experts on ALL T tokens, O(E·T)
// GEMM rows), this runs each routed (token,expert) pair exactly once, so the
// total GEMM work is O(T·K) rows regardless of E. The routing/permutation/
// combine bookkeeping is done by the banked `moe_dispatch` kernels.
//
// WEIGHT / TENSOR LAYOUTS (identical to the dense moe_mlp_block_bf16):
//   hidden   : [T, H]      bf16, row-major          (input activations)
//   router_w : [E, H]      bf16, row-major          (router; gemm wants [E,H])
//   wgu      : [E, 2I, H]  bf16, row-major per expert (gate||up; gate=[0,I),
//                                                       up=[I,2I), N=2I, K=H)
//   wdown    : [E, H, I]   bf16, row-major per expert (down;    N=H,  K=I)
//   out      : [T, H]      bf16, row-major          (output activations)
//
// DISPATCH BUFFERS (all device scratch, allocated/freed internally):
//   num_routes = T * K.
//   sorted_route_ids    [num_routes] i32 — route ids grouped by expert; row r
//                        of the permuted layout holds original route
//                        sorted_route_ids[r] (= token*K + k).
//   route_to_sorted_row [num_routes] i32 — inverse map: route id -> its row in
//                        the permuted (expert-contiguous) layout. Drives the
//                        weighted combine back to token order.
//   counts_dev          [E] i32 — exact per-expert routed-row counts (device).
//   gate_up   [num_routes, 2I] bf16 — grouped gate||up GEMM output.
//   mlp       [num_routes, I]  bf16 — swiglu(gate,up).
//   down      [num_routes, H]  bf16 — grouped down GEMM output (expert order).
//
// HOST EXPERT OFFSETS: `grouped_gemm_bf16` needs the [E+1] prefix sum on the
// HOST to size/launch each per-expert cuBLAS call. `moe_bucket_exact` produces
// the per-expert counts on the DEVICE (`counts_dev`), so we cudaMemcpy those E
// ints D2H and cudaStreamSynchronize before building `expert_offsets_host`
// (prefix sum, [0]=0, [E]=num_routes). This is the one unavoidable host sync in
// the routed path.

#include <cstdint>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace pie_cuda_device::forward {

// Drop-in alternative to moe_mlp_block_bf16 (same args, same semantics), routed.
cudaError_t moe_sparse_block_bf16(
    cublasHandle_t cublas, cudaStream_t stream,
    const void* hidden, const void* router_w, const void* wgu, const void* wdown, void* out,
    int num_tokens, int hidden_size, int intermediate, int num_experts, int top_k);

}  // namespace pie_cuda_device::forward

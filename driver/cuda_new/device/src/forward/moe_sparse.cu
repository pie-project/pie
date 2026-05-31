#include "moe_sparse.cuh"

#include "../kernels/moe.cuh"           // topk_softmax_bf16, chunked_swiglu_bf16
#include "../kernels/moe_dispatch.cuh"  // moe_bucket_exact, gather_*, combine
#include "../ops/gemm.cuh"              // gemm_act_x_wt_bf16 (router)
#include "../ops/grouped_gemm.cuh"      // grouped_gemm_bf16

#include <cuda_bf16.h>
#include <vector>

// See moe_sparse.cuh for the layout / dispatch-buffer / host-offsets docs. This
// composes over the banked moe_dispatch kernels:
//   moe_bucket_exact                 -> sorted_route_ids, route_to_sorted_row,
//                                       per-expert counts (device)
//   gather_moe_aligned_inputs_bf16   -> permute hidden rows into expert order
//   token_batched_weighted_sum_aligned_bf16 -> weighted top-K combine back
// plus ops::grouped_gemm_bf16 for the per-expert gate||up and down GEMMs. The
// only host sync is the counts D2H copy that builds the grouped-GEMM HOST
// expert-offsets prefix sum.

namespace pie_cuda_device::forward {

cudaError_t moe_sparse_block_bf16(
    cublasHandle_t cublas, cudaStream_t stream,
    const void* hidden, const void* router_w, const void* wgu, const void* wdown, void* out,
    int num_tokens, int hidden_size, int intermediate, int num_experts, int top_k) {
    const int T = num_tokens, H = hidden_size, I = intermediate, E = num_experts, K = top_k;
    const int R = T * K;  // num_routes = routed (token,expert) pairs
    constexpr size_t es = sizeof(std::uint16_t);  // bf16

    // Scratch. Routing metadata, the permuted expert-contiguous activations
    // (gate||up, swiglu, down), and the routing weights / index buffers.
    enum {
        LOGITS,    // [T, E]      bf16   router logits
        IDX,       // [T, K]      i32    top-K expert ids   (= [R] flat route->expert)
        W,         // [T, K]      f32    renormalized routing weights (= [R] flat)
        SORTED,    // [R]         i32    sorted_route_ids (expert-contiguous order)
        INVMAP,    // [R]         i32    route_to_sorted_row (inverse permute)
        COUNTS,    // [E]         i32    per-expert routed-row counts (device)
        AIN,       // [R, H]      bf16   gathered hidden rows in expert order
        GATE_UP,   // [R, 2I]     bf16   grouped gate||up GEMM out
        MLP,       // [R, I]      bf16   swiglu(gate,up)
        DOWN,      // [R, H]      bf16   grouped down GEMM out (expert order)
        NBUF
    };
    const size_t sizes[NBUF] = {
        (size_t)T * E * es,
        (size_t)R * sizeof(int),
        (size_t)R * sizeof(float),
        (size_t)R * sizeof(int),
        (size_t)R * sizeof(int),
        (size_t)E * sizeof(int),
        (size_t)R * H * es,
        (size_t)R * 2 * I * es,
        (size_t)R * I * es,
        (size_t)R * H * es,
    };
    void* b[NBUF] = {};
    for (int i = 0; i < NBUF; ++i) {
        if (cudaError_t e = cudaMalloc(&b[i], sizes[i]); e != cudaSuccess) {
            for (int j = 0; j < i; ++j) cudaFree(b[j]);
            return e;
        }
    }

    auto cleanup = [&]() { for (int i = 0; i < NBUF; ++i) cudaFree(b[i]); };

    // router -> top-K softmax. topk_idx [T,K] flattens to [R] route->expert,
    // topk_w [T,K] flattens to [R] route weights — exactly what the dispatch /
    // combine kernels consume.
    ops::gemm_act_x_wt_bf16(cublas, hidden, router_w, b[LOGITS], T, E, H, 0.f);
    kernels::topk_softmax_bf16(b[LOGITS], static_cast<int*>(b[IDX]),
                               static_cast<float*>(b[W]), T, E, K, stream);

    // dispatch: exact per-expert bucketing of the R routes. Produces the
    // expert-contiguous row order (SORTED), the inverse permute (INVMAP) used by
    // the combine, and the exact per-expert counts (COUNTS, device).
    kernels::moe_bucket_exact(
        static_cast<const int*>(b[IDX]),
        static_cast<int*>(b[SORTED]),
        static_cast<int*>(b[INVMAP]),
        static_cast<int*>(b[COUNTS]),
        R, E, stream);

    // HOST expert-offsets for grouped_gemm: copy device counts D2H + sync, then
    // prefix-sum on the host ([0]=0, [E]=R). This is the one host sync the
    // routed path needs (grouped_gemm_bf16 reads counts host-side to size each
    // per-expert cuBLAS call).
    std::vector<int> counts_host(E);
    if (cudaError_t e = cudaMemcpyAsync(counts_host.data(), b[COUNTS],
                                        (size_t)E * sizeof(int),
                                        cudaMemcpyDeviceToHost, stream);
        e != cudaSuccess) { cleanup(); return e; }
    if (cudaError_t e = cudaStreamSynchronize(stream); e != cudaSuccess) {
        cleanup(); return e;
    }
    std::vector<int> expert_offsets_host(E + 1);
    expert_offsets_host[0] = 0;
    for (int e = 0; e < E; ++e) expert_offsets_host[e + 1] = expert_offsets_host[e] + counts_host[e];
    // expert_offsets_host[E] == R by construction (every route is bucketed).

    // gather: permute hidden rows into expert-contiguous order. aligned_rows=R
    // (exact bucketing, no padding) so every row is a real route — gather maps
    // SORTED[row] -> token = route/K and copies hidden[token].
    kernels::gather_moe_aligned_inputs_bf16(
        hidden, static_cast<const int*>(b[SORTED]), b[AIN],
        /*num_routes=*/R, /*aligned_rows=*/R, /*top_k=*/K, /*hidden=*/H, stream);

    // grouped gate||up: wgu [E, 2I, H] -> gate_up [R, 2I]  (N=2I, K=H)
    if (cudaError_t e = ops::grouped_gemm_bf16(
            cublas, stream, b[AIN], wgu, expert_offsets_host.data(), b[GATE_UP],
            /*total_rows=*/R, E, /*N=*/2 * I, /*K=*/H);
        e != cudaSuccess) { cleanup(); return e; }

    // swiglu over all R rows: silu(gate) * up -> mlp [R, I]
    kernels::chunked_swiglu_bf16(b[GATE_UP], b[MLP], R, I, stream);

    // grouped down: wdown [E, H, I] -> down [R, H]  (N=H, K=I)
    if (cudaError_t e = ops::grouped_gemm_bf16(
            cublas, stream, b[MLP], wdown, expert_offsets_host.data(), b[DOWN],
            /*total_rows=*/R, E, /*N=*/H, /*K=*/I);
        e != cudaSuccess) { cleanup(); return e; }

    // combine: out[t,h] = Σ_k topk_w[t*K+k] * down[route_to_sorted_row[t*K+k], h].
    // INVMAP maps each token's K routes to their rows in the expert-contiguous
    // DOWN buffer; this reproduces the dense block's weighted top-K combine.
    kernels::token_batched_weighted_sum_aligned_bf16(
        out, b[DOWN], static_cast<const float*>(b[W]),
        static_cast<const int*>(b[INVMAP]), T, K, H, stream);

    cudaError_t e = cudaStreamSynchronize(stream);
    cleanup();
    return e;
}

}  // namespace pie_cuda_device::forward

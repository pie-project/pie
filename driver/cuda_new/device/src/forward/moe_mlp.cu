#include "moe_mlp.cuh"

#include "../kernels/moe.cuh"  // topk_softmax_bf16, chunked_swiglu_bf16
#include "../ops/gemm.cuh"

#include <cuda_bf16.h>

namespace pie_cuda_device::forward {

namespace {

// out[t,h] = Σ_k topk_w[t,k] · ffn_all[idx[t,k], t, h]   (weighted top-K combine)
__global__ void moe_combine_topk_kernel(
    const __nv_bfloat16* __restrict__ ffn_all,  // [E, T, H]
    const int* __restrict__ topk_idx,           // [T, K]
    const float* __restrict__ topk_w,           // [T, K]
    __nv_bfloat16* __restrict__ out,            // [T, H]
    int T, int H, int K) {
    const int t = blockIdx.x;
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || h >= H) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        const int e = topk_idx[t * K + k];
        acc += topk_w[t * K + k] *
               __bfloat162float(ffn_all[((long long)e * T + t) * H + h]);
    }
    out[(long long)t * H + h] = __float2bfloat16(acc);
}

}  // namespace

cudaError_t moe_mlp_block_bf16(
    cublasHandle_t cublas, cudaStream_t stream,
    const void* hidden, const void* router_w, const void* wgu, const void* wdown, void* out,
    int num_tokens, int hidden_size, int intermediate, int num_experts, int top_k) {
    const int T = num_tokens, H = hidden_size, I = intermediate, E = num_experts, K = top_k;
    constexpr size_t es = sizeof(std::uint16_t);  // bf16

    // Scratch: router logits, top-K idx/weights, per-expert gate||up + mlp, and
    // the dense [E,T,H] expert outputs.
    enum { LOGITS, IDX, W, GATE_UP, MLP, FFN_ALL, NBUF };
    const size_t sizes[NBUF] = {
        (size_t)T * E * es,        // logits  bf16
        (size_t)T * K * sizeof(int),
        (size_t)T * K * sizeof(float),
        (size_t)T * 2 * I * es,    // gate||up
        (size_t)T * I * es,        // mlp
        (size_t)E * T * H * es,    // ffn_all
    };
    void* b[NBUF] = {};
    for (int i = 0; i < NBUF; ++i) {
        if (cudaError_t e = cudaMalloc(&b[i], sizes[i]); e != cudaSuccess) {
            for (int j = 0; j < i; ++j) cudaFree(b[j]);
            return e;
        }
    }

    // router → top-K
    ops::gemm_act_x_wt_bf16(cublas, hidden, router_w, b[LOGITS], T, E, H, 0.f);
    kernels::topk_softmax_bf16(b[LOGITS], static_cast<int*>(b[IDX]),
                               static_cast<float*>(b[W]), T, E, K, stream);

    // dense experts: ffn_all[e] = down(silu(gate) * up)
    for (int e = 0; e < E; ++e) {
        const void* wgu_e = static_cast<const char*>(wgu) + (size_t)e * 2 * I * H * es;
        const void* wdown_e = static_cast<const char*>(wdown) + (size_t)e * H * I * es;
        void* ffn_e = static_cast<char*>(b[FFN_ALL]) + (size_t)e * T * H * es;
        ops::gemm_act_x_wt_bf16(cublas, hidden, wgu_e, b[GATE_UP], T, 2 * I, H, 0.f);
        kernels::chunked_swiglu_bf16(b[GATE_UP], b[MLP], T, I, stream);
        ops::gemm_act_x_wt_bf16(cublas, b[MLP], wdown_e, ffn_e, T, H, I, 0.f);
    }

    // weighted top-K combine
    constexpr int BLOCK = 128;
    dim3 grid(T, (H + BLOCK - 1) / BLOCK);
    moe_combine_topk_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(b[FFN_ALL]), static_cast<const int*>(b[IDX]),
        static_cast<const float*>(b[W]), static_cast<__nv_bfloat16*>(out), T, H, K);

    cudaError_t e = cudaStreamSynchronize(stream);
    for (int i = 0; i < NBUF; ++i) cudaFree(b[i]);
    return e;
}

}  // namespace pie_cuda_device::forward

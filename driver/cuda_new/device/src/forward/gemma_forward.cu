#include "gemma_forward.cuh"

#include "../kernels/argmax.cuh"
#include "../kernels/embed.cuh"
#include "../kernels/gemma.cuh"   // rmsnorm_gemma_bf16, logit_softcap_bf16
#include "../ops/gemm.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace pie_cuda_device::forward {

namespace {

// In-place bf16 elementwise scale: y[i] = round_bf16(y[i] * scale). Applied to
// the embedding output for Gemma's √H scaling (ASSUMPTION A1). Done in fp32 and
// re-rounded so it matches the CPU reference's `bf16_rt(emb * scale)` exactly.
__global__ void scale_inplace_bf16_kernel(__nv_bfloat16* __restrict__ y,
                                          float scale, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
    if (i >= n) return;
    y[i] = __float2bfloat16(__bfloat162float(y[i]) * scale);
}

void scale_inplace_bf16(void* y, float scale, std::size_t n,
                        cudaStream_t stream) {
    if (n == 0) return;
    constexpr int BLOCK = 256;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    scale_inplace_bf16_kernel<<<blocks, BLOCK, 0, stream>>>(
        static_cast<__nv_bfloat16*>(y), scale, n);
}

}  // namespace

cudaError_t gemma_forward_bf16(
    cublasHandle_t cublas, cudaStream_t stream,
    const std::int32_t* token_ids, const GemmaForwardWeights& w,
    const std::int32_t* positions,
    void* k_pages, void* v_pages,
    const std::uint32_t* qo_indptr, const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr, const std::uint32_t* kv_last_page_lens,
    void* out_logits, std::int32_t* out_token_ids,
    int num_tokens, int num_requests, const GemmaForwardDims& d) {
    // Reject DEFERRED features so a caller can never silently believe they are
    // active (ASSUMPTIONS A3, A8).
    if (d.qk_norm != 0) return cudaErrorInvalidValue;          // A3: qk-norm off
    if (d.altup_num_inputs != 1) return cudaErrorInvalidValue;  // A8: AltUp off

    const int T = num_tokens;
    const int H = d.hidden;
    const int Hq = d.n_q_heads * d.head_dim;
    const int Hkv = d.n_kv_heads * d.head_dim;
    const int I = d.intermediate;
    const int V = d.vocab;
    constexpr std::size_t es = sizeof(std::uint16_t);  // bf16

    // --- Activation scratch: residual stream `hidden` + a final `normed`
    //     buffer + the 10-slot LayerScratch reused across layers. One alloc
    //     pass; freed at the end. Mirrors moe_forward's single-alloc strategy.
    enum {
        HIDDEN, NORMED,                          // model-level
        S_NORMED, S_Q, S_K, S_V, S_ATTN, S_O,    // LayerScratch attention half
        S_GATE, S_UP, S_MLP, S_MLP_OUT,          // LayerScratch MLP half
        NBUF
    };
    const std::size_t sizes[NBUF] = {
        (std::size_t)T * H * es,    // hidden
        (std::size_t)T * H * es,    // normed (final-norm output)
        (std::size_t)T * H * es,    // s.normed
        (std::size_t)T * Hq * es,   // s.q
        (std::size_t)T * Hkv * es,  // s.k
        (std::size_t)T * Hkv * es,  // s.v
        (std::size_t)T * Hq * es,   // s.attn
        (std::size_t)T * H * es,    // s.o
        (std::size_t)T * I * es,    // s.gate
        (std::size_t)T * I * es,    // s.up
        (std::size_t)T * I * es,    // s.mlp
        (std::size_t)T * H * es,    // s.mlp_out
    };
    void* b[NBUF] = {};
    for (int i = 0; i < NBUF; ++i) {
        if (cudaError_t e = cudaMalloc(&b[i], sizes[i]); e != cudaSuccess) {
            for (int j = 0; j < i; ++j) cudaFree(b[j]);
            return e;
        }
    }
    void* hidden = b[HIDDEN];
    void* normed = b[NORMED];
    const LayerScratch s{b[S_NORMED], b[S_Q], b[S_K], b[S_V], b[S_ATTN],
                         b[S_O], b[S_GATE], b[S_UP], b[S_MLP], b[S_MLP_OUT]};

    // --- embed: token_ids -> residual stream, then Gemma's √H scale (A1). ---
    kernels::embed_bf16(token_ids, w.embed, hidden, T, H, V, stream);
    scale_inplace_bf16(hidden, d.embed_scale, (std::size_t)T * H, stream);

    // --- N Gemma sandwich layers, each on its own KV slice + window. ---
    const std::size_t kv_layer_stride_bytes =
        (std::size_t)d.num_pages * d.page_size * Hkv * es;  // A9: NHD
    for (int L = 0; L < d.n_layers; ++L) {
        void* kL = static_cast<char*>(k_pages) + (std::size_t)L * kv_layer_stride_bytes;
        void* vL = static_cast<char*>(v_pages) + (std::size_t)L * kv_layer_stride_bytes;
        const int window_left =
            (d.window_left != nullptr) ? d.window_left[L] : d.window_left_all;  // A5
        decoder_layer_gemma_inplace(
            cublas, stream, hidden, w.layers[L], positions, kL, vL,
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens, s,
            T, num_requests, H, d.n_q_heads, d.n_kv_heads, d.head_dim, I,
            d.page_size, d.rms_eps, d.rope_theta, window_left,
            d.attn_logit_softcap);
    }

    // --- final norm (rmsnorm_gemma, (1+w)) -> lm_head -> final softcap -> argmax. ---
    kernels::rmsnorm_gemma_bf16(hidden, w.final_norm, normed, T, H, d.rms_eps, stream);  // A2
    ops::gemm_act_x_wt_bf16(cublas, normed, w.lm_head, out_logits, T, V, H, 0.f);
    kernels::logit_softcap_bf16(out_logits, d.final_logit_softcap,
                                (std::size_t)T * V, stream);  // A6 (no-op if <=0)
    kernels::argmax_bf16(out_logits, out_token_ids, T, V, stream);

    cudaError_t e = cudaStreamSynchronize(stream);
    for (int i = 0; i < NBUF; ++i) cudaFree(b[i]);
    return e;
}

}  // namespace pie_cuda_device::forward

#include "nemotron_forward.cuh"

#include "nemotron_block.cuh"            // nemotron_mamba_block_bf16
#include "../kernels/argmax.cuh"
#include "../kernels/embed.cuh"
#include "../kernels/kv_append.cuh"
#include "../kernels/residual_add.cuh"
#include "../kernels/rmsnorm.cuh"
#include "../kernels/rope.cuh"
#include "../kernels/swiglu.cuh"
#include "../ops/attention_naive_paged.cuh"
#include "../ops/gemm.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <vector>

// Composition mirrors driver/cuda/src/model/nemotron_h_forward.cpp's per-layer
// loop (single fresh-prefill request, TP==1). Each layer is a SINGLE mixer with
// one pre-norm; the attention layer is attention-only and the FFN layer is its
// own layer (see nemotron_forward.cuh for the structure + assumptions). All
// math lives in the lifted/banked primitives; this file sequences them and owns
// the residual stream + per-layer state caches.

namespace pie_cuda_device::forward {

namespace {

constexpr std::size_t kBf16 = sizeof(std::uint16_t);

// hidden[i] += (post[i] - pre[i]). Adds the Mamba mixer delta (block output
// minus its pre-norm input) into the real residual stream. See assumption A2.
__global__ void add_delta_kernel(__nv_bfloat16* __restrict__ hidden,
                                 const __nv_bfloat16* __restrict__ post,
                                 const __nv_bfloat16* __restrict__ pre,
                                 std::size_t n) {
    const std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float d = __bfloat162float(post[i]) - __bfloat162float(pre[i]);
    hidden[i] = __float2bfloat16(__bfloat162float(hidden[i]) + d);
}

// Attention sublayer (pre-norm -> qkv -> rope -> KV append -> naive paged attn
// -> o_proj -> residual). Runs IN PLACE on `hidden`. Mirrors llama_layer's
// attention half, but is a standalone layer (no FFN follows here).
cudaError_t attn_layer(
    cublasHandle_t cublas, cudaStream_t stream,
    void* hidden, const NemotronAttnWeights& w,
    const std::int32_t* positions,
    void* k_pages, void* v_pages,
    const std::uint32_t* qo_indptr, const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr, const std::uint32_t* kv_last_page_lens,
    int T, int H, int n_q, int n_kv, int head_dim, int page_size,
    float rms_eps, float rope_theta) {
    const int Hq = n_q * head_dim;
    const int Hkv = n_kv * head_dim;

    enum { NORMED, Q, K, V, ATTN, O, NBUF };
    const std::size_t sizes[NBUF] = {
        (std::size_t)T * H * kBf16,
        (std::size_t)T * Hq * kBf16,
        (std::size_t)T * Hkv * kBf16,
        (std::size_t)T * Hkv * kBf16,
        (std::size_t)T * Hq * kBf16,
        (std::size_t)T * H * kBf16,
    };
    void* b[NBUF] = {};
    for (int i = 0; i < NBUF; ++i)
        if (cudaError_t e = cudaMalloc(&b[i], sizes[i]); e != cudaSuccess) {
            for (int j = 0; j < i; ++j) cudaFree(b[j]);
            return e;
        }

    kernels::rmsnorm_bf16(hidden, w.attn_norm, b[NORMED], T, H, rms_eps, stream);
    ops::gemm_act_x_wt_bf16(cublas, b[NORMED], w.wq, b[Q], T, Hq, H, 0.f);
    ops::gemm_act_x_wt_bf16(cublas, b[NORMED], w.wk, b[K], T, Hkv, H, 0.f);
    ops::gemm_act_x_wt_bf16(cublas, b[NORMED], w.wv, b[V], T, Hkv, H, 0.f);
    kernels::rope_bf16(b[Q], b[K], positions, T, n_q, n_kv, head_dim, rope_theta,
                       /*interleaved=*/false, stream);
    kernels::write_kv_to_pages_bf16(k_pages, v_pages, b[K], b[V], qo_indptr,
                                    kv_page_indices, kv_page_indptr, kv_last_page_lens,
                                    T, /*num_requests=*/1, page_size, n_kv, head_dim,
                                    /*hnd_layout=*/false, stream);
    ops::attention_naive_paged_bf16(
        b[Q], k_pages, v_pages, b[ATTN], qo_indptr, kv_page_indices, kv_page_indptr,
        kv_last_page_lens, T, /*num_requests=*/1, n_q, n_kv, head_dim, page_size, stream,
        /*window_left=*/-1, /*sm_scale=*/-1.f, /*logits_soft_cap=*/0.f, nullptr);
    ops::gemm_act_x_wt_bf16(cublas, b[ATTN], w.wo, b[O], T, H, Hq, 0.f);
    kernels::residual_add_bf16(hidden, b[O], (std::size_t)T * H, stream);

    cudaError_t e = cudaStreamSynchronize(stream);
    for (int i = 0; i < NBUF; ++i) cudaFree(b[i]);
    return e;
}

// Dense SwiGLU FFN sublayer (pre-norm -> silu(gate)*up -> down -> residual).
// Runs IN PLACE on `hidden`. See assumption A1 (stand-in for upstream MoE).
cudaError_t ffn_layer(
    cublasHandle_t cublas, cudaStream_t stream,
    void* hidden, const NemotronFfnWeights& w,
    int T, int H, int I, float rms_eps) {
    enum { NORMED, GATE, UP, MLP, OUT, NBUF };
    const std::size_t sizes[NBUF] = {
        (std::size_t)T * H * kBf16,
        (std::size_t)T * I * kBf16,
        (std::size_t)T * I * kBf16,
        (std::size_t)T * I * kBf16,
        (std::size_t)T * H * kBf16,
    };
    void* b[NBUF] = {};
    for (int i = 0; i < NBUF; ++i)
        if (cudaError_t e = cudaMalloc(&b[i], sizes[i]); e != cudaSuccess) {
            for (int j = 0; j < i; ++j) cudaFree(b[j]);
            return e;
        }

    kernels::rmsnorm_bf16(hidden, w.ffn_norm, b[NORMED], T, H, rms_eps, stream);
    ops::gemm_act_x_wt_bf16(cublas, b[NORMED], w.w_gate, b[GATE], T, I, H, 0.f);
    ops::gemm_act_x_wt_bf16(cublas, b[NORMED], w.w_up, b[UP], T, I, H, 0.f);
    kernels::swiglu_bf16(b[GATE], b[UP], b[MLP], T * I, stream);
    ops::gemm_act_x_wt_bf16(cublas, b[MLP], w.w_down, b[OUT], T, H, I, 0.f);
    kernels::residual_add_bf16(hidden, b[OUT], (std::size_t)T * H, stream);

    cudaError_t e = cudaStreamSynchronize(stream);
    for (int i = 0; i < NBUF; ++i) cudaFree(b[i]);
    return e;
}

// Mamba mixer layer (see assumption A2). Net effect:
//   normed = rmsnorm(hidden, pre_norm)          (layer pre-norm)
//   hidden += out_proj( mixer(normed) )
// The banked block runs in place doing `arg += out_proj(mixer(arg))`, so we
// hand it a COPY of `normed` and add (block_out - normed) into `hidden`.
cudaError_t mamba_layer(
    cublasHandle_t cublas, cudaStream_t stream,
    void* hidden, const void* pre_norm, const NemotronMambaWeights& w,
    int T, int H, int num_heads, int head_dim, int state_size, int n_groups,
    int conv_kernel, float rms_eps, float time_step_min) {
    const std::size_t bytes = (std::size_t)T * H * kBf16;
    const std::size_t n = (std::size_t)T * H;
    void* normed = nullptr;  // rmsnorm(hidden); block input/output (mutated)
    void* pre = nullptr;     // snapshot of normed (pre-block) for the delta
    if (cudaError_t e = cudaMalloc(&normed, bytes); e != cudaSuccess) return e;
    if (cudaError_t e = cudaMalloc(&pre, bytes); e != cudaSuccess) {
        cudaFree(normed);
        return e;
    }

    // Layer pre-norm into `normed`, then snapshot into `pre`.
    kernels::rmsnorm_bf16(hidden, pre_norm, normed, T, H, rms_eps, stream);
    cudaMemcpyAsync(pre, normed, bytes, cudaMemcpyDeviceToDevice, stream);
    if (cudaError_t e = cudaStreamSynchronize(stream); e != cudaSuccess) {
        cudaFree(normed); cudaFree(pre);
        return e;
    }

    // Banked Mamba block: normed += out_proj(mixer(normed)).
    if (cudaError_t e = nemotron_mamba_block_bf16(
            cublas, stream, normed, w, T, H, num_heads, head_dim, state_size,
            n_groups, conv_kernel, rms_eps, time_step_min);
        e != cudaSuccess) {
        cudaFree(normed); cudaFree(pre);
        return e;
    }

    // hidden += (normed - pre) = out_proj(mixer(rmsnorm(hidden))).
    constexpr int BLK = 256;
    add_delta_kernel<<<(unsigned)((n + BLK - 1) / BLK), BLK, 0, stream>>>(
        static_cast<__nv_bfloat16*>(hidden), static_cast<const __nv_bfloat16*>(normed),
        static_cast<const __nv_bfloat16*>(pre), n);

    cudaError_t e = cudaStreamSynchronize(stream);
    cudaFree(normed); cudaFree(pre);
    return e;
}

}  // namespace

cudaError_t nemotron_forward_bf16(
    cublasHandle_t cublas, cudaStream_t stream,
    const std::int32_t* token_ids, const NemotronForwardWeights& w,
    const std::int32_t* positions,
    void* out_logits, std::int32_t* out_token_ids,
    int num_tokens, const NemotronForwardDims& dims) {
    const int T = num_tokens;
    const int H = dims.hidden;
    const int V = dims.vocab;

    // ---- CSR paging for a single contiguous request over [0, T) ------------
    const int n_pages = (T + dims.page_size - 1) / dims.page_size;
    const int last_page_len = T - (n_pages - 1) * dims.page_size;  // 1..page_size
    std::vector<std::uint32_t> qo_indptr_h = {0u, (std::uint32_t)T};
    std::vector<std::uint32_t> kv_page_indptr_h = {0u, (std::uint32_t)n_pages};
    std::vector<std::uint32_t> kv_page_indices_h(n_pages);
    for (int p = 0; p < n_pages; ++p) kv_page_indices_h[p] = (std::uint32_t)p;
    std::vector<std::uint32_t> kv_last_page_lens_h = {(std::uint32_t)last_page_len};

    // Count attention layers (each gets its own KV page pool).
    int n_attn = 0;
    for (int L = 0; L < dims.n_layers; ++L)
        if (dims.kinds[L] == 'A') ++n_attn;

    const int Hkv = dims.attn_n_kv_heads * dims.attn_head_dim;
    const std::size_t kv_layer_elems =
        (std::size_t)n_pages * dims.page_size * dims.attn_n_kv_heads * dims.attn_head_dim;

    // ---- Allocations: residual, final-norm scratch, CSR, per-attn KV --------
    void* hidden = nullptr;
    void* normed = nullptr;
    void *d_qo = nullptr, *d_kpi = nullptr, *d_kpp = nullptr, *d_klpl = nullptr;
    void *k_pool = nullptr, *v_pool = nullptr;
    auto cleanup = [&] {
        cudaFree(hidden); cudaFree(normed);
        cudaFree(d_qo); cudaFree(d_kpi); cudaFree(d_kpp); cudaFree(d_klpl);
        cudaFree(k_pool); cudaFree(v_pool);
    };
#define ALLOC(ptr, bytes)                                                       \
    if (cudaError_t e = cudaMalloc(&(ptr), (bytes)); e != cudaSuccess) {        \
        cleanup();                                                              \
        return e;                                                              \
    }
    ALLOC(hidden, (std::size_t)T * H * kBf16);
    ALLOC(normed, (std::size_t)T * H * kBf16);
    ALLOC(d_qo, qo_indptr_h.size() * sizeof(std::uint32_t));
    ALLOC(d_kpi, kv_page_indices_h.size() * sizeof(std::uint32_t));
    ALLOC(d_kpp, kv_page_indptr_h.size() * sizeof(std::uint32_t));
    ALLOC(d_klpl, kv_last_page_lens_h.size() * sizeof(std::uint32_t));
    if (n_attn > 0) {
        ALLOC(k_pool, (std::size_t)n_attn * kv_layer_elems * kBf16);
        ALLOC(v_pool, (std::size_t)n_attn * kv_layer_elems * kBf16);
        cudaMemsetAsync(k_pool, 0, (std::size_t)n_attn * kv_layer_elems * kBf16, stream);
        cudaMemsetAsync(v_pool, 0, (std::size_t)n_attn * kv_layer_elems * kBf16, stream);
    }
#undef ALLOC
    (void)Hkv;

    cudaMemcpyAsync(d_qo, qo_indptr_h.data(),
                    qo_indptr_h.size() * sizeof(std::uint32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_kpi, kv_page_indices_h.data(),
                    kv_page_indices_h.size() * sizeof(std::uint32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_kpp, kv_page_indptr_h.data(),
                    kv_page_indptr_h.size() * sizeof(std::uint32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_klpl, kv_last_page_lens_h.data(),
                    kv_last_page_lens_h.size() * sizeof(std::uint32_t),
                    cudaMemcpyHostToDevice, stream);

    // ---- embed: token_ids -> residual stream (no scale) --------------------
    kernels::embed_bf16(token_ids, w.embed, hidden, T, H, V, stream);

    // ---- per-layer dispatch ------------------------------------------------
    cudaError_t rc = cudaStreamSynchronize(stream);
    int attn_slot = 0;
    for (int L = 0; L < dims.n_layers && rc == cudaSuccess; ++L) {
        const NemotronLayerWeights& Lw = w.layers[L];
        switch (dims.kinds[L]) {
            case 'M': {
                rc = mamba_layer(cublas, stream, hidden, Lw.mamba_pre_norm, Lw.mamba,
                                 T, H, dims.mamba_num_heads, dims.mamba_head_dim,
                                 dims.mamba_state_size, dims.mamba_n_groups,
                                 dims.mamba_conv_kernel, dims.rms_eps,
                                 dims.time_step_min);
                break;
            }
            case 'A': {
                void* kL = static_cast<char*>(k_pool) +
                           (std::size_t)attn_slot * kv_layer_elems * kBf16;
                void* vL = static_cast<char*>(v_pool) +
                           (std::size_t)attn_slot * kv_layer_elems * kBf16;
                ++attn_slot;
                rc = attn_layer(cublas, stream, hidden, Lw.attn, positions, kL, vL,
                                (const std::uint32_t*)d_qo, (const std::uint32_t*)d_kpi,
                                (const std::uint32_t*)d_kpp, (const std::uint32_t*)d_klpl,
                                T, H, dims.attn_n_q_heads, dims.attn_n_kv_heads,
                                dims.attn_head_dim, dims.page_size, dims.rms_eps,
                                dims.rope_theta);
                break;
            }
            case 'F': {
                rc = ffn_layer(cublas, stream, hidden, Lw.ffn, T, H,
                               dims.ffn_intermediate, dims.rms_eps);
                break;
            }
            default:
                rc = cudaErrorInvalidValue;
                break;
        }
    }
    if (rc != cudaSuccess) {
        cleanup();
        return rc;
    }

    // ---- final norm -> lm_head -> greedy argmax ----------------------------
    kernels::rmsnorm_bf16(hidden, w.final_norm, normed, T, H, dims.rms_eps, stream);
    ops::gemm_act_x_wt_bf16(cublas, normed, w.lm_head, out_logits, T, V, H, 0.f);
    kernels::argmax_bf16(out_logits, out_token_ids, T, V, stream);

    rc = cudaStreamSynchronize(stream);
    cleanup();
    return rc;
}

}  // namespace pie_cuda_device::forward

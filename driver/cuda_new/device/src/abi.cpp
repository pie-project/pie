// ABI entry points for libpie_cuda_device.
//
// Phase 0: lifecycle + introspection are implemented for real so the
// seam is exercisable from Rust today. The construction/hot-path/graph
// entries are stubs that return PIE_ERR_INTERNAL with a clear message;
// each names the driver/cuda source it will be ported from in phase 1.
//
// Every entry is noexcept: C++ exceptions must not cross the boundary.
// Errors are reported via the return code + pie_cuda_last_error().

#include "../include/pie_cuda_device.h"
#include "context.hpp"
#include "kernels/argmax.cuh"
#include "kernels/causal_conv1d.cuh"
#include "kernels/dequant_fp8.cuh"
#include "kernels/dequant_wna16.cuh"
#include "kernels/rope_partial.cuh"
#include "kernels/dtype_cast.cuh"
#include "kernels/embed.cuh"
#include "kernels/gather_rows.cuh"
#include "kernels/gemma.cuh"
#include "kernels/kv_append.cuh"
#include "kernels/moe.cuh"
#include "kernels/rope_yarn.cuh"
#include "kernels/residual_add.cuh"
#include "kernels/rmsnorm.cuh"
#include "kernels/rope.cuh"
#include "kernels/sample.cuh"
#include "kernels/swiglu.cuh"
#include "forward/llama_forward.cuh"
#include "forward/llama_layer.cuh"
#include "forward/deepseek_forward.cuh"
#include "forward/gemma_forward.cuh"
#include "forward/mla_block.cuh"
#include "forward/mla_forward.cuh"
#include "forward/moe_mlp.cuh"
#include "forward/moe_forward.cuh"
#include "forward/nemotron_forward.cuh"
#include "forward/moe_sparse.cuh"
#include "forward/nemotron_block.cuh"
#include "kernels/altup.cuh"
#include "kernels/ssm_scan.cuh"
#include "ops/grouped_gemm.cuh"
#include "qgemm/qgemm.h"
#include "ops/attention_naive_paged.cuh"
#include "ops/gemm.cuh"
#include "workspace.hpp"

#include <cuda_runtime.h>

#include <exception>
#include <string>
#include <vector>

using pie_cuda_device::last_error;
using pie_cuda_device::set_last_error;

namespace {

PieStatus cuda_fail(const char* what, cudaError_t err) {
    set_last_error(std::string(what) + ": " + cudaGetErrorString(err));
    return PIE_ERR_CUDA;
}

// Uniform stub for not-yet-ported entries. `from` is the driver/cuda
// source that holds the logic to lift in phase 1.
PieStatus not_yet(const char* entry, const char* from) {
    set_last_error(std::string(entry) + ": not implemented (phase 1: port from " +
                   from + ")");
    return PIE_ERR_INTERNAL;
}

}  // namespace

extern "C" {

uint32_t pie_cuda_abi_version(void) { return PIE_CUDA_DEVICE_ABI_VERSION; }

const char* pie_cuda_last_error(void) { return last_error(); }

PieStatus pie_cuda_ctx_create(int32_t device_ordinal, PieDevCtx** out_ctx) try {
    if (out_ctx == nullptr) return PIE_ERR_INVALID_ARG;
    *out_ctx = nullptr;

    if (cudaError_t e = cudaSetDevice(device_ordinal); e != cudaSuccess)
        return cuda_fail("cudaSetDevice", e);

    auto* ctx = new PieDevCtx{};
    ctx->device_ordinal = device_ordinal;
    if (cudaError_t e = cudaStreamCreate(&ctx->stream); e != cudaSuccess) {
        delete ctx;
        return cuda_fail("cudaStreamCreate", e);
    }
    if (cublasStatus_t s = cublasCreate(&ctx->cublas); s != CUBLAS_STATUS_SUCCESS) {
        cudaStreamDestroy(ctx->stream);
        delete ctx;
        set_last_error("cublasCreate failed");
        return PIE_ERR_CUDA;
    }
    cublasSetStream(ctx->cublas, ctx->stream);

    *out_ctx = ctx;
    set_last_error({});
    return PIE_OK;
} catch (const std::exception& e) {
    set_last_error(std::string("pie_cuda_ctx_create: ") + e.what());
    return PIE_ERR_INTERNAL;
}

PieStatus pie_cuda_ctx_destroy(PieDevCtx* ctx) {
    if (ctx == nullptr) return PIE_OK;
    if (ctx->cublas) cublasDestroy(ctx->cublas);
    if (ctx->stream) cudaStreamDestroy(ctx->stream);
    delete ctx;
    return PIE_OK;
}

PieStatus pie_cuda_mem_info(PieDevCtx* ctx, size_t* out_free, size_t* out_total) {
    if (ctx == nullptr || out_free == nullptr || out_total == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (cudaError_t e = cudaSetDevice(ctx->device_ordinal); e != cudaSuccess)
        return cuda_fail("cudaSetDevice", e);
    if (cudaError_t e = cudaMemGetInfo(out_free, out_total); e != cudaSuccess)
        return cuda_fail("cudaMemGetInfo", e);
    return PIE_OK;
}

PieStatus pie_cuda_device_props(PieDevCtx* ctx, int32_t* out_sm_count,
                                int32_t* out_major, int32_t* out_minor) {
    if (ctx == nullptr || out_sm_count == nullptr || out_major == nullptr || out_minor == nullptr)
        return PIE_ERR_INVALID_ARG;
    cudaDeviceProp prop{};
    if (cudaError_t e = cudaGetDeviceProperties(&prop, ctx->device_ordinal); e != cudaSuccess)
        return cuda_fail("cudaGetDeviceProperties", e);
    *out_sm_count = prop.multiProcessorCount;
    *out_major = prop.major;
    *out_minor = prop.minor;
    return PIE_OK;
}

// --- raw device memory ---

PieStatus pie_cuda_malloc(PieDevCtx* ctx, size_t nbytes, void** out_ptr) {
    if (ctx == nullptr || out_ptr == nullptr) return PIE_ERR_INVALID_ARG;
    *out_ptr = nullptr;
    if (cudaError_t e = cudaSetDevice(ctx->device_ordinal); e != cudaSuccess)
        return cuda_fail("cudaSetDevice", e);
    void* p = nullptr;
    if (cudaError_t e = cudaMalloc(&p, nbytes); e != cudaSuccess)
        return e == cudaErrorMemoryAllocation ? (set_last_error("cudaMalloc: out of memory"), PIE_ERR_OOM)
                                              : cuda_fail("cudaMalloc", e);
    *out_ptr = p;
    return PIE_OK;
}

PieStatus pie_cuda_free(PieDevCtx* ctx, void* ptr) {
    if (ctx == nullptr) return PIE_ERR_INVALID_ARG;
    if (ptr != nullptr) {
        if (cudaError_t e = cudaFree(ptr); e != cudaSuccess) return cuda_fail("cudaFree", e);
    }
    return PIE_OK;
}

PieStatus pie_cuda_memcpy_h2d(PieDevCtx* ctx, void* dst, const void* src, size_t nbytes) {
    if (ctx == nullptr || dst == nullptr || src == nullptr) return PIE_ERR_INVALID_ARG;
    if (cudaError_t e = cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, ctx->stream);
        e != cudaSuccess)
        return cuda_fail("cudaMemcpyAsync H2D", e);
    return PIE_OK;
}

PieStatus pie_cuda_memcpy_d2h(PieDevCtx* ctx, void* dst, const void* src, size_t nbytes) {
    if (ctx == nullptr || dst == nullptr || src == nullptr) return PIE_ERR_INVALID_ARG;
    if (cudaError_t e = cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToHost, ctx->stream);
        e != cudaSuccess)
        return cuda_fail("cudaMemcpyAsync D2H", e);
    return PIE_OK;
}

PieStatus pie_cuda_memcpy_d2d(PieDevCtx* ctx, void* dst, const void* src, size_t nbytes) {
    if (ctx == nullptr || dst == nullptr || src == nullptr) return PIE_ERR_INVALID_ARG;
    if (cudaError_t e = cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, ctx->stream);
        e != cudaSuccess)
        return cuda_fail("cudaMemcpyAsync D2D", e);
    return PIE_OK;
}

PieStatus pie_cuda_stream_sync(PieDevCtx* ctx) {
    if (ctx == nullptr) return PIE_ERR_INVALID_ARG;
    if (cudaError_t e = cudaStreamSynchronize(ctx->stream); e != cudaSuccess)
        return cuda_fail("cudaStreamSynchronize", e);
    return PIE_OK;
}

// --- lifted kernels ---

PieStatus pie_cuda_rmsnorm_bf16(PieDevCtx* ctx, const void* x, const void* weight,
                                void* y, int32_t num_rows, int32_t hidden, float eps) {
    if (ctx == nullptr || x == nullptr || weight == nullptr || y == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (num_rows <= 0 || hidden <= 0) return PIE_ERR_INVALID_ARG;
    pie_cuda_device::kernels::rmsnorm_bf16(x, weight, y, num_rows, hidden, eps, ctx->stream);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return cuda_fail("pie_cuda_rmsnorm_bf16 launch", e);
    return PIE_OK;
}

PieStatus pie_cuda_residual_add_bf16(PieDevCtx* ctx, void* y, const void* x, size_t n) {
    if (ctx == nullptr || y == nullptr || x == nullptr) return PIE_ERR_INVALID_ARG;
    pie_cuda_device::kernels::residual_add_bf16(y, x, n, ctx->stream);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return cuda_fail("pie_cuda_residual_add_bf16 launch", e);
    return PIE_OK;
}

PieStatus pie_cuda_swiglu_bf16(PieDevCtx* ctx, const void* gate, const void* up,
                               void* y, int32_t num_elements) {
    if (ctx == nullptr || gate == nullptr || up == nullptr || y == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (num_elements <= 0) return PIE_ERR_INVALID_ARG;
    pie_cuda_device::kernels::swiglu_bf16(gate, up, y, num_elements, ctx->stream);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return cuda_fail("pie_cuda_swiglu_bf16 launch", e);
    return PIE_OK;
}

PieStatus pie_cuda_rope_bf16(PieDevCtx* ctx, void* q, void* k, const int32_t* positions,
                             int32_t num_tokens, int32_t num_q_heads, int32_t num_kv_heads,
                             int32_t head_dim, float theta, int32_t interleaved) {
    if (ctx == nullptr || q == nullptr || k == nullptr || positions == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (num_tokens <= 0 || num_q_heads <= 0 || num_kv_heads <= 0 ||
        head_dim <= 0 || (head_dim & 1))
        return PIE_ERR_INVALID_ARG;
    pie_cuda_device::kernels::rope_bf16(q, k, positions, num_tokens, num_q_heads,
                                        num_kv_heads, head_dim, theta,
                                        interleaved != 0, ctx->stream);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return cuda_fail("pie_cuda_rope_bf16 launch", e);
    return PIE_OK;
}

PieStatus pie_cuda_gemm_bf16(PieDevCtx* ctx, const void* act, const void* w, void* y,
                             int32_t M, int32_t N, int32_t K, float beta) {
    if (ctx == nullptr || act == nullptr || w == nullptr || y == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (M <= 0 || N <= 0 || K <= 0) return PIE_ERR_INVALID_ARG;
    cublasStatus_t s = pie_cuda_device::ops::gemm_act_x_wt_bf16(ctx->cublas, act, w, y,
                                                                M, N, K, beta);
    if (s != CUBLAS_STATUS_SUCCESS) {
        set_last_error("pie_cuda_gemm_bf16: cublasGemmEx failed (status " +
                       std::to_string(static_cast<int>(s)) + ")");
        return PIE_ERR_CUDA;
    }
    return PIE_OK;
}

PieStatus pie_cuda_embed_bf16(PieDevCtx* ctx, const int32_t* token_ids, const void* weight,
                              void* y, int32_t num_tokens, int32_t hidden, int32_t vocab) {
    if (ctx == nullptr || token_ids == nullptr || weight == nullptr || y == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (num_tokens <= 0 || hidden <= 0 || vocab <= 0) return PIE_ERR_INVALID_ARG;
    pie_cuda_device::kernels::embed_bf16(token_ids, weight, y, num_tokens, hidden, vocab,
                                         ctx->stream);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return cuda_fail("pie_cuda_embed_bf16 launch", e);
    return PIE_OK;
}

PieStatus pie_cuda_argmax_bf16(PieDevCtx* ctx, const void* logits, int32_t* token_ids,
                               int32_t num_rows, int32_t vocab) {
    if (ctx == nullptr || logits == nullptr || token_ids == nullptr) return PIE_ERR_INVALID_ARG;
    if (num_rows <= 0 || vocab <= 0) return PIE_ERR_INVALID_ARG;
    pie_cuda_device::kernels::argmax_bf16(logits, token_ids, num_rows, vocab, ctx->stream);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return cuda_fail("pie_cuda_argmax_bf16 launch", e);
    return PIE_OK;
}

PieStatus pie_cuda_attention_naive_paged_bf16(
    PieDevCtx* ctx, const void* q, const void* k_pages, const void* v_pages, void* o,
    const uint32_t* qo_indptr_d, const uint32_t* kv_page_indices_d,
    const uint32_t* kv_page_indptr_d, const uint32_t* kv_last_page_lens_d,
    int32_t total_tokens, int32_t num_requests, int32_t num_q_heads, int32_t num_kv_heads,
    int32_t head_dim, int32_t page_size, int32_t window_left, float sm_scale) {
    if (ctx == nullptr || q == nullptr || k_pages == nullptr || v_pages == nullptr ||
        o == nullptr || qo_indptr_d == nullptr || kv_page_indices_d == nullptr ||
        kv_page_indptr_d == nullptr || kv_last_page_lens_d == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (total_tokens <= 0 || num_requests <= 0 || num_q_heads <= 0 || num_kv_heads <= 0 ||
        head_dim <= 0 || page_size <= 0)
        return PIE_ERR_INVALID_ARG;
    pie_cuda_device::ops::attention_naive_paged_bf16(
        q, k_pages, v_pages, o, qo_indptr_d, kv_page_indices_d, kv_page_indptr_d,
        kv_last_page_lens_d, total_tokens, num_requests, num_q_heads, num_kv_heads,
        head_dim, page_size, ctx->stream, window_left, sm_scale,
        /*logits_soft_cap=*/0.f, /*lse_out=*/nullptr);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return cuda_fail("pie_cuda_attention_naive_paged_bf16 launch", e);
    return PIE_OK;
}

PieStatus pie_cuda_write_kv_to_pages_bf16(
    PieDevCtx* ctx, void* k_pages, void* v_pages, const void* k_curr, const void* v_curr,
    const uint32_t* qo_indptr_d, const uint32_t* kv_page_indices_d,
    const uint32_t* kv_page_indptr_d, const uint32_t* kv_last_page_lens_d,
    int32_t total_tokens, int32_t num_requests, int32_t page_size, int32_t num_kv_heads,
    int32_t head_dim, int32_t hnd_layout) {
    if (ctx == nullptr || k_pages == nullptr || v_pages == nullptr || k_curr == nullptr ||
        v_curr == nullptr || qo_indptr_d == nullptr || kv_page_indices_d == nullptr ||
        kv_page_indptr_d == nullptr || kv_last_page_lens_d == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (total_tokens <= 0 || num_requests <= 0 || page_size <= 0 || num_kv_heads <= 0 ||
        head_dim <= 0)
        return PIE_ERR_INVALID_ARG;
    pie_cuda_device::kernels::write_kv_to_pages_bf16(
        k_pages, v_pages, k_curr, v_curr, qo_indptr_d, kv_page_indices_d, kv_page_indptr_d,
        kv_last_page_lens_d, total_tokens, num_requests, page_size, num_kv_heads, head_dim,
        hnd_layout != 0, ctx->stream);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return cuda_fail("pie_cuda_write_kv_to_pages_bf16 launch", e);
    return PIE_OK;
}

PieStatus pie_cuda_sample_temp_bf16(
    PieDevCtx* ctx, const void* logits, const float* temperatures, const float* top_ps,
    const int32_t* top_ks, const float* min_ps, const uint32_t* seeds, int32_t* out,
    int32_t num_rows, int32_t vocab) {
    // top_ps / top_ks / min_ps are optional (NULL = filter off).
    if (ctx == nullptr || logits == nullptr || temperatures == nullptr ||
        seeds == nullptr || out == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (num_rows <= 0 || vocab <= 0) return PIE_ERR_INVALID_ARG;
    pie_cuda_device::kernels::sample_temp_bf16(logits, temperatures, top_ps, top_ks, min_ps,
                                               seeds, out, num_rows, vocab, ctx->stream);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return cuda_fail("pie_cuda_sample_temp_bf16 launch", e);
    return PIE_OK;
}

PieStatus pie_cuda_cast_fp16_to_bf16(PieDevCtx* ctx, const void* src, void* dst, size_t n) {
    if (ctx == nullptr || src == nullptr || dst == nullptr) return PIE_ERR_INVALID_ARG;
    pie_cuda_device::kernels::cast_fp16_to_bf16(src, dst, n, ctx->stream);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return cuda_fail("pie_cuda_cast_fp16_to_bf16 launch", e);
    return PIE_OK;
}

PieStatus pie_cuda_cast_fp32_to_bf16(PieDevCtx* ctx, const void* src, void* dst, size_t n) {
    if (ctx == nullptr || src == nullptr || dst == nullptr) return PIE_ERR_INVALID_ARG;
    pie_cuda_device::kernels::cast_fp32_to_bf16(src, dst, n, ctx->stream);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return cuda_fail("pie_cuda_cast_fp32_to_bf16 launch", e);
    return PIE_OK;
}

PieStatus pie_cuda_cast_bf16_to_fp32(PieDevCtx* ctx, const void* src, void* dst, size_t n) {
    if (ctx == nullptr || src == nullptr || dst == nullptr) return PIE_ERR_INVALID_ARG;
    pie_cuda_device::kernels::cast_bf16_to_fp32(src, dst, n, ctx->stream);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return cuda_fail("pie_cuda_cast_bf16_to_fp32 launch", e);
    return PIE_OK;
}

PieStatus pie_cuda_gather_bf16_rows(
    PieDevCtx* ctx, const uint16_t* src, const int32_t* row_indices, uint16_t* dst,
    int32_t num_dst_rows, int32_t vocab) {
    if (ctx == nullptr || src == nullptr || row_indices == nullptr || dst == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (num_dst_rows <= 0 || vocab <= 0) return PIE_ERR_INVALID_ARG;
    pie_cuda_device::kernels::gather_bf16_rows(src, row_indices, dst, num_dst_rows, vocab,
                                               ctx->stream);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return cuda_fail("pie_cuda_gather_bf16_rows launch", e);
    return PIE_OK;
}

PieStatus pie_cuda_rmsnorm_gemma_bf16(PieDevCtx* ctx, const void* x, const void* weight,
                                      void* y, int32_t num_rows, int32_t hidden, float eps) {
    if (ctx == nullptr || x == nullptr || weight == nullptr || y == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (num_rows <= 0 || hidden <= 0) return PIE_ERR_INVALID_ARG;
    pie_cuda_device::kernels::rmsnorm_gemma_bf16(x, weight, y, num_rows, hidden, eps, ctx->stream);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return cuda_fail("pie_cuda_rmsnorm_gemma_bf16 launch", e);
    return PIE_OK;
}

PieStatus pie_cuda_geglu_tanh_bf16(PieDevCtx* ctx, const void* gate, const void* up,
                                   void* y, int32_t num_elements) {
    if (ctx == nullptr || gate == nullptr || up == nullptr || y == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (num_elements <= 0) return PIE_ERR_INVALID_ARG;
    pie_cuda_device::kernels::geglu_tanh_bf16(gate, up, y, num_elements, ctx->stream);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return cuda_fail("pie_cuda_geglu_tanh_bf16 launch", e);
    return PIE_OK;
}

PieStatus pie_cuda_logit_softcap_bf16(PieDevCtx* ctx, void* x, float cap, size_t n) {
    if (ctx == nullptr || x == nullptr) return PIE_ERR_INVALID_ARG;
    pie_cuda_device::kernels::logit_softcap_bf16(x, cap, n, ctx->stream);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return cuda_fail("pie_cuda_logit_softcap_bf16 launch", e);
    return PIE_OK;
}

PieStatus pie_cuda_dequant_fp8_e4m3_to_bf16(PieDevCtx* ctx, const uint8_t* fp8_in,
                                            void* bf16_out, float scale, size_t n) {
    if (ctx == nullptr || fp8_in == nullptr || bf16_out == nullptr) return PIE_ERR_INVALID_ARG;
    pie_cuda_device::kernels::dequant_fp8_e4m3_to_bf16(fp8_in, bf16_out, scale, n, ctx->stream);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return cuda_fail("pie_cuda_dequant_fp8_e4m3_to_bf16 launch", e);
    return PIE_OK;
}

PieStatus pie_cuda_rope_yarn_bf16(
    PieDevCtx* ctx, void* q, void* k, const int32_t* positions, int32_t num_tokens,
    int32_t num_q_heads, int32_t num_kv_heads, int32_t head_dim, float theta, float factor,
    float low_freq_factor, float high_freq_factor, int32_t original_max_position) {
    if (ctx == nullptr || q == nullptr || k == nullptr || positions == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (num_tokens <= 0 || num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
        (head_dim & 1))
        return PIE_ERR_INVALID_ARG;
    pie_cuda_device::kernels::rope_yarn_bf16(q, k, positions, num_tokens, num_q_heads,
                                             num_kv_heads, head_dim, theta, factor,
                                             low_freq_factor, high_freq_factor,
                                             original_max_position, ctx->stream);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return cuda_fail("pie_cuda_rope_yarn_bf16 launch", e);
    return PIE_OK;
}

PieStatus pie_cuda_topk_softmax_bf16(
    PieDevCtx* ctx, const void* logits, int32_t* topk_idx, float* topk_w, int32_t N,
    int32_t num_experts, int32_t K) {
    if (ctx == nullptr || logits == nullptr || topk_idx == nullptr || topk_w == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (N <= 0 || num_experts <= 0 || K <= 0 || K > num_experts) return PIE_ERR_INVALID_ARG;
    pie_cuda_device::kernels::topk_softmax_bf16(logits, topk_idx, topk_w, N, num_experts, K,
                                                ctx->stream);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return cuda_fail("pie_cuda_topk_softmax_bf16 launch", e);
    return PIE_OK;
}

PieStatus pie_cuda_chunked_swiglu_bf16(PieDevCtx* ctx, const void* packed, void* y,
                                       int32_t N, int32_t I) {
    if (ctx == nullptr || packed == nullptr || y == nullptr) return PIE_ERR_INVALID_ARG;
    if (N <= 0 || I <= 0) return PIE_ERR_INVALID_ARG;
    pie_cuda_device::kernels::chunked_swiglu_bf16(packed, y, N, I, ctx->stream);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return cuda_fail("pie_cuda_chunked_swiglu_bf16 launch", e);
    return PIE_OK;
}

PieStatus pie_cuda_rope_partial_bf16(PieDevCtx* ctx, void* q, void* k, const int32_t* positions,
    int32_t num_tokens, int32_t num_q_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t rotary_dim, float theta) {
    if (ctx == nullptr || q == nullptr || k == nullptr || positions == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (num_tokens <= 0 || num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
        rotary_dim <= 0 || rotary_dim > head_dim)
        return PIE_ERR_INVALID_ARG;
    pie_cuda_device::kernels::rope_partial_bf16(q, k, positions, num_tokens, num_q_heads,
                                                num_kv_heads, head_dim, rotary_dim, theta,
                                                ctx->stream);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return cuda_fail("pie_cuda_rope_partial_bf16 launch", e);
    return PIE_OK;
}

PieStatus pie_cuda_causal_conv1d_prefill_bf16(PieDevCtx* ctx, const void* x, const void* weight,
    const void* bias, void* y, int32_t N, int32_t C, int32_t K) {
    if (ctx == nullptr || x == nullptr || weight == nullptr || y == nullptr)
        return PIE_ERR_INVALID_ARG;  // bias may be null
    if (N <= 0 || C <= 0 || K <= 0) return PIE_ERR_INVALID_ARG;
    pie_cuda_device::kernels::causal_conv1d_prefill_bf16(x, weight, bias, y, /*state_out=*/nullptr,
                                                         N, C, K, ctx->stream);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return cuda_fail("pie_cuda_causal_conv1d_prefill_bf16 launch", e);
    return PIE_OK;
}

PieStatus pie_cuda_dequant_wna16_int4b8_to_bf16(PieDevCtx* ctx, const int32_t* packed,
    const void* scale_bf16, void* out_bf16, int32_t out_dim, int32_t in_dim, int32_t group_size) {
    if (ctx == nullptr || packed == nullptr || scale_bf16 == nullptr || out_bf16 == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (out_dim <= 0 || in_dim <= 0 || group_size <= 0) return PIE_ERR_INVALID_ARG;
    pie_cuda_device::kernels::dequant_wna16_int4b8_to_bf16(packed, scale_bf16, out_bf16, out_dim,
                                                           in_dim, group_size, ctx->stream);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return cuda_fail("pie_cuda_dequant_wna16_int4b8_to_bf16 launch", e);
    return PIE_OK;
}

PieStatus pie_cuda_llama_layer_bf16(
    PieDevCtx* ctx, void* hidden, const PieLlamaLayerWeights* w, const int32_t* positions,
    void* k_pages, void* v_pages, const uint32_t* qo_indptr_d,
    const uint32_t* kv_page_indices_d, const uint32_t* kv_page_indptr_d,
    const uint32_t* kv_last_page_lens_d, int32_t num_tokens, int32_t num_requests,
    int32_t hidden_size, int32_t n_q_heads, int32_t n_kv_heads, int32_t head_dim,
    int32_t intermediate, int32_t page_size, float rms_eps, float rope_theta) {
    if (ctx == nullptr || hidden == nullptr || w == nullptr || positions == nullptr ||
        k_pages == nullptr || v_pages == nullptr || qo_indptr_d == nullptr ||
        kv_page_indices_d == nullptr || kv_page_indptr_d == nullptr ||
        kv_last_page_lens_d == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (num_tokens <= 0 || num_requests <= 0 || hidden_size <= 0 || n_q_heads <= 0 ||
        n_kv_heads <= 0 || head_dim <= 0 || intermediate <= 0 || page_size <= 0)
        return PIE_ERR_INVALID_ARG;
    pie_cuda_device::forward::LlamaLayerWeights lw{
        w->attn_norm, w->wq, w->wk, w->wv, w->wo, w->ffn_norm, w->w_gate, w->w_up, w->w_down,
        w->q_norm, w->k_norm, w->q_bias, w->k_bias, w->v_bias};
    cudaError_t e = pie_cuda_device::forward::llama_layer_bf16(
        ctx->cublas, ctx->stream, hidden, lw, positions, k_pages, v_pages, qo_indptr_d,
        kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d, num_tokens, num_requests,
        hidden_size, n_q_heads, n_kv_heads, head_dim, intermediate, page_size, rms_eps,
        rope_theta);
    if (e != cudaSuccess) return cuda_fail("pie_cuda_llama_layer_bf16", e);
    return PIE_OK;
}

PieStatus pie_cuda_llama_forward_bf16(
    PieDevCtx* ctx, PieWorkspace* ws, const int32_t* token_ids, const PieLlamaWeights* w,
    const int32_t* positions, void* kv_k, void* kv_v, const uint32_t* qo_indptr_d,
    const uint32_t* kv_page_indices_d, const uint32_t* kv_page_indptr_d,
    const uint32_t* kv_last_page_lens_d, void* out_logits, int32_t* out_token_ids,
    int32_t num_tokens, int32_t num_requests, int32_t hidden_size, int32_t n_q_heads,
    int32_t n_kv_heads, int32_t head_dim, int32_t intermediate, int32_t page_size,
    int32_t num_kv_pages, int32_t vocab, float rms_eps, float rope_theta) {
    if (ctx == nullptr || ws == nullptr || token_ids == nullptr || w == nullptr ||
        positions == nullptr || kv_k == nullptr || kv_v == nullptr || qo_indptr_d == nullptr ||
        kv_page_indices_d == nullptr || kv_page_indptr_d == nullptr ||
        kv_last_page_lens_d == nullptr || out_logits == nullptr || out_token_ids == nullptr ||
        w->layers == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (num_tokens <= 0 || w->n_layers <= 0 || hidden_size <= 0 || vocab <= 0 || page_size <= 0)
        return PIE_ERR_INVALID_ARG;

    // Map the C ABI weight array → the C++ forward struct (identical layout,
    // distinct types).
    std::vector<pie_cuda_device::forward::LlamaLayerWeights> layers((size_t)w->n_layers);
    for (int L = 0; L < w->n_layers; ++L) {
        const PieLlamaLayerWeights& src = w->layers[L];
        layers[(size_t)L] = {src.attn_norm, src.wq, src.wk, src.wv, src.wo,
                             src.ffn_norm, src.w_gate, src.w_up, src.w_down,
                             src.q_norm, src.k_norm, src.q_bias, src.k_bias, src.v_bias};
    }
    pie_cuda_device::forward::LlamaWeights lw{
        w->embed, layers.data(), w->n_layers, w->final_norm, w->lm_head};
    cudaError_t e = pie_cuda_device::forward::llama_forward_bf16(
        ctx->cublas, ctx->stream, ws, token_ids, lw, positions, kv_k, kv_v, qo_indptr_d,
        kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d, out_logits, out_token_ids,
        num_tokens, num_requests, hidden_size, n_q_heads, n_kv_heads, head_dim, intermediate,
        page_size, num_kv_pages, vocab, rms_eps, rope_theta);
    if (e != cudaSuccess) return cuda_fail("pie_cuda_llama_forward_bf16", e);
    return PIE_OK;
}

PieStatus pie_cuda_moe_mlp_block_bf16(
    PieDevCtx* ctx, const void* hidden, const void* router_w, const void* wgu,
    const void* wdown, void* out, int32_t num_tokens, int32_t hidden_size,
    int32_t intermediate, int32_t num_experts, int32_t top_k) {
    if (ctx == nullptr || hidden == nullptr || router_w == nullptr || wgu == nullptr ||
        wdown == nullptr || out == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (num_tokens <= 0 || hidden_size <= 0 || intermediate <= 0 || num_experts <= 0 ||
        top_k <= 0 || top_k > num_experts)
        return PIE_ERR_INVALID_ARG;
    cudaError_t e = pie_cuda_device::forward::moe_mlp_block_bf16(
        ctx->cublas, ctx->stream, hidden, router_w, wgu, wdown, out, num_tokens, hidden_size,
        intermediate, num_experts, top_k);
    if (e != cudaSuccess) return cuda_fail("pie_cuda_moe_mlp_block_bf16", e);
    return PIE_OK;
}

PieStatus pie_cuda_mla_block_bf16(
    PieDevCtx* ctx, void* hidden, const PieMlaLayerWeights* w, const int32_t* positions,
    void* ckv_pages, void* kpe_pages, const uint32_t* qo_indptr_d,
    const uint32_t* kv_page_indices_d, const uint32_t* kv_page_indptr_d,
    const uint32_t* kv_last_page_lens_d, int32_t num_tokens, int32_t num_requests,
    int32_t hidden_size, int32_t num_heads, int32_t q_lora_rank, int32_t kv_lora_rank,
    int32_t qk_nope_head_dim, int32_t qk_rope_head_dim, int32_t v_head_dim,
    int32_t page_size, float rms_eps, float sm_scale, float rope_theta) {
    if (ctx == nullptr || hidden == nullptr || w == nullptr || positions == nullptr ||
        ckv_pages == nullptr || kpe_pages == nullptr || qo_indptr_d == nullptr ||
        kv_page_indices_d == nullptr || kv_page_indptr_d == nullptr ||
        kv_last_page_lens_d == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (num_tokens <= 0 || num_requests <= 0 || hidden_size <= 0 || num_heads <= 0 ||
        q_lora_rank <= 0 || kv_lora_rank <= 0 || qk_nope_head_dim <= 0 ||
        qk_rope_head_dim <= 0 || v_head_dim <= 0 || page_size <= 0)
        return PIE_ERR_INVALID_ARG;
    pie_cuda_device::forward::MlaLayerWeights mw{
        w->attn_norm, w->w_q_a, w->q_a_ln, w->w_q_b, w->w_kv_a, w->kv_a_ln,
        w->w_uk, w->w_uv, w->w_o};
    cudaError_t e = pie_cuda_device::forward::mla_block_bf16(
        ctx->cublas, ctx->stream, hidden, mw, positions, ckv_pages, kpe_pages, qo_indptr_d,
        kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d, num_tokens, num_requests,
        hidden_size, num_heads, q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim,
        v_head_dim, page_size, rms_eps, sm_scale, rope_theta);
    if (e != cudaSuccess) return cuda_fail("pie_cuda_mla_block_bf16", e);
    return PIE_OK;
}

PieStatus pie_cuda_mla_forward_bf16(
    PieDevCtx* ctx, const int32_t* token_ids, const PieMlaWeights* w, const int32_t* positions,
    void* ckv_pages, void* kpe_pages, const uint32_t* qo_indptr_d,
    const uint32_t* kv_page_indices_d, const uint32_t* kv_page_indptr_d,
    const uint32_t* kv_last_page_lens_d, void* out_logits, int32_t* out_token_ids,
    int32_t num_tokens, int32_t num_requests, int32_t hidden_size, int32_t num_heads,
    int32_t q_lora_rank, int32_t kv_lora_rank, int32_t qk_nope_head_dim,
    int32_t qk_rope_head_dim, int32_t v_head_dim, int32_t vocab, int32_t page_size,
    int32_t num_pages, float rms_eps, float sm_scale, float rope_theta) {
    if (ctx == nullptr || token_ids == nullptr || w == nullptr || positions == nullptr ||
        ckv_pages == nullptr || kpe_pages == nullptr || qo_indptr_d == nullptr ||
        kv_page_indices_d == nullptr || kv_page_indptr_d == nullptr ||
        kv_last_page_lens_d == nullptr || out_logits == nullptr || out_token_ids == nullptr ||
        w->layers == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (num_tokens <= 0 || num_requests <= 0 || w->n_layers <= 0 || hidden_size <= 0 ||
        vocab <= 0 || page_size <= 0 || kv_lora_rank <= 0)
        return PIE_ERR_INVALID_ARG;

    // Map the C ABI per-layer weight array → the C++ forward struct.
    std::vector<pie_cuda_device::forward::MlaLayerWeights> layers((size_t)w->n_layers);
    for (int L = 0; L < w->n_layers; ++L) {
        const PieMlaLayerWeights& s = w->layers[L];
        layers[(size_t)L] = {s.attn_norm, s.w_q_a, s.q_a_ln, s.w_q_b, s.w_kv_a, s.kv_a_ln,
                             s.w_uk, s.w_uv, s.w_o};
    }
    pie_cuda_device::forward::MlaForwardWeights fw{
        w->embed, layers.data(), w->final_norm, w->lm_head};
    pie_cuda_device::forward::MlaForwardDims dims{
        w->n_layers, hidden_size, num_heads, q_lora_rank, kv_lora_rank, qk_nope_head_dim,
        qk_rope_head_dim, v_head_dim, vocab, page_size, num_pages, rms_eps, sm_scale, rope_theta};
    cudaError_t e = pie_cuda_device::forward::mla_forward_bf16(
        ctx->cublas, ctx->stream, token_ids, fw, positions, ckv_pages, kpe_pages, qo_indptr_d,
        kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d, out_logits, out_token_ids,
        num_tokens, num_requests, dims);
    if (e != cudaSuccess) return cuda_fail("pie_cuda_mla_forward_bf16", e);
    return PIE_OK;
}

PieStatus pie_cuda_deepseek_forward_bf16(
    PieDevCtx* ctx, const int32_t* token_ids, const PieDeepseekWeights* w,
    const int32_t* positions, void* ckv_pages, void* kpe_pages, const uint32_t* qo_indptr_d,
    const uint32_t* kv_page_indices_d, const uint32_t* kv_page_indptr_d,
    const uint32_t* kv_last_page_lens_d, void* out_logits, int32_t* out_token_ids,
    int32_t num_tokens, int32_t num_requests, int32_t first_k_dense, int32_t hidden_size,
    int32_t num_heads, int32_t q_lora_rank, int32_t kv_lora_rank, int32_t qk_nope_head_dim,
    int32_t qk_rope_head_dim, int32_t v_head_dim, int32_t dense_inter, int32_t moe_inter,
    int32_t num_experts, int32_t top_k, int32_t vocab, int32_t page_size, int32_t num_pages,
    float rms_eps, float sm_scale, float rope_theta) {
    if (ctx == nullptr || token_ids == nullptr || w == nullptr || positions == nullptr ||
        ckv_pages == nullptr || kpe_pages == nullptr || qo_indptr_d == nullptr ||
        kv_page_indices_d == nullptr || kv_page_indptr_d == nullptr ||
        kv_last_page_lens_d == nullptr || out_logits == nullptr || out_token_ids == nullptr ||
        w->layers == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (num_tokens <= 0 || num_requests <= 0 || w->n_layers <= 0 || hidden_size <= 0 ||
        kv_lora_rank <= 0 || vocab <= 0 || page_size <= 0)
        return PIE_ERR_INVALID_ARG;

    std::vector<pie_cuda_device::forward::DeepseekLayerWeights> layers((size_t)w->n_layers);
    for (int L = 0; L < w->n_layers; ++L) {
        const PieDeepseekLayerWeights& s = w->layers[L];
        const PieMlaLayerWeights& a = s.attn;
        pie_cuda_device::forward::MlaLayerWeights attn{
            a.attn_norm, a.w_q_a, a.q_a_ln, a.w_q_b, a.w_kv_a, a.kv_a_ln, a.w_uk, a.w_uv, a.w_o};
        layers[(size_t)L] = {attn, s.ffn_norm, s.w_gate, s.w_up, s.w_down, s.router_w, s.wgu,
                             s.wdown};
    }
    pie_cuda_device::forward::DeepseekWeights fw{
        w->embed, layers.data(), w->final_norm, w->lm_head};
    pie_cuda_device::forward::DeepseekDims dims{
        w->n_layers, first_k_dense, hidden_size, num_heads, q_lora_rank, kv_lora_rank,
        qk_nope_head_dim, qk_rope_head_dim, v_head_dim, dense_inter, moe_inter, num_experts,
        top_k, vocab, page_size, num_pages, rms_eps, sm_scale, rope_theta};
    cudaError_t e = pie_cuda_device::forward::deepseek_forward_bf16(
        ctx->cublas, ctx->stream, token_ids, fw, positions, ckv_pages, kpe_pages, qo_indptr_d,
        kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d, out_logits, out_token_ids,
        num_tokens, num_requests, dims);
    if (e != cudaSuccess) return cuda_fail("pie_cuda_deepseek_forward_bf16", e);
    return PIE_OK;
}

PieStatus pie_cuda_gemma_forward_bf16(
    PieDevCtx* ctx, PieWorkspace* ws, const int32_t* token_ids, const PieGemmaWeights* w,
    const int32_t* positions,
    void* k_pages, void* v_pages, const uint32_t* qo_indptr_d, const uint32_t* kv_page_indices_d,
    const uint32_t* kv_page_indptr_d, const uint32_t* kv_last_page_lens_d, void* out_logits,
    int32_t* out_token_ids, int32_t num_tokens, int32_t num_requests, int32_t hidden_size,
    int32_t n_q_heads, int32_t n_kv_heads, int32_t head_dim, int32_t intermediate, int32_t vocab,
    int32_t page_size, int32_t num_pages, const int32_t* window_left_host, int32_t window_left_all,
    float attn_logit_softcap, float final_logit_softcap, float embed_scale, float rms_eps,
    float rope_theta, int32_t qk_norm, int32_t altup_num_inputs) {
    if (ctx == nullptr || ws == nullptr || token_ids == nullptr || w == nullptr || positions == nullptr ||
        k_pages == nullptr || v_pages == nullptr || qo_indptr_d == nullptr ||
        kv_page_indices_d == nullptr || kv_page_indptr_d == nullptr ||
        kv_last_page_lens_d == nullptr || out_logits == nullptr || out_token_ids == nullptr ||
        w->layers == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (num_tokens <= 0 || num_requests <= 0 || w->n_layers <= 0 || hidden_size <= 0 ||
        vocab <= 0 || page_size <= 0)
        return PIE_ERR_INVALID_ARG;

    std::vector<pie_cuda_device::forward::GemmaLayerWeights> layers((size_t)w->n_layers);
    for (int L = 0; L < w->n_layers; ++L) {
        const PieGemmaLayerWeights& s = w->layers[L];
        layers[(size_t)L] = {s.input_ln, s.post_attn_ln, s.pre_ffn_ln, s.post_ffn_ln, s.wq, s.wk,
                             s.wv, s.wo, s.w_gate, s.w_up, s.w_down};
    }
    pie_cuda_device::forward::GemmaForwardWeights fw{
        w->embed, layers.data(), w->final_norm, w->lm_head};
    pie_cuda_device::forward::GemmaForwardDims dims{
        w->n_layers, hidden_size, n_q_heads, n_kv_heads, head_dim, intermediate, vocab, page_size,
        num_pages, window_left_host, window_left_all, attn_logit_softcap, final_logit_softcap,
        embed_scale, rms_eps, rope_theta, qk_norm, altup_num_inputs};
    cudaError_t e = pie_cuda_device::forward::gemma_forward_bf16(
        ctx->cublas, ctx->stream, ws, token_ids, fw, positions, k_pages, v_pages, qo_indptr_d,
        kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d, out_logits, out_token_ids,
        num_tokens, num_requests, dims);
    if (e != cudaSuccess) return cuda_fail("pie_cuda_gemma_forward_bf16", e);
    return PIE_OK;
}

PieStatus pie_cuda_nemotron_forward_bf16(
    PieDevCtx* ctx, const int32_t* token_ids, const PieNemotronWeights* w,
    const int32_t* positions, void* out_logits, int32_t* out_token_ids, int32_t num_tokens,
    const char* kinds_host, int32_t hidden_size, int32_t vocab, int32_t mamba_num_heads,
    int32_t mamba_head_dim, int32_t mamba_state_size, int32_t mamba_n_groups,
    int32_t mamba_conv_kernel, float time_step_min, int32_t attn_n_q_heads,
    int32_t attn_n_kv_heads, int32_t attn_head_dim, int32_t page_size, float rope_theta,
    int32_t ffn_intermediate, float rms_eps) {
    if (ctx == nullptr || token_ids == nullptr || w == nullptr || positions == nullptr ||
        out_logits == nullptr || out_token_ids == nullptr || w->layers == nullptr ||
        kinds_host == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (num_tokens <= 0 || w->n_layers <= 0 || hidden_size <= 0 || vocab <= 0)
        return PIE_ERR_INVALID_ARG;

    std::vector<pie_cuda_device::forward::NemotronLayerWeights> layers((size_t)w->n_layers);
    for (int L = 0; L < w->n_layers; ++L) {
        const PieNemotronLayerWeights& s = w->layers[L];
        const PieNemotronMambaWeights& m = s.mamba;
        layers[(size_t)L] = {
            s.kind, s.mamba_pre_norm,
            {m.in_proj_w, m.conv_w, m.conv_bias, m.a_log, m.d, m.dt_bias, m.norm_weight, m.out_proj_w},
            {s.attn.attn_norm, s.attn.wq, s.attn.wk, s.attn.wv, s.attn.wo},
            {s.ffn.ffn_norm, s.ffn.w_gate, s.ffn.w_up, s.ffn.w_down}};
    }
    pie_cuda_device::forward::NemotronForwardWeights fw{
        w->embed, layers.data(), w->final_norm, w->lm_head};
    pie_cuda_device::forward::NemotronForwardDims dims{
        w->n_layers, kinds_host, hidden_size, vocab, mamba_num_heads, mamba_head_dim,
        mamba_state_size, mamba_n_groups, mamba_conv_kernel, time_step_min, attn_n_q_heads,
        attn_n_kv_heads, attn_head_dim, page_size, rope_theta, ffn_intermediate, rms_eps};
    cudaError_t e = pie_cuda_device::forward::nemotron_forward_bf16(
        ctx->cublas, ctx->stream, token_ids, fw, positions, out_logits, out_token_ids, num_tokens,
        dims);
    if (e != cudaSuccess) return cuda_fail("pie_cuda_nemotron_forward_bf16", e);
    return PIE_OK;
}

PieStatus pie_cuda_altup_predict_bf16(
    PieDevCtx* ctx, const void* streams, const float* coefs, void* predictions,
    int32_t k_streams, int32_t num_tokens, int32_t hidden_size) {
    if (ctx == nullptr || streams == nullptr || coefs == nullptr || predictions == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (k_streams <= 0 || num_tokens <= 0 || hidden_size <= 0) return PIE_ERR_INVALID_ARG;
    pie_cuda_device::kernels::altup_predict_bf16(streams, coefs, predictions,
                                                 k_streams, num_tokens, hidden_size, ctx->stream);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return cuda_fail("pie_cuda_altup_predict_bf16 launch", e);
    return PIE_OK;
}

PieStatus pie_cuda_altup_correct_bf16(
    PieDevCtx* ctx, const void* predictions, const void* activated,
    const float* correction_coefs_p1, void* corrected, int32_t k_streams,
    int32_t num_tokens, int32_t hidden_size, int32_t active_idx) {
    if (ctx == nullptr || predictions == nullptr || activated == nullptr ||
        correction_coefs_p1 == nullptr || corrected == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (k_streams <= 0 || num_tokens <= 0 || hidden_size <= 0 ||
        active_idx < 0 || active_idx >= k_streams)
        return PIE_ERR_INVALID_ARG;
    pie_cuda_device::kernels::altup_correct_bf16(predictions, activated, correction_coefs_p1,
                                                 corrected, k_streams, num_tokens, hidden_size,
                                                 active_idx, ctx->stream);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return cuda_fail("pie_cuda_altup_correct_bf16 launch", e);
    return PIE_OK;
}

PieStatus pie_cuda_grouped_gemm_bf16(
    PieDevCtx* ctx, const void* x, const void* w, const int32_t* expert_offsets_host,
    void* y, int32_t total_rows, int32_t num_experts, int32_t n_out, int32_t k_in) {
    if (ctx == nullptr || x == nullptr || w == nullptr || expert_offsets_host == nullptr ||
        y == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (total_rows <= 0 || num_experts <= 0 || n_out <= 0 || k_in <= 0)
        return PIE_ERR_INVALID_ARG;
    cudaError_t e = pie_cuda_device::ops::grouped_gemm_bf16(
        ctx->cublas, ctx->stream, x, w, expert_offsets_host, y, total_rows, num_experts, n_out,
        k_in);
    if (e != cudaSuccess) return cuda_fail("pie_cuda_grouped_gemm_bf16", e);
    return PIE_OK;
}

PieStatus pie_cuda_moe_forward_bf16(
    PieDevCtx* ctx, const int32_t* token_ids, const PieMoeWeights* w, const int32_t* positions,
    void* kv_k, void* kv_v, const uint32_t* qo_indptr_d, const uint32_t* kv_page_indices_d,
    const uint32_t* kv_page_indptr_d, const uint32_t* kv_last_page_lens_d, void* out_logits,
    int32_t* out_token_ids, int32_t num_tokens, int32_t num_requests, int32_t num_kv_pages,
    int32_t hidden_size, int32_t n_q_heads, int32_t n_kv_heads, int32_t head_dim,
    int32_t intermediate, int32_t num_experts, int32_t top_k, int32_t vocab, int32_t page_size,
    float rms_eps, float rope_theta) {
    if (ctx == nullptr || token_ids == nullptr || w == nullptr || positions == nullptr ||
        kv_k == nullptr || kv_v == nullptr || qo_indptr_d == nullptr ||
        kv_page_indices_d == nullptr || kv_page_indptr_d == nullptr ||
        kv_last_page_lens_d == nullptr || out_logits == nullptr || out_token_ids == nullptr ||
        w->layers == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (num_tokens <= 0 || num_requests <= 0 || w->n_layers <= 0 || hidden_size <= 0 ||
        num_experts <= 0 || top_k <= 0 || top_k > num_experts || vocab <= 0 || page_size <= 0)
        return PIE_ERR_INVALID_ARG;

    std::vector<pie_cuda_device::forward::MoeLayerWeights> layers((size_t)w->n_layers);
    for (int L = 0; L < w->n_layers; ++L) {
        const PieMoeLayerWeights& s = w->layers[L];
        layers[(size_t)L] = {s.attn_norm, s.wq, s.wk, s.wv, s.wo, s.ffn_norm, s.router_w,
                             s.wgu, s.wdown};
    }
    pie_cuda_device::forward::MoeForwardWeights fw{
        w->embed, layers.data(), w->n_layers, w->final_norm, w->lm_head};
    pie_cuda_device::forward::MoeForwardDims dims{
        hidden_size, n_q_heads, n_kv_heads, head_dim, intermediate, num_experts, top_k,
        w->n_layers, vocab, page_size, rms_eps, rope_theta};
    cudaError_t e = pie_cuda_device::forward::moe_forward_bf16(
        ctx->cublas, ctx->stream, token_ids, fw, positions, kv_k, kv_v, qo_indptr_d,
        kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d, out_logits, out_token_ids,
        num_tokens, num_requests, num_kv_pages, dims);
    if (e != cudaSuccess) return cuda_fail("pie_cuda_moe_forward_bf16", e);
    return PIE_OK;
}

PieStatus pie_cuda_ssm_selective_scan_bf16(
    PieDevCtx* ctx, const void* conv_out, const void* dt, const float* a, const float* d,
    const float* dt_bias, const float* dt_precomputed, const float* da_precomputed,
    void* ssm_state_base, const int32_t* slot_ids, const uint32_t* qo_indptr, void* y,
    int32_t num_requests, int32_t num_heads, int32_t head_dim, int32_t state_size,
    int32_t n_groups, int32_t conv_dim, int32_t intermediate, float time_step_min) {
    if (ctx == nullptr || conv_out == nullptr || dt == nullptr || a == nullptr || d == nullptr ||
        dt_bias == nullptr || ssm_state_base == nullptr || qo_indptr == nullptr || y == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (num_requests <= 0 || num_heads <= 0 || head_dim <= 0 || state_size <= 0 ||
        n_groups <= 0 || conv_dim <= 0 || intermediate <= 0)
        return PIE_ERR_INVALID_ARG;
    pie_cuda_device::kernels::ssm_selective_scan_bf16(
        conv_out, dt, a, d, dt_bias, dt_precomputed, da_precomputed, ssm_state_base, slot_ids,
        qo_indptr, y, num_requests, num_heads, head_dim, state_size, n_groups, conv_dim,
        intermediate, time_step_min, ctx->stream);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        return cuda_fail("pie_cuda_ssm_selective_scan_bf16 launch", e);
    return PIE_OK;
}

PieStatus pie_cuda_qgemm_w4a16_bf16(
    PieDevCtx* ctx, const void* act_bf16, const int32_t* qweight_packed,
    const void* scales_bf16, void* out_bf16, int32_t m, int32_t n, int32_t k,
    int32_t group_size, int32_t* workspace, int32_t sms) {
    if (ctx == nullptr || act_bf16 == nullptr || qweight_packed == nullptr ||
        scales_bf16 == nullptr || out_bf16 == nullptr || workspace == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (m <= 0 || n <= 0 || k <= 0) return PIE_ERR_INVALID_ARG;
    cudaError_t e = pie_cuda_device::qgemm::w4a16_bf16_gemm(
        ctx->stream, act_bf16, qweight_packed, scales_bf16, out_bf16, m, n, k, group_size,
        workspace, sms);
    if (e != cudaSuccess) return cuda_fail("pie_cuda_qgemm_w4a16_bf16", e);
    return PIE_OK;
}

PieStatus pie_cuda_qgemm_w4a16_repack(
    PieDevCtx* ctx, const int32_t* qweight_rowmajor_packed, int32_t* qweight_out,
    int32_t n, int32_t k) {
    if (ctx == nullptr || qweight_rowmajor_packed == nullptr || qweight_out == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (n <= 0 || k <= 0) return PIE_ERR_INVALID_ARG;
    cudaError_t e = pie_cuda_device::qgemm::w4a16_repack(
        ctx->stream, qweight_rowmajor_packed, qweight_out, n, k);
    if (e != cudaSuccess) return cuda_fail("pie_cuda_qgemm_w4a16_repack", e);
    return PIE_OK;
}

int32_t pie_cuda_qgemm_w4a16_workspace_ints(int32_t n, int32_t max_m) {
    return pie_cuda_device::qgemm::w4a16_workspace_ints(n, max_m);
}

PieStatus pie_cuda_qgemm_w8a16_fp8_bf16(
    PieDevCtx* ctx, const void* act_bf16, const void* qweight_fp8, const void* scales_bf16,
    void* out_bf16, int32_t m, int32_t n, int32_t k, int32_t group_size, int32_t* workspace,
    int32_t sms) {
    if (ctx == nullptr || act_bf16 == nullptr || qweight_fp8 == nullptr ||
        scales_bf16 == nullptr || out_bf16 == nullptr || workspace == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (m <= 0 || n <= 0 || k <= 0) return PIE_ERR_INVALID_ARG;
    cudaError_t e = pie_cuda_device::qgemm::w8a16_fp8_bf16_gemm(
        ctx->stream, act_bf16, qweight_fp8, scales_bf16, out_bf16, m, n, k, group_size, workspace,
        sms);
    if (e != cudaSuccess) return cuda_fail("pie_cuda_qgemm_w8a16_fp8_bf16", e);
    return PIE_OK;
}

PieStatus pie_cuda_qgemm_w8a16_fp8_repack(
    PieDevCtx* ctx, const int32_t* qweight_rowmajor_packed, int32_t* qweight_out, int32_t n,
    int32_t k) {
    if (ctx == nullptr || qweight_rowmajor_packed == nullptr || qweight_out == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (n <= 0 || k <= 0) return PIE_ERR_INVALID_ARG;
    cudaError_t e = pie_cuda_device::qgemm::w8a16_fp8_repack(
        ctx->stream, qweight_rowmajor_packed, qweight_out, n, k);
    if (e != cudaSuccess) return cuda_fail("pie_cuda_qgemm_w8a16_fp8_repack", e);
    return PIE_OK;
}

int32_t pie_cuda_qgemm_w8a16_fp8_workspace_ints(int32_t n, int32_t max_m) {
    return pie_cuda_device::qgemm::w8a16_fp8_workspace_ints(n, max_m);
}

PieStatus pie_cuda_moe_sparse_block_bf16(
    PieDevCtx* ctx, const void* hidden, const void* router_w, const void* wgu,
    const void* wdown, void* out, int32_t num_tokens, int32_t hidden_size,
    int32_t intermediate, int32_t num_experts, int32_t top_k) {
    if (ctx == nullptr || hidden == nullptr || router_w == nullptr || wgu == nullptr ||
        wdown == nullptr || out == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (num_tokens <= 0 || hidden_size <= 0 || intermediate <= 0 || num_experts <= 0 ||
        top_k <= 0 || top_k > num_experts)
        return PIE_ERR_INVALID_ARG;
    cudaError_t e = pie_cuda_device::forward::moe_sparse_block_bf16(
        ctx->cublas, ctx->stream, hidden, router_w, wgu, wdown, out, num_tokens, hidden_size,
        intermediate, num_experts, top_k);
    if (e != cudaSuccess) return cuda_fail("pie_cuda_moe_sparse_block_bf16", e);
    return PIE_OK;
}

PieStatus pie_cuda_nemotron_mamba_block_bf16(
    PieDevCtx* ctx, void* hidden, const PieNemotronMambaWeights* w, int32_t num_tokens,
    int32_t hidden_size, int32_t num_heads, int32_t head_dim, int32_t state_size,
    int32_t n_groups, int32_t conv_kernel, float rms_eps, float time_step_min) {
    if (ctx == nullptr || hidden == nullptr || w == nullptr || w->in_proj_w == nullptr ||
        w->conv_w == nullptr || w->a_log == nullptr || w->d == nullptr ||
        w->dt_bias == nullptr || w->norm_weight == nullptr || w->out_proj_w == nullptr)
        return PIE_ERR_INVALID_ARG;
    if (num_tokens <= 0 || hidden_size <= 0 || num_heads <= 0 || head_dim <= 0 ||
        state_size <= 0 || n_groups <= 0 || conv_kernel <= 0)
        return PIE_ERR_INVALID_ARG;
    pie_cuda_device::forward::NemotronMambaWeights mw{
        w->in_proj_w, w->conv_w, w->conv_bias, w->a_log, w->d, w->dt_bias, w->norm_weight,
        w->out_proj_w};
    cudaError_t e = pie_cuda_device::forward::nemotron_mamba_block_bf16(
        ctx->cublas, ctx->stream, hidden, mw, num_tokens, hidden_size, num_heads, head_dim,
        state_size, n_groups, conv_kernel, rms_eps, time_step_min);
    if (e != cudaSuccess) return cuda_fail("pie_cuda_nemotron_mamba_block_bf16", e);
    return PIE_OK;
}

// --- construction (phase 2 in Rust drives these; impl ported phase 1) ---

PieStatus pie_weights_bind(PieDevCtx*, PieArchId, void*, PieWeights** out) {
    if (out) *out = nullptr;
    return not_yet("pie_weights_bind", "model/loaded_model.cpp + loader/");
}
PieStatus pie_weights_destroy(PieWeights*) { return PIE_OK; }

PieStatus pie_kv_alloc(PieDevCtx*, const PieKvLayout*, PieKvCache** out) {
    if (out) *out = nullptr;
    return not_yet("pie_kv_alloc", "kv_cache.cpp / mla_cache.cpp");
}
PieStatus pie_kv_destroy(PieKvCache*) { return PIE_OK; }

PieStatus pie_ws_alloc(PieDevCtx* ctx, const PieWorkspaceDims* dims, PieWorkspace** out) {
    if (ctx == nullptr || dims == nullptr || out == nullptr) return PIE_ERR_INVALID_ARG;
    *out = nullptr;
    if (dims->max_tokens <= 0 || dims->hidden_size <= 0) return PIE_ERR_INVALID_ARG;
    if (cudaError_t e = cudaSetDevice(ctx->device_ordinal); e != cudaSuccess)
        return cuda_fail("cudaSetDevice", e);

    auto* ws = new PieWorkspace{};
    ws->max_tokens = dims->max_tokens;
    ws->hidden = dims->hidden_size;
    ws->n_q_heads = dims->num_heads;
    ws->n_kv_heads = dims->num_kv_heads;
    ws->head_dim = dims->head_dim;
    ws->intermediate = dims->intermediate_size;
    ws->vocab = dims->vocab_size;

    const size_t es = sizeof(std::uint16_t);  // bf16
    const size_t T = (size_t)dims->max_tokens;
    const size_t H = (size_t)dims->hidden_size;
    const size_t Hq = (size_t)dims->num_heads * dims->head_dim;
    const size_t Hkv = (size_t)dims->num_kv_heads * dims->head_dim;
    const size_t I = (size_t)dims->intermediate_size;
    void** ptrs[11] = {&ws->hidden_buf, &ws->normed, &ws->q, &ws->k, &ws->v, &ws->attn,
                       &ws->o, &ws->gate, &ws->up, &ws->mlp, &ws->mlp_out};
    const size_t bytes[11] = {T * H * es, T * H * es, T * Hq * es, T * Hkv * es, T * Hkv * es,
                              T * Hq * es, T * H * es, T * I * es, T * I * es, T * I * es,
                              T * H * es};
    for (int i = 0; i < 11; ++i) *ptrs[i] = nullptr;
    for (int i = 0; i < 11; ++i) {
        if (cudaMalloc(ptrs[i], bytes[i]) != cudaSuccess) {
            for (int j = 0; j < 11; ++j)
                if (*ptrs[j]) cudaFree(*ptrs[j]);
            delete ws;
            set_last_error("pie_ws_alloc: cudaMalloc failed");
            return PIE_ERR_OOM;
        }
    }
    *out = ws;
    return PIE_OK;
}

PieStatus pie_ws_destroy(PieWorkspace* ws) {
    if (ws == nullptr) return PIE_OK;
    void* ptrs[11] = {ws->hidden_buf, ws->normed, ws->q, ws->k, ws->v, ws->attn,
                      ws->o, ws->gate, ws->up, ws->mlp, ws->mlp_out};
    for (void* p : ptrs)
        if (p) cudaFree(p);
    delete ws;
    return PIE_OK;
}

size_t pie_kv_page_bytes(const PieKvLayout*) {
    // Pure arithmetic — ported from kv_cache.cpp::kv_cache_device_bytes_per_page.
    return 0;
}

// --- hot path (impl ported phase 1; driven by Rust executor phase 3) ---

PieStatus pie_upload_inputs(PieWorkspace*, const PieForwardInputs*) {
    return not_yet("pie_upload_inputs", "executor/persistent_inputs.cpp");
}
PieStatus pie_prepare(PieArchId, PieWorkspace*, const PiePrepareInputs*) {
    return not_yet("pie_prepare", "model/<arch>_model.cpp::prepare (IModel)");
}
PieStatus pie_body(PieArchId, PieWeights*, PieWorkspace*, PieKvCache*,
                   const PieForwardInputs*) {
    return not_yet("pie_body", "model/<arch>_forward.cpp + ops/ + kernels/");
}
PieStatus pie_sample(PieWorkspace*, const PieSampleParams*, int32_t*) {
    return not_yet("pie_sample", "sampling_dispatch.cpp + kernels/sample_*.cu");
}

// --- graph (mechanics ported phase 1; policy in Rust phase 3) ---

PieStatus pie_graph_capture(PieDevCtx*, PieArchId, PieWeights*, PieWorkspace*,
                            PieKvCache*, const PieForwardInputs*,
                            PieGraphExec** out) {
    if (out) *out = nullptr;
    return not_yet("pie_graph_capture", "executor/forward_graph + executor.cpp capture");
}
PieStatus pie_graph_launch(PieGraphExec*, PieDevCtx*) {
    return not_yet("pie_graph_launch", "executor.cpp graph replay");
}
PieStatus pie_graph_destroy(PieGraphExec*) { return PIE_OK; }

}  // extern "C"

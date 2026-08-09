// The two host bridges `qwen3_vl_tower.cu` forward-declares:
// the cuBLAS GEMM and the flashinfer vision attention. ADAPTED from
// driver-cuda/csrc/src/model/qwen3_vl/qwen3_vl_vision_adapter.cpp
// (2026-08-09, the VL tower bridge) with ONE change: the dedicated
// attention workspace is raw-allocated here (32 MiB float / 16 MiB int /
// a pinned int mirror — the same sizes the driver passes its
// `AttentionWorkspace`), because the owning class is a DRIVER object and
// this archive only knows the view. Everything else — the paged
// single-plan layout, the shape-keyed caching, the non-causal plan — is
// the adapter's, verbatim. The OLD driver keeps its copy until phase E
// deletes it. Do not diverge.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "attention_workspace_view.hpp"
#include "attn/attention_flashinfer.hpp"
#include "gemm/gemm.hpp"

namespace pie_cuda_driver::model {

// y[M,N] = x[M,K] @ W[N,K]^T (W row-major [N,K]); bf16 in/out, fp32
// accumulate. beta=1 fuses a residual add (the o-/fc2-projection
// epilogues).
void qwen3vl_vis_gemm_bf16(cublasHandle_t blas, const void* x, const void* W,
                           void* y, int M, int N, int K, float beta) {
    kernels::gemm::act_x_wt_bf16(blas, x, W, y, M, N, K, beta);
}

namespace {
struct VisAttnRes {
    AttentionWorkspaceView ws{};
    kernels::attn::PrefillPlanCachePtr plan;
    bool ready = false;
    // Plan/index signature: (num_seqs, total_tokens, per-seq len[0], NH,
    // HEAD). Vision images in one batch are equal-sized, so this captures
    // the shape.
    int sig_seqs = -1, sig_total = -1, sig_len0 = -1, sig_NH = -1, sig_HD = -1;
    std::uint32_t *qo_d = nullptr, *kvpi_d = nullptr, *kvidx_d = nullptr,
                  *klpl_d = nullptr;
    std::mutex mu;
};
VisAttnRes& vis_attn_res() {
    static VisAttnRes v;
    return v;
}
constexpr int kVisPageSize = 16;

void ensure_ws(VisAttnRes& st) {
    if (st.ready) return;
    constexpr std::size_t kFloatBytes = 32u << 20;
    constexpr std::size_t kIntBytes = 16u << 20;
    void* fb = nullptr;
    void* ib = nullptr;
    void* pl = nullptr;
    if (cudaMalloc(&fb, kFloatBytes) != cudaSuccess ||
        cudaMalloc(&ib, kIntBytes) != cudaSuccess ||
        cudaMallocHost(&pl, kIntBytes) != cudaSuccess) {
        throw std::runtime_error("qwen3vl_vis_attn: workspace allocation failed");
    }
    st.ws.float_buffer = fb;
    st.ws.float_bytes = kFloatBytes;
    st.ws.int_buffer = ib;
    st.ws.int_bytes = kIntBytes;
    st.ws.page_locked_int = pl;
    st.plan = kernels::attn::make_prefill_plan();
    st.ready = true;
}
}  // namespace

// Non-causal MHA over `num_seqs` independent images (block-diagonal):
// image i has `seqlens[i]` patches at row offset Σ_{j<i} seqlens[j] in
// q/k/v ([Σ, NH, HEAD]). One flashinfer multi-sequence prefill — each
// query attends only within its own image.
void qwen3vl_vis_attn(const void* q, void* k, void* v, void* o,
                      int num_seqs, const int* seqlens, int NH, int HEAD,
                      cudaStream_t S) {
    auto& st = vis_attn_res();
    std::lock_guard<std::mutex> lk(st.mu);
    const int ps = kVisPageSize;
    int total = 0;
    for (int i = 0; i < num_seqs; ++i) total += seqlens[i];
    ensure_ws(st);
    std::vector<std::uint32_t> qo(num_seqs + 1, 0), kvpi(num_seqs + 1, 0),
        klpl(num_seqs);
    for (int i = 0; i < num_seqs; ++i) {
        const int pages_i = (seqlens[i] + ps - 1) / ps;
        qo[i + 1] = qo[i] + (std::uint32_t)seqlens[i];
        kvpi[i + 1] = kvpi[i] + (std::uint32_t)pages_i;
        klpl[i] = (std::uint32_t)(seqlens[i] - (pages_i - 1) * ps);
    }
    const int total_pages = (int)kvpi[num_seqs];
    const int len0 = num_seqs > 0 ? seqlens[0] : 0;
    const bool changed = st.sig_seqs != num_seqs || st.sig_total != total ||
                         st.sig_len0 != len0 || st.sig_NH != NH ||
                         st.sig_HD != HEAD;
    if (changed) {
        for (void* p :
             {(void*)st.qo_d, (void*)st.kvpi_d, (void*)st.kvidx_d, (void*)st.klpl_d})
            if (p) cudaFree(p);
        std::vector<std::uint32_t> kvidx(total_pages);
        for (int i = 0; i < total_pages; ++i) kvidx[i] = (std::uint32_t)i;
        auto up = [&](std::uint32_t** d, const std::vector<std::uint32_t>& h) {
            cudaMalloc(d, h.size() * sizeof(std::uint32_t));
            cudaMemcpy(*d, h.data(), h.size() * sizeof(std::uint32_t),
                       cudaMemcpyHostToDevice);
        };
        up(&st.qo_d, qo);
        up(&st.kvpi_d, kvpi);
        up(&st.kvidx_d, kvidx);
        up(&st.klpl_d, klpl);
        kernels::attn::plan_attention_flashinfer_prefill_bf16(
            *st.plan, qo.data(), kvpi.data(), klpl.data(),
            /*total_tokens=*/total, num_seqs, NH, NH, HEAD, ps, st.ws, S,
            /*enable_cuda_graph=*/false, /*window_left=*/-1,
            /*full_attention_variant=*/false, /*hnd_layout=*/false,
            /*causal_mask=*/false);
        st.sig_seqs = num_seqs;
        st.sig_total = total;
        st.sig_len0 = len0;
        st.sig_NH = NH;
        st.sig_HD = HEAD;
    }
    const float sm_scale = 1.0f / std::sqrt((float)HEAD);
    kernels::attn::dispatch_attention_flashinfer_prefill_bf16(
        *st.plan, q, k, v, o, st.qo_d, st.kvidx_d, st.kvpi_d, st.klpl_d, st.ws,
        S, /*logits_soft_cap=*/0.f, sm_scale, /*lse_out=*/nullptr);
}

}  // namespace pie_cuda_driver::model

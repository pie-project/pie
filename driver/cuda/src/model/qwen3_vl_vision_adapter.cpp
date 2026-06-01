// Host bridge: Qwen3VLVisionWeights (bound DeviceTensors) → QwenVisRawWeights.
// See qwen3_vl_vision_adapter.hpp. Compiled by the host C++ compiler (g++), so
// it may include the toml++-heavy model headers. Mirrors gemma4_vision_adapter.cpp.

#include "model/qwen3_vl_vision_adapter.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "attention_workspace.hpp"
#include "ops/attention_flashinfer.hpp"
#include "ops/gemm.hpp"

namespace pie_cuda_driver::model {

// cuBLAS bf16 GEMM bridge for the vision forward (declared in the CUDA-only
// qwen3_vl_vision_forward TU). Defined here — in the g++-compiled adapter — so
// the heavy ops/gemm.hpp stays out of that .cu.
// y[M,N] = x[M,K] @ W[N,K]^T (W row-major [N,K]); bf16 in/out, fp32 accumulate.
void qwen3vl_vis_gemm_bf16(cublasHandle_t blas, const void* x, const void* W,
                           void* y, int M, int N, int K, float beta) {
    // beta=1 fuses a residual add (used by the o-/fc2-projection epilogues).
    ops::gemm_act_x_wt_bf16(blas, x, W, y, M, N, K, beta);
}

// ── Vision attention via flashinfer (the same flash kernel the LLM uses) ──────
// Vision attention is non-causal MHA over all N patches, head_dim 64. We model
// it as a single-sequence flashinfer prefill: q is [N, NH, HEAD]; k/v are the
// same contiguous buffers viewed as a paged KV cache ([N/ps, ps, NH, HEAD], NHD)
// with one request spanning all pages. RoPE is already applied to q/k upstream
// (k_rope_vis), so flashinfer runs with external positions (no pos-enc), and
// `causal_mask=false` gives full bidirectional attention. Replaces the per-head
// cuBLAS QK/softmax/AV loop, which materialized [N,N] scores in gmem.
//
// Owns a dedicated workspace + plan (the LLM's are occupied during prefill) and
// caches the constant index arrays + plan keyed by N. Forward passes for one
// model are serialized by the engine; the mutex guards first-touch.
namespace {
struct VisAttnRes {
    AttentionWorkspace ws;
    ops::PrefillPlanCachePtr plan;
    bool ready = false;
    // Plan/index signature: (num_seqs, total_tokens, per-seq len[0], NH, HEAD).
    // Vision images in one batch are equal-sized, so this captures the shape.
    int sig_seqs = -1, sig_total = -1, sig_len0 = -1, sig_NH = -1, sig_HD = -1;
    std::uint32_t *qo_d = nullptr, *kvpi_d = nullptr, *kvidx_d = nullptr, *klpl_d = nullptr;
    std::mutex mu;
};
VisAttnRes& vis_attn() { static VisAttnRes v; return v; }
constexpr int kVisPageSize = 16;  // image patch counts are multiples; flashinfer-friendly
}  // namespace

// Non-causal MHA over `num_seqs` independent images (block-diagonal): image i has
// `seqlens[i]` patches at row offset Σ_{j<i} seqlens[j] in q/k/v ([Σ, NH, HEAD]).
// One flashinfer multi-sequence prefill — each query attends only within its own
// image. q/k/v rows are also a paged KV view ([Σ/ps, ps, NH, HEAD]).
void qwen3vl_vis_attn(const void* q, void* k, void* v, void* o,
                      int num_seqs, const int* seqlens, int NH, int HEAD, cudaStream_t S) {
    auto& st = vis_attn();
    std::lock_guard<std::mutex> lk(st.mu);
    const int ps = kVisPageSize;
    int total = 0; for (int i = 0; i < num_seqs; ++i) total += seqlens[i];
    if (!st.ready) {
        st.ws = AttentionWorkspace::allocate();
        st.plan = ops::make_prefill_plan();
        st.ready = true;
    }
    // Build host indptr/index arrays for the multi-sequence paged layout.
    std::vector<std::uint32_t> qo(num_seqs + 1, 0), kvpi(num_seqs + 1, 0), klpl(num_seqs);
    for (int i = 0; i < num_seqs; ++i) {
        const int pages_i = (seqlens[i] + ps - 1) / ps;
        qo[i + 1] = qo[i] + (std::uint32_t)seqlens[i];
        kvpi[i + 1] = kvpi[i] + (std::uint32_t)pages_i;
        klpl[i] = (std::uint32_t)(seqlens[i] - (pages_i - 1) * ps);
    }
    const int total_pages = (int)kvpi[num_seqs];
    const int len0 = num_seqs > 0 ? seqlens[0] : 0;
    // Rebuild index arrays + re-plan on any shape change (cheap; shape is stable
    // across the 24 layers and across same-resolution batches).
    const bool changed = st.sig_seqs != num_seqs || st.sig_total != total ||
                         st.sig_len0 != len0 || st.sig_NH != NH || st.sig_HD != HEAD;
    if (changed) {
        for (void* p : {(void*)st.qo_d, (void*)st.kvpi_d, (void*)st.kvidx_d, (void*)st.klpl_d})
            if (p) cudaFree(p);
        std::vector<std::uint32_t> kvidx(total_pages);
        for (int i = 0; i < total_pages; ++i) kvidx[i] = (std::uint32_t)i;
        auto up = [&](std::uint32_t** d, const std::vector<std::uint32_t>& h) {
            cudaMalloc(d, h.size() * sizeof(std::uint32_t));
            cudaMemcpy(*d, h.data(), h.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice);
        };
        up(&st.qo_d, qo); up(&st.kvpi_d, kvpi); up(&st.kvidx_d, kvidx); up(&st.klpl_d, klpl);
        ops::plan_attention_flashinfer_prefill_bf16(
            *st.plan, qo.data(), kvpi.data(), klpl.data(), /*total_tokens=*/total, num_seqs,
            NH, NH, HEAD, ps, st.ws, S, /*enable_cuda_graph=*/false, /*window_left=*/-1,
            /*full_attention_variant=*/false, /*hnd_layout=*/false, /*causal_mask=*/false);
        st.sig_seqs = num_seqs; st.sig_total = total; st.sig_len0 = len0;
        st.sig_NH = NH; st.sig_HD = HEAD;
    }
    const float sm_scale = 1.0f / std::sqrt((float)HEAD);
    ops::dispatch_attention_flashinfer_prefill_bf16(
        *st.plan, q, k, v, o, st.qo_d, st.kvidx_d, st.kvpi_d, st.klpl_d,
        st.ws, S, /*logits_soft_cap=*/0.f, sm_scale, /*lse_out=*/nullptr);
}

namespace {
using bf = __nv_bfloat16;
const bf* P(const DeviceTensor* t) {
    return t ? static_cast<const bf*>(t->data()) : nullptr;
}
QVisLinear to_lin(const DeviceTensor* w, const DeviceTensor* b) {
    QVisLinear r;
    r.w = P(w);
    r.b = P(b);
    return r;
}
QVisLayerNorm to_ln(const DeviceTensor* g, const DeviceTensor* b) {
    QVisLayerNorm r;
    r.g = P(g);
    r.b = P(b);
    return r;
}
QVisMerger to_merger(const Qwen3VLVisionMergerWeights& m) {
    QVisMerger r;
    r.norm = to_ln(m.norm_weight, m.norm_bias);
    r.fc1 = to_lin(m.fc1_weight, m.fc1_bias);
    r.fc2 = to_lin(m.fc2_weight, m.fc2_bias);
    r.is_postshuffle = m.use_postshuffle_norm;
    return r;
}
// Read a shape dimension, defaulting if the tensor or axis is absent.
int dim_or(const DeviceTensor* t, int axis, int fallback) {
    if (!t) return fallback;
    const auto& s = t->shape();
    return axis < static_cast<int>(s.size()) ? static_cast<int>(s[axis]) : fallback;
}
}  // namespace

QwenVisRawWeights to_vis_raw_qwen(const Qwen3VLVisionWeights& w) {
    QwenVisRawWeights r;
    r.patch = to_lin(w.patch_weight, w.patch_bias);
    r.pos_embed = P(w.pos_embed);

    // Dims from config (with shape-derived fallbacks, robust to config gaps).
    r.hidden = w.config.hidden_size;
    r.heads = w.config.num_heads;
    r.head_dim = r.heads > 0 ? r.hidden / r.heads : 64;
    r.intermediate = w.config.intermediate_size;
    r.patch_size = w.config.patch_size;
    r.temporal_patch_size = w.config.temporal_patch_size;
    r.spatial_merge_size = w.config.spatial_merge_size;
    r.in_channels = w.config.in_channels;
    r.out_hidden = w.config.out_hidden_size;
    r.num_pos_embed = dim_or(w.pos_embed, 0, w.config.num_position_embeddings);
    // num_grid_per_side = int(sqrt(num_pos_embed)) (48 for 2304).
    r.num_grid_per_side = static_cast<int>(0.5 + std::sqrt((double)r.num_pos_embed));
    r.ln_eps = 1e-6f;
    r.rope_theta = 10000.0f;

    r.blocks.reserve(w.layers.size());
    for (const auto& L : w.layers) {
        QVisBlock o;
        o.norm1 = to_ln(L.norm1_weight, L.norm1_bias);
        o.norm2 = to_ln(L.norm2_weight, L.norm2_bias);
        o.qkv = to_lin(L.qkv_weight, L.qkv_bias);
        o.o = to_lin(L.proj_weight, L.proj_bias);
        o.fc1 = to_lin(L.fc1_weight, L.fc1_bias);
        o.fc2 = to_lin(L.fc2_weight, L.fc2_bias);
        r.blocks.push_back(o);
    }

    r.merger = to_merger(w.merger);
    r.deepstack.reserve(w.deepstack.size());
    for (const auto& m : w.deepstack) r.deepstack.push_back(to_merger(m));
    r.deepstack_layer_idx = w.deepstack_layer_idx;

    if (r.hidden != r.heads * r.head_dim)
        throw std::runtime_error("to_vis_raw_qwen: hidden != heads*head_dim");
    return r;
}

}  // namespace pie_cuda_driver::model

#include "model/gemma4.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/embed.hpp"
#include "kernels/gather_rows.hpp"
#include "kernels/kv_paged.hpp"
#include "kernels/moe_dispatch.hpp"
#include "kernels/residual_add.hpp"
#include "kernels/rmsnorm.hpp"
#include "kernels/rope.hpp"
#include "kernels/scalar_mul.hpp"
#include "kernels/softcap.hpp"
#include "kernels/swiglu.hpp"
#include "kernels/topk_softmax.hpp"
#include "ops/attention_naive_paged.hpp"

namespace pie_cuda_driver::model {

namespace {

const DeviceTensor& must(const Engine& e, const std::string& name) {
    if (!e.has(name)) {
        throw std::runtime_error("gemma4: missing weight '" + name + "'");
    }
    return e.get(name);
}

// HF Gemma-4 prefixes language-model tensors with `model.language_model.`
// — different from Llama / Qwen which use `model.`. The bind helpers
// below take the prefix as a string so the call sites read like the
// other binders.
constexpr const char* kPrefix = "model.language_model.";

}  // namespace

Gemma4MoeMlpWorkspace Gemma4MoeMlpWorkspace::allocate(
    int max_tokens, int hidden, int num_experts, int top_k,
    int moe_intermediate)
{
    Gemma4MoeMlpWorkspace ws;
    const std::size_t N    = static_cast<std::size_t>(max_tokens);
    const std::size_t maxR = N * top_k;
    const std::size_t H    = static_cast<std::size_t>(hidden);
    const std::size_t I    = static_cast<std::size_t>(moe_intermediate);

    ws.router_x      = DeviceBuffer<std::uint16_t>::alloc(N * H);
    ws.router_logits = DeviceBuffer<std::uint16_t>::alloc(N * num_experts);
    ws.topk_idx      = DeviceBuffer<std::int32_t>::alloc(N * top_k);
    ws.topk_weights  = DeviceBuffer<float>::alloc(N * top_k);

    ws.moe_input    = DeviceBuffer<std::uint16_t>::alloc(N * H);
    ws.expert_in    = DeviceBuffer<std::uint16_t>::alloc(maxR * H);
    ws.expert_gate_up = DeviceBuffer<std::uint16_t>::alloc(maxR * 2 * I);
    ws.expert_act   = DeviceBuffer<std::uint16_t>::alloc(maxR * I);
    ws.expert_out   = DeviceBuffer<std::uint16_t>::alloc(maxR * H);
    ws.expert_idx   = DeviceBuffer<std::int32_t>::alloc(maxR);
    ws.expert_w     = DeviceBuffer<float>::alloc(maxR);
    ws.moe_out      = DeviceBuffer<std::uint16_t>::alloc(N * H);

    ws.a_gu_ptrs    = DeviceBuffer<const std::uint16_t*>::alloc(top_k);
    ws.b_gu_ptrs    = DeviceBuffer<const std::uint16_t*>::alloc(top_k);
    ws.c_gu_ptrs    = DeviceBuffer<std::uint16_t*>::alloc(top_k);
    ws.a_dn_ptrs    = DeviceBuffer<const std::uint16_t*>::alloc(top_k);
    ws.b_dn_ptrs    = DeviceBuffer<const std::uint16_t*>::alloc(top_k);
    ws.c_dn_ptrs    = DeviceBuffer<std::uint16_t*>::alloc(top_k);
    ws.batch_weights = DeviceBuffer<float>::alloc(top_k);
    return ws;
}

Gemma4Weights bind_gemma4(const Engine& engine) {
    const auto& cfg = engine.hf_config();
    if (cfg.layer_types.empty()) {
        throw std::runtime_error(
            "gemma4: HfConfig.layer_types is empty — Gemma-4 requires "
            "the per-layer attention type from the HF config.");
    }
    const int L = cfg.num_hidden_layers;
    if (static_cast<int>(cfg.layer_types.size()) != L) {
        throw std::runtime_error("gemma4: layer_types size != num_hidden_layers");
    }

    Gemma4Weights w;
    // Reserve up front: pointers into `owned_router_combined_scales`
    // get cached on each `Gemma4LayerWeights`, so any vector reallocation
    // would dangle them. One slot per layer when MoE is on.
    if (cfg.gemma4_enable_moe) {
        w.owned_router_combined_scales.reserve(
            static_cast<std::size_t>(cfg.num_hidden_layers));
    }
    const std::string p = kPrefix;
    w.embed           = &must(engine, p + "embed_tokens.weight");
    // PLE (Per-Layer Embeddings) machinery is optional — Gemma-4 E2B /
    // E4B / 31B all ship it (`hidden_size_per_layer_input > 0`) but the
    // 26B-A4B MoE variant disables it (`hidden_size_per_layer_input ==
    // 0`). Skip the PLE tensors when the config says they're inert.
    if (cfg.gemma_hidden_size_per_layer_input > 0) {
        w.embed_per_layer = &must(engine, p + "embed_tokens_per_layer.weight");
        w.ple_model_proj  = &must(engine, p + "per_layer_model_projection.weight");
        w.ple_model_norm  = &must(engine, p + "per_layer_projection_norm.weight");
    }
    w.final_norm      = &must(engine, p + "norm.weight");
    if (engine.has("lm_head.weight")) {
        w.lm_head = &engine.get("lm_head.weight");
    } else if (cfg.tie_word_embeddings) {
        w.lm_head = w.embed;
    } else {
        throw std::runtime_error(
            "gemma4: lm_head missing and tie_word_embeddings=false");
    }

    // Per-layer dimensions. HF stores `head_dim` (sliding) and
    // `global_head_dim` (full); we read both from the config but our
    // HfConfig only carries the single `head_dim`. Recompute from
    // first-layer Q-proj shape: full-attention layers have
    // `q_proj = [num_q*global_head_dim, hidden]`, sliding have
    // `[num_q*head_dim, hidden]`.
    const int sliding_head_dim = cfg.head_dim;
    int global_head_dim = cfg.head_dim;
    {
        // First full-attention layer's q_proj reveals global_head_dim.
        for (int i = 0; i < L; ++i) {
            if (cfg.layer_types[i] == "full_attention") {
                const std::string q_name = p + "layers." + std::to_string(i) +
                                           ".self_attn.q_proj.weight";
                if (engine.has(q_name)) {
                    const auto& qt = engine.get(q_name);
                    const auto& s = qt.shape();
                    if (!s.empty()) {
                        global_head_dim = static_cast<int>(s[0]) /
                                          cfg.num_attention_heads;
                    }
                }
                break;
            }
        }
    }

    // Determine which layers are KV-shared. HF: last
    // `num_kv_shared_layers` layers reuse from earlier; given E2B has
    // 35 layers, 20 shared, the boundary is index 14. The `kv_source`
    // for a shared layer is the most recent non-shared layer of the
    // *same* attention type.
    const int num_kv_shared = std::max(0, [&]{
        // num_kv_shared_layers isn't carried in HfConfig today; we
        // derive it from the layer_types vector and the implicit
        // "last N reuse from earlier" rule. Fall back to 0 (every
        // layer computes its own K/V) when the field is missing.
        return engine.hf_config().num_kv_shared_layers;
    }());
    const int first_shared = L - num_kv_shared;  // first shared layer index

    w.layers.resize(static_cast<std::size_t>(L));
    w.per_layer_head_dim.resize(L);
    w.per_layer_intermediate.resize(L);
    w.per_layer_num_kv_heads.resize(L);
    w.kv_source_layer.resize(L);
    w.per_layer_window_left.resize(L);
    w.per_layer_rope_theta.resize(L);
    w.per_layer_partial_rotary_factor.resize(L, 1.0f);

    for (int i = 0; i < L; ++i) {
        const std::string lp = p + "layers." + std::to_string(i) + ".";
        auto& Lw = w.layers[i];
        const bool is_full = (cfg.layer_types[i] == "full_attention");
        const bool is_shared = (i >= first_shared);

        Lw.is_full   = is_full;
        Lw.is_shared = is_shared;
        Lw.head_dim  = is_full ? global_head_dim : sliding_head_dim;
        w.per_layer_head_dim[i] = Lw.head_dim;
        // 26B-A4B's "k_eq_v" mode flips full-attention layers onto a
        // narrower KV head count (`num_global_key_value_heads`) and
        // skips `v_proj.weight` entirely — V is taken from the raw
        // k_proj output, before k_norm/RoPE, then v-norm. Sliding
        // layers stay on the standard `num_key_value_heads` and have
        // their own v_proj.
        Lw.use_k_as_v = cfg.gemma4_attention_k_eq_v && is_full;
        Lw.num_kv_heads = Lw.use_k_as_v
                              ? cfg.gemma4_num_global_key_value_heads
                              : cfg.num_key_value_heads;
        w.per_layer_num_kv_heads[i] = Lw.num_kv_heads;

        // KV source: same layer when not shared; most recent non-shared
        // layer of the same type when shared.
        if (!is_shared) {
            Lw.kv_source = i;
        } else {
            int src = -1;
            for (int j = first_shared - 1; j >= 0; --j) {
                if (cfg.layer_types[j] == cfg.layer_types[i]) { src = j; break; }
            }
            if (src < 0) {
                throw std::runtime_error(
                    "gemma4: no source layer found for shared layer " +
                    std::to_string(i));
            }
            Lw.kv_source = src;
        }
        w.kv_source_layer[i] = Lw.kv_source;

        // Per-layer window_left: sliding layers limit context to the
        // configured `sliding_window`; full layers run unbounded.
        w.per_layer_window_left[i] = is_full ? -1 : cfg.sliding_window;

        // Per-layer rope_theta + partial_rotary_factor. Gemma-4 nests
        // these under `rope_parameters[layer_type]` in HF; we expand
        // into vectors at parse time.
        if (i < static_cast<int>(cfg.gemma_per_layer_rope_theta.size())) {
            w.per_layer_rope_theta[i]            = cfg.gemma_per_layer_rope_theta[i];
            w.per_layer_partial_rotary_factor[i] =
                cfg.gemma_per_layer_partial_rotary_factor[i];
        } else {
            w.per_layer_rope_theta[i] =
                (is_full || cfg.rope_local_base_freq <= 0.f)
                    ? cfg.rope_theta
                    : cfg.rope_local_base_freq;
        }

        // Norms (4 per layer).
        Lw.attn_norm_pre  = &must(engine, lp + "input_layernorm.weight");
        Lw.attn_norm_post = &must(engine, lp + "post_attention_layernorm.weight");
        Lw.mlp_norm_pre   = &must(engine, lp + "pre_feedforward_layernorm.weight");
        Lw.mlp_norm_post  = &must(engine, lp + "post_feedforward_layernorm.weight");

        // Q is always present.
        Lw.q_proj = &must(engine, lp + "self_attn.q_proj.weight");
        Lw.q_norm = &must(engine, lp + "self_attn.q_norm.weight");

        // Even on shared layers HF keeps k_proj/v_proj/k_norm in the
        // file (redundant). Bind them when present so the schema
        // tolerates either dump style; the forward only consults them
        // when `is_shared == false`. On 26B-A4B's `use_k_as_v` full
        // layers v_proj is genuinely absent (V is derived from raw
        // k_proj), so a missing v_proj is expected there.
        if (engine.has(lp + "self_attn.k_proj.weight")) {
            Lw.k_proj = &engine.get(lp + "self_attn.k_proj.weight");
        }
        if (engine.has(lp + "self_attn.v_proj.weight")) {
            Lw.v_proj = &engine.get(lp + "self_attn.v_proj.weight");
        }
        if (engine.has(lp + "self_attn.k_norm.weight")) {
            Lw.k_norm = &engine.get(lp + "self_attn.k_norm.weight");
        }
        Lw.o_proj = &must(engine, lp + "self_attn.o_proj.weight");

        // MLP (intermediate may be 2× when use_double_wide_mlp + shared).
        Lw.gate_proj = &must(engine, lp + "mlp.gate_proj.weight");
        Lw.up_proj   = &must(engine, lp + "mlp.up_proj.weight");
        Lw.down_proj = &must(engine, lp + "mlp.down_proj.weight");
        Lw.intermediate = static_cast<int>(Lw.gate_proj->shape()[0]);
        w.per_layer_intermediate[i] = Lw.intermediate;

        // PLE per-layer triple. HF names match `per_layer_input_gate`,
        // `per_layer_projection`, `post_per_layer_input_norm`. Optional
        // — see top of `bind_gemma4`.
        if (cfg.gemma_hidden_size_per_layer_input > 0) {
            Lw.ple_input_gate = &must(engine, lp + "per_layer_input_gate.weight");
            Lw.ple_projection = &must(engine, lp + "per_layer_projection.weight");
            Lw.ple_norm       = &must(engine, lp + "post_per_layer_input_norm.weight");
        }

        // Per-layer learnable scalar.
        Lw.layer_scalar = engine.has(lp + "layer_scalar")
                              ? &engine.get(lp + "layer_scalar")
                              : nullptr;

        // ── Sparse-MoE block (Gemma-4 26B-A4B only) ───────────────────
        // The MoE variant runs in parallel with the dense MLP (HF
        // `Gemma4TextDecoderLayer.forward`). When `enable_moe_block` is
        // false the dense path runs alone and these pointers stay null.
        if (cfg.gemma4_enable_moe) {
            Lw.router_proj            = &must(engine, lp + "router.proj.weight");
            Lw.router_per_expert_scale = &must(engine, lp + "router.per_expert_scale");
            Lw.moe_gate_up_proj       = &must(engine, lp + "experts.gate_up_proj");
            Lw.moe_down_proj          = &must(engine, lp + "experts.down_proj");
            Lw.mlp_norm_post_dense    = &must(engine, lp + "post_feedforward_layernorm_1.weight");
            Lw.moe_norm_pre           = &must(engine, lp + "pre_feedforward_layernorm_2.weight");
            Lw.moe_norm_post          = &must(engine, lp + "post_feedforward_layernorm_2.weight");
            // The router pipeline does `(rmsnorm_no_scale(x) * scale) *
            // (1/sqrt(H))` then a linear. Bake `1/sqrt(H)` into the
            // per-channel `scale` here so the forward collapses the
            // first three steps into a single rmsnorm-with-weight call.
            const auto& raw_scale = must(engine, lp + "router.scale");
            const std::int64_t H64 = raw_scale.numel();
            const float inv_sqrt_h = 1.f / std::sqrt(static_cast<float>(H64));
            std::vector<std::uint16_t> host(static_cast<std::size_t>(H64));
            CUDA_CHECK(cudaMemcpy(host.data(), raw_scale.data(),
                                  H64 * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost));
            for (auto& bits : host) {
                std::uint32_t f32_bits = static_cast<std::uint32_t>(bits) << 16;
                float v;
                std::memcpy(&v, &f32_bits, sizeof(float));
                v *= inv_sqrt_h;
                std::memcpy(&f32_bits, &v, sizeof(float));
                bits = static_cast<std::uint16_t>(f32_bits >> 16);
            }
            DeviceTensor combined = DeviceTensor::allocate(DType::BF16, {H64});
            CUDA_CHECK(cudaMemcpy(combined.data(), host.data(),
                                  H64 * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice));
            w.owned_router_combined_scales.push_back(std::move(combined));
            Lw.router_scale = &w.owned_router_combined_scales.back();
        }

        if (is_full) w.full_layer_indices.push_back(i);
    }

    return w;
}

namespace {

// Parity-dump helper: write a bf16 tensor of `numel` elements to
// `<dir>/<tag>.bin` as raw bf16 bytes. We only record the *first* fire
// of a session (typically the prefill) — subsequent decode fires would
// overwrite the prefill's intermediates with decode-shaped tensors,
// which is not what the PyTorch parity harness compares against.
inline bool& dbg_first_fire_flag() {
    static bool first = true;
    return first;
}
// True only when `PIE_GEMMA4_DUMP_DIR` is set in the environment.
// Cached on first call so per-fire checks are a single bool load.
inline bool dbg_dumps_enabled() {
    static const bool enabled = std::getenv("PIE_GEMMA4_DUMP_DIR") != nullptr;
    return enabled;
}
inline void dbg_dump_bf16(const char* tag, const void* dev_ptr,
                          std::size_t numel) {
    if (!dbg_dumps_enabled()) return;
    if (!dbg_first_fire_flag()) return;
    static const char* dir = std::getenv("PIE_GEMMA4_DUMP_DIR");
    std::vector<std::uint16_t> tmp(numel);
    cudaMemcpy(tmp.data(), dev_ptr, numel * 2, cudaMemcpyDeviceToHost);
    std::string path = std::string(dir) + "/" + tag + ".bin";
    std::ofstream out(path, std::ios::binary);
    if (!out) return;
    out.write(reinterpret_cast<const char*>(tmp.data()), numel * 2);
}
// Sync-then-dump: paired with `dbg_dump_bf16` to guarantee the kernel
// preceding the dump has finished. **Only syncs when dumping is on.**
// Replaces the previous pattern of unconditional `cudaDeviceSynchronize()
// + dbg_dump_bf16(...)` which stalled the GPU on every layer of every
// fire even with no dump directory configured (the dumps no-op'd, but
// the syncs did not — they were the dominant per-step overhead in
// Gemma-4 release builds).
inline void dbg_sync_dump_bf16(const char* tag, const void* dev_ptr,
                               std::size_t numel) {
    if (!dbg_dumps_enabled()) return;
    if (!dbg_first_fire_flag()) return;
    cudaDeviceSynchronize();
    dbg_dump_bf16(tag, dev_ptr, numel);
}

// Read a single bf16 tensor's first element to host. Used for the
// per-layer learnable scalar — there's one float per layer; copying
// one bf16 value at startup is cheap and lets the forward avoid a
// small device-side mul.
inline float read_bf16_scalar(const DeviceTensor& t) {
    if (t.empty()) return 1.f;
    std::uint16_t bits = 0;
    cudaMemcpy(&bits, t.data(), sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    const std::uint32_t f32_bits = static_cast<std::uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &f32_bits, sizeof(float));
    return f;
}

// Per-expert routing lists from device-side topk decisions. Mirrors
// `qwen3_5_moe_forward.cpp::build_routing` — kept local because the
// layer-weights schema differs.
struct ExpertRouting {
    std::vector<std::vector<std::int32_t>> token_idx;
    std::vector<std::vector<float>>        weights;
};
ExpertRouting build_routing(
    const std::vector<std::int32_t>& topk_idx_h,
    const std::vector<float>& topk_w_h,
    int N, int K, int E)
{
    ExpertRouting r;
    r.token_idx.assign(E, {});
    r.weights.assign(E, {});
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            const int e = topk_idx_h[n * K + k];
            if (e < 0 || e >= E) continue;
            r.token_idx[e].push_back(n);
            r.weights[e].push_back(topk_w_h[n * K + k]);
        }
    }
    return r;
}

// MoE block for Gemma-4 26B-A4B: parallel branch alongside the dense
// MLP. Computes `branch_2 = post_ff_norm_2(experts(pre_ff_norm_2(y),
// router(y)))` and writes it into `moe_ws.moe_out`.
void gemma4_moe_block(
    const Gemma4LayerWeights& Lw,
    const HfConfig& cfg,
    Qwen3Workspace& ws,
    Gemma4MoeMlpWorkspace& moe_ws,
    int N,
    ops::CublasHandle& cublas, cudaStream_t stream)
{
    const int H  = cfg.hidden_size;
    const int E  = cfg.num_experts;
    const int K  = cfg.num_experts_per_tok;
    const int Im = cfg.moe_intermediate_size;
    const float eps = cfg.rms_norm_eps;

    // ── Router ────────────────────────────────────────────────────
    // Step 1+2: rmsnorm-no-scale(y) * (router_scale * 1/sqrt(H)).
    // The combined scale was baked at bind time, so this collapses to
    // a single weighted-rmsnorm call.
    kernels::launch_rmsnorm_bf16(
        ws.y.data(), Lw.router_scale->data(), moe_ws.router_x.data(),
        N, H, eps, stream);
    // Step 3: linear projection to expert logits.
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        moe_ws.router_x.data(), Lw.router_proj->data(),
        moe_ws.router_logits.data(), N, E, H);
    // Steps 4+5: softmax over E → top-K → renormalise.
    kernels::launch_topk_softmax_bf16(
        moe_ws.router_logits.data(),
        moe_ws.topk_idx.data(), moe_ws.topk_weights.data(),
        N, E, K, stream);
    // Step 6: per-expert scalar gain on the chosen weights.
    kernels::launch_apply_per_expert_scale_bf16(
        moe_ws.topk_idx.data(), moe_ws.topk_weights.data(),
        Lw.router_per_expert_scale->data(),
        N, K, stream);

    // ── MoE input ────────────────────────────────────────────────
    // pre_feedforward_layernorm_2(y) → moe_input. Note: HF flattens the
    // residual `y`, NOT the dense MLP's pre-norm (`ws.norm_x` was
    // already overwritten by the dense path).
    kernels::launch_rmsnorm_bf16(
        ws.y.data(), Lw.moe_norm_pre->data(), moe_ws.moe_input.data(),
        N, H, eps, stream);

    // D2H sync the routing decisions for the prefill / multi-token
    // dispatch loop. The dense Gemma-4 forward is non-graph-capturable
    // (per-layer head_dim, attention_factor lookups all run host code),
    // so an extra sync here is in the noise.
    std::vector<std::int32_t> topk_idx_h((std::size_t)N * K);
    std::vector<float>        topk_w_h((std::size_t)N * K);
    CUDA_CHECK(cudaMemcpyAsync(topk_idx_h.data(), moe_ws.topk_idx.data(),
        topk_idx_h.size() * sizeof(std::int32_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(topk_w_h.data(), moe_ws.topk_weights.data(),
        topk_w_h.size() * sizeof(float),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Zero `moe_out` before scatter-add accumulation.
    CUDA_CHECK(cudaMemsetAsync(moe_ws.moe_out.data(), 0,
        (std::size_t)N * H * sizeof(std::uint16_t), stream));

    const auto routing = build_routing(topk_idx_h, topk_w_h, N, K, E);
    const std::size_t expert_stride_gu =
        static_cast<std::size_t>(2) * Im * H;  // bf16 elements per expert
    const std::size_t expert_stride_dn =
        static_cast<std::size_t>(H) * Im;

    for (int e = 0; e < E; ++e) {
        const auto& tok_idx = routing.token_idx[e];
        const auto& wts     = routing.weights[e];
        const int Ne = static_cast<int>(tok_idx.size());
        if (Ne == 0) continue;

        CUDA_CHECK(cudaMemcpyAsync(
            moe_ws.expert_idx.data(), tok_idx.data(),
            Ne * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            moe_ws.expert_w.data(), wts.data(),
            Ne * sizeof(float), cudaMemcpyHostToDevice, stream));

        kernels::launch_gather_bf16_rows(
            static_cast<const std::uint16_t*>(moe_ws.moe_input.data()),
            moe_ws.expert_idx.data(),
            moe_ws.expert_in.data(),
            Ne, H, stream);

        const auto* gate_up_w = static_cast<const std::uint16_t*>(
                                    Lw.moe_gate_up_proj->data())
                                + e * expert_stride_gu;
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            moe_ws.expert_in.data(), gate_up_w,
            moe_ws.expert_gate_up.data(), Ne, 2 * Im, H);

        kernels::launch_chunked_geglu_tanh_bf16(
            moe_ws.expert_gate_up.data(),
            moe_ws.expert_act.data(),
            Ne, Im, stream);

        const auto* down_w = static_cast<const std::uint16_t*>(
                                 Lw.moe_down_proj->data())
                             + e * expert_stride_dn;
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            moe_ws.expert_act.data(), down_w,
            moe_ws.expert_out.data(), Ne, H, Im);

        kernels::launch_scatter_add_weighted_bf16(
            moe_ws.moe_out.data(), moe_ws.expert_out.data(),
            moe_ws.expert_idx.data(), moe_ws.expert_w.data(),
            Ne, H, stream);
    }
}

}  // namespace

void gemma4_forward_paged(
    const Gemma4Weights& w,
    const HfConfig& cfg,
    const Gemma4ForwardCfg& fwd_cfg,
    Qwen3Workspace& ws,
    Gemma4MoeMlpWorkspace& moe_ws,
    KvCache& cache,
    AttentionWorkspace& attn_ws,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int N,
    int R,
    bool is_pure_decode,
    const std::uint8_t* /*custom_mask_d*/,
    const std::int32_t* /*custom_mask_indptr_d*/)
{
    const int H        = cfg.hidden_size;
    const int L        = cfg.num_hidden_layers;
    const int V        = cfg.vocab_size;
    const int ple_dim  = cfg.gemma_hidden_size_per_layer_input;
    const float eps    = cfg.rms_norm_eps;
    cudaStream_t stream = nullptr;

    const bool use_decode_path = is_pure_decode && !fwd_cfg.force_prefill_path;

    // ── 1. Embed + √hidden scale ──────────────────────────────────────────
    // Dump input tokens to disk so the parity harness can confirm
    // it's running HF on the same prefix. Only on the first fire
    // (the prefill) — subsequent decode fires would clobber the
    // file with a single-token dump.
    {
        const char* dir = std::getenv("PIE_GEMMA4_DUMP_DIR");
        if (dir != nullptr && dbg_first_fire_flag()) {
            std::vector<std::int32_t> tmp(N);
            cudaMemcpy(tmp.data(), token_ids, N * sizeof(std::int32_t),
                       cudaMemcpyDeviceToHost);
            std::ofstream out(std::string(dir) + "/tokens.bin", std::ios::binary);
            if (out) out.write(reinterpret_cast<const char*>(tmp.data()),
                               N * sizeof(std::int32_t));
        }
    }
    kernels::launch_embed_bf16(
        token_ids, w.embed->data(), ws.y.data(), N, H, V, stream);
    dbg_sync_dump_bf16("embed_pre_scale", ws.y.data(),
                  static_cast<std::size_t>(N) * H);
    kernels::launch_scalar_mul_bf16(
        ws.y.data(), std::sqrt(static_cast<float>(H)),
        static_cast<std::size_t>(N) * H, stream);
    dbg_sync_dump_bf16("embed_post_scale", ws.y.data(),
                  static_cast<std::size_t>(N) * H);

    // ── 2. Per-layer inputs (PLE) ────────────────────────────────────────
    // Compute once per fire; sliced per layer below. Skipped entirely
    // when `ple_dim == 0` (Gemma-4 26B-A4B disables PLE; the per-layer
    // residual block at step 3c is also gated below).
    //
    //     per_layer_token = embed_per_layer[token_ids]              [N, L*ple_dim]
    //     per_layer_token *= sqrt(ple_dim)
    //     per_layer_proj   = inputs_embeds @ ple_model_proj.T       [N, L*ple_dim]
    //     per_layer_proj  *= 1/sqrt(hidden)
    //     per_layer_proj   = rms_norm(per_layer_proj, ple_model_norm)  [per ple_dim row]
    //     per_layer_inputs = (per_layer_proj + per_layer_token) * 1/sqrt(2)
    //
    // Allocate dedicated scratch for these — `ws.gate`/`ws.up` would
    // get clobbered by the per-layer MLP GEMM in step 3 before the
    // PLE block reads back the slice for layer N. Earlier versions
    // shared the buffers and silently produced wrong PLE residuals
    // for every token (most visibly token 0).
    const int per_layer_total = L * ple_dim;
    DeviceTensor per_layer_token_buf;
    DeviceTensor per_layer_proj_buf;
    void* per_layer_token = nullptr;
    void* per_layer_proj  = nullptr;
    if (ple_dim > 0) {
        per_layer_token_buf = DeviceTensor::allocate(
            DType::BF16, {N, per_layer_total});
        per_layer_proj_buf = DeviceTensor::allocate(
            DType::BF16, {N, per_layer_total});
        per_layer_token = per_layer_token_buf.data();
        per_layer_proj  = per_layer_proj_buf.data();
        // Embed lookup into the per-layer table.
        kernels::launch_embed_bf16(
            token_ids, w.embed_per_layer->data(), per_layer_token,
            N, per_layer_total, V, stream);
        kernels::launch_scalar_mul_bf16(
            per_layer_token, std::sqrt(static_cast<float>(ple_dim)),
            static_cast<std::size_t>(N) * per_layer_total, stream);

        // Project the main embedding to the per-layer subspace.
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.y.data(), w.ple_model_proj->data(), per_layer_proj,
            N, per_layer_total, H);
        kernels::launch_scalar_mul_bf16(
            per_layer_proj, 1.0f / std::sqrt(static_cast<float>(H)),
            static_cast<std::size_t>(N) * per_layer_total, stream);

        // RMSNorm per ple_dim row. We reshape mentally to
        // [N*L, ple_dim] and run our row-wise rmsnorm at that shape.
        kernels::launch_rmsnorm_bf16(
            per_layer_proj, w.ple_model_norm->data(), per_layer_proj,
            N * L, ple_dim, eps, stream);

        // (per_layer_proj + per_layer_token) * 1/sqrt(2). residual_add
        // gives us in-place add; then scale.
        kernels::launch_residual_add_bf16(
            per_layer_proj, per_layer_token,
            static_cast<std::size_t>(N) * per_layer_total, stream);
        kernels::launch_scalar_mul_bf16(
            per_layer_proj, 1.0f / std::sqrt(2.0f),
            static_cast<std::size_t>(N) * per_layer_total, stream);
    }
    // After this block, `per_layer_proj` holds the [N, L*ple_dim]
    // per-layer signal. We slice out `per_layer_proj[:, l*ple_dim :
    // (l+1)*ple_dim]` per layer below by pointer-arithmetic.
    auto* per_layer_base = static_cast<std::uint8_t*>(per_layer_proj);
    constexpr int bf16_bytes = 2;
    dbg_sync_dump_bf16("per_layer_inputs", per_layer_proj,
                  static_cast<std::size_t>(N) * L * ple_dim);

    // Hoist decode plan(s). Gemma-4 has dual head_dim so we plan twice
    // — once at sliding head_dim, once at global head_dim. Both reuse
    // the same workspace (flashinfer's plan info encodes only memory
    // offsets, which don't conflict across kernels run sequentially).
    // We call the kernel-level plan inline per-layer instead — cheaper
    // for small batches than maintaining two cached plans.

    // ── 3. Layer loop ────────────────────────────────────────────────────
    int debug_max_layers = L;
    if (const char* lim = getenv("PIE_GEMMA4_MAX_LAYERS")) {
        debug_max_layers = std::min(L, std::atoi(lim));
    }
    for (int l = 0; l < debug_max_layers; ++l) {
        const auto& layer = w.layers[l];
        const bool dump_this = (l == 0);
        // Pair the sync with the dump so a release run (no dump dir
        // env var) skips both — the standalone syncs that used to
        // precede each dump_l0 call were the dominant per-step
        // overhead on Gemma-4 (~3-15 ms per fire across 30 layers).
        auto dump_l0 = [&](const char* tag, const void* p, std::size_t n) {
            if (!dump_this || !dbg_dumps_enabled() || !dbg_first_fire_flag()) return;
            cudaDeviceSynchronize();
            std::string t = std::string("L0_") + tag;
            dbg_dump_bf16(t.c_str(), p, n);
        };
        // Per-layer dims sharded by tp_size on TP runs. The head/intermediate
        // counts must be divisible by tp_size — guarded at engine load.
        const int T  = (fwd_cfg.tp_size > 0) ? fwd_cfg.tp_size : 1;
        const int d  = layer.head_dim;
        const int num_q_heads_local  = cfg.num_attention_heads / T;
        // KV-head count is now per-layer: 26B-A4B's full-attention
        // layers use num_global_key_value_heads; sliding layers and
        // every other Gemma-4 family use the standard num_key_value_heads.
        const int num_kv_heads_local = layer.num_kv_heads / T;
        const int Hq = num_q_heads_local * d;
        const int Hk = num_kv_heads_local * d;
        const int I  = layer.intermediate / T;
        NcclComm* tp = (T > 1) ? fwd_cfg.tp_comm : nullptr;

        // ── 3a. Attention block ─────────────────────────────────────────
        kernels::launch_rmsnorm_bf16(
            ws.y.data(), layer.attn_norm_pre->data(), ws.norm_x.data(),
            N, H, eps, stream);
        dump_l0("attn_norm_pre", ws.norm_x.data(),
                static_cast<std::size_t>(N) * H);

        // Q-projection always runs.
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.norm_x.data(), layer.q_proj->data(), ws.q.data(),
            N, Hq, H);
        // K/V projections only on non-shared layers. On 26B-A4B's
        // `use_k_as_v` full layers there's no v_proj — V is the raw
        // k_proj output (before k_norm/RoPE), then v-norm. Copy K to V
        // here so the v-norm step below can apply in-place on V.
        if (!layer.is_shared) {
            ops::gemm_act_x_wt_bf16(cublas.handle(),
                ws.norm_x.data(), layer.k_proj->data(), ws.k.data(),
                N, Hk, H);
            if (layer.use_k_as_v) {
                CUDA_CHECK(cudaMemcpyAsync(
                    ws.v.data(), ws.k.data(),
                    static_cast<std::size_t>(N) * Hk *
                        sizeof(std::uint16_t),
                    cudaMemcpyDeviceToDevice, stream));
            } else {
                ops::gemm_act_x_wt_bf16(cublas.handle(),
                    ws.norm_x.data(), layer.v_proj->data(), ws.v.data(),
                    N, Hk, H);
            }
        }

        // Pre-norm dumps for parity.
        if (l == 0 && !layer.is_shared) {
            dump_l0("v_pre_norm", ws.v.data(),
                    static_cast<std::size_t>(N) * num_kv_heads_local * d);
            dump_l0("q_pre_norm", ws.q.data(),
                    static_cast<std::size_t>(N) * num_q_heads_local * d);
        }

        // Per-head Q/K RMSNorm (Gemma-4 always has it).
        if (getenv("PIE_NO_QK_NORM") == nullptr) {
            kernels::launch_rmsnorm_bf16(
                ws.q.data(), layer.q_norm->data(), ws.q.data(),
                N * num_q_heads_local, d, eps, stream);
            if (!layer.is_shared) {
                kernels::launch_rmsnorm_bf16(
                    ws.k.data(), layer.k_norm->data(), ws.k.data(),
                    N * num_kv_heads_local, d, eps, stream);
                // V-Norm: pure RMSNorm (no learnable scale) on V before the
                // KV write. Gemma-4 trained against this; skipping it
                // produces gibberish even though softmax stays well-formed.
                kernels::launch_rmsnorm_no_scale_bf16(
                    ws.v.data(), ws.v.data(),
                    N * num_kv_heads_local, d, eps, stream);
            }
        }
        if (l == 0 && !layer.is_shared) {
            dump_l0("v_post_norm", ws.v.data(),
                    static_cast<std::size_t>(N) * num_kv_heads_local * d);
            dump_l0("q_post_norm", ws.q.data(),
                    static_cast<std::size_t>(N) * num_q_heads_local * d);
        }

        // RoPE: partial rotary on full-attention layers
        // (`partial_rotary_factor < 1`), full rotation otherwise.
        const float prf = w.per_layer_partial_rotary_factor[l];
        const int rotary_dim = static_cast<int>(prf * d);
        const bool partial = (prf < 1.0f) && (rotary_dim > 0);

        if (!layer.is_shared) {
            if (partial) {
                kernels::launch_rope_partial_bf16(
                    ws.q.data(), ws.k.data(), positions,
                    N, num_q_heads_local, num_kv_heads_local, d,
                    rotary_dim, w.per_layer_rope_theta[l], stream);
            } else {
                kernels::launch_rope_bf16(
                    ws.q.data(), ws.k.data(), positions,
                    N, num_q_heads_local, num_kv_heads_local, d,
                    w.per_layer_rope_theta[l], stream);
            }
        } else {
            // Shared layers: only Q gets RoPE'd here; K was rotated at
            // its source layer (where it was written to the cache).
            if (partial) {
                kernels::launch_rope_partial_bf16(
                    ws.q.data(), ws.q.data(), positions,
                    N, num_q_heads_local, /*num_kv_heads=*/0, d,
                    rotary_dim, w.per_layer_rope_theta[l], stream);
            } else {
                kernels::launch_rope_bf16(
                    ws.q.data(), ws.q.data(), positions,
                    N, num_q_heads_local, /*num_kv_heads=*/0, d,
                    w.per_layer_rope_theta[l], stream);
            }
        }

        // KV write only on non-shared layers — shared layers attend
        // through the source slot's already-populated pages.
        if (!layer.is_shared) {
            kernels::launch_write_kv_to_pages_bf16(
                cache.k(l), cache.v(l), ws.k.data(), ws.v.data(),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                N, R, cache.page_size(), num_kv_heads_local, d, stream);
        }

        // Plan + dispatch attention. Shared layers dispatch against the
        // source slot's tensors (KvCache redirects via `kv_source_layer`).
        // Gemma-4 full-attention layers run at HEAD_DIM=512, which
        // flashinfer 0.6.x's TC prefill template rejects ("Invalid
        // configuration: NUM_MMA_D_QK=32"). For prefill at 512 we fall
        // back to a naive paged-attention kernel (much slower but
        // correct); decode at 512 still uses flashinfer.
        ops::DecodePlanCachePtr decode_plan;
        if (use_decode_path) {
            decode_plan = ops::make_decode_plan();
            ops::plan_attention_flashinfer_decode_bf16(
                *decode_plan, kv_page_indptr_h, R,
                num_q_heads_local, num_kv_heads_local, d,
                cache.page_size(), attn_ws, stream);
            ops::dispatch_attention_flashinfer_decode_bf16(
                *decode_plan,
                ws.q.data(), cache.k(l), cache.v(l), ws.attn_out.data(),
                kv_page_indices, kv_page_indptr, kv_last_page_lens,
                attn_ws, stream,
                /*window_left=*/w.per_layer_window_left[l],
                /*logits_soft_cap=*/0.f,
                /*sm_scale=*/1.0f);  // Gemma-4: q/k norm absorbs 1/sqrt(d)
        } else if (d == 512) {
            ops::launch_attention_naive_paged_bf16(
                ws.q.data(), cache.k(l), cache.v(l), ws.attn_out.data(),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                N, R, num_q_heads_local, num_kv_heads_local, d,
                cache.page_size(), stream,
                /*window_left=*/w.per_layer_window_left[l],
                /*sm_scale=*/1.0f);
        } else {
            ops::launch_attention_flashinfer_prefill_bf16(
                ws.q.data(), cache.k(l), cache.v(l), ws.attn_out.data(),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indptr_h,
                N, R, num_q_heads_local, num_kv_heads_local, d,
                cache.page_size(), attn_ws, stream,
                /*window_left=*/w.per_layer_window_left[l],
                /*logits_soft_cap=*/0.f,
                /*sm_scale=*/1.0f);
        }

        dump_l0("attn_out", ws.attn_out.data(),
                static_cast<std::size_t>(N) * Hq);

        // o_proj → norm_x scratch, post-attn norm, residual-add y. Under
        // TP this is row-parallel: all-reduce the partial sums before
        // post-norm sees them.
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.attn_out.data(), layer.o_proj->data(), ws.norm_x.data(),
            N, H, Hq, /*beta=*/0.f);
        if (T > 1) {
            tp->all_reduce_bf16(ws.norm_x.data(),
                static_cast<std::size_t>(N) * H, ncclSum, stream);
        }
        dump_l0("o_proj_out", ws.norm_x.data(),
                static_cast<std::size_t>(N) * H);
        kernels::launch_rmsnorm_bf16(
            ws.norm_x.data(), layer.attn_norm_post->data(), ws.norm_y.data(),
            N, H, eps, stream);
        dump_l0("attn_norm_post", ws.norm_y.data(),
                static_cast<std::size_t>(N) * H);
        kernels::launch_residual_add_bf16(
            ws.y.data(), ws.norm_y.data(),
            static_cast<std::size_t>(N) * H, stream);
        dump_l0("post_attn_y", ws.y.data(),
                static_cast<std::size_t>(N) * H);

        // ── 3b. MLP block ──────────────────────────────────────────────
        kernels::launch_rmsnorm_bf16(
            ws.y.data(), layer.mlp_norm_pre->data(), ws.norm_x.data(),
            N, H, eps, stream);
        dump_l0("mlp_norm_pre", ws.norm_x.data(),
                static_cast<std::size_t>(N) * H);

        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.norm_x.data(), layer.gate_proj->data(), ws.gate.data(),
            N, I, H);
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.norm_x.data(), layer.up_proj->data(),   ws.up.data(),
            N, I, H);
        kernels::launch_geglu_tanh_bf16(
            ws.gate.data(), ws.up.data(), ws.gate.data(),
            N * I, stream);
        dump_l0("mlp_geglu", ws.gate.data(),
                static_cast<std::size_t>(N) * I);
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.gate.data(), layer.down_proj->data(), ws.norm_x.data(),
            N, H, I, /*beta=*/0.f);
        if (T > 1) {
            tp->all_reduce_bf16(ws.norm_x.data(),
                static_cast<std::size_t>(N) * H, ncclSum, stream);
        }
        dump_l0("mlp_down", ws.norm_x.data(),
                static_cast<std::size_t>(N) * H);

        // Gemma-4 26B-A4B's MoE block runs **alongside** the dense MLP
        // and the two branches' post-norms are summed before the final
        // `post_feedforward_layernorm`. On the dense E2B / E4B / 31B
        // variants `cfg.gemma4_enable_moe` is false and we keep the
        // straight-line dense path.
        const bool moe_active = cfg.gemma4_enable_moe &&
                                layer.router_proj != nullptr;
        if (moe_active) {
            // branch_1 = post_feedforward_layernorm_1(dense_out)
            kernels::launch_rmsnorm_bf16(
                ws.norm_x.data(), layer.mlp_norm_post_dense->data(),
                ws.norm_y.data(), N, H, eps, stream);
            // experts → moe_ws.moe_out (raw, no post-norm).
            gemma4_moe_block(layer, cfg, ws, moe_ws, N, cublas, stream);
            // branch_2 = post_feedforward_layernorm_2(moe_out) → norm_x
            // (norm_x's prior contents — dense_out — are no longer
            // needed).
            kernels::launch_rmsnorm_bf16(
                moe_ws.moe_out.data(), layer.moe_norm_post->data(),
                ws.norm_x.data(), N, H, eps, stream);
            // combined = branch_1 + branch_2 (in norm_y).
            kernels::launch_residual_add_bf16(
                ws.norm_y.data(), ws.norm_x.data(),
                static_cast<std::size_t>(N) * H, stream);
            // final = post_feedforward_layernorm(combined) → norm_x.
            kernels::launch_rmsnorm_bf16(
                ws.norm_y.data(), layer.mlp_norm_post->data(),
                ws.norm_x.data(), N, H, eps, stream);
            dump_l0("mlp_norm_post", ws.norm_x.data(),
                    static_cast<std::size_t>(N) * H);
            kernels::launch_residual_add_bf16(
                ws.y.data(), ws.norm_x.data(),
                static_cast<std::size_t>(N) * H, stream);
        } else {
            kernels::launch_rmsnorm_bf16(
                ws.norm_x.data(), layer.mlp_norm_post->data(), ws.norm_y.data(),
                N, H, eps, stream);
            dump_l0("mlp_norm_post", ws.norm_y.data(),
                    static_cast<std::size_t>(N) * H);
            kernels::launch_residual_add_bf16(
                ws.y.data(), ws.norm_y.data(),
                static_cast<std::size_t>(N) * H, stream);
        }
        dump_l0("post_mlp_y", ws.y.data(),
                static_cast<std::size_t>(N) * H);

        // ── 3c. PLE residual ───────────────────────────────────────────
        // Wrapped in a block so debugging can disable the whole step
        // (env `PIE_NO_PLE=1`) without touching the surrounding flow.
        // Skipped on Gemma-4 26B-A4B (`hidden_size_per_layer_input == 0`
        // → `ple_dim == 0`), which doesn't ship the per-layer triple at
        // all.
        if (ple_dim > 0 && getenv("PIE_NO_PLE") == nullptr) {
        dump_l0("ple_residual_in", ws.y.data(),
                static_cast<std::size_t>(N) * H);
        // Gather this layer's slice of per_layer_inputs:
        //   ple_signal[n, :] = per_layer_proj[n, l*ple_dim:(l+1)*ple_dim]
        // The base tensor is already laid out as [N, L*ple_dim]; we
        // pass the offset pointer to the GEMM as the "input" matrix
        // with stride = L*ple_dim. cuBLAS row-major convention here
        // means the kernel reads N rows of `ple_dim` cols at offset
        // l*ple_dim within each row.
        //
        // ple_gate = ple_input_gate @ y_norm (using attn output in y)
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.y.data(), layer.ple_input_gate->data(), ws.norm_x.data(),
            N, ple_dim, H);
        dump_l0("ple_gate_pre_gelu", ws.norm_x.data(),
                static_cast<std::size_t>(N) * ple_dim);
        // GeGLU(tanh) but with the per-layer input acting as the "up"
        // signal. We don't have a strided-input GeGLU kernel; instead
        // use a temporary stride-pack: copy the layer's PLE slice into
        // ws.norm_y so it's contiguous, then geglu over [N, ple_dim].
        // The slice is `[N, ple_dim]` from a `[N, L*ple_dim]` buffer.
        const std::size_t row_stride = static_cast<std::size_t>(per_layer_total);
        const std::size_t slice_off  = static_cast<std::size_t>(l) * ple_dim;
        for (int n = 0; n < N; ++n) {
            // Pack PLE slice for token n into ws.norm_y[n].
            const auto* src = per_layer_base +
                              (n * row_stride + slice_off) * bf16_bytes;
            auto* dst = static_cast<std::uint8_t*>(ws.norm_y.data()) +
                        static_cast<std::size_t>(n) * ple_dim * bf16_bytes;
            CUDA_CHECK(cudaMemcpyAsync(dst, src,
                                       ple_dim * bf16_bytes,
                                       cudaMemcpyDeviceToDevice, stream));
        }
        dump_l0("ple_signal_slice", ws.norm_y.data(),
                static_cast<std::size_t>(N) * ple_dim);
        kernels::launch_geglu_tanh_bf16(
            ws.norm_x.data(), ws.norm_y.data(), ws.norm_x.data(),
            N * ple_dim, stream);
        dump_l0("ple_gated", ws.norm_x.data(),
                static_cast<std::size_t>(N) * ple_dim);
        // Project back to hidden, post-norm, add to residual.
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.norm_x.data(), layer.ple_projection->data(), ws.norm_y.data(),
            N, H, ple_dim, /*beta=*/0.f);
        kernels::launch_rmsnorm_bf16(
            ws.norm_y.data(), layer.ple_norm->data(), ws.norm_y.data(),
            N, H, eps, stream);
        kernels::launch_residual_add_bf16(
            ws.y.data(), ws.norm_y.data(),
            static_cast<std::size_t>(N) * H, stream);
        }  // end PLE-bypass guard

        // Parity dump: residual stream after attention/MLP/PLE for
        // the first few layers.
        if (l < 4) {
            char tag[32];
            std::snprintf(tag, sizeof tag, "layer_%d_post_ple_y", l);
            dbg_sync_dump_bf16(tag, ws.y.data(),
                               static_cast<std::size_t>(N) * H);
        }

        // ── 3d. Per-layer learnable scalar ────────────────────────────
        if (layer.layer_scalar && getenv("PIE_NO_LAYER_SCALAR") == nullptr) {
            // Read the bf16 scalar to host once at this point. The
            // overhead is one cudaMemcpy of 2 bytes per layer per
            // fire — small enough to ignore at the scale of full
            // forward latency. Cache could hoist this; not needed
            // for correctness.
            const float s = read_bf16_scalar(*layer.layer_scalar);
            if (std::abs(s - 1.f) > 1e-6f) {
                kernels::launch_scalar_mul_bf16(
                    ws.y.data(), s,
                    static_cast<std::size_t>(N) * H, stream);
            }
        }

        // Post-layer_scalar dump for parity comparison against HF's
        // `hidden_states[layer+1]` (which is after the scalar mul).
        if (l < 4) {
            char tag[32];
            std::snprintf(tag, sizeof tag, "layer_%d_post_scalar_y", l);
            dbg_sync_dump_bf16(tag, ws.y.data(),
                               static_cast<std::size_t>(N) * H);
        }
    }

    // ── 4. Final norm, lm_head, optional softcap ─────────────────────
    kernels::launch_rmsnorm_bf16(
        ws.y.data(), w.final_norm->data(), ws.norm_x.data(),
        N, H, eps, stream);
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), w.lm_head->data(), ws.logits.data(),
        N, V, H);
    if (fwd_cfg.final_logit_softcap > 0.f) {
        kernels::launch_logit_softcap_bf16(
            ws.logits.data(), fwd_cfg.final_logit_softcap,
            static_cast<std::size_t>(N) * V, stream);
    }
    dbg_sync_dump_bf16("logits_last", static_cast<const std::uint16_t*>(ws.logits.data())
                                  + static_cast<std::size_t>(N - 1) * V,
                  static_cast<std::size_t>(V));
    // After the first fire, freeze the dumps so subsequent decode
    // fires don't overwrite the prefill intermediates we want to
    // parity-check.
    dbg_first_fire_flag() = false;
}

}  // namespace pie_cuda_driver::model

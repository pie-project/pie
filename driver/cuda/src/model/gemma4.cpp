#include "model/gemma4.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/embed.hpp"
#include "kernels/kv_paged.hpp"
#include "kernels/residual_add.hpp"
#include "kernels/rmsnorm.hpp"
#include "kernels/rope.hpp"
#include "kernels/scalar_mul.hpp"
#include "kernels/softcap.hpp"
#include "kernels/swiglu.hpp"
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
    const std::string p = kPrefix;
    w.embed           = &must(engine, p + "embed_tokens.weight");
    w.embed_per_layer = &must(engine, p + "embed_tokens_per_layer.weight");
    w.ple_model_proj  = &must(engine, p + "per_layer_model_projection.weight");
    w.ple_model_norm  = &must(engine, p + "per_layer_projection_norm.weight");
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
        // when `is_shared == false`.
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
        // `per_layer_projection`, `post_per_layer_input_norm`.
        Lw.ple_input_gate = &must(engine, lp + "per_layer_input_gate.weight");
        Lw.ple_projection = &must(engine, lp + "per_layer_projection.weight");
        Lw.ple_norm       = &must(engine, lp + "post_per_layer_input_norm.weight");

        // Per-layer learnable scalar.
        Lw.layer_scalar = engine.has(lp + "layer_scalar")
                              ? &engine.get(lp + "layer_scalar")
                              : nullptr;

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
inline void dbg_dump_bf16(const char* tag, const void* dev_ptr,
                          std::size_t numel) {
    static const char* dir = std::getenv("PIE_GEMMA4_DUMP_DIR");
    if (dir == nullptr) return;
    if (!dbg_first_fire_flag()) return;
    std::vector<std::uint16_t> tmp(numel);
    cudaMemcpy(tmp.data(), dev_ptr, numel * 2, cudaMemcpyDeviceToHost);
    std::string path = std::string(dir) + "/" + tag + ".bin";
    std::ofstream out(path, std::ios::binary);
    if (!out) return;
    out.write(reinterpret_cast<const char*>(tmp.data()), numel * 2);
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

}  // namespace

void gemma4_forward_paged(
    const Gemma4Weights& w,
    const HfConfig& cfg,
    const Gemma4ForwardCfg& fwd_cfg,
    Qwen3Workspace& ws,
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
    cudaDeviceSynchronize();
    dbg_dump_bf16("embed_pre_scale", ws.y.data(),
                  static_cast<std::size_t>(N) * H);
    kernels::launch_scalar_mul_bf16(
        ws.y.data(), std::sqrt(static_cast<float>(H)),
        static_cast<std::size_t>(N) * H, stream);
    cudaDeviceSynchronize();
    dbg_dump_bf16("embed_post_scale", ws.y.data(),
                  static_cast<std::size_t>(N) * H);

    // ── 2. Per-layer inputs (PLE) ────────────────────────────────────────
    // Compute once per fire; sliced per layer below.
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
    DeviceTensor per_layer_token_buf = DeviceTensor::allocate(
        DType::BF16, {N, per_layer_total});
    DeviceTensor per_layer_proj_buf = DeviceTensor::allocate(
        DType::BF16, {N, per_layer_total});
    void* per_layer_token = per_layer_token_buf.data();
    void* per_layer_proj  = per_layer_proj_buf.data();
    {
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
    cudaDeviceSynchronize();
    dbg_dump_bf16("per_layer_inputs", per_layer_proj,
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
        auto dump_l0 = [&](const char* tag, const void* p, std::size_t n) {
            if (!dump_this) return;
            std::string t = std::string("L0_") + tag;
            dbg_dump_bf16(t.c_str(), p, n);
        };
        const int d  = layer.head_dim;
        const int Hq = cfg.num_attention_heads * d;
        const int Hk = cfg.num_key_value_heads * d;
        const int I  = layer.intermediate;

        // ── 3a. Attention block ─────────────────────────────────────────
        const void* attn_residual = ws.y.data();

        kernels::launch_rmsnorm_bf16(
            ws.y.data(), layer.attn_norm_pre->data(), ws.norm_x.data(),
            N, H, eps, stream);
        cudaDeviceSynchronize();
        dump_l0("attn_norm_pre", ws.norm_x.data(),
                static_cast<std::size_t>(N) * H);

        // Q-projection always runs.
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.norm_x.data(), layer.q_proj->data(), ws.q.data(),
            N, Hq, H);
        // K/V projections only on non-shared layers.
        if (!layer.is_shared) {
            ops::gemm_act_x_wt_bf16(cublas.handle(),
                ws.norm_x.data(), layer.k_proj->data(), ws.k.data(),
                N, Hk, H);
            ops::gemm_act_x_wt_bf16(cublas.handle(),
                ws.norm_x.data(), layer.v_proj->data(), ws.v.data(),
                N, Hk, H);
        }

        // Pre-norm dumps for parity.
        if (l == 0 && !layer.is_shared) {
            cudaDeviceSynchronize();
            dump_l0("v_pre_norm", ws.v.data(),
                    static_cast<std::size_t>(N) * cfg.num_key_value_heads * d);
            dump_l0("q_pre_norm", ws.q.data(),
                    static_cast<std::size_t>(N) * cfg.num_attention_heads * d);
        }

        // Per-head Q/K RMSNorm (Gemma-4 always has it).
        if (getenv("PIE_NO_QK_NORM") == nullptr) {
            kernels::launch_rmsnorm_bf16(
                ws.q.data(), layer.q_norm->data(), ws.q.data(),
                N * cfg.num_attention_heads, d, eps, stream);
            if (!layer.is_shared) {
                kernels::launch_rmsnorm_bf16(
                    ws.k.data(), layer.k_norm->data(), ws.k.data(),
                    N * cfg.num_key_value_heads, d, eps, stream);
                // V-Norm: pure RMSNorm (no learnable scale) on V before the
                // KV write. Gemma-4 trained against this; skipping it
                // produces gibberish even though softmax stays well-formed.
                kernels::launch_rmsnorm_no_scale_bf16(
                    ws.v.data(), ws.v.data(),
                    N * cfg.num_key_value_heads, d, eps, stream);
            }
        }
        if (l == 0 && !layer.is_shared) {
            cudaDeviceSynchronize();
            dump_l0("v_post_norm", ws.v.data(),
                    static_cast<std::size_t>(N) * cfg.num_key_value_heads * d);
            dump_l0("q_post_norm", ws.q.data(),
                    static_cast<std::size_t>(N) * cfg.num_attention_heads * d);
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
                    N, cfg.num_attention_heads, cfg.num_key_value_heads, d,
                    rotary_dim, w.per_layer_rope_theta[l], stream);
            } else {
                kernels::launch_rope_bf16(
                    ws.q.data(), ws.k.data(), positions,
                    N, cfg.num_attention_heads, cfg.num_key_value_heads, d,
                    w.per_layer_rope_theta[l], stream);
            }
        } else {
            // Shared layers: only Q gets RoPE'd here; K was rotated at
            // its source layer (where it was written to the cache).
            if (partial) {
                kernels::launch_rope_partial_bf16(
                    ws.q.data(), ws.q.data(), positions,
                    N, cfg.num_attention_heads, /*num_kv_heads=*/0, d,
                    rotary_dim, w.per_layer_rope_theta[l], stream);
            } else {
                kernels::launch_rope_bf16(
                    ws.q.data(), ws.q.data(), positions,
                    N, cfg.num_attention_heads, /*num_kv_heads=*/0, d,
                    w.per_layer_rope_theta[l], stream);
            }
        }

        // KV write only on non-shared layers — shared layers attend
        // through the source slot's already-populated pages.
        if (!layer.is_shared) {
            kernels::launch_write_kv_to_pages_bf16(
                cache.k(l), cache.v(l), ws.k.data(), ws.v.data(),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                N, R, cache.page_size(), cfg.num_key_value_heads, d, stream);
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
                cfg.num_attention_heads, cfg.num_key_value_heads, d,
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
                N, R, cfg.num_attention_heads, cfg.num_key_value_heads, d,
                cache.page_size(), stream,
                /*window_left=*/w.per_layer_window_left[l],
                /*sm_scale=*/1.0f);
        } else {
            ops::launch_attention_flashinfer_prefill_bf16(
                ws.q.data(), cache.k(l), cache.v(l), ws.attn_out.data(),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indptr_h,
                N, R, cfg.num_attention_heads, cfg.num_key_value_heads, d,
                cache.page_size(), attn_ws, stream,
                /*window_left=*/w.per_layer_window_left[l],
                /*logits_soft_cap=*/0.f,
                /*sm_scale=*/1.0f);
        }

        cudaDeviceSynchronize();
        dump_l0("attn_out", ws.attn_out.data(),
                static_cast<std::size_t>(N) * Hq);

        // o_proj → norm_x scratch, post-attn norm, residual-add y.
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.attn_out.data(), layer.o_proj->data(), ws.norm_x.data(),
            N, H, Hq, /*beta=*/0.f);
        cudaDeviceSynchronize();
        dump_l0("o_proj_out", ws.norm_x.data(),
                static_cast<std::size_t>(N) * H);
        kernels::launch_rmsnorm_bf16(
            ws.norm_x.data(), layer.attn_norm_post->data(), ws.norm_y.data(),
            N, H, eps, stream);
        cudaDeviceSynchronize();
        dump_l0("attn_norm_post", ws.norm_y.data(),
                static_cast<std::size_t>(N) * H);
        kernels::launch_residual_add_bf16(
            ws.y.data(), ws.norm_y.data(),
            static_cast<std::size_t>(N) * H, stream);
        cudaDeviceSynchronize();
        dump_l0("post_attn_y", ws.y.data(),
                static_cast<std::size_t>(N) * H);

        // ── 3b. MLP block ──────────────────────────────────────────────
        kernels::launch_rmsnorm_bf16(
            ws.y.data(), layer.mlp_norm_pre->data(), ws.norm_x.data(),
            N, H, eps, stream);
        cudaDeviceSynchronize();
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
        cudaDeviceSynchronize();
        dump_l0("mlp_geglu", ws.gate.data(),
                static_cast<std::size_t>(N) * I);
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.gate.data(), layer.down_proj->data(), ws.norm_x.data(),
            N, H, I, /*beta=*/0.f);
        cudaDeviceSynchronize();
        dump_l0("mlp_down", ws.norm_x.data(),
                static_cast<std::size_t>(N) * H);
        kernels::launch_rmsnorm_bf16(
            ws.norm_x.data(), layer.mlp_norm_post->data(), ws.norm_y.data(),
            N, H, eps, stream);
        cudaDeviceSynchronize();
        dump_l0("mlp_norm_post", ws.norm_y.data(),
                static_cast<std::size_t>(N) * H);
        kernels::launch_residual_add_bf16(
            ws.y.data(), ws.norm_y.data(),
            static_cast<std::size_t>(N) * H, stream);
        cudaDeviceSynchronize();
        dump_l0("post_mlp_y", ws.y.data(),
                static_cast<std::size_t>(N) * H);

        // ── 3c. PLE residual ───────────────────────────────────────────
        // Wrapped in a block so debugging can disable the whole step
        // (env `PIE_NO_PLE=1`) without touching the surrounding flow.
        if (getenv("PIE_NO_PLE") == nullptr) {
        cudaDeviceSynchronize();
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
        cudaDeviceSynchronize();
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
        cudaDeviceSynchronize();
        dump_l0("ple_signal_slice", ws.norm_y.data(),
                static_cast<std::size_t>(N) * ple_dim);
        kernels::launch_geglu_tanh_bf16(
            ws.norm_x.data(), ws.norm_y.data(), ws.norm_x.data(),
            N * ple_dim, stream);
        cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
            char tag[32];
            std::snprintf(tag, sizeof tag, "layer_%d_post_ple_y", l);
            dbg_dump_bf16(tag, ws.y.data(),
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
            cudaDeviceSynchronize();
            char tag[32];
            std::snprintf(tag, sizeof tag, "layer_%d_post_scalar_y", l);
            dbg_dump_bf16(tag, ws.y.data(),
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
    cudaDeviceSynchronize();
    dbg_dump_bf16("logits_last", static_cast<const std::uint16_t*>(ws.logits.data())
                                  + static_cast<std::size_t>(N - 1) * V,
                  static_cast<std::size_t>(V));
    // After the first fire, freeze the dumps so subsequent decode
    // fires don't overwrite the prefill intermediates we want to
    // parity-check.
    dbg_first_fire_flag() = false;
}

}  // namespace pie_cuda_driver::model

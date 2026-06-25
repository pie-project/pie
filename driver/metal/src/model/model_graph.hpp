#pragma once

// ModelGraph — the polymorphic per-architecture forward interface, plus the
// data-driven factory that selects the concrete builder from `ModelConfig`.
//
// Mirrors the cuda driver's `IModel` (model/imodel.hpp) and portable's
// `compute_()` graph dispatch: one runtime switch on `PieArch` picks the
// shared Llama-like builder or the Gemma builder. Adding an arch is a
// directory-local change here — no compile-time coupling elsewhere.
//
// A `forward()` call builds a *lazy* MLX graph (the returned `Tensor` is not
// yet evaluated) producing the logits `[n_slots, vocab]` (token-major, the
// locked convention — see ops/tensor.hpp). The executor (beta) owns
// evaluation + sampling; this dir owns only the math of the forward pass.

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "ops/tensor.hpp"
#include "config.hpp"
#include "weights.hpp"
#include "model/model_kv.hpp"   // KvCacheView — model->KV accessor seam

namespace pie_metal_driver {

class LinearStateCache;  // driver/metal/src/linear_state_cache.hpp (delta-owned)

namespace model {

// ForwardBatch — per-fire inputs the executor stages for the graph. The
// MLX `Tensor`s are device-resident i32 arrays the builder threads into the
// graph; the scalars/flags describe batch geometry and the chosen sampling
// fast-path. This is the proposed seam with beta's executor (reconcile the
// exact field set on integration — see #mac).
struct ForwardBatch {
    // ── Token / position inputs ──
    Tensor token_ids;   // i32 [n_total]
    Tensor positions;   // i32 [n_total]

    // Rows of the final hidden state that produce logits (the last token of
    // each request for decode; a subset for prefill). i32 [n_slots].
    Tensor logit_rows;

    // ── Paged attention metadata (layout owned by delta's KV header) ──
    // Index/pointer arrays addressing the paged K/V cache for this batch.
    Tensor kv_page_indices;    // i32 [total_pages_in_batch]
    Tensor kv_page_indptr;     // i32 [n_requests + 1]
    Tensor kv_last_page_lens;  // i32 [n_requests]
    Tensor qo_indptr;          // i32 [n_requests + 1]
    // Physical KV write slots for this batch's new tokens. i32 [n_total].
    Tensor kv_write_indices;

    // ── Geometry / flags ──
    std::int32_t n_total     = 0;   // total tokens this fire
    std::int32_t n_requests  = 0;
    std::int32_t n_slots     = 0;   // logit rows
    bool         pure_decode = false;

    // ── qwen3.6 hybrid linear-attention (Gated DeltaNet) seam ──
    // Populated by the executor (alpha) only for hybrid models; null/absent on
    // every other arch (zero effect on gemma4 / qwen2 / llama). The graph owns
    // the decoder-layer -> linear-layer-ordinal mapping (counts 'l' layers in
    // arch_spec.layer_pattern) and passes these straight to beta's
    // gated_delta_net.
    //   lin_cache     : delta's per-request conv+recurrent state store (null =
    //                   non-hybrid -> the graph never touches a linear layer).
    //   slot_ids      : i32 [n_requests], each request's persistent state slot.
    //   qo_indptr_host: [n_requests + 1] host CSR token spans for the varlen
    //                   (prefill / mixed) path; empty on the pure-decode path.
    LinearStateCache*       lin_cache = nullptr;
    std::optional<Tensor>   slot_ids;
    std::vector<int>        qo_indptr_host;

    // ── host-side paged-KV CSR (full-attention paged_attention) ──
    // Same host metadata that backs the kv_page_indptr / kv_last_page_lens
    // device tensors, kept on the host so paged_attention can read its
    // per-request loop bounds without a mid-forward GPU->CPU readback (those
    // .eval() syncs are pipeline bubbles at batch=1 AND block mx::compile of
    // the decode step). Populated by the executor for every arch; empty only
    // on paths that build ForwardBatch without paged KV.
    std::vector<int>        kv_page_indptr_host;
    std::vector<int>        kv_last_page_lens_host;
};

class ModelGraph {
public:
    virtual ~ModelGraph() = default;

    // Build the lazy forward graph; returns logits `[n_slots, vocab]`
    // (token-major: one row per sampling slot; sampler does argmax over axis=1).
    virtual Tensor forward(const ForwardBatch& batch, KvCacheView& kv) = 0;

    virtual const ModelConfig& config() const = 0;
};

// Data-driven factory: switches on `cfg.arch` to construct the right concrete
// graph (LlamaLikeGraph for llama/qwen/mistral/MoE, GemmaGraph for gemma2/3).
// Throws on PieArch::Unknown / unimplemented archs.
std::unique_ptr<ModelGraph> make_model_graph(const ModelConfig& cfg,
                                             ModelWeights weights);

// Data-driven weight binding: switches on `cfg.arch` to pick the matching
// `bind_*` builder (bind_llama_like for llama/qwen/mistral/MoE, bind_gemma for
// gemma2/3), resolving HF tensor names from `src` into a ModelWeights. This is
// the single entry point the loader (delta) / runtime assembly (alpha) calls —
// no need to branch on arch at the call site. Throws on unimplemented archs.
ModelWeights bind_weights(const WeightSource& src, const ModelConfig& cfg);

}  // namespace model
}  // namespace pie_metal_driver

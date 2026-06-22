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

#include "ops/tensor.hpp"
#include "config.hpp"
#include "weights.hpp"
#include "model/model_kv.hpp"   // KvCacheView — model->KV accessor seam

namespace pie_metal_driver {
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
};

class ModelGraph {
public:
    virtual ~ModelGraph() = default;

    // Build the lazy forward graph; returns logits `[vocab, n_slots]`.
    virtual Tensor forward(const ForwardBatch& batch, KvCacheView& kv) = 0;

    virtual const ModelConfig& config() const = 0;
};

// Data-driven factory: switches on `cfg.arch` to construct the right concrete
// graph (LlamaLikeGraph for llama/qwen/mistral/MoE, GemmaGraph for gemma2/3).
// Throws on PieArch::Unknown / unimplemented archs.
std::unique_ptr<ModelGraph> make_model_graph(const ModelConfig& cfg,
                                             ModelWeights weights);

}  // namespace model
}  // namespace pie_metal_driver

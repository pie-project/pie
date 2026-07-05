#pragma once
// gemma4_heap_bind.hpp — gemma4 weight-staging + per-ordinal arg-table binding (alpha).
//
// The gemma4 analog of delta's qwen3.6 heap_bind. Split into a pure NAME REGISTRY
// (gemma4_heap_bind.cpp, testable vs the real checkpoint with no Metal) and the Metal
// staging/bind that lands every weight/state/KV/IO slot into alpha's resident heap.
//
// Lane split vs the scratch ACTIVATION dataflow:
//   * here (delta-style): load-once weights, KV pages (L0..14 non-shared only, with the
//     KV-share source indirection on the shared layers' Sdpa), the per-token IO scalars,
//     and the const geometry params (gemma4_consts.cpp).
//   * gemma4_decode_run.cpp wires the per-dispatch activation X/Out (no_recycle dataflow)
//     + the per-layer ple_input slice offset.
//
// Checkpoint facts (verified vs ~/models/4bit-tied/gemma-4-E2B-it):
//   * Text decoder under `model.language_model.layers.{L}.*`; final norm
//     `model.language_model.norm.weight`. Skip audio_tower/vision_tower/embed_*.
//   * TIED embeddings: top-level 4-bit `lm_head.{weight,scales,biases}` is the canonical
//     stored side, bound to BOTH the main EmbedGather and the LmHead matmul.
//   * PLE tables: `model.language_model.embed_tokens_per_layer.{triple}` (per-token gather,
//     row width = n_layers*ple_dim = 8960), `per_layer_model_projection.{triple}`,
//     `per_layer_projection_norm.weight` (256-wide).
//   * Per layer: input_layernorm, post_attention_layernorm, pre_feedforward_layernorm,
//     post_feedforward_layernorm, post_per_layer_input_norm, self_attn.{q,k,v,o}_proj
//     (4-bit triples), self_attn.{q,k}_norm.weight, mlp.{gate,up,down}_proj (4-bit triples),
//     per_layer_input_gate / per_layer_projection (4-bit triples), layer_scalar (bf16[1]).
//   * k/v_proj + k_norm are stored for ALL 35 layers but only L0..14 (non-shared) compute
//     them — the shared layers' copies are vestigial; we stage/bind ONLY the non-shared set
//     (the DAG never emits Qmv{K,V}/KNorm/KvAppend for shared layers).
//
// All quant linears are affine g64/b4.

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "gemma4_abi.hpp"
#include "gemma4_decode_step.hpp"  // Gemma4Dispatch
#include "mtl4_context.hpp"
#include "safetensors_view.hpp"

namespace pie_metal_driver::raw_metal::gemma4 {

// ── 1. Pure name registry (no Metal) ─────────────────────────────────────────

struct WeightBind {
    uint8_t     bind_index;  // bind::<Kind> slot
    std::string tensor;      // HF checkpoint tensor name
};

// HF tensor-name helpers (the verified gemma4 layout).
std::string gemma4_layer_prefix(int layer);   // model.language_model.layers.{L}.
std::vector<std::string> quant_triple(const std::string& base);  // .weight/.scales/.biases

// Load-once WEIGHT tensors a single dispatch binds (empty for weight-less kinds whose
// inputs are scratch/KV/IO/const). `layer` = -1 for the layer-less PLE precompute / tail.
std::vector<WeightBind> gemma4_weight_binds(Kernel kind, int layer, const Gemma4Geometry& g);

// Complete ordered unique tensor list to stage for the gemma4 text decode (tied lm_head
// once; non-shared k/v/k_norm only; skips audio/vision towers).
std::vector<std::string> gemma4_weight_tensors(const Gemma4Geometry& g);

// ── 2. Staging + bind (Metal) ────────────────────────────────────────────────

struct Gemma4Kv {            // per non-shared layer (L0..14): own paged K/V cache
    SlotHandle k_pages, v_pages;
};

struct BoundGemma4 {
    std::unordered_map<std::string, SlotHandle> weights;
    std::vector<Gemma4Kv> kv;                  // size n_layers; only non-shared populated
    SlotHandle io_token, io_position, io_seqlen;
    SlotHandle logits;        // LmHead output (bf16, vocab) — I3
    SlotHandle logits_capped; // FinalSoftcap output (bf16, vocab) — argmax/golden reads this
};

// Stage every weight/KV/IO slot into the resident heap. `max_ctx` sizes the KV cache.
BoundGemma4 stage_gemma4_weights(RawMetalContext& ctx, const SafetensorsView& view,
                                 const Gemma4Geometry& g, int max_ctx);

// Walk the DAG; bind weight / KV / IO slots for each dispatch by ordinal (the KV-share
// source indirection on shared layers' Sdpa lives here). Scratch activations are wired by
// the caller (gemma4_decode_run).
void bind_gemma4_weights(RawMetalContext& ctx, const BoundGemma4& b,
                         const std::vector<Gemma4Dispatch>& dag, const Gemma4Geometry& g);

// Bind every geometry-derived const param by ordinal over the DAG (RmsParams plus_one=0,
// sdpa scale=1.0 + per-layer window, per-type rope theta, qmv K/N, geglu/softcap/etc.).
// Returns the number of const slots bound. `max_ctx` matches the staged KV stride.
int bind_gemma4_consts(RawMetalContext& ctx, const std::vector<Gemma4Dispatch>& dag,
                       const Gemma4Geometry& g, int max_ctx);

}  // namespace pie_metal_driver::raw_metal::gemma4

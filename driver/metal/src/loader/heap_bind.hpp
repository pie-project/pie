#pragma once
// heap_bind.hpp — delta's weight-staging + arg-table binding for the raw-Metal decode.
//
// Two layers, split so the NAME REGISTRY is pure C++ (testable vs the real checkpoint
// with no Metal dependency — see heap_bind_probe), while the actual heap_alloc/memcpy/
// arg_bind wiring lives in heap_bind.cpp behind alpha's RawMetalContext.
//
//   1. NAME REGISTRY (heap_bind_names.cpp, pure):
//        weight_binds(kind, layer, geom)  -> the (bind_index, HF tensor) pairs a single
//                                            dispatch needs as LOAD-ONCE weights.
//        decode_weight_tensors(geom)      -> the complete unique tensor list to stage
//                                            (text decode only; skips model.visual.* /
//                                            mtp.*; tied lm_head appears ONCE).
//   2. STAGING + BIND (heap_bind.cpp, Metal):
//        stage_decode_weights(...)        -> heap_alloc + memcpy every weight/state/KV/IO
//                                            slot, returns a BoundDecode.
//        bind_decode_dag(...)             -> walk beta's build_decode_dag, bind each
//                                            dispatch's weight/state/KV/IO slots by ORDINAL
//                                            (beta separately binds the X/Out scratch).
//
// Ownership seam (manager): delta lays out the heap + binds the LOAD-ONCE weights, the
// persistent GDN state, the KV pages, and the IO scalars. beta binds the per-dispatch
// activation/scratch X/Out (his WAR/WAW ping-pong schedule) over the SAME ordinals.
//
// Checkpoint facts baked in (verified vs ~/models/4bit-tied/Qwen3.5-0.8B, alpha+delta):
//   * Decoder weights are under `model.language_model.layers.{L}.*` (note the
//     `.language_model.` infix), final norm `model.language_model.norm.weight`.
//   * embed_tokens.weight is ABSENT — tied; the canonical stored side is the top-level
//     4-bit `lm_head.{weight,scales,biases}`, bound to BOTH bind::Embed and QmvLmHead.
//   * GDN in_proj_a/in_proj_b are DENSE bf16 [16,1024] (NOT 4-bit); everything else in
//     {in_proj_qkv,in_proj_z,out_proj} + attn/mlp projections is 4-bit g64.
//   * conv1d has NO bias (`linear_attn.conv1d.bias` absent) — GdnCore::ConvB is bound to
//     a zeroed slot.
//   * Skip `model.visual.*` (Qwen3-VL vision tower) + `mtp.*`.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "decode_abi.hpp"

namespace pie::metal {

// ── 1. Pure name registry (no Metal) ─────────────────────────────────────────

// One load-once weight a dispatch binds: which kernel bind-index <- which HF tensor.
struct WeightBind {
    uint8_t     bind_index;  // bind::<Kind> slot
    std::string tensor;      // HF checkpoint tensor name
};

// The load-once WEIGHT tensors a single dispatch (kind at `layer`) binds. `layer` = -1
// for singletons (EmbedGather/FinalRms/QmvLmHead/Argmax). Returns empty for weight-less
// kinds (QSplit/Rope/RopeK/Sdpa-Q/AttnGate/SiluMul/Residual/KvAppend/Argmax) — their
// non-weight slots (scratch X/Out, KV pages, IO scalars, const params) are bound
// elsewhere (beta's scratch; delta's KV/IO in bind_decode_dag).
std::vector<WeightBind> weight_binds(Kernel kind, int layer, const DecodeGeometry& g,
                                     bool gdn_prep = false);

// Complete ordered list of UNIQUE weight tensors to stage for text decode (tied lm_head
// once; skips visual/mtp). Order: lm_head bundle, final norm, then per layer in model
// order. Used for staging + Weights-region byte accounting (vs heap_layout).
std::vector<std::string> decode_weight_tensors(const DecodeGeometry& g);

// HF tensor-name helpers (the verified `.language_model.` layout).
std::string layer_prefix(int layer);                 // model.language_model.layers.{L}.
std::string final_norm_name();                       // model.language_model.norm.weight
std::vector<std::string> quant_triple(const std::string& base);  // .weight/.scales/.biases

}  // namespace pie::metal

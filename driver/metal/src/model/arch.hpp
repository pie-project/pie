#pragma once

// PieArch — the set of model architectures the metal driver understands,
// plus the data-driven mapping from an HF `config.json` `model_type` string
// onto one of them. Mirrors driver/portable's hf_config PieArch enum and
// driver/cuda's arch-detect chain, pared down to the architectures the
// metal driver implements first (Llama-like + Qwen3, then Gemma2/3).
//
// Arch selection is purely data-driven: the loader parses `model_type`
// (and, for multimodal wrappers, the inner text_config.model_type) and
// calls `hf_model_type_to_pie_arch`. Nothing downstream branches on a
// compile-time arch constant — the graph dispatcher (model_graph.cpp)
// switches on this enum at runtime, exactly like portable's compute_().

#include <string>

namespace pie_metal_driver::model {

enum class PieArch {
    // ── Llama-like dense family (shared graph builder) ──
    Llama3,     // HF "llama"
    Qwen2,      // HF "qwen2"   — adds Q/K/V bias
    Qwen3,      // HF "qwen3"   — adds per-head Q/K RMSNorm
    Mistral3,   // HF "mistral" / "mistral3" — all-layer sliding window

    // ── Llama-like MoE (shared builder + MoE FFN) ──
    Qwen3Moe,   // HF "qwen3_moe"
    Mixtral,    // HF "mixtral"

    // ── Gemma dense family (separate builder) ──
    Gemma2,     // HF "gemma2" — softcaps, alternating SWA
    Gemma3,     // HF "gemma3" / "gemma3_text" — per-head qk-norm, every-6 SWA
    Gemma4,     // HF "gemma4" / "gemma4_text" — PLE, KV-share, parallel MoE,
                // per-attn-type rope, per-layer scalar

    // ── Qwen3.6 (Qwen3.5 hybrid family) — own builder ──
    Qwen36,     // HF "qwen3_5"/"qwen3_5_moe"/"qwen3_6"/"qwen3.6" — gated
                // DeltaNet linear-attn + full-attn, output-gate, partial rope, MoE

    Unknown,
};

// Human-readable arch name (for logs / errors).
const char* pie_arch_name(PieArch a);

// Map an HF `model_type` (e.g. "qwen3", "llama") onto its PieArch. Returns
// PieArch::Unknown for types the metal driver does not implement yet, so the
// caller can emit a clear "unsupported architecture" diagnostic rather than
// silently mis-dispatching.
PieArch hf_model_type_to_pie_arch(const std::string& hf_model_type);

// True for archs routed through the shared Llama-like dense/MoE builder
// (everything except the Gemma family). Used by the dispatcher.
bool is_llama_like(PieArch a);

}  // namespace pie_metal_driver::model

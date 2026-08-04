#pragma once

#include "kernels/affine_format.hpp"

/// Decode geometry for the llama-shaped families.
///
/// This is the plain transformer decoder every one of `llama`, `llama3`,
/// `mistral`, `qwen2`, `qwen3` and `qwen3_moe` is: an RMSNorm sandwich, GQA
/// with NEOX RoPE, and a gated MLP. Nothing here is new to this driver --
/// gemma4 is this shape plus a norm sandwich, per-layer embeddings and a
/// sliding/full split, and qwen3.5 is this shape plus interleaved linear
/// attention. What makes those two families their own is exactly what they
/// add; a family that adds nothing still needs its own geometry, because the
/// *shape* is the thing that varies.
///
/// Two axes vary across the families this covers, and both are fields rather
/// than separate families:
///
///  * **QK-norm.** Qwen3 RMS-normalises q and k per head before the rotation;
///    llama, llama3 and mistral do not. The checkpoint says which by shipping
///    `self_attn.q_norm` or not.
///  * **The FFN.** Dense SwiGLU, or a routed mixture. Qwen3-MoE is Qwen3's
///    attention with the MLP replaced, so it is this geometry with `n_experts`
///    set rather than a fourth family.
///
/// Weights arrive as MLX affine-U4 triplets, the same as every other family
/// here (`mlx_lm convert -q --q-bits 4 --q-group-size 64 --q-mode affine`).

#include <cstddef>
#include <string>

#include "../shared_kernels.hpp"

namespace pie::metal::llama {

struct LlamaGeometry {
    int hidden = 4096;
    int n_layers = 32;
    int vocab = 128256;
    float eps = 1e-5f;

    /// Whether `lm_head` IS `embed_tokens`. Llama 3 8B ships both; Qwen3's
    /// smaller sizes tie them. The contract maps a tied head onto
    /// `shared_embedding` and an untied one onto its own `lm_head`, so this
    /// picks which pair of dispatch kinds the ends of the DAG use.
    bool tied_embeddings = false;

    int n_q_heads = 32;
    int n_kv_heads = 8;
    int head_dim = 128;
    /// The affine width and group every quantized kernel name here is spelled
    /// with. See `AffineFormat`: they are one fact, the config states it, and
    /// a pipeline built for the wrong pair answers instead of failing.
    AffineFormat quant{4, 64};

    /// Qwen3 RMS-normalises q and k per head before the rotation. Not a scale
    /// difference -- the norm is over `head_dim` with its own learned weight,
    /// and omitting it on a checkpoint that has one is a wrong model that still
    /// produces fluent text.
    bool qk_norm = false;

    float rope_theta = 500000.0f;
    /// Linear position scale, 1.0 for none. Under `llama3` this is that
    /// schedule's `factor` instead, and it divides FREQUENCIES rather than
    /// positions -- which is why the two cannot share a code path and why
    /// approximating one with the other runs and is wrong past the original
    /// context.
    float rope_scale = 1.0f;
    /// Llama 3.1's piecewise schedule, when `rope_freq_table` is true.
    /// `rope_scaling_factor` is its `factor` -- kept apart from `rope_scale`
    /// because the two act on different things: this one divides the
    /// frequencies inside the table, that one divides the position outside it,
    /// and letting one field mean both applies the factor twice.
    float rope_scaling_factor = 1.0f;
    float rope_low_freq_factor = 1.0f;
    float rope_high_freq_factor = 4.0f;
    int rope_original_max_position = 8192;
    /// Whether the rotary frequencies are a TABLE rather than a geometric
    /// series in `rope_theta`. One predicate, asked by the PSO choice, the
    /// constant binding and the kernel's ABI alike -- they must agree, and a
    /// disagreement is a rotation by the wrong angle rather than a crash.
    bool rope_freq_table = false;

    /// Dense SwiGLU width. Unused when `is_moe()`.
    int intermediate = 14336;

    /// Routed mixture. `n_experts == 0` is a dense FFN, which is the whole
    /// difference between qwen3 and qwen3_moe on this side.
    int n_experts = 0;
    int experts_per_token = 0;
    int moe_intermediate = 0;

    /// Tokens the KV cache is sized for. The append and attention strides are
    /// read off it, so there is no useful default: a stride guessed here is
    /// silently wrong attention.
    int kv_max_ctx = 0;

    int max_tokens = 1;
    int max_requests = 1;
    int max_slots = 1;
    int kv_page_size = 32;
    int total_pages = 1;
    bool paged_kv_enabled = false;

    bool is_moe() const { return n_experts > 0 && experts_per_token > 0; }
    /// The width one expert's gate/up produce, or the dense width.
    int ffn_width() const { return is_moe() ? moe_intermediate : intermediate; }
    /// Full rotary: every family here rotates the whole head.
    int rotary_dims() const { return head_dim; }
    int q_width() const { return n_q_heads * head_dim; }
    int kv_width() const { return n_kv_heads * head_dim; }
};

/// Build the geometry a config describes, or report why it cannot be built.
///
/// Returns false rather than filling in defaults: a checkpoint whose shape this
/// driver silently guessed at would produce plausible-looking wrong tokens.
template <class Facts>
inline bool geometry_from_facts(const Facts& f, LlamaGeometry& out, std::string* err) {
    const auto refuse = [&](const std::string& why) {
        if (err != nullptr) *err = "llama geometry: " + why;
        return false;
    };
    if (!f.present()) return refuse("config carried no decoder shape");
    if (f.n_layers <= 0 || f.hidden <= 0) {
        return refuse("n_layers and hidden_size must both be positive");
    }
    if (f.n_q_heads <= 0 || f.n_kv_heads <= 0) {
        return refuse("head counts must be positive");
    }
    if (f.n_q_heads % f.n_kv_heads != 0) {
        return refuse("n_q_heads " + std::to_string(f.n_q_heads) +
                      " is not a multiple of n_kv_heads " + std::to_string(f.n_kv_heads) +
                      ", which GQA requires");
    }
    // `llama3` is a table (see `llama3_inv_freq`); linear and default are the
    // plain geometric series. Anything else -- longrope, dynamic -- is refused,
    // because approximating a per-channel schedule with the linear scale runs
    // and is wrong past the original context length.
    if (!f.rope_scaling_kind.empty() && f.rope_scaling_kind != "linear" &&
        f.rope_scaling_kind != "default" && f.rope_scaling_kind != "llama3") {
        return refuse("rope_scaling type '" + f.rope_scaling_kind +
                      "' is not implemented; only an absent, linear or llama3 scaling is");
    }

    out.n_layers = f.n_layers;
    out.hidden = f.hidden;
    out.vocab = f.vocab;
    out.eps = f.eps;
    out.n_q_heads = f.n_q_heads;
    out.n_kv_heads = f.n_kv_heads;
    out.head_dim = f.head_dim > 0 ? f.head_dim : f.hidden / f.n_q_heads;
    out.qk_norm = f.qk_norm;
    out.tied_embeddings = f.tied_embeddings;
    out.rope_theta = f.rope_theta;
    out.rope_scale = f.rope_scale > 0.0f ? f.rope_scale : 1.0f;
    out.rope_freq_table = f.rope_scaling_kind == "llama3";
    out.rope_low_freq_factor = f.rope_low_freq_factor;
    out.rope_high_freq_factor = f.rope_high_freq_factor;
    out.rope_original_max_position = f.rope_original_max_position;
    if (out.rope_freq_table) {
        // The table divides frequencies; the kernel's own `scale` divides
        // positions. Leaving `factor` in both applies it twice, which is a
        // rope that runs at a plausible-looking wrong rate.
        out.rope_scaling_factor = out.rope_scale;
        if (out.rope_low_freq_factor == out.rope_high_freq_factor) {
            return refuse("llama3 rope_scaling needs low_freq_factor != high_freq_factor; "
                          "both are " + std::to_string(out.rope_low_freq_factor));
        }
        if (out.rope_original_max_position <= 0) {
            return refuse("llama3 rope_scaling needs a positive "
                          "original_max_position_embeddings");
        }
        out.rope_scale = 1.0f;
    }
    out.intermediate = f.intermediate;
    out.n_experts = f.n_experts;
    out.experts_per_token = f.experts_per_token;
    out.moe_intermediate = f.moe_intermediate;

    if (out.head_dim <= 0) {
        return refuse("head_dim is absent and hidden/n_q_heads is not positive either");
    }
    if (out.vocab <= 0) return refuse("vocab_size must be positive");
    if (out.n_experts > 0 || out.experts_per_token > 0) {
        if (out.experts_per_token <= 0) {
            return refuse("a routed FFN needs num_experts_per_tok");
        }
        if (out.experts_per_token > out.n_experts) {
            return refuse("experts_per_token " + std::to_string(out.experts_per_token) +
                          " exceeds n_experts " + std::to_string(out.n_experts));
        }
        if (out.moe_intermediate <= 0) {
            return refuse("a routed FFN needs moe_intermediate_size");
        }
        // `router_topk` holds the chosen logits in a fixed threadgroup array
        // and clamps k to its size. Clamping silently would route with fewer
        // experts than the config asks for while every consumer still strides
        // by the configured k, so the refusal the kernel documents lives here.
        if (out.experts_per_token > shared_kernels::kRouterMaxTopK) {
            return refuse("experts_per_token " + std::to_string(out.experts_per_token) +
                          " exceeds the router's top-k limit of " +
                          std::to_string(shared_kernels::kRouterMaxTopK));
        }
        // One lane per expert, one threadgroup per row: the expert count is
        // bounded by the threadgroup size. `router_topk_dispatch` clamps, which
        // would route among the first 1024 experts and never mention it.
        if (out.n_experts > shared_kernels::kRouterMaxExperts) {
            return refuse("n_experts " + std::to_string(out.n_experts) +
                          " exceeds the " + std::to_string(shared_kernels::kRouterMaxExperts) +
                          " a single threadgroup can rank");
        }
        // The router softmaxes the SELECTED logits, so its weights sum to one
        // over the chosen experts -- exactly `norm_topk_prob: true`. A config
        // that says false wants weights from the softmax over ALL experts,
        // which sum to less than one and scale the FFN's whole contribution
        // down. Same tokens, quietly wrong magnitudes; refuse instead.
        if (!f.norm_topk_prob) {
            return refuse("norm_topk_prob is false; the router normalizes over "
                          "the selected experts only");
        }
    } else if (out.intermediate <= 0) {
        return refuse("a dense FFN needs intermediate_size");
    }
    return true;
}

}  // namespace pie::metal::llama

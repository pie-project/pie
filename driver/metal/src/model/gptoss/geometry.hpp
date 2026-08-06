#pragma once

/// GPT-OSS decode geometry.
///
/// Read off `openai/gpt-oss-20b`, whose shape the defaults carry. Everything is
/// runtime state: compiling one checkpoint's shape into the type is what stopped
/// a second family from loading at all, and this is the third.
///
/// Four things make this family's shape different from the two already here,
/// and all four are in the schedule rather than in the arithmetic:
///
///  * **Every layer is a sparse MoE.** A router picks `experts_per_token` of
///    `n_experts`, and only those experts' weights are read. So the MLP's cost
///    is `experts_per_token / n_experts` of a dense one, but its DISPATCH shape
///    depends on a value the GPU computes — which is why the routed matmul
///    cannot be a plain `qmv`.
///  * **Attention has per-head sinks.** One learned scalar per query head, added
///    to the softmax denominator as though there were an extra key. It makes a
///    head able to attend to nothing.
///  * **The projections are biased.** `attention_bias: true`, and the experts
///    carry biases too, so every matvec here has an additive epilogue that
///    neither other family needs.
///  * **Alternating sliding/full attention**, and the window is 128 — small
///    enough that it binds at almost every realistic context length, unlike
///    gemma4's 512.

#include <cstddef>
#include <string>

#include "../shared_kernels.hpp"

namespace pie::metal::gptoss {

struct GptOssGeometry {
    int hidden = 2880;
    int n_layers = 24;
    int vocab = 201088;
    float eps = 1e-5f;
    /// gpt-oss ships a separate `lm_head`; it does not reuse the embedding.
    bool tied_embeddings = false;

    int n_q_heads = 64;
    int n_kv_heads = 8;
    int head_dim = 64;

    /// YaRN. `factor` scales the effective context; the driver needs the same
    /// four numbers `mlx_lm`'s `YarnRoPE` takes.
    float rope_theta = 150000.0f;
    float rope_factor = 32.0f;
    float rope_beta_fast = 32.0f;
    float rope_beta_slow = 1.0f;
    int rope_original_max_position = 4096;

    /// Sliding layers attend the last `sliding_window` positions. Layer 0 is
    /// sliding and they alternate, which `is_sliding` states directly rather
    /// than deriving from an interval the way gemma4 does -- gpt-oss's
    /// `layer_types` is a strict alternation, not every-Nth.
    int sliding_window = 128;

    // ── MoE ──
    int n_experts = 32;
    int experts_per_token = 4;
    /// One expert's MLP width. NOT the dense intermediate: the layer computes
    /// `experts_per_token` of these and sums them.
    int intermediate = 2880;
    /// The clamp gpt-oss applies inside its SwiGLU, and the sigmoid's gain.
    float swiglu_limit = 7.0f;
    float swiglu_alpha = 1.702f;

    /// The router's quantization width, which is NOT the bank's: `mlx_lm`'s
    /// quantization predicate leaves a 32 x 2880 router at 8 bits while
    /// everything around it goes to 4, but a checkpoint quantized uniformly
    /// ships a 4-bit one. Nothing in `config.json` says which, so this is
    /// solved from the staged tensors (`router_bits_from_extents`) rather than
    /// assumed -- an 8-bit kernel over 4-bit nibbles reads half the row as the
    /// wrong values and routes to the wrong experts, which survives as fluent
    /// wrong text rather than as a crash.
    int router_bits = 8;

    /// The attention/embedding projections' quantization width, which is a
    /// third thing again: `mlx-community/gpt-oss-20b-MXFP4-Q8` declares a
    /// global `{mxfp4, 4, group 32}` and then overrides `embed_tokens`, every
    /// `q/k/v/o_proj`, every `mlp.router` and `lm_head` back to
    /// `{affine, 8, group 64}`. Two `4`s were hardcoded against that -- the
    /// `b_4` PSO suffix here and the `kGptOssBase` pair that builds the
    /// prefill GEMM tables -- so an 8-bit checkpoint's projections were read
    /// by 4-bit matvecs at half the stride. That is not a crash: it is noise
    /// that looks like text. Solved from the staged tensors for the same
    /// reason `router_bits` is, and by the same arithmetic -- these
    /// projections are affine group-64 whatever their width, which is exactly
    /// what `router_bits_from_extents` assumes.
    int proj_bits = 4;

    /// The expert bank is read in the MXFP4 the checkpoint published, rather
    /// than dequantized and re-quantized to affine U4 at load. Solved the same
    /// way `router_bits` is -- from the tensors that were staged -- because
    /// `config.json` says "mxfp4" for a checkpoint the loader may still have
    /// converted, and the only honest witness of which happened is what is in
    /// the heap. Chooses both what is bound and which matvec runs.
    bool mxfp4_experts = false;

    int max_tokens = 1;
    int max_requests = 1;
    int max_slots = 1;
    int kv_page_size = 32;
    int total_pages = 1;
    int kv_max_ctx = 0;
    bool paged_kv_enabled = false;

    /// Sliding and full alternate, starting sliding. Verified against the
    /// checkpoint's `layer_types`, which lists exactly that.
    bool is_sliding(int layer) const { return (layer % 2) == 0; }
    bool is_full_attn(int layer) const { return !is_sliding(layer); }

    int q_dim() const { return n_q_heads * head_dim; }
    int kv_dim() const { return n_kv_heads * head_dim; }
    int gqa_factor() const { return n_kv_heads > 0 ? n_q_heads / n_kv_heads : 1; }
};

/// Build the geometry a config describes, or report why it cannot be built.
///
/// Returns false rather than filling in defaults: a checkpoint whose shape this
/// driver silently guessed at would produce plausible-looking wrong tokens.
template <class Facts>
inline bool geometry_from_facts(const Facts& f, GptOssGeometry& out, std::string* err) {
    if (!f.present()) {
        if (err) *err = "gpt-oss geometry: config carried no decoder shape";
        return false;
    }
    if (f.n_experts <= 0 || f.experts_per_token <= 0) {
        if (err) *err = "gpt-oss geometry: every layer is a sparse MoE, so both "
                        "`num_local_experts` and `num_experts_per_tok` must be positive";
        return false;
    }
    if (f.experts_per_token > f.n_experts) {
        if (err) *err = "gpt-oss geometry: `num_experts_per_tok` exceeds `num_local_experts`";
        return false;
    }
    // The same two bounds llama's routed branch refuses, because it is the same
    // `router_topk`: it holds the chosen logits in a fixed threadgroup array and
    // ranks one expert per lane, and it CLAMPS to both. Clamping silently would
    // route with fewer experts than the config asks for, or among the first
    // 1024, while every consumer keeps striding by the configured k --
    // `go_kind_width` sizes the expert stacks at `experts_per_token *
    // intermediate` whatever the kernel actually filled.
    if (f.experts_per_token > shared_kernels::kRouterMaxTopK) {
        if (err) *err = "gpt-oss geometry: `num_experts_per_tok` " +
                        std::to_string(f.experts_per_token) +
                        " exceeds the router's top-k limit of " +
                        std::to_string(shared_kernels::kRouterMaxTopK);
        return false;
    }
    if (f.n_experts > shared_kernels::kRouterMaxExperts) {
        if (err) *err = "gpt-oss geometry: `num_local_experts` " + std::to_string(f.n_experts) +
                        " exceeds the " + std::to_string(shared_kernels::kRouterMaxExperts) +
                        " a single threadgroup can rank";
        return false;
    }
    if (f.n_kv_heads <= 0 || f.n_q_heads % f.n_kv_heads != 0) {
        if (err) *err = "gpt-oss geometry: the query heads do not divide evenly into the "
                        "key/value heads, which grouped attention assumes";
        return false;
    }
    out.n_layers = f.n_layers;
    out.hidden = f.hidden;
    out.vocab = f.vocab;
    out.eps = f.eps;
    out.n_q_heads = f.n_q_heads;
    out.n_kv_heads = f.n_kv_heads;
    out.head_dim = f.head_dim;
    out.sliding_window = f.sliding_window;
    out.n_experts = f.n_experts;
    out.experts_per_token = f.experts_per_token;
    out.intermediate = f.intermediate;
    out.swiglu_limit = f.swiglu_limit;
    out.rope_theta = f.rope_theta;
    out.rope_factor = f.rope_factor;
    out.rope_beta_fast = f.rope_beta_fast;
    out.rope_beta_slow = f.rope_beta_slow;
    out.rope_original_max_position = f.rope_original_max_position;
    return true;
}

}  // namespace pie::metal::gptoss

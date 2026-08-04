#pragma once

#include "kernels/affine_format.hpp"

/// Gemma 4 decode geometry.
///
/// Ported from the bring-up harness (`tools/rawmetal/gemma4_abi.hpp`), whose
/// defaults were read off `google/gemma-4-E2B-it` and which this file keeps as
/// the defaults. Everything here is runtime state: the driver used to compile
/// one checkpoint's shape into the type, and that is what stopped a second
/// family from loading at all.
///
/// Three things make this family's shape different from qwen3.5's, and all
/// three are in the schedule rather than the kernels:
///
///  * **head_dim is per attention type.** Sliding layers use `head_dim`; full
///    layers use `global_head_dim`. So are the RoPE base frequency and the
///    partial-rotary factor.
///  * **The tail of the stack shares KV.** The last `num_kv_shared_layers`
///    layers re-attend the most recent earlier layer of the SAME attention
///    type; they have no k/v projections of their own (`mlx_lm`'s
///    `gemma4_text.py` does not even construct them).
///  * **The MLP doubles over exactly that range** (`use_double_wide_mlp`).

#include <cstddef>
#include <string>

namespace pie::metal::gemma4 {

struct Gemma4Geometry {
    int hidden = 1536;
    int n_layers = 35;
    int vocab = 262144;
    float eps = 1e-6f;
    bool tied_embeddings = true;
    /// Gemma 4 stores plain RMS weights, NOT the `(1 + w)` gain earlier Gemmas
    /// used. Getting this wrong is an ~80% per-norm error that the residual
    /// stream compounds — it is the defect that cost qwen3.5 bring-up a day.
    bool norm_plus_one = false;

    int n_q_heads = 8;
    int n_kv_heads = 1;
    /// Sliding layers. Full layers use `global_head_dim`.
    int head_dim = 256;
    /// The affine width and group every quantized kernel name here is spelled
    /// with. See `AffineFormat`: they are one fact, the config states it, and
    /// a pipeline built for the wrong pair answers instead of failing.
    AffineFormat quant{4, 64};

    /// The affine format of the DENSE FFN and the router, when it differs.
    ///
    /// mlx_lm quantizes per tensor, and gemma-4-26B's predicate spares its
    /// dense `mlp.{gate,up,down}_proj` and its `router.proj` at 8 bits while
    /// everything else is 4. One format for the whole checkpoint would run a
    /// 4-bit pipeline over 8-bit bytes: it compiles, it binds, it dispatches,
    /// it is fast, and every token is wrong. `{0, 0}` means "one format", which
    /// is every other checkpoint here.
    AffineFormat ffn_quant{0, 0};
    bool has_alt_quant() const { return ffn_quant.bits != 0 && ffn_quant.group != 0; }

    int global_head_dim = 512;
    /// Full layers rotate a quarter of their head; sliding layers rotate all of it.
    float full_partial_rotary = 0.25f;

    float rope_theta_global = 1.0e6f;
    float rope_theta_local = 1.0e4f;

    int sliding_window = 512;

    /// Base width; doubles on the KV-shared range when `double_wide_mlp`.
    int intermediate = 6144;
    bool double_wide_mlp = true;

    /// Per-Layer Embeddings: a second table `n_layers * per_layer_emb_dim` wide.
    int per_layer_emb_dim = 256;

    int num_kv_shared_layers = 20;

    // ── the mixture (gemma-4-26B-A4B), absent on the dense members ──
    //
    // The routed branch sits BESIDE the dense MLP rather than replacing it:
    // both read the post-attention residual, and their outputs are added. So
    // `intermediate` still means the dense width and `moe_intermediate` is a
    // second, much narrower one -- 2112 against 704 on the 26B.
    bool enable_moe = false;
    int n_experts = 0;
    int experts_per_token = 0;
    int moe_intermediate = 0;
    bool is_moe() const { return enable_moe && n_experts > 0 && experts_per_token > 0; }

    /// Full-attention layers take V from the K PROJECTION instead of projecting
    /// their own, and carry a KV head count of their own.
    ///
    /// `V = v_norm(k_proj(x))` while `K = rope(k_norm(k_proj(x)))`: the two
    /// diverge after one shared matvec. Those layers ship no `v_proj` at all,
    /// so this is not an optimisation to skip -- it is which weights exist.
    bool attention_k_eq_v = false;
    int n_global_kv_heads = 0;
    bool k_is_v(int layer) const { return attention_k_eq_v && is_full_attn(layer); }

    /// The KV head count of a layer. A genuinely per-layer axis on this family:
    /// the 26B carries 2 on its full-attention layers against 8 on its sliding
    /// ones, so one scalar either over-allocates the cache or truncates it.
    int n_kv_heads_of(int layer) const {
        return (attention_k_eq_v && is_full_attn(layer) && n_global_kv_heads > 0)
                   ? n_global_kv_heads
                   : n_kv_heads;
    }

    /// `out = cap * tanh(logits / cap)`; 0 disables.
    float final_softcap = 30.0f;

    /// Tokens the KV cache is sized for. The SDPA and append strides are read
    /// off it ([n_kv_heads, kv_max_ctx, head_dim]), so it has no useful default:
    /// a stride guessed here is silently wrong attention.
    int kv_max_ctx = 0;

    int max_tokens = 1;
    int max_requests = 1;
    int max_slots = 1;
    int kv_page_size = 32;
    int total_pages = 1;
    bool paged_kv_enabled = false;

    /// One full-attention layer every `full_attn_interval`-th, counting from 1 —
    /// so layers 4, 9, 14, ... on E2B. Verified against the checkpoint's
    /// `layer_types`, which lists exactly those.
    int full_attn_interval = 5;
    bool is_full_attn(int layer) const {
        return full_attn_interval <= 1 || ((layer + 1) % full_attn_interval) == 0;
    }
    bool is_sliding(int layer) const { return !is_full_attn(layer); }

    /// Attention-type-dependent shapes.
    int head_dim_of(int layer) const { return is_full_attn(layer) ? global_head_dim : head_dim; }
    float rope_theta_of(int layer) const {
        return is_full_attn(layer) ? rope_theta_global : rope_theta_local;
    }
    int rotary_dims_of(int layer) const {
        const int hd = head_dim_of(layer);
        return is_full_attn(layer) ? static_cast<int>(full_partial_rotary * float(hd)) : hd;
    }

    int first_kv_shared() const { return n_layers - num_kv_shared_layers; }
    bool is_kv_shared(int layer) const {
        return num_kv_shared_layers > 0 && layer >= first_kv_shared();
    }
    /// Which layer's KV pages `layer` reads: itself when it owns them, else the
    /// most recent earlier owning layer of the same attention type. -1 means the
    /// config describes a stack whose shared layers have no source, which is a
    /// config error rather than something to paper over at run time.
    int kv_source(int layer) const {
        if (!is_kv_shared(layer)) return layer;
        const bool want_sliding = is_sliding(layer);
        for (int j = first_kv_shared() - 1; j >= 0; --j) {
            if (is_sliding(j) == want_sliding) return j;
        }
        return -1;
    }

    /// The MLP is double-wide exactly where the KV is shared.
    int intermediate_of(int layer) const {
        return double_wide_mlp && is_kv_shared(layer) ? 2 * intermediate : intermediate;
    }

    int n_full_attn() const {
        int n = 0;
        for (int L = 0; L < n_layers; ++L) n += is_full_attn(L) ? 1 : 0;
        return n;
    }
    /// Layers that actually own KV pages — the only ones the KV region sizes for.
    int n_kv_owning() const {
        int n = 0;
        for (int L = 0; L < n_layers; ++L) n += is_kv_shared(L) ? 0 : 1;
        return n;
    }
};

/// Build the geometry a config describes, or report why it cannot be built.
///
/// Returns false rather than filling in defaults: a checkpoint whose shape this
/// driver silently guessed at would produce plausible-looking wrong tokens.
template <class Facts>
inline bool geometry_from_facts(const Facts& f, Gemma4Geometry& out, std::string* err) {
    if (!f.present()) {
        if (err) *err = "gemma4 geometry: config carried no text_config shape";
        return false;
    }
    if (f.full_attn_interval < 0) {
        if (err) {
            *err = "gemma4 geometry: `layer_types` is not a regular interval, which the "
                   "decode DAG's schedule assumes";
        }
        return false;
    }
    out.n_layers = f.n_layers;
    out.hidden = f.hidden;
    out.intermediate = f.intermediate;
    out.n_q_heads = f.n_q_heads;
    out.n_kv_heads = f.n_kv_heads;
    out.head_dim = f.head_dim;
    out.global_head_dim = f.global_head_dim > 0 ? f.global_head_dim : f.head_dim;
    out.sliding_window = f.sliding_window;
    out.num_kv_shared_layers = f.num_kv_shared_layers;
    out.per_layer_emb_dim = f.per_layer_emb_dim;
    out.double_wide_mlp = f.double_wide_mlp;
    out.final_softcap = f.final_softcap;
    out.rope_theta_global = f.rope_theta_full;
    out.rope_theta_local = f.rope_theta_sliding;
    out.full_partial_rotary = f.full_partial_rotary;
    out.enable_moe = f.enable_moe;
    out.n_experts = f.n_experts;
    out.experts_per_token = f.experts_per_token;
    out.moe_intermediate = f.moe_intermediate;
    out.attention_k_eq_v = f.attention_k_eq_v;
    out.n_global_kv_heads = f.n_global_kv_heads;
    if (f.full_attn_interval > 0) out.full_attn_interval = f.full_attn_interval;
    // Two ways a config can claim a mixture it does not describe. Both refuse
    // rather than fall back to dense: a driver that quietly ran the dense half
    // of a mixture would produce fluent, wrong text.
    if (out.enable_moe && (out.n_experts <= 0 || out.experts_per_token <= 0)) {
        if (err) {
            *err = "gemma4 geometry: `enable_moe_block` is set but the config names no "
                   "experts to route between";
        }
        return false;
    }
    if (out.is_moe() && out.moe_intermediate <= 0) {
        if (err) {
            *err = "gemma4 geometry: a mixture of " + std::to_string(out.n_experts) +
                   " experts with no `moe_intermediate_size` -- the routed projections "
                   "have no width";
        }
        return false;
    }
    // A shared layer with no source is a config this driver cannot schedule.
    for (int L = 0; L < out.n_layers; ++L) {
        if (out.is_kv_shared(L) && out.kv_source(L) < 0) {
            if (err) {
                *err = "gemma4 geometry: layer " + std::to_string(L) +
                       " shares KV but no earlier layer of its attention type owns any";
            }
            return false;
        }
    }
    return true;
}

}  // namespace pie::metal::gemma4

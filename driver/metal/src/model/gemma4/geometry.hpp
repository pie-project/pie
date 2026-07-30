#pragma once

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

    /// `out = cap * tanh(logits / cap)`; 0 disables.
    float final_softcap = 30.0f;

    int q_group = 64;
    int q_bits = 4;

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

}  // namespace pie::metal::gemma4

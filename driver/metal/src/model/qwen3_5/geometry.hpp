#pragma once

#include "kernels/affine_format.hpp"

#include <cstddef>

namespace pie::metal {

struct DecodeGeometry {
    int hidden = 1024;
    int n_layers = 24;
    int vocab = 248320;
    float eps = 1e-6f;
    bool tied_embeddings = true;

    int n_q_heads = 8;
    int n_kv_heads = 2;
    int head_dim = 256;
    /// The affine width and group every quantized kernel name here is spelled
    /// with. See `AffineFormat`: they are one fact, the config states it, and
    /// a pipeline built for the wrong pair answers instead of failing.
    AffineFormat quant{4, 64};
    /// The format the ROUTING projections are in when it is not the model's.
    ///
    /// mlx_lm quantizes per tensor and spares the two that decide where a token
    /// goes: Qwen3.6-35B-A3B's `mlp.gate` (hidden -> one logit per expert) and
    /// `mlp.shared_expert_gate` (hidden -> one) are 8-bit inside a 4-bit
    /// checkpoint. Read as 4-bit they still produce finite, plausible numbers --
    /// the router's logits came out at cosine 0.84 to mlx-lm's and the shared
    /// gate's at cosine 1.0 and 0.56 of the magnitude, which is a mixture
    /// routing to almost the right experts and weighting them wrongly.
    ///
    /// Both are hidden-by-a-few projections, so they are dispatched as matvecs
    /// off a second pipeline table rather than being given an alternate GEMM:
    /// 256 columns against the 17408 of an FFN is not where a prefill's time is.
    AffineFormat alt_quant{0, 0};
    bool has_alt_quant() const { return alt_quant.bits != 0 && alt_quant.group != 0; }
    int rotary_dims = 64;     // derived from partial_rotary_factor * head_dim
    float rope_theta = 1e7f;  // `config.json`'s rope_parameters overrides
    int mrope_section[3] = {11, 11, 10};

    int gdn_k_heads = 16;
    int gdn_v_heads = 16;
    int gdn_k_dim = 128;
    int gdn_v_dim = 128;
    int gdn_conv_k = 4;
    int gdn_conv_dim = 6144;
    int gdn_v_total = 2048;

    int intermediate = 3584;

    /// Routed mixture. `n_experts == 0` is a dense FFN. Same three fields the
    /// llama geometry carries, and for the same reason: the difference between
    /// a dense and a routed decoder is these, not a different family.
    int n_experts = 0;
    int experts_per_token = 0;
    /// False routes with weights from the softmax over ALL experts, which sum
    /// to less than one. See `RouterParams::softmax_over_all`.
    bool norm_topk_prob = true;
    int moe_intermediate = 0;
    /// The bank is stored in the MXFP4 the checkpoint published rather than
    /// re-quantized to affine U4 at load. It changes what is BOUND -- MXFP4 has
    /// block exponents and no zero point, so there is no `.biases` to bind --
    /// which is why a codec belongs in the geometry and not only in a kernel
    /// name. Solved from the staged tensors, never assumed.
    bool mxfp4_experts = false;
    /// The dense FFN every routed member runs BESIDE the bank, on every token,
    /// added to the mixture under a one-scalar-per-token sigmoid gate. Zero
    /// only for a routing that has none -- and no member of this family
    /// actually ships that way, which is why it is not optional in practice.
    int shared_intermediate = 0;

    int max_tokens = 1;
    int max_requests = 1;
    int max_slots = 1;
    int kv_page_size = 32;
    int total_pages = 1;
    bool paged_kv_enabled = false;

    // Which layers use full attention rather than linear/sliding. qwen3.5 puts
    // one every `full_attn_interval`; a family with no linear attention sets the
    // interval to 1, which makes every layer qualify. Runtime, not `constexpr`,
    // because the interval is a property of the checkpoint and this driver is no
    // longer built around exactly one of them.
    int full_attn_interval = 4;
    bool is_full_attn(int layer) const {
        return full_attn_interval <= 1 ||
               (layer % full_attn_interval) == (full_attn_interval - 1);
    }

    std::size_t gdn_conv_stride_bytes() const {
        return std::size_t(gdn_conv_dim) * std::size_t(gdn_conv_k) * 4u;
    }

    std::size_t gdn_recurrent_stride_bytes() const {
        return std::size_t(gdn_v_heads) * std::size_t(gdn_v_dim) *
               std::size_t(gdn_k_dim) * 4u;
    }

    bool is_moe() const { return n_experts > 0 && experts_per_token > 0; }
    bool has_shared_expert() const { return is_moe() && shared_intermediate > 0; }
    /// The width one expert's gate/up produce, or the dense width.
    int ffn_width() const { return is_moe() ? moe_intermediate : intermediate; }
};

}  // namespace pie::metal

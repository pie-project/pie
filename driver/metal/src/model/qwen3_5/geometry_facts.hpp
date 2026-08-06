#pragma once

/// Turning a config's facts into this family's `DecodeGeometry`.
///
/// Split from `geometry.hpp` because it needs the router's kernel limits and
/// `geometry.hpp` is a leaf that `decode_abi.hpp` includes before `Kernel`
/// exists. The POD stays where the ABI can see it; the part that can REFUSE
/// lives here, one layer up, where it may read what the kernels can do.

#include <string>

#include "../shared_kernels.hpp"
#include "geometry.hpp"

namespace pie::metal {

/// Build the geometry a config describes, or say why it cannot be built.
///
/// The whole point is that it can FAIL. This family's geometry used to be a
/// default-constructed struct whose defaults were one preview checkpoint's
/// dimensions, so every other checkpoint of the family ran at the wrong shape
/// and said nothing -- the loader binds by name and a name says nothing about
/// a dimension. Anything not derivable is refused rather than defaulted.
template <class Facts>
inline bool geometry_from_facts(const Facts& f, DecodeGeometry& out, std::string* err) {
    const auto refuse = [&](const std::string& why) {
        if (err != nullptr) *err = "qwen3.5 geometry: " + why;
        return false;
    };
    if (!f.present()) return refuse("config carried no decoder shape");
    if (f.n_layers <= 0 || f.hidden <= 0) {
        return refuse("num_hidden_layers and hidden_size must both be positive");
    }
    if (f.n_q_heads <= 0 || f.n_kv_heads <= 0) {
        return refuse("attention head counts must be positive");
    }
    if (f.n_q_heads % f.n_kv_heads != 0) {
        return refuse("num_attention_heads " + std::to_string(f.n_q_heads) +
                      " is not a multiple of num_key_value_heads " +
                      std::to_string(f.n_kv_heads) + ", which GQA requires");
    }
    if (f.vocab <= 0) return refuse("vocab_size must be positive");

    out.n_layers = f.n_layers;
    out.hidden = f.hidden;
    out.vocab = f.vocab;
    out.eps = f.eps;
    out.n_q_heads = f.n_q_heads;
    out.n_kv_heads = f.n_kv_heads;
    out.head_dim = f.head_dim > 0 ? f.head_dim : f.hidden / f.n_q_heads;
    out.tied_embeddings = f.tied_embeddings;
    out.intermediate = f.intermediate;
    if (out.head_dim <= 0) {
        return refuse("head_dim is absent and hidden/num_attention_heads is not positive");
    }

    // ── the linear-attention block ──
    //
    // Refused rather than defaulted for the same reason as everything else,
    // but the stakes are higher here: the convolution and recurrent state
    // strides are computed from these, and a stride that is wrong reads one
    // head's state as another's. That is not a crash, it is a fluent model
    // with the wrong memory.
    if (f.gdn_k_heads <= 0 || f.gdn_v_heads <= 0 || f.gdn_k_dim <= 0 || f.gdn_v_dim <= 0) {
        return refuse("the linear-attention block needs linear_num_key_heads, "
                      "linear_num_value_heads, linear_key_head_dim and "
                      "linear_value_head_dim");
    }
    if (f.gdn_conv_k <= 0) return refuse("the linear-attention block needs "
                                         "linear_conv_kernel_dim");
    out.gdn_k_heads = f.gdn_k_heads;
    out.gdn_v_heads = f.gdn_v_heads;
    out.gdn_k_dim = f.gdn_k_dim;
    out.gdn_v_dim = f.gdn_v_dim;
    out.gdn_conv_k = f.gdn_conv_k;
    // DERIVED, not read. The convolution runs over q, k and v concatenated and
    // the value total is what the output projection consumes, so a config
    // cannot state either of them inconsistently with the head counts -- and
    // neither can this driver.
    out.gdn_v_total = f.gdn_v_heads * f.gdn_v_dim;
    out.gdn_conv_dim = 2 * f.gdn_k_heads * f.gdn_k_dim + out.gdn_v_total;

    if (f.full_attn_interval < 0) {
        return refuse("layer_types lists an irregular full-attention pattern; this "
                      "driver places one every fixed interval and will not round "
                      "an irregular stack to a regular one");
    }
    if (f.full_attn_interval == 0) {
        return refuse("the config states no full_attention_interval and no layer_types; "
                      "which layers are linear cannot be guessed");
    }
    out.full_attn_interval = f.full_attn_interval;

    // ── the FFN ──
    if (f.n_experts > 0 || f.experts_per_token > 0) {
        if (f.experts_per_token <= 0) return refuse("a routed FFN needs num_experts_per_tok");
        if (f.experts_per_token > f.n_experts) {
            return refuse("num_experts_per_tok " + std::to_string(f.experts_per_token) +
                          " exceeds num_experts " + std::to_string(f.n_experts));
        }
        if (f.moe_intermediate <= 0) return refuse("a routed FFN needs moe_intermediate_size");
        if (f.experts_per_token > shared_kernels::kRouterMaxTopK) {
            return refuse("num_experts_per_tok " + std::to_string(f.experts_per_token) +
                          " exceeds the router's top-k limit of " +
                          std::to_string(shared_kernels::kRouterMaxTopK));
        }
        if (f.n_experts > shared_kernels::kRouterMaxExperts) {
            return refuse("num_experts " + std::to_string(f.n_experts) + " exceeds the " +
                          std::to_string(shared_kernels::kRouterMaxExperts) +
                          " a single threadgroup can rank");
        }
        // Not refused any more: `RouterParams::softmax_over_all` gives the
        // router the other denominator, so a config that wants the softmax
        // over every expert gets it rather than being turned away.
        out.norm_topk_prob = f.norm_topk_prob;
        // A shared expert runs beside the routed bank on every token. Every
        // released routed member of this family has one, so it is carried
        // rather than refused. A shared width that is not a multiple of the
        // quantization group is still refused, for the same reason the routed
        // width is: the packed weights would not line up.
        if (f.shared_expert_intermediate < 0) {
            return refuse("shared_expert_intermediate_size is negative");
        }
        if (f.decoder_sparse_step != 1) {
            return refuse("decoder_sparse_step " + std::to_string(f.decoder_sparse_step) +
                          " routes only some layers; this driver routes every layer or none");
        }
        if (f.mlp_only_layer_count != 0) {
            return refuse("mlp_only_layers exempts " + std::to_string(f.mlp_only_layer_count) +
                          " layers from routing; this driver routes every layer or none");
        }
        out.n_experts = f.n_experts;
        out.experts_per_token = f.experts_per_token;
        out.moe_intermediate = f.moe_intermediate;
        out.shared_intermediate = f.shared_expert_intermediate;
    } else if (f.intermediate <= 0) {
        return refuse("a dense FFN needs intermediate_size");
    }
    return true;
}


}  // namespace pie::metal

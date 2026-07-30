#pragma once

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
    int q_group = 64;
    int q_bits = 4;

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
};

}  // namespace pie::metal

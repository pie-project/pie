// Host bridge: Qwen3VLVisionWeights (bound DeviceTensors) → QwenVisRawWeights.
// See qwen3_vl_vision_adapter.hpp. Compiled by the host C++ compiler (g++), so
// it may include the toml++-heavy model headers. Mirrors gemma4_vision_adapter.cpp.

#include "model/qwen3_vl_vision_adapter.hpp"

#include <cmath>
#include <stdexcept>

namespace pie_cuda_driver::model {

namespace {
using bf = __nv_bfloat16;
const bf* P(const DeviceTensor* t) {
    return t ? static_cast<const bf*>(t->data()) : nullptr;
}
QVisLinear to_lin(const DeviceTensor* w, const DeviceTensor* b) {
    QVisLinear r;
    r.w = P(w);
    r.b = P(b);
    return r;
}
QVisLayerNorm to_ln(const DeviceTensor* g, const DeviceTensor* b) {
    QVisLayerNorm r;
    r.g = P(g);
    r.b = P(b);
    return r;
}
QVisMerger to_merger(const Qwen3VLVisionMergerWeights& m) {
    QVisMerger r;
    r.norm = to_ln(m.norm_weight, m.norm_bias);
    r.fc1 = to_lin(m.fc1_weight, m.fc1_bias);
    r.fc2 = to_lin(m.fc2_weight, m.fc2_bias);
    r.is_postshuffle = m.use_postshuffle_norm;
    return r;
}
// Read a shape dimension, defaulting if the tensor or axis is absent.
int dim_or(const DeviceTensor* t, int axis, int fallback) {
    if (!t) return fallback;
    const auto& s = t->shape();
    return axis < static_cast<int>(s.size()) ? static_cast<int>(s[axis]) : fallback;
}
}  // namespace

QwenVisRawWeights to_vis_raw_qwen(const Qwen3VLVisionWeights& w) {
    QwenVisRawWeights r;
    r.patch = to_lin(w.patch_weight, w.patch_bias);
    r.pos_embed = P(w.pos_embed);

    // Dims from config (with shape-derived fallbacks, robust to config gaps).
    r.hidden = w.config.hidden_size;
    r.heads = w.config.num_heads;
    r.head_dim = r.heads > 0 ? r.hidden / r.heads : 64;
    r.intermediate = w.config.intermediate_size;
    r.patch_size = w.config.patch_size;
    r.temporal_patch_size = w.config.temporal_patch_size;
    r.spatial_merge_size = w.config.spatial_merge_size;
    r.in_channels = w.config.in_channels;
    r.out_hidden = w.config.out_hidden_size;
    r.num_pos_embed = dim_or(w.pos_embed, 0, w.config.num_position_embeddings);
    // num_grid_per_side = int(sqrt(num_pos_embed)) (48 for 2304).
    r.num_grid_per_side = static_cast<int>(0.5 + std::sqrt((double)r.num_pos_embed));
    r.ln_eps = 1e-6f;
    r.rope_theta = 10000.0f;

    r.blocks.reserve(w.layers.size());
    for (const auto& L : w.layers) {
        QVisBlock o;
        o.norm1 = to_ln(L.norm1_weight, L.norm1_bias);
        o.norm2 = to_ln(L.norm2_weight, L.norm2_bias);
        o.qkv = to_lin(L.qkv_weight, L.qkv_bias);
        o.o = to_lin(L.proj_weight, L.proj_bias);
        o.fc1 = to_lin(L.fc1_weight, L.fc1_bias);
        o.fc2 = to_lin(L.fc2_weight, L.fc2_bias);
        r.blocks.push_back(o);
    }

    r.merger = to_merger(w.merger);
    r.deepstack.reserve(w.deepstack.size());
    for (const auto& m : w.deepstack) r.deepstack.push_back(to_merger(m));
    r.deepstack_layer_idx = w.deepstack_layer_idx;

    if (r.hidden != r.heads * r.head_dim)
        throw std::runtime_error("to_vis_raw_qwen: hidden != heads*head_dim");
    return r;
}

}  // namespace pie_cuda_driver::model

// Host bridge: Gemma4VisionWeights (bound DeviceTensors) → VisRawWeights.
// See gemma4_vision_adapter.hpp. Compiled by the host C++ compiler (g++), so
// it may include the toml++-heavy model headers.

#include "model/gemma4_vision_adapter.hpp"

#include <stdexcept>

namespace pie_cuda_driver::model {

namespace {
using bf = __nv_bfloat16;
const bf* P(const DeviceTensor* t) {
    return t ? static_cast<const bf*>(t->data()) : nullptr;
}
VisClipRaw to_clip(const Gemma4ClippedLinear& c) {
    VisClipRaw r;
    r.w = P(c.weight);
    r.imin = P(c.input_min);
    r.imax = P(c.input_max);
    r.omin = P(c.output_min);
    r.omax = P(c.output_max);
    return r;
}
// Read a shape dimension, defaulting if the tensor or axis is absent.
int dim_or(const DeviceTensor* t, int axis, int fallback) {
    if (!t) return fallback;
    const auto& s = t->shape();
    return axis < static_cast<int>(s.size()) ? static_cast<int>(s[axis]) : fallback;
}
}  // namespace

VisRawWeights to_vis_raw(const Gemma4VisionWeights& w) {
    VisRawWeights r;
    r.patch_w = P(w.patch_input_proj);
    r.pos_table = P(w.patch_position_embedding);
    r.embed_proj = P(w.embed_vision_projection);

    r.hidden = w.config.hidden_size;
    r.heads = w.config.num_attention_heads;
    r.intermediate = w.config.intermediate_size;
    r.pool_kernel = w.config.pooling_kernel_size;
    r.eps = w.config.rms_norm_eps;
    r.theta = w.config.rope_theta;
    // Derive from tensor shapes (robust to config gaps):
    //   position_embedding_table : [2, position_embedding_size, hidden]
    //   embedding_projection     : [text_hidden, hidden]
    r.pos_table_size = dim_or(w.patch_position_embedding, 1, 10240);
    r.text_hidden = dim_or(w.embed_vision_projection, 0, 2560);

    r.layers.reserve(w.layers.size());
    for (const auto& L : w.layers) {
        VisLayerRaw o;
        o.in_ln = P(L.input_layernorm);
        o.post_attn_ln = P(L.post_attention_layernorm);
        o.pre_ff_ln = P(L.pre_feedforward_layernorm);
        o.post_ff_ln = P(L.post_feedforward_layernorm);
        o.q_norm = P(L.q_norm);
        o.k_norm = P(L.k_norm);
        o.q = to_clip(L.q_proj);
        o.k = to_clip(L.k_proj);
        o.v = to_clip(L.v_proj);
        o.o = to_clip(L.o_proj);
        o.gate = to_clip(L.gate_proj);
        o.up = to_clip(L.up_proj);
        o.down = to_clip(L.down_proj);
        r.layers.push_back(o);
    }
    return r;
}

void run_gemma4_vision(const Gemma4VisionWeights& w,
                       const __nv_bfloat16* pixel,
                       const float* pos,
                       const int* grp,
                       int n_patch,
                       int out_len,
                       __nv_bfloat16* out_proj,
                       cudaStream_t stream) {
    run_gemma4_vision(to_vis_raw(w), pixel, pos, grp, n_patch, out_len, out_proj, stream);
}

}  // namespace pie_cuda_driver::model

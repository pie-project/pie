// The tower launcher's body: rebuild `QwenVisRawWeights` from the flat
// tables and hand off to the parity-anchored C++ walk. Marshalling only —
// every launch and every byte of host prep is `qwen3_vl_tower.cu`'s.

#include "vision/qwen3_vl_tower_c.hpp"

#include <cmath>

#include "vision/qwen3_vl_tower.hpp"

namespace pie_cuda_driver::kernels::vision {

namespace {

using model::QVisBlock;
using model::QVisLayerNorm;
using model::QVisLinear;
using model::QVisMerger;

using bf = __nv_bfloat16;

const bf* as_bf(const void* p) { return static_cast<const bf*>(p); }

QVisLayerNorm ln(const void* g, const void* b) {
    QVisLayerNorm out;
    out.g = as_bf(g);
    out.b = as_bf(b);
    return out;
}

QVisLinear lin(const void* w, const void* b) {
    QVisLinear out;
    out.w = as_bf(w);
    out.b = as_bf(b);
    return out;
}

// The six-pointer merger table: [norm.g, norm.b, fc1.w, fc1.b, fc2.w, fc2.b].
QVisMerger merger_of(const void* const* t, bool postshuffle) {
    QVisMerger m;
    m.norm = ln(t[0], t[1]);
    m.fc1 = lin(t[2], t[3]);
    m.fc2 = lin(t[4], t[5]);
    m.is_postshuffle = postshuffle;
    return m;
}

}  // namespace

void qwen3vl_scatter(
    const void* patch_w, const void* patch_b, const void* pos_embed,
    const void* const* block_w, int depth,
    const void* const* merger_w,
    const void* const* deepstack_w, const int* deepstack_layers,
    int hidden, int heads, int intermediate, int patch_size,
    int temporal_patch, int merge_size, int in_channels, int out_hidden,
    int num_pos_embed, float ln_eps, float rope_theta,
    const float* pixels_h, const std::uint32_t* pixel_byte_indptr_h,
    const std::uint32_t* grids_h, const std::uint32_t* anchor_rows_h,
    int num_images,
    void* hidden_rows, int n_rows,
    void* deepstack_scratch, int num_deep,
    cublasHandle_t blas, cudaStream_t stream) {
    model::QwenVisRawWeights w;
    w.patch = lin(patch_w, patch_b);
    w.pos_embed = as_bf(pos_embed);
    w.blocks.resize(static_cast<std::size_t>(depth));
    for (int i = 0; i < depth; ++i) {
        const void* const* t = block_w + static_cast<std::size_t>(i) * 12;
        QVisBlock& blk = w.blocks[static_cast<std::size_t>(i)];
        blk.norm1 = ln(t[0], t[1]);
        blk.norm2 = ln(t[2], t[3]);
        blk.qkv = lin(t[4], t[5]);
        blk.o = lin(t[6], t[7]);
        blk.fc1 = lin(t[8], t[9]);
        blk.fc2 = lin(t[10], t[11]);
    }
    w.merger = merger_of(merger_w, /*postshuffle=*/false);
    w.deepstack.resize(static_cast<std::size_t>(num_deep));
    w.deepstack_layer_idx.resize(static_cast<std::size_t>(num_deep));
    for (int d = 0; d < num_deep; ++d) {
        w.deepstack[static_cast<std::size_t>(d)] = merger_of(
            deepstack_w + static_cast<std::size_t>(d) * 6, /*postshuffle=*/true);
        w.deepstack_layer_idx[static_cast<std::size_t>(d)] = deepstack_layers[d];
    }
    w.hidden = hidden;
    w.heads = heads;
    w.head_dim = heads > 0 ? hidden / heads : 0;
    w.intermediate = intermediate;
    w.patch_size = patch_size;
    w.temporal_patch_size = temporal_patch;
    w.spatial_merge_size = merge_size;
    w.in_channels = in_channels;
    w.out_hidden = out_hidden;
    w.num_pos_embed = num_pos_embed;
    w.num_grid_per_side =
        static_cast<int>(std::lround(std::sqrt(static_cast<double>(num_pos_embed))));
    w.ln_eps = ln_eps;
    w.rope_theta = rope_theta;

    model::Qwen3VLVisionInputs in;
    in.weights = &w;
    in.pixels_h = pixels_h;
    in.pixel_byte_indptr_h = pixel_byte_indptr_h;
    in.grids_h = grids_h;
    in.anchor_rows_h = anchor_rows_h;
    in.num_images = num_images;

    model::scatter_qwen3vl_vision(
        in, static_cast<bf*>(hidden_rows), n_rows, out_hidden,
        static_cast<bf*>(deepstack_scratch), num_deep, blas, stream);
}

}  // namespace pie_cuda_driver::kernels::vision

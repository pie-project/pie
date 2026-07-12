#pragma once

namespace pie_cuda_driver {

struct DsaCache {
    // TODO(dsa): unfinished, not dead — GLM5's forward takes a DsaCache& by
    // reference today, so this stub keeps that call site compiling. Fill in
    // the real DeepSeek Sparse Attention indexer-cache allocation here.
    static DsaCache allocate(int num_layers, int max_position_embeddings, int index_head_dim) {
        (void)num_layers; (void)max_position_embeddings; (void)index_head_dim;
        return DsaCache{};
    }
};

}  // namespace pie_cuda_driver

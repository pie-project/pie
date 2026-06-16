#pragma once

namespace pie_cuda_driver {

struct DsaCache {
    static DsaCache allocate(int num_layers, int max_position_embeddings, int index_head_dim) {
        (void)num_layers; (void)max_position_embeddings; (void)index_head_dim;
        return DsaCache{};
    }
};

}  // namespace pie_cuda_driver

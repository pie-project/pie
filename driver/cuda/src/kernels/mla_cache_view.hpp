#pragma once

// Neutral per-layer MLA pool descriptor. Kernel-owned so kernels/mla_paged.*
// stays a leaf (no include on store/): store/mla_cache.hpp includes this
// header and builds the view from its MlaCache, kernels/mla_paged.* consumes
// it as a plain value type without knowing about MlaCache at all.

namespace pie_cuda_driver {

struct MlaCacheLayerView {
    int layer = 0;
    int num_pages = 0;
    int page_size = 0;
    int kv_lora_rank = 0;
    int qk_rope_head_dim = 0;
    void* ckv_pages = nullptr;
    void* kpe_pages = nullptr;
};

}  // namespace pie_cuda_driver

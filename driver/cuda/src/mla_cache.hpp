#pragma once

// Paged MLA cache for DeepSeek/Kimi-style latent attention.
//
// Unlike the standard KV cache, MLA stores one latent vector plus one rotary
// key vector per token:
//   ckv_pages: [num_pages, page_size, kv_lora_rank]
//   kpe_pages: [num_pages, page_size, qk_rope_head_dim]
//
// FlashInfer's MLA kernels consume this layout directly.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "tensor.hpp"

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

class MlaCache {
public:
    static MlaCache allocate(int num_layers,
                             int num_pages,
                             int page_size,
                             int kv_lora_rank,
                             int qk_rope_head_dim,
                             DType dtype = DType::BF16);

    MlaCache() = default;
    MlaCache(const MlaCache&) = delete;
    MlaCache& operator=(const MlaCache&) = delete;
    MlaCache(MlaCache&&) noexcept = default;
    MlaCache& operator=(MlaCache&&) noexcept = default;

    int num_layers() const noexcept { return num_layers_; }
    int num_pages() const noexcept { return num_pages_; }
    int page_size() const noexcept { return page_size_; }
    int kv_lora_rank() const noexcept { return kv_lora_rank_; }
    int qk_rope_head_dim() const noexcept { return qk_rope_head_dim_; }
    DType dtype() const noexcept { return dtype_; }

    void* ckv(int layer) { return ckv_layers_[layer].data(); }
    void* kpe(int layer) { return kpe_layers_[layer].data(); }
    const void* ckv(int layer) const { return ckv_layers_[layer].data(); }
    const void* kpe(int layer) const { return kpe_layers_[layer].data(); }

    MlaCacheLayerView layer_view(int layer);

    struct PageBuffer {
        void* data = nullptr;
        std::size_t page_bytes = 0;
    };
    std::vector<PageBuffer> page_buffers(int layer);

private:
    int num_layers_ = 0;
    int num_pages_ = 0;
    int page_size_ = 0;
    int kv_lora_rank_ = 0;
    int qk_rope_head_dim_ = 0;
    DType dtype_ = DType::BF16;
    std::vector<DeviceTensor> ckv_layers_;
    std::vector<DeviceTensor> kpe_layers_;
};

}  // namespace pie_cuda_driver

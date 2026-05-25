#include "mla_cache.hpp"

#include <stdexcept>
#include <utility>

namespace pie_cuda_driver {

MlaCache MlaCache::allocate(int num_layers,
                            int num_pages,
                            int page_size,
                            int kv_lora_rank,
                            int qk_rope_head_dim,
                            DType dtype)
{
    if (num_layers <= 0 || num_pages <= 0 || page_size <= 0 ||
        kv_lora_rank <= 0 || qk_rope_head_dim <= 0) {
        throw std::runtime_error("mla_cache: invalid allocation dimensions");
    }
    if (dtype != DType::BF16 && dtype != DType::FP16) {
        throw std::runtime_error("mla_cache: only bf16/fp16 storage is supported");
    }

    MlaCache c;
    c.num_layers_ = num_layers;
    c.num_pages_ = num_pages;
    c.page_size_ = page_size;
    c.kv_lora_rank_ = kv_lora_rank;
    c.qk_rope_head_dim_ = qk_rope_head_dim;
    c.dtype_ = dtype;
    c.ckv_layers_.reserve(num_layers);
    c.kpe_layers_.reserve(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        c.ckv_layers_.push_back(DeviceTensor::allocate(
            dtype, {num_pages, page_size, kv_lora_rank}));
        c.kpe_layers_.push_back(DeviceTensor::allocate(
            dtype, {num_pages, page_size, qk_rope_head_dim}));
    }
    return c;
}

MlaCacheLayerView MlaCache::layer_view(int layer) {
    return MlaCacheLayerView{
        .layer = layer,
        .num_pages = num_pages_,
        .page_size = page_size_,
        .kv_lora_rank = kv_lora_rank_,
        .qk_rope_head_dim = qk_rope_head_dim_,
        .ckv_pages = ckv_layers_[layer].data(),
        .kpe_pages = kpe_layers_[layer].data(),
    };
}

std::vector<MlaCache::PageBuffer> MlaCache::page_buffers(int layer) {
    const std::size_t elem = dtype_bytes(dtype_);
    return {
        PageBuffer{
            .data = ckv_layers_[layer].data(),
            .page_bytes = static_cast<std::size_t>(page_size_) *
                          static_cast<std::size_t>(kv_lora_rank_) * elem,
        },
        PageBuffer{
            .data = kpe_layers_[layer].data(),
            .page_bytes = static_cast<std::size_t>(page_size_) *
                          static_cast<std::size_t>(qk_rope_head_dim_) * elem,
        },
    };
}

}  // namespace pie_cuda_driver

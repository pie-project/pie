#include "dsv4_compress_cache.hpp"

#include "cuda_check.hpp"

#include <cstdint>

namespace pie_cuda_driver {

namespace {

int compressor_coff(int ratio) { return ratio == 4 ? 2 : 1; }

}  // namespace

DsV4CompressCache DsV4CompressCache::allocate(const HfConfig& cfg,
                                              int num_pages,
                                              int page_size) {
    DsV4CompressCache cache;
    if (cfg.dsv4_compress_ratios.empty() || num_pages <= 0 || page_size <= 0) {
        return cache;
    }
    const int L = cfg.num_hidden_layers;
    const int head_dim = cfg.head_dim;
    cache.page_size_ = page_size;
    cache.layers_.resize(static_cast<std::size_t>(L));
    for (int li = 0; li < L; ++li) {
        const int ratio =
            li < static_cast<int>(cfg.dsv4_compress_ratios.size())
                ? cfg.dsv4_compress_ratios[static_cast<std::size_t>(li)]
                : 0;
        if (ratio <= 0) continue;
        const int width = compressor_coff(ratio) * head_dim;
        auto& layer = cache.layers_[static_cast<std::size_t>(li)];
        layer.state_width = width;
        layer.state_kv =
            DeviceTensor::allocate(DType::BF16, {num_pages, page_size, width});
        layer.state_score =
            DeviceTensor::allocate(DType::BF16, {num_pages, page_size, width});
        layer.comp_kv =
            DeviceTensor::allocate(DType::BF16, {num_pages, page_size, head_dim});
        // The compressor accumulates into `state_*` across a block and the
        // attention kernel can look at compressed slots whose block has not
        // closed yet. Both are reads of pages this request has not written,
        // so the cache should start at zero rather than at whatever the
        // allocator handed back.
        //
        // These tensors live in the elastic KV arena, which only *reserves*
        // virtual address space here; physical pages are committed on demand.
        // Writing the whole range is therefore expected to fail whenever the
        // arena is larger than what is currently committed, and the zeroing is
        // best-effort: `dsv4_zero_compress_pages` re-zeros the pages a request
        // actually touches once they are backed.
        for (DeviceTensor* t : {&layer.state_kv, &layer.state_score, &layer.comp_kv}) {
            if (t->nbytes() == 0 || t->data() == nullptr) continue;
            if (cudaMemset(t->data(), 0, t->nbytes()) != cudaSuccess) {
                cudaGetLastError();
                break;
            }
        }
    }
    return cache;
}

std::size_t dsv4_compress_bytes_per_token(const HfConfig& cfg) {
    std::size_t total = 0;
    const std::size_t head_dim = static_cast<std::size_t>(cfg.head_dim);
    for (int ratio : cfg.dsv4_compress_ratios) {
        if (ratio <= 0) continue;
        const std::size_t width =
            static_cast<std::size_t>(compressor_coff(ratio)) * head_dim;
        total += (2 * width + head_dim) * sizeof(std::uint16_t);
    }
    return total;
}

}  // namespace pie_cuda_driver

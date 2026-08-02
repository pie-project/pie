#pragma once

// store/: long-lived device memory pools.
//
// DeepSeek-V4 compressor caches. A V4 layer with `compress_ratio > 0` runs a
// second attention over *compressed* KV entries: one entry per `ratio` tokens,
// each a per-dimension softmax pool over the `coff * ratio` tokens ending at a
// boundary position (`(position + 1) % ratio == 0`).
//
// Two pieces of per-token state have to survive across forward passes so that
// decode can keep producing entries:
//
//   * `state_kv` / `state_score` — the compressor's `wkv` / `wgate`
//     projections of every token, `coff * head_dim` wide. A boundary token
//     pools the projections of the whole window, most of which were written
//     during earlier forward passes. vLLM calls this the "partial state"
//     cache (`CompressorStateCache`).
//   * `comp_kv` — the finished compressed entry, `head_dim` wide, written at
//     the boundary token's own slot so that the sparse attention can find
//     entry `c` at absolute position `(c + 1) * ratio - 1`.
//
// All three share the KV cache's page geometry and page tables, so a request's
// compressor state is freed exactly when its KV pages are.

#include <cstddef>
#include <vector>

#include "tensor.hpp"
#include "model/config.hpp"

namespace pie_cuda_driver {

class DsV4CompressCache {
public:
    static DsV4CompressCache allocate(const HfConfig& cfg,
                                      int num_pages,
                                      int page_size);

    DsV4CompressCache() = default;
    DsV4CompressCache(const DsV4CompressCache&) = delete;
    DsV4CompressCache& operator=(const DsV4CompressCache&) = delete;
    DsV4CompressCache(DsV4CompressCache&&) noexcept = default;
    DsV4CompressCache& operator=(DsV4CompressCache&&) noexcept = default;

    bool empty() const noexcept { return layers_.empty(); }
    int page_size() const noexcept { return page_size_; }

    bool has_layer(int layer) const noexcept {
        return layer >= 0 && layer < static_cast<int>(layers_.size()) &&
               !layers_[static_cast<std::size_t>(layer)].state_kv.empty();
    }
    void* state_kv(int layer) {
        return layers_[static_cast<std::size_t>(layer)].state_kv.data();
    }
    void* state_score(int layer) {
        return layers_[static_cast<std::size_t>(layer)].state_score.data();
    }
    void* comp_kv(int layer) {
        return layers_[static_cast<std::size_t>(layer)].comp_kv.data();
    }
    int state_width(int layer) const noexcept {
        return layers_[static_cast<std::size_t>(layer)].state_width;
    }

private:
    struct Layer {
        DeviceTensor state_kv;     // [num_pages, page_size, coff*head_dim] BF16
        DeviceTensor state_score;  // [num_pages, page_size, coff*head_dim] BF16
        DeviceTensor comp_kv;      // [num_pages, page_size, head_dim] BF16
        int state_width = 0;       // coff * head_dim
    };
    std::vector<Layer> layers_;
    int page_size_ = 0;
};

// Device bytes the compressor caches consume per KV token, summed over every
// compressing layer. Fed to the memory planner so page count accounts for it.
std::size_t dsv4_compress_bytes_per_token(const HfConfig& cfg);

}  // namespace pie_cuda_driver

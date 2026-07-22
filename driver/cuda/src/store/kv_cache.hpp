#pragma once

// store/: long-lived device memory pools (KV/MLA/DSA/recurrent-state caches, swap).
//
// Paged KV cache pool. The default physical layout is one pair of
// [num_pages, page_size, num_kv_heads, head_dim] tensors per layer. Native
// BF16 can opt into FlashInfer's HND layout:
// [num_pages, num_kv_heads, page_size, head_dim].
// The runtime hands us page-index lists in each wire request; we read/write
// through that translation.

#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>

#include "kv_cache_format.hpp"
#include "kernels/kv_cache_view.hpp"
#include "../tensor.hpp"

namespace pie_cuda_driver {

class CudaArenaAllocator;

class KvCache {
public:
    // Homogeneous: every layer gets the same `[num_pages, page_size,
    // num_kv_heads, head_dim]` shape. Used by every model whose
    // attention is uniform across layers (qwen, llama, mistral, gemma-2/3,
    // olmo-3, phi-3).
    static KvCache allocate(int num_layers,
                            int num_pages,
                            int page_size,
                            int num_kv_heads,
                            int head_dim,
                            DType dtype = DType::BF16);

    static KvCache allocate(int num_layers,
                            int num_pages,
                            int page_size,
                            int num_kv_heads,
                            int head_dim,
                            KvCacheFormat format);

    // Per-layer head_dim with optional KV-cache sharing. When
    // `kv_source_layer[L] != L` no physical tensor is allocated for
    // that slot — `k(L)` / `v(L)` redirect to `kv_source_layer[L]`.
    // Used by Gemma-4 (dual head_dim 256/512 + 20 shared layers).
    // `kv_source_layer` may be empty, in which case every layer is its
    // own source.
    //
    // `per_layer_num_kv_heads` overrides the scalar `num_kv_heads` per
    // layer. Required by Gemma-4 26B-A4B's `attention_k_eq_v` mode,
    // where full-attention layers use `num_global_key_value_heads`
    // (typically 2) while sliding layers stay at the scalar
    // `num_key_value_heads` (8). Pass an empty vector for the
    // homogeneous case.
    static KvCache allocate_per_layer(int num_layers,
                                      int num_pages,
                                      int page_size,
                                      int num_kv_heads,
                                      const std::vector<int>& per_layer_head_dim,
                                      const std::vector<int>& kv_source_layer,
                                      const std::vector<int>& per_layer_num_kv_heads = {},
                                      DType dtype = DType::BF16);

    static KvCache allocate_per_layer(int num_layers,
                                      int num_pages,
                                      int page_size,
                                      int num_kv_heads,
                                      const std::vector<int>& per_layer_head_dim,
                                      const std::vector<int>& kv_source_layer,
                                      const std::vector<int>& per_layer_num_kv_heads,
                                      KvCacheFormat format);

    KvCache() = default;
    KvCache(const KvCache&) = delete;
    KvCache& operator=(const KvCache&) = delete;
    KvCache(KvCache&&) noexcept = default;
    KvCache& operator=(KvCache&&) noexcept = default;

    int num_layers() const noexcept { return num_layers_; }
    int num_pages() const noexcept { return num_pages_; }
    int page_size() const noexcept { return page_size_; }
    const KvCacheFormat& format() const noexcept { return format_; }
    int num_kv_heads() const noexcept { return num_kv_heads_; }
    // Homogeneous head_dim — matches the constructor argument. For
    // per-layer-head_dim allocations call `head_dim_at(layer)` instead.
    int head_dim() const noexcept { return head_dim_; }
    int head_dim_at(int layer) const noexcept {
        return per_layer_head_dim_.empty() ? head_dim_
                                           : per_layer_head_dim_[layer];
    }
    bool hnd_layout() const noexcept { return hnd_layout_; }
    // Per-layer `num_kv_heads`. Falls back to the scalar when the
    // per-layer override was not supplied at allocation time.
    int num_kv_heads_at(int layer) const noexcept {
        return per_layer_num_kv_heads_.empty() ? num_kv_heads_
                                               : per_layer_num_kv_heads_[layer];
    }
    KvCacheLayerView layer_view(int layer);

    // Per-layer accessors. Layout depends on `format()` and `hnd_layout()`:
    //   BF16 NHD / INT8 / FP8: [num_pages, page_size, num_kv_heads, head_dim_at(layer)]
    //   BF16 HND:              [num_pages, num_kv_heads, page_size, head_dim_at(layer)]
    //   FP4 packed:        [num_pages, page_size, num_kv_heads, ceil(head_dim_at(layer)/2)]
    // Resolves through `kv_source_layer_` so shared slots return their
    // source's tensor.
    void* k(int layer)       { return k_layers_[resolve_(layer)].data(); }
    void* v(int layer)       { return v_layers_[resolve_(layer)].data(); }
    const void* k(int layer) const { return k_layers_[resolve_(layer)].data(); }
    const void* v(int layer) const { return v_layers_[resolve_(layer)].data(); }

    // Optional side scale pages. Per-token-head scales have layout
    // [num_pages, page_size, num_kv_heads]. FP4 block scales have layout
    // [num_pages, page_size, num_kv_heads, ceil(head_dim / block_size)].
    void* k_scale(int layer);
    void* v_scale(int layer);
    const void* k_scale(int layer) const;
    const void* v_scale(int layer) const;

    // BF16 pages consumed by attention. For native BF16 these alias k()/v().
    // Quantized formats use private BF16 scratch populated by the format-aware
    // attention wrappers.
    void* k_for_attention(int layer);
    void* v_for_attention(int layer);
    const void* k_for_attention(int layer) const;
    const void* v_for_attention(int layer) const;

    struct PageBuffer {
        void* data = nullptr;
        std::size_t page_bytes = 0;
    };
    std::vector<PageBuffer> page_buffers(int layer);

    void set_elastic_allocator(
        std::shared_ptr<CudaArenaAllocator> allocator) noexcept;
    void ensure_pages(int pages);
    void trim_pages(int pages);
    std::size_t committed_bytes() const noexcept;

private:
    int resolve_(int layer) const noexcept {
        return kv_source_layer_.empty() ? layer : kv_source_layer_[layer];
    }

    int num_layers_ = 0;
    int num_pages_ = 0;
    int page_size_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    bool hnd_layout_ = false;
    KvCacheFormat format_;
    std::vector<DeviceTensor> k_layers_;
    std::vector<DeviceTensor> v_layers_;
    std::vector<DeviceTensor> k_scale_layers_;
    std::vector<DeviceTensor> v_scale_layers_;
    std::vector<DeviceTensor> k_bf16_layers_;
    std::vector<DeviceTensor> v_bf16_layers_;
    // Empty for homogeneous allocations. When populated:
    //   * `per_layer_head_dim_[L]` is layer L's K/V head_dim.
    //   * `kv_source_layer_[L]` is the slot index whose physical
    //     tensor backs L (typically L; smaller for shared layers).
    std::vector<int> per_layer_head_dim_;
    std::vector<int> kv_source_layer_;
    std::vector<int> per_layer_num_kv_heads_;
    std::shared_ptr<CudaArenaAllocator> elastic_allocator_;
};

struct HfConfig;

// Device bytes for a single KV cache page in the given format. Includes
// the dequant BF16 scratch tier when format != native bf16.
std::size_t kv_cache_device_bytes_per_page(const KvCacheFormat& format,
                                           int page_size,
                                           int num_kv_heads,
                                           int head_dim);

// Per-page bytes for a homogeneous transformer where every layer has
// the same head_dim/kv_heads. Equivalent to multiplying the per-layer
// page bytes by num_hidden_layers.
std::size_t kv_page_bytes_homogeneous(const HfConfig& cfg,
                                      int tp_size,
                                      const KvCacheFormat& format);

// Per-page bytes for a heterogeneous stack where layers may have
// different head_dim or share physical pages (kv_source_layer).
std::size_t kv_page_bytes_per_layer(const HfConfig& cfg,
                                    const std::vector<int>& per_layer_head_dim,
                                    const std::vector<int>& kv_source_layer,
                                    int tp_size,
                                    const KvCacheFormat& format);

}  // namespace pie_cuda_driver

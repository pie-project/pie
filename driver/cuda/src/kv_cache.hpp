#pragma once

// Paged KV cache pool. One pair of [num_pages, page_size, num_kv_heads,
// head_dim] tensors per layer. The runtime hands us page-index lists in
// each BPIQ request; we read/write through that translation.

#include <cstdint>
#include <vector>

#include "tensor.hpp"

namespace pie_cuda_driver {

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

    KvCache() = default;
    KvCache(const KvCache&) = delete;
    KvCache& operator=(const KvCache&) = delete;
    KvCache(KvCache&&) noexcept = default;
    KvCache& operator=(KvCache&&) noexcept = default;

    int num_layers() const noexcept { return num_layers_; }
    int num_pages() const noexcept { return num_pages_; }
    int page_size() const noexcept { return page_size_; }
    int num_kv_heads() const noexcept { return num_kv_heads_; }
    // Homogeneous head_dim — matches the constructor argument. For
    // per-layer-head_dim allocations call `head_dim_at(layer)` instead.
    int head_dim() const noexcept { return head_dim_; }
    int head_dim_at(int layer) const noexcept {
        return per_layer_head_dim_.empty() ? head_dim_
                                           : per_layer_head_dim_[layer];
    }
    // Per-layer `num_kv_heads`. Falls back to the scalar when the
    // per-layer override was not supplied at allocation time.
    int num_kv_heads_at(int layer) const noexcept {
        return per_layer_num_kv_heads_.empty() ? num_kv_heads_
                                               : per_layer_num_kv_heads_[layer];
    }

    // Per-layer accessors. Layout: [num_pages, page_size, num_kv_heads, head_dim_at(layer)].
    // Resolves through `kv_source_layer_` so shared slots return their
    // source's tensor.
    void* k(int layer)       { return k_layers_[resolve_(layer)].data(); }
    void* v(int layer)       { return v_layers_[resolve_(layer)].data(); }
    const void* k(int layer) const { return k_layers_[resolve_(layer)].data(); }
    const void* v(int layer) const { return v_layers_[resolve_(layer)].data(); }

private:
    int resolve_(int layer) const noexcept {
        return kv_source_layer_.empty() ? layer : kv_source_layer_[layer];
    }

    int num_layers_ = 0;
    int num_pages_ = 0;
    int page_size_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    std::vector<DeviceTensor> k_layers_;
    std::vector<DeviceTensor> v_layers_;
    // Empty for homogeneous allocations. When populated:
    //   * `per_layer_head_dim_[L]` is layer L's K/V head_dim.
    //   * `kv_source_layer_[L]` is the slot index whose physical
    //     tensor backs L (typically L; smaller for shared layers).
    std::vector<int> per_layer_head_dim_;
    std::vector<int> kv_source_layer_;
    std::vector<int> per_layer_num_kv_heads_;
};

}  // namespace pie_cuda_driver

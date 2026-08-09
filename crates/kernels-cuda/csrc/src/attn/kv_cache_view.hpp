#pragma once

// A neutral per-layer KV descriptor, so a cache owner in the driver can
// expose its storage to a kernel without this crate importing `store/`.
// That one-way rule is what the kernels-cuda / driver-cuda split is: the
// driver reaches down, nothing here reaches up.

#include <cstdint>

#include "tensor.hpp"

namespace pie_cuda_driver {

enum class KvCacheScheme : std::uint8_t {
    Native,
    Fp8PerTensor,
    Int8PerTokenHead,
    Fp8PerTokenHead,
    Fp4Block,
};

struct KvCacheLayerView {
    int layer = 0;
    int source_layer = 0;
    int num_pages = 0;
    int page_size = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    KvCacheScheme scheme = KvCacheScheme::Native;
    DType storage_dtype = DType::BF16;
    int block_size = 0;
    void* k_pages = nullptr;
    void* v_pages = nullptr;
    void* k_scales = nullptr;
    void* v_scales = nullptr;
    void* k_bf16_pages = nullptr;
    void* v_bf16_pages = nullptr;
    // Quest per-page key envelopes, [num_pages, num_kv_heads, head_dim] bf16
    // each. Null unless envelopes were explicitly enabled on the cache: they
    // cost 8 bytes per (page, kv_head, dim) against the page's own
    // `page_size * 2`, i.e. 4/page_size of the KV cache, so they are never
    // allocated for models that no program asks to observe.
    std::uint16_t* k_env_min = nullptr;
    std::uint16_t* k_env_max = nullptr;
    bool hnd_layout = false;
    bool native_bf16 = false;

    bool has_envelopes() const noexcept {
        return k_env_min != nullptr && k_env_max != nullptr;
    }

    bool is_native_bf16() const noexcept { return native_bf16; }
};

}  // namespace pie_cuda_driver

#pragma once

// kernels/: leaf CUDA launch contracts consumed by ops/ and model forwards.
// This neutral per-layer KV descriptor lets cache owners expose storage to
// kernels without importing store/ into this leaf module.

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
    bool hnd_layout = false;
    bool native_bf16 = false;

    bool is_native_bf16() const noexcept { return native_bf16; }
};

}  // namespace pie_cuda_driver

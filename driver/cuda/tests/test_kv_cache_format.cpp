#include <cassert>
#include <iostream>

#include "kv_cache_format.hpp"

using pie_cuda_driver::KvCacheScaleLayout;
using pie_cuda_driver::KvCacheScheme;
using pie_cuda_driver::kv_cache_format_from_string;

int main() {
    const int page = 16;
    const int heads = 2;
    const int dim = 128;

    auto bf16 = kv_cache_format_from_string("auto");
    assert(bf16.scheme == KvCacheScheme::Native);
    assert(bf16.kv_bytes_per_page(page, heads, dim) == 16u * 2u * 128u * 2u);
    assert(bf16.scale_bytes_per_page(page, heads, dim) == 0u);
    assert(kv_cache_format_from_string("bf16").scheme == KvCacheScheme::Native);
    assert(kv_cache_format_from_string("bfloat16").scheme == KvCacheScheme::Native);

    auto e4m3 = kv_cache_format_from_string("fp8_e4m3");
    assert(e4m3.kv_bytes_per_page(page, heads, dim) == 16u * 2u * 128u);
    assert(e4m3.total_bytes_per_page(page, heads, dim) == 2u * 16u * 2u * 128u);

    auto e5m2 = kv_cache_format_from_string("fp8_e5m2");
    assert(e5m2.scheme == KvCacheScheme::Fp8PerTensor);
    assert(e5m2.kv_bytes_per_page(page, heads, dim) == 16u * 2u * 128u);

    auto int8 = kv_cache_format_from_string("int8_per_token_head");
    assert(int8.scheme == KvCacheScheme::Int8PerTokenHead);
    assert(int8.scale_layout == KvCacheScaleLayout::PerTokenHead);
    assert(int8.scale_bytes_per_page(page, heads, dim) == 16u * 2u * 4u);

    auto fp8_pth = kv_cache_format_from_string("fp8_per_token_head");
    assert(fp8_pth.scale_bytes_per_page(page, heads, dim) == 16u * 2u * 4u);

    auto fp4 = kv_cache_format_from_string("fp4_e2m1");
    assert(fp4.storage_head_dim(dim) == 64);
    assert(fp4.kv_bytes_per_page(page, heads, dim) == 16u * 2u * 64u);
    assert(fp4.scale_bytes_per_page(page, heads, dim) == 16u * 2u * 8u * 4u);
    assert(kv_cache_format_from_string("nvfp4").scheme == KvCacheScheme::Fp4Block);

    bool threw = false;
    try {
        (void)kv_cache_format_from_string("turboquant");
    } catch (...) {
        threw = true;
    }
    assert(threw);

    std::cout << "kv_cache_format ok\n";
    return 0;
}

#include "kv_cache.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

#include "../model/config.hpp"
#include "elastic.hpp"

namespace pie_cuda_driver {

namespace {

bool env_requests_hnd_kv_layout() {
    const char* value = std::getenv("PIE_CUDA_KV_LAYOUT");
    if (value == nullptr || value[0] == '\0') {
        value = std::getenv("PIE_CUDA_KV_CACHE_LAYOUT");
    }
    if (value == nullptr) return false;
    const std::string layout(value);
    return layout == "HND" || layout == "hnd";
}

std::vector<std::int64_t> kv_storage_shape(
    int num_pages,
    int page_size,
    int num_kv_heads,
    int storage_head_dim,
    bool hnd_layout)
{
    if (hnd_layout) {
        return {num_pages, num_kv_heads, page_size, storage_head_dim};
    }
    return {num_pages, page_size, num_kv_heads, storage_head_dim};
}

}  // namespace

KvCache KvCache::allocate(int num_layers,
                          int num_pages,
                          int page_size,
                          int num_kv_heads,
                          int head_dim,
                          DType dtype)
{
    KvCacheFormat f;
    f.storage_dtype = dtype;
    f.name = dtype == DType::BF16 ? "bf16" : dtype_name(dtype);
    return allocate(num_layers, num_pages, page_size, num_kv_heads, head_dim, f);
}

KvCache KvCache::allocate(int num_layers,
                          int num_pages,
                          int page_size,
                          int num_kv_heads,
                          int head_dim,
                          KvCacheFormat format)
{
    KvCache c;
    c.num_layers_ = num_layers;
    c.num_pages_ = num_pages;
    c.page_size_ = page_size;
    c.num_kv_heads_ = num_kv_heads;
    c.head_dim_ = head_dim;
    c.format_ = std::move(format);
    c.hnd_layout_ = c.format_.is_native_bf16() && env_requests_hnd_kv_layout();

    c.k_layers_.reserve(num_layers);
    c.v_layers_.reserve(num_layers);
    c.k_scale_layers_.reserve(num_layers);
    c.v_scale_layers_.reserve(num_layers);
    c.k_bf16_layers_.reserve(num_layers);
    c.v_bf16_layers_.reserve(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        const auto storage_hd = c.format_.storage_head_dim(head_dim);
        c.k_layers_.push_back(DeviceTensor::allocate(
            c.format_.storage_dtype,
            kv_storage_shape(num_pages, page_size, num_kv_heads,
                             storage_hd, c.hnd_layout_)));
        c.v_layers_.push_back(DeviceTensor::allocate(
            c.format_.storage_dtype,
            kv_storage_shape(num_pages, page_size, num_kv_heads,
                             storage_hd, c.hnd_layout_)));
        if (c.format_.scale_layout == KvCacheScaleLayout::PerTokenHead) {
            c.k_scale_layers_.push_back(DeviceTensor::allocate(
                DType::FP32, {num_pages, page_size, num_kv_heads}));
            c.v_scale_layers_.push_back(DeviceTensor::allocate(
                DType::FP32, {num_pages, page_size, num_kv_heads}));
        } else if (c.format_.scale_layout == KvCacheScaleLayout::PerTokenHeadBlock) {
            const int bs = c.format_.block_size > 0 ? c.format_.block_size : 16;
            const int blocks = (head_dim + bs - 1) / bs;
            c.k_scale_layers_.push_back(DeviceTensor::allocate(
                DType::FP32, {num_pages, page_size, num_kv_heads, blocks}));
            c.v_scale_layers_.push_back(DeviceTensor::allocate(
                DType::FP32, {num_pages, page_size, num_kv_heads, blocks}));
        } else {
            c.k_scale_layers_.emplace_back();
            c.v_scale_layers_.emplace_back();
        }
        if (c.format_.is_native_bf16()) {
            c.k_bf16_layers_.emplace_back();
            c.v_bf16_layers_.emplace_back();
        } else {
            c.k_bf16_layers_.push_back(DeviceTensor::allocate(
                DType::BF16, {num_pages, page_size, num_kv_heads, head_dim}));
            c.v_bf16_layers_.push_back(DeviceTensor::allocate(
                DType::BF16, {num_pages, page_size, num_kv_heads, head_dim}));
        }
    }
    return c;
}

KvCache KvCache::allocate_per_layer(int num_layers,
                                    int num_pages,
                                    int page_size,
                                    int num_kv_heads,
                                    const std::vector<int>& per_layer_head_dim,
                                    const std::vector<int>& kv_source_layer,
                                    const std::vector<int>& per_layer_num_kv_heads,
                                    DType dtype)
{
    KvCacheFormat f;
    f.storage_dtype = dtype;
    f.name = dtype == DType::BF16 ? "bf16" : dtype_name(dtype);
    return allocate_per_layer(num_layers, num_pages, page_size, num_kv_heads,
                              per_layer_head_dim, kv_source_layer,
                              per_layer_num_kv_heads, f);
}

KvCache KvCache::allocate_per_layer(int num_layers,
                                    int num_pages,
                                    int page_size,
                                    int num_kv_heads,
                                    const std::vector<int>& per_layer_head_dim,
                                    const std::vector<int>& kv_source_layer,
                                    const std::vector<int>& per_layer_num_kv_heads,
                                    KvCacheFormat format)
{
    if (!per_layer_head_dim.empty() &&
        static_cast<int>(per_layer_head_dim.size()) != num_layers) {
        throw std::runtime_error("kv_cache: per_layer_head_dim size mismatch");
    }
    if (!kv_source_layer.empty() &&
        static_cast<int>(kv_source_layer.size()) != num_layers) {
        throw std::runtime_error("kv_cache: kv_source_layer size mismatch");
    }
    if (!per_layer_num_kv_heads.empty() &&
        static_cast<int>(per_layer_num_kv_heads.size()) != num_layers) {
        throw std::runtime_error("kv_cache: per_layer_num_kv_heads size mismatch");
    }

    KvCache c;
    c.num_layers_ = num_layers;
    c.num_pages_ = num_pages;
    c.page_size_ = page_size;
    c.num_kv_heads_ = num_kv_heads;
    c.head_dim_ = per_layer_head_dim.empty() ? 0 : per_layer_head_dim[0];
    c.format_ = std::move(format);
    c.hnd_layout_ = c.format_.is_native_bf16() && env_requests_hnd_kv_layout();
    c.per_layer_head_dim_ = per_layer_head_dim;
    c.kv_source_layer_ = kv_source_layer;
    c.per_layer_num_kv_heads_ = per_layer_num_kv_heads;

    // Allocate physical storage at every slot — even shared slots get
    // an empty placeholder so the vector index matches `layer`. Slots
    // whose `kv_source_layer != self` get a zero-byte view that we
    // never read or write through (the resolver redirects `k(L)` to
    // the source slot before any access). The placeholder keeps the
    // accessor lookup O(1).
    c.k_layers_.reserve(num_layers);
    c.v_layers_.reserve(num_layers);
    c.k_scale_layers_.reserve(num_layers);
    c.v_scale_layers_.reserve(num_layers);
    c.k_bf16_layers_.reserve(num_layers);
    c.v_bf16_layers_.reserve(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        const bool is_source = kv_source_layer.empty() || kv_source_layer[i] == i;
        if (is_source) {
            const int hd = per_layer_head_dim.empty() ? c.head_dim_
                                                      : per_layer_head_dim[i];
            const int kvh = per_layer_num_kv_heads.empty()
                                ? num_kv_heads
                                : per_layer_num_kv_heads[i];
            const auto storage_hd = c.format_.storage_head_dim(hd);
            c.k_layers_.push_back(DeviceTensor::allocate(
                c.format_.storage_dtype,
                kv_storage_shape(num_pages, page_size, kvh,
                                 storage_hd, c.hnd_layout_)));
            c.v_layers_.push_back(DeviceTensor::allocate(
                c.format_.storage_dtype,
                kv_storage_shape(num_pages, page_size, kvh,
                                 storage_hd, c.hnd_layout_)));
            if (c.format_.scale_layout == KvCacheScaleLayout::PerTokenHead) {
                c.k_scale_layers_.push_back(DeviceTensor::allocate(
                    DType::FP32, {num_pages, page_size, kvh}));
                c.v_scale_layers_.push_back(DeviceTensor::allocate(
                    DType::FP32, {num_pages, page_size, kvh}));
            } else if (c.format_.scale_layout == KvCacheScaleLayout::PerTokenHeadBlock) {
                const int bs = c.format_.block_size > 0 ? c.format_.block_size : 16;
                const int blocks = (hd + bs - 1) / bs;
                c.k_scale_layers_.push_back(DeviceTensor::allocate(
                    DType::FP32, {num_pages, page_size, kvh, blocks}));
                c.v_scale_layers_.push_back(DeviceTensor::allocate(
                    DType::FP32, {num_pages, page_size, kvh, blocks}));
            } else {
                c.k_scale_layers_.emplace_back();
                c.v_scale_layers_.emplace_back();
            }
            if (c.format_.is_native_bf16()) {
                c.k_bf16_layers_.emplace_back();
                c.v_bf16_layers_.emplace_back();
            } else {
                c.k_bf16_layers_.push_back(DeviceTensor::allocate(
                    DType::BF16, {num_pages, page_size, kvh, hd}));
                c.v_bf16_layers_.push_back(DeviceTensor::allocate(
                    DType::BF16, {num_pages, page_size, kvh, hd}));
            }
        } else {
            c.k_layers_.emplace_back();  // empty
            c.v_layers_.emplace_back();
            c.k_scale_layers_.emplace_back();
            c.v_scale_layers_.emplace_back();
            c.k_bf16_layers_.emplace_back();
            c.v_bf16_layers_.emplace_back();
        }
    }
    return c;
}

void* KvCache::k_scale(int layer) {
    auto& t = k_scale_layers_[resolve_(layer)];
    return t.empty() ? nullptr : t.data();
}

void* KvCache::v_scale(int layer) {
    auto& t = v_scale_layers_[resolve_(layer)];
    return t.empty() ? nullptr : t.data();
}

const void* KvCache::k_scale(int layer) const {
    const auto& t = k_scale_layers_[resolve_(layer)];
    return t.empty() ? nullptr : t.data();
}

const void* KvCache::v_scale(int layer) const {
    const auto& t = v_scale_layers_[resolve_(layer)];
    return t.empty() ? nullptr : t.data();
}

void* KvCache::k_for_attention(int layer) {
    const int src = resolve_(layer);
    return format_.is_native_bf16() ? k_layers_[src].data()
                                    : k_bf16_layers_[src].data();
}

void* KvCache::v_for_attention(int layer) {
    const int src = resolve_(layer);
    return format_.is_native_bf16() ? v_layers_[src].data()
                                    : v_bf16_layers_[src].data();
}

const void* KvCache::k_for_attention(int layer) const {
    const int src = resolve_(layer);
    return format_.is_native_bf16() ? k_layers_[src].data()
                                    : k_bf16_layers_[src].data();
}

const void* KvCache::v_for_attention(int layer) const {
    const int src = resolve_(layer);
    return format_.is_native_bf16() ? v_layers_[src].data()
                                    : v_bf16_layers_[src].data();
}

KvCacheLayerView KvCache::layer_view(int layer) {
    const int src = resolve_(layer);
    const int hd = head_dim_at(src);
    const int kvh = num_kv_heads_at(src);
    auto data_or_null = [](DeviceTensor& t) -> void* {
        return t.empty() ? nullptr : t.data();
    };
    return KvCacheLayerView{
        .layer = layer,
        .source_layer = src,
        .num_pages = num_pages_,
        .page_size = page_size_,
        .num_kv_heads = kvh,
        .head_dim = hd,
        .scheme = format_.scheme,
        .storage_dtype = format_.storage_dtype,
        .block_size = format_.block_size,
        .k_pages = k_layers_[src].data(),
        .v_pages = v_layers_[src].data(),
        .k_scales = data_or_null(k_scale_layers_[src]),
        .v_scales = data_or_null(v_scale_layers_[src]),
        .k_bf16_pages = format_.is_native_bf16() ? k_layers_[src].data()
                                                 : k_bf16_layers_[src].data(),
        .v_bf16_pages = format_.is_native_bf16() ? v_layers_[src].data()
                                                 : v_bf16_layers_[src].data(),
        .hnd_layout = hnd_layout_,
        .native_bf16 = format_.is_native_bf16(),
    };
}

std::vector<KvCache::PageBuffer> KvCache::page_buffers(int layer) {
    const int src = resolve_(layer);
    const int hd = head_dim_at(src);
    const int kvh = num_kv_heads_at(src);
    std::vector<PageBuffer> out;
    out.reserve(format_.has_side_scales() ? 4 : 2);
    out.push_back({k_layers_[src].data(),
                   format_.kv_bytes_per_page(page_size_, kvh, hd)});
    out.push_back({v_layers_[src].data(),
                   format_.kv_bytes_per_page(page_size_, kvh, hd)});
    const std::size_t scale_bytes =
        format_.scale_bytes_per_page(page_size_, kvh, hd);
    if (scale_bytes > 0) {
        out.push_back({k_scale_layers_[src].data(), scale_bytes});
        out.push_back({v_scale_layers_[src].data(), scale_bytes});
    }
    return out;
}

void KvCache::set_elastic_allocator(
    std::shared_ptr<CudaArenaAllocator> allocator) noexcept {
    elastic_allocator_ = std::move(allocator);
}

void KvCache::ensure_pages(int pages) {
    if (elastic_allocator_ == nullptr || num_pages_ <= 0) return;
    elastic_allocator_->ensure_fraction(
        static_cast<std::size_t>(std::clamp(pages, 0, num_pages_)),
        static_cast<std::size_t>(num_pages_));
}

void KvCache::trim_pages(int pages) {
    if (elastic_allocator_ == nullptr || num_pages_ <= 0) return;
    elastic_allocator_->trim_fraction(
        static_cast<std::size_t>(std::clamp(pages, 0, num_pages_)),
        static_cast<std::size_t>(num_pages_));
}

std::size_t KvCache::committed_bytes() const noexcept {
    return elastic_allocator_ == nullptr
        ? 0
        : elastic_allocator_->committed_bytes();
}

std::size_t kv_cache_device_bytes_per_page(const KvCacheFormat& format,
                                           int page_size,
                                           int num_kv_heads,
                                           int head_dim) {
    std::size_t bytes =
        format.total_bytes_per_page(page_size, num_kv_heads, head_dim);
    if (!format.is_native_bf16()) {
        bytes += 2 * static_cast<std::size_t>(page_size) *
                 static_cast<std::size_t>(num_kv_heads) *
                 static_cast<std::size_t>(head_dim) *
                 dtype_bytes(DType::BF16);
    }
    return bytes;
}

std::size_t kv_page_bytes_homogeneous(const HfConfig& cfg,
                                      int tp_size,
                                      const KvCacheFormat& format) {
    const int kv_heads = cfg.num_key_value_heads / std::max(1, tp_size);
    return static_cast<std::size_t>(cfg.num_hidden_layers) *
           kv_cache_device_bytes_per_page(
               format, 1, kv_heads, cfg.head_dim_kernel);
}

std::size_t kv_page_bytes_per_layer(const HfConfig& cfg,
                                    const std::vector<int>& per_layer_head_dim,
                                    const std::vector<int>& kv_source_layer,
                                    int tp_size,
                                    const KvCacheFormat& format) {
    std::size_t per_token = 0;
    const int kv_heads = cfg.num_key_value_heads / std::max(1, tp_size);
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        const bool is_source = kv_source_layer.empty() || kv_source_layer[i] == i;
        if (!is_source) continue;
        const int hd = per_layer_head_dim.empty()
            ? cfg.head_dim_kernel
            : per_layer_head_dim[i];
        per_token += kv_cache_device_bytes_per_page(format, 1, kv_heads, hd);
    }
    return per_token;
}

}  // namespace pie_cuda_driver

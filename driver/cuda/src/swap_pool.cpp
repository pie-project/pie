#include "swap_pool.hpp"

#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>

#include "cuda_check.hpp"

namespace pie_cuda_driver {

SwapPool SwapPool::allocate(int num_layers,
                            int num_pages,
                            int page_size,
                            int num_kv_heads,
                            int head_dim,
                            DType dtype)
{
    SwapPool s;
    s.num_layers_ = num_layers;
    s.num_pages_ = num_pages;
    s.page_size_ = page_size;
    s.num_kv_heads_ = num_kv_heads;
    s.head_dim_ = head_dim;
    s.page_bytes_ = static_cast<std::size_t>(page_size) *
                    static_cast<std::size_t>(num_kv_heads) *
                    static_cast<std::size_t>(head_dim) *
                    dtype_bytes(dtype);

    if (num_pages <= 0 || num_layers <= 0) return s;

    CUDA_CHECK(cudaStreamCreateWithFlags(&s.stream_, cudaStreamNonBlocking));

    s.k_host_pools_.reserve(num_layers);
    s.v_host_pools_.reserve(num_layers);
    const std::size_t pool_bytes =
        s.page_bytes_ * static_cast<std::size_t>(num_pages);
    for (int i = 0; i < num_layers; ++i) {
        void* k = nullptr;
        void* v = nullptr;
        CUDA_CHECK(cudaMallocHost(&k, pool_bytes));
        CUDA_CHECK(cudaMallocHost(&v, pool_bytes));
        s.k_host_pools_.push_back(k);
        s.v_host_pools_.push_back(v);
    }
    return s;
}

SwapPool::SwapPool(SwapPool&& o) noexcept
    : num_layers_(o.num_layers_),
      num_pages_(o.num_pages_),
      page_size_(o.page_size_),
      num_kv_heads_(o.num_kv_heads_),
      head_dim_(o.head_dim_),
      page_bytes_(o.page_bytes_),
      k_host_pools_(std::move(o.k_host_pools_)),
      v_host_pools_(std::move(o.v_host_pools_)),
      stream_(o.stream_) {
    o.num_pages_ = 0;
    o.stream_ = nullptr;
}

SwapPool& SwapPool::operator=(SwapPool&& o) noexcept {
    if (this != &o) {
        for (auto* p : k_host_pools_) cudaFreeHost(p);
        for (auto* p : v_host_pools_) cudaFreeHost(p);
        if (stream_) cudaStreamDestroy(stream_);
        num_layers_ = o.num_layers_;
        num_pages_ = o.num_pages_;
        page_size_ = o.page_size_;
        num_kv_heads_ = o.num_kv_heads_;
        head_dim_ = o.head_dim_;
        page_bytes_ = o.page_bytes_;
        k_host_pools_ = std::move(o.k_host_pools_);
        v_host_pools_ = std::move(o.v_host_pools_);
        stream_ = o.stream_;
        o.num_pages_ = 0;
        o.stream_ = nullptr;
    }
    return *this;
}

SwapPool::~SwapPool() {
    for (auto* p : k_host_pools_) cudaFreeHost(p);
    for (auto* p : v_host_pools_) cudaFreeHost(p);
    if (stream_) cudaStreamDestroy(stream_);
}

namespace {

void check_pairs(std::size_t a, std::size_t b) {
    if (a != b) {
        throw std::runtime_error(
            "swap_pool: src/dst page count mismatch (" +
            std::to_string(a) + " vs " + std::to_string(b) + ")");
    }
}

inline void* page_addr(void* base, std::uint32_t page_idx, std::size_t page_bytes) {
    return static_cast<std::uint8_t*>(base) +
           static_cast<std::size_t>(page_idx) * page_bytes;
}

}  // namespace

void SwapPool::copy_d2h(KvCache& cache,
                        std::span<const std::uint32_t> src_gpu,
                        std::span<const std::uint32_t> dst_host)
{
    check_pairs(src_gpu.size(), dst_host.size());
    for (int layer = 0; layer < num_layers_; ++layer) {
        void* k_dev = cache.k(layer);
        void* v_dev = cache.v(layer);
        for (std::size_t i = 0; i < src_gpu.size(); ++i) {
            CUDA_CHECK(cudaMemcpyAsync(
                page_addr(k_host_pools_[layer], dst_host[i], page_bytes_),
                page_addr(k_dev, src_gpu[i], page_bytes_),
                page_bytes_, cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaMemcpyAsync(
                page_addr(v_host_pools_[layer], dst_host[i], page_bytes_),
                page_addr(v_dev, src_gpu[i], page_bytes_),
                page_bytes_, cudaMemcpyDeviceToHost, stream_));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void SwapPool::copy_h2d(KvCache& cache,
                        std::span<const std::uint32_t> src_host,
                        std::span<const std::uint32_t> dst_gpu)
{
    check_pairs(src_host.size(), dst_gpu.size());
    for (int layer = 0; layer < num_layers_; ++layer) {
        void* k_dev = cache.k(layer);
        void* v_dev = cache.v(layer);
        for (std::size_t i = 0; i < src_host.size(); ++i) {
            CUDA_CHECK(cudaMemcpyAsync(
                page_addr(k_dev, dst_gpu[i], page_bytes_),
                page_addr(k_host_pools_[layer], src_host[i], page_bytes_),
                page_bytes_, cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(
                page_addr(v_dev, dst_gpu[i], page_bytes_),
                page_addr(v_host_pools_[layer], src_host[i], page_bytes_),
                page_bytes_, cudaMemcpyHostToDevice, stream_));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void SwapPool::copy_d2d(KvCache& cache,
                        std::span<const std::uint32_t> src_gpu,
                        std::span<const std::uint32_t> dst_gpu)
{
    check_pairs(src_gpu.size(), dst_gpu.size());
    for (int layer = 0; layer < num_layers_; ++layer) {
        void* k_dev = cache.k(layer);
        void* v_dev = cache.v(layer);
        for (std::size_t i = 0; i < src_gpu.size(); ++i) {
            CUDA_CHECK(cudaMemcpyAsync(
                page_addr(k_dev, dst_gpu[i], page_bytes_),
                page_addr(k_dev, src_gpu[i], page_bytes_),
                page_bytes_, cudaMemcpyDeviceToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(
                page_addr(v_dev, dst_gpu[i], page_bytes_),
                page_addr(v_dev, src_gpu[i], page_bytes_),
                page_bytes_, cudaMemcpyDeviceToDevice, stream_));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void SwapPool::copy_h2h(std::span<const std::uint32_t> src_host,
                        std::span<const std::uint32_t> dst_host)
{
    check_pairs(src_host.size(), dst_host.size());
    for (int layer = 0; layer < num_layers_; ++layer) {
        auto* k_pool = k_host_pools_[layer];
        auto* v_pool = v_host_pools_[layer];
        for (std::size_t i = 0; i < src_host.size(); ++i) {
            std::memcpy(
                page_addr(k_pool, dst_host[i], page_bytes_),
                page_addr(k_pool, src_host[i], page_bytes_),
                page_bytes_);
            std::memcpy(
                page_addr(v_pool, dst_host[i], page_bytes_),
                page_addr(v_pool, src_host[i], page_bytes_),
                page_bytes_);
        }
    }
}

}  // namespace pie_cuda_driver

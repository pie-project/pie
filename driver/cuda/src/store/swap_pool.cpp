#include "swap_pool.hpp"

#include <cstring>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>

#include "../cuda_check.hpp"

namespace pie_cuda_driver {

void SwapPool::synchronize() const {
    if (stream_ != nullptr) CUDA_CHECK(cudaStreamSynchronize(stream_));
}

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

    const std::size_t one_page_bytes =
        static_cast<std::size_t>(page_size) *
        static_cast<std::size_t>(num_kv_heads) *
        static_cast<std::size_t>(head_dim) *
        dtype_bytes(dtype);
    s.page_bytes_ = 2 * static_cast<std::size_t>(num_layers) * one_page_bytes;
    if (num_pages <= 0 || num_layers <= 0) return s;

    CUDA_CHECK(cudaStreamCreateWithFlags(&s.stream_, cudaStreamNonBlocking));
    s.host_pools_.reserve(num_layers);
    for (int layer = 0; layer < num_layers; ++layer) {
        std::vector<HostBuffer> pools;
        pools.reserve(2);
        for (int i = 0; i < 2; ++i) {
            void* p = nullptr;
            CUDA_CHECK(cudaMallocHost(
                &p, one_page_bytes * static_cast<std::size_t>(num_pages)));
            pools.push_back({p, one_page_bytes});
        }
        s.host_pools_.push_back(std::move(pools));
    }
    return s;
}

SwapPool SwapPool::allocate_for_cache(const KvCache& cache, int num_pages)
{
    SwapPool s;
    s.num_layers_ = cache.num_layers();
    s.num_pages_ = num_pages;
    s.page_size_ = cache.page_size();
    s.num_kv_heads_ = cache.num_kv_heads();
    s.head_dim_ = cache.head_dim();

    if (num_pages <= 0 || cache.num_layers() <= 0) return s;

    CUDA_CHECK(cudaStreamCreateWithFlags(&s.stream_, cudaStreamNonBlocking));

    s.host_pools_.reserve(cache.num_layers());
    for (int layer = 0; layer < cache.num_layers(); ++layer) {
        auto dev_buffers = const_cast<KvCache&>(cache).page_buffers(layer);
        std::vector<HostBuffer> layer_pools;
        layer_pools.reserve(dev_buffers.size());
        for (const auto& db : dev_buffers) {
            s.page_bytes_ += db.page_bytes;
            void* p = nullptr;
            const std::size_t pool_bytes =
                db.page_bytes * static_cast<std::size_t>(num_pages);
            CUDA_CHECK(cudaMallocHost(&p, pool_bytes));
            layer_pools.push_back({p, db.page_bytes});
        }
        s.host_pools_.push_back(std::move(layer_pools));
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
      host_pools_(std::move(o.host_pools_)),
      stream_(o.stream_) {
    o.num_pages_ = 0;
    o.stream_ = nullptr;
}

SwapPool& SwapPool::operator=(SwapPool&& o) noexcept {
    if (this != &o) {
        for (auto& layer : host_pools_) {
            for (auto& p : layer) cudaFreeHost(p.data);
        }
        if (stream_) cudaStreamDestroy(stream_);
        num_layers_ = o.num_layers_;
        num_pages_ = o.num_pages_;
        page_size_ = o.page_size_;
        num_kv_heads_ = o.num_kv_heads_;
        head_dim_ = o.head_dim_;
        page_bytes_ = o.page_bytes_;
        host_pools_ = std::move(o.host_pools_);
        stream_ = o.stream_;
        o.num_pages_ = 0;
        o.stream_ = nullptr;
    }
    return *this;
}

SwapPool::~SwapPool() {
    for (auto& layer : host_pools_) {
        for (auto& p : layer) cudaFreeHost(p.data);
    }
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
    copy_d2h_async(cache, src_gpu, dst_host);
    synchronize();
}

void SwapPool::copy_d2h_async(KvCache& cache,
                              std::span<const std::uint32_t> src_gpu,
                              std::span<const std::uint32_t> dst_host)
{
    check_pairs(src_gpu.size(), dst_host.size());
    for (int layer = 0; layer < num_layers_; ++layer) {
        auto dev_buffers = cache.page_buffers(layer);
        auto& host_buffers = host_pools_[layer];
        for (std::size_t i = 0; i < src_gpu.size(); ++i) {
            for (std::size_t b = 0; b < dev_buffers.size(); ++b) {
                CUDA_CHECK(cudaMemcpyAsync(
                    page_addr(host_buffers[b].data, dst_host[i], host_buffers[b].page_bytes),
                    page_addr(dev_buffers[b].data, src_gpu[i], dev_buffers[b].page_bytes),
                    dev_buffers[b].page_bytes, cudaMemcpyDeviceToHost, stream_));
            }
        }
    }
}

void SwapPool::copy_h2d(KvCache& cache,
                        std::span<const std::uint32_t> src_host,
                        std::span<const std::uint32_t> dst_gpu)
{
    copy_h2d_async(cache, src_host, dst_gpu);
    synchronize();
}

void SwapPool::copy_h2d_async(KvCache& cache,
                              std::span<const std::uint32_t> src_host,
                              std::span<const std::uint32_t> dst_gpu)
{
    check_pairs(src_host.size(), dst_gpu.size());
    for (int layer = 0; layer < num_layers_; ++layer) {
        auto dev_buffers = cache.page_buffers(layer);
        auto& host_buffers = host_pools_[layer];
        for (std::size_t i = 0; i < src_host.size(); ++i) {
            for (std::size_t b = 0; b < dev_buffers.size(); ++b) {
                CUDA_CHECK(cudaMemcpyAsync(
                    page_addr(dev_buffers[b].data, dst_gpu[i], dev_buffers[b].page_bytes),
                    page_addr(host_buffers[b].data, src_host[i], host_buffers[b].page_bytes),
                    dev_buffers[b].page_bytes, cudaMemcpyHostToDevice, stream_));
            }
        }
    }
}

void SwapPool::copy_d2d(KvCache& cache,
                        std::span<const std::uint32_t> src_gpu,
                        std::span<const std::uint32_t> dst_gpu)
{
    copy_d2d_async(cache, src_gpu, dst_gpu);
    synchronize();
}

void SwapPool::copy_d2d_async(KvCache& cache,
                              std::span<const std::uint32_t> src_gpu,
                              std::span<const std::uint32_t> dst_gpu)
{
    check_pairs(src_gpu.size(), dst_gpu.size());
    for (int layer = 0; layer < num_layers_; ++layer) {
        auto dev_buffers = cache.page_buffers(layer);
        for (std::size_t i = 0; i < src_gpu.size(); ++i) {
            for (const auto& db : dev_buffers) {
                CUDA_CHECK(cudaMemcpyAsync(
                    page_addr(db.data, dst_gpu[i], db.page_bytes),
                    page_addr(db.data, src_gpu[i], db.page_bytes),
                    db.page_bytes, cudaMemcpyDeviceToDevice, stream_));
            }
        }
    }
}

void SwapPool::copy_h2h(std::span<const std::uint32_t> src_host,
                        std::span<const std::uint32_t> dst_host)
{
    copy_h2h_async(src_host, dst_host);
}

void SwapPool::copy_h2h_async(std::span<const std::uint32_t> src_host,
                              std::span<const std::uint32_t> dst_host)
{
    check_pairs(src_host.size(), dst_host.size());
    for (int layer = 0; layer < num_layers_; ++layer) {
        auto& host_buffers = host_pools_[layer];
        for (std::size_t i = 0; i < src_host.size(); ++i) {
            for (const auto& hb : host_buffers) {
                std::memcpy(
                    page_addr(hb.data, dst_host[i], hb.page_bytes),
                    page_addr(hb.data, src_host[i], hb.page_bytes),
                    hb.page_bytes);
            }
        }
    }
}

}  // namespace pie_cuda_driver

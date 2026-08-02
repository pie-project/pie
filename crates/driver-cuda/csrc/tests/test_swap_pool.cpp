#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "store/kv_cache.hpp"
#include "store/kv_cache_format.hpp"
#include "store/swap_pool.hpp"

namespace {

constexpr int kLayers = 2;
constexpr int kGpuPages = 4;
constexpr int kHostPages = 4;
constexpr int kPageSize = 3;
constexpr int kHeads = 2;
constexpr int kHeadDim = 5;

std::vector<std::uint8_t> pattern(int layer, int buffer, std::size_t n) {
    std::vector<std::uint8_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = static_cast<std::uint8_t>(
            17 * layer + 37 * buffer + 13 * (i / 7) + i);
    }
    return out;
}

void fill_cache(pie_cuda_driver::KvCache& cache,
                std::vector<std::vector<std::vector<std::uint8_t>>>& expected_page0) {
    expected_page0.resize(kLayers);
    for (int layer = 0; layer < kLayers; ++layer) {
        auto buffers = cache.page_buffers(layer);
        expected_page0[layer].resize(buffers.size());
        for (std::size_t b = 0; b < buffers.size(); ++b) {
            const std::size_t total = buffers[b].page_bytes * kGpuPages;
            auto host = pattern(layer, static_cast<int>(b), total);
            expected_page0[layer][b].assign(
                host.begin(), host.begin() + static_cast<std::ptrdiff_t>(buffers[b].page_bytes));
            CUDA_CHECK(cudaMemcpy(buffers[b].data, host.data(), total,
                                  cudaMemcpyHostToDevice));
        }
    }
}

void zero_page(pie_cuda_driver::KvCache& cache, int page) {
    for (int layer = 0; layer < kLayers; ++layer) {
        for (const auto& buffer : cache.page_buffers(layer)) {
            CUDA_CHECK(cudaMemset(
                static_cast<std::uint8_t*>(buffer.data) +
                    static_cast<std::size_t>(page) * buffer.page_bytes,
                0, buffer.page_bytes));
        }
    }
}

void expect_page(pie_cuda_driver::KvCache& cache,
                 int page,
                 const std::vector<std::vector<std::vector<std::uint8_t>>>& expected,
                 const std::string& label) {
    for (int layer = 0; layer < kLayers; ++layer) {
        auto buffers = cache.page_buffers(layer);
        for (std::size_t b = 0; b < buffers.size(); ++b) {
            std::vector<std::uint8_t> got(buffers[b].page_bytes);
            CUDA_CHECK(cudaMemcpy(
                got.data(),
                static_cast<std::uint8_t*>(buffers[b].data) +
                    static_cast<std::size_t>(page) * buffers[b].page_bytes,
                buffers[b].page_bytes, cudaMemcpyDeviceToHost));
            if (got != expected[layer][b]) {
                throw std::runtime_error(
                    label + ": layer=" + std::to_string(layer) +
                    " buffer=" + std::to_string(b) + " page=" +
                    std::to_string(page) + " mismatch");
            }
        }
    }
}

}  // namespace

int main() {
    try {
        auto fmt = pie_cuda_driver::kv_cache_format_from_string("int8_per_token_head");
        auto cache = pie_cuda_driver::KvCache::allocate(
            kLayers, kGpuPages, kPageSize, kHeads, kHeadDim, fmt);
        std::vector<std::vector<std::vector<std::uint8_t>>> expected_page0;
        fill_cache(cache, expected_page0);

        auto swap = pie_cuda_driver::SwapPool::allocate_for_cache(cache, kHostPages);

        const std::uint32_t gpu0[] = {0};
        const std::uint32_t gpu1[] = {1};
        const std::uint32_t gpu2[] = {2};
        const std::uint32_t gpu3[] = {3};
        const std::uint32_t slot0[] = {0};
        const std::uint32_t slot1[] = {1};

        swap.copy_d2h_async(cache, gpu0, slot0);
        swap.synchronize();

        zero_page(cache, 1);
        swap.copy_h2d_async(cache, slot0, gpu1);
        swap.synchronize();
        expect_page(cache, 1, expected_page0, "h2d");

        zero_page(cache, 2);
        swap.copy_d2d_async(cache, gpu0, gpu2);
        swap.synchronize();
        expect_page(cache, 2, expected_page0, "d2d");

        zero_page(cache, 3);
        swap.copy_h2h_async(slot0, slot1);
        swap.copy_h2d_async(cache, slot1, gpu3);
        swap.synchronize();
        expect_page(cache, 3, expected_page0, "h2h+h2d");

        std::puts("swap_pool ok");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_swap_pool failed: %s\n", e.what());
        return 1;
    }
}

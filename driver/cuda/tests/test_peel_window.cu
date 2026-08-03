// Peel device-window campaign, kernel A/B #1: the device-window
// explicit KV write must reproduce the host-window form byte for byte
// on every window shape — full, empty, prefix, suffix, interior — in
// both page layouts. The host form windows by CALLER pointer offsets;
// the devwin form takes base pointers and a {start, len} device word.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/kv_cache_view.hpp"
#include "kernels/kv_paged.hpp"

using namespace pie_cuda_driver;

namespace {

constexpr int kLanes = 7;
constexpr int kPages = 16;
constexpr int kPageSize = 4;
constexpr int kHkv = 2;
constexpr int kDim = 8;
constexpr long long kRow = static_cast<long long>(kHkv) * kDim;
constexpr long long kCacheElems =
    static_cast<long long>(kPages) * kPageSize * kRow;

// The host-window reference: offset pointers + count, exactly the
// interpreter's `bf16_row(..., win_start, Hk)` calls.
void run_host_window(KvCacheLayerView layer, const std::uint16_t* k_d,
                     const std::uint16_t* v_d, const std::uint32_t* wp_d,
                     const std::uint32_t* wo_d, int start, int len,
                     cudaStream_t s) {
    const std::uint16_t* k_off = k_d + static_cast<long long>(start) * kRow;
    const std::uint16_t* v_off = v_d + static_cast<long long>(start) * kRow;
    kernels::launch_write_kv_explicit_bf16(
        layer, k_off, v_off, wp_d + start, wo_d + start, len, s,
        /*row_valid=*/nullptr);
}

}  // namespace

int main() {
    cudaStream_t s{};
    CUDA_CHECK(cudaStreamCreate(&s));

    std::vector<std::uint16_t> k_h(kLanes * kRow), v_h(kLanes * kRow);
    for (std::size_t i = 0; i < k_h.size(); ++i) {
        k_h[i] = static_cast<std::uint16_t>(0x3f80 + i);
        v_h[i] = static_cast<std::uint16_t>(0x4000 + i * 3);
    }
    std::vector<std::uint32_t> wp_h(kLanes), wo_h(kLanes);
    for (int b = 0; b < kLanes; ++b) {
        wp_h[b] = static_cast<std::uint32_t>((b * 5 + 3) % kPages);
        wo_h[b] = static_cast<std::uint32_t>((b * 7 + 1) % kPageSize);
    }

    std::uint16_t *k_d{}, *v_d{}, *cache_a_k{}, *cache_a_v{}, *cache_b_k{},
        *cache_b_v{};
    std::uint32_t *wp_d{}, *wo_d{}, *win_d{};
    CUDA_CHECK(cudaMalloc(&k_d, k_h.size() * 2));
    CUDA_CHECK(cudaMalloc(&v_d, v_h.size() * 2));
    CUDA_CHECK(cudaMalloc(&cache_a_k, kCacheElems * 2));
    CUDA_CHECK(cudaMalloc(&cache_a_v, kCacheElems * 2));
    CUDA_CHECK(cudaMalloc(&cache_b_k, kCacheElems * 2));
    CUDA_CHECK(cudaMalloc(&cache_b_v, kCacheElems * 2));
    CUDA_CHECK(cudaMalloc(&wp_d, kLanes * 4));
    CUDA_CHECK(cudaMalloc(&wo_d, kLanes * 4));
    CUDA_CHECK(cudaMalloc(&win_d, 2 * 4));
    CUDA_CHECK(cudaMemcpy(k_d, k_h.data(), k_h.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(v_d, v_h.data(), v_h.size() * 2,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(wp_d, wp_h.data(), kLanes * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(wo_d, wo_h.data(), kLanes * 4,
                          cudaMemcpyHostToDevice));

    const int windows[][2] = {
        {0, kLanes},  // full
        {0, 0},       // empty
        {0, 3},       // prefix
        {4, 3},       // suffix
        {2, 3},       // interior
    };
    bool ok = true;
    for (const bool hnd : {false, true}) {
        for (const auto& w : windows) {
            KvCacheLayerView layer{};
            layer.num_pages = kPages;
            layer.page_size = kPageSize;
            layer.num_kv_heads = kHkv;
            layer.head_dim = kDim;
            layer.hnd_layout = hnd;
            layer.native_bf16 = true;

            CUDA_CHECK(cudaMemset(cache_a_k, 0xAB, kCacheElems * 2));
            CUDA_CHECK(cudaMemset(cache_a_v, 0xAB, kCacheElems * 2));
            CUDA_CHECK(cudaMemset(cache_b_k, 0xAB, kCacheElems * 2));
            CUDA_CHECK(cudaMemset(cache_b_v, 0xAB, kCacheElems * 2));

            layer.k_pages = cache_a_k;
            layer.v_pages = cache_a_v;
            run_host_window(layer, k_d, v_d, wp_d, wo_d, w[0], w[1], s);

            layer.k_pages = cache_b_k;
            layer.v_pages = cache_b_v;
            const std::uint32_t win_h[2] = {
                static_cast<std::uint32_t>(w[0]),
                static_cast<std::uint32_t>(w[1])};
            CUDA_CHECK(cudaMemcpyAsync(win_d, win_h, 8,
                                       cudaMemcpyHostToDevice, s));
            kernels::launch_write_kv_explicit_bf16_devwin(
                layer, k_d, v_d, wp_d, wo_d, win_d, kLanes, s,
                /*row_valid=*/nullptr);

            CUDA_CHECK(cudaStreamSynchronize(s));
            std::vector<std::uint16_t> a(kCacheElems), b(kCacheElems);
            CUDA_CHECK(cudaMemcpy(a.data(), cache_a_k, kCacheElems * 2,
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(b.data(), cache_b_k, kCacheElems * 2,
                                  cudaMemcpyDeviceToHost));
            const bool k_eq = std::memcmp(a.data(), b.data(),
                                          kCacheElems * 2) == 0;
            CUDA_CHECK(cudaMemcpy(a.data(), cache_a_v, kCacheElems * 2,
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(b.data(), cache_b_v, kCacheElems * 2,
                                  cudaMemcpyDeviceToHost));
            const bool v_eq = std::memcmp(a.data(), b.data(),
                                          kCacheElems * 2) == 0;
            std::printf("hnd=%d win=(%d,%d): k=%s v=%s\n", hnd ? 1 : 0,
                        w[0], w[1], k_eq ? "eq" : "NE", v_eq ? "eq" : "NE");
            ok = ok && k_eq && v_eq;
        }
    }
    std::printf("%s\n", ok ? "PEEL-WINDOW-KERNEL-1-OK" : "MISMATCH");
    return ok ? 0 : 1;
}

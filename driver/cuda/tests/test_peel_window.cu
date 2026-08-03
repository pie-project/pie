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
#include "kernels/rope.hpp"
#include "kernels/split_packed.hpp"

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
    // ── Kernel 2: split_qkv ──────────────────────────────────────────
    {
        constexpr int kQ = 12, kKv = 6;
        constexpr int kStride = kQ + 2 * kKv;
        std::vector<std::uint16_t> packed_h(kLanes * kStride);
        for (std::size_t i = 0; i < packed_h.size(); ++i) {
            packed_h[i] = static_cast<std::uint16_t>(0x3c00 + i * 5);
        }
        std::uint16_t *packed_d{}, *qa{}, *ka{}, *va{}, *qb{}, *kb{}, *vb{};
        CUDA_CHECK(cudaMalloc(&packed_d, packed_h.size() * 2));
        CUDA_CHECK(cudaMemcpy(packed_d, packed_h.data(),
                              packed_h.size() * 2, cudaMemcpyHostToDevice));
        for (auto** buf : {&qa, &qb}) CUDA_CHECK(cudaMalloc(buf, kLanes * kQ * 2));
        for (auto** buf : {&ka, &va, &kb, &vb}) {
            CUDA_CHECK(cudaMalloc(buf, kLanes * kKv * 2));
        }
        for (const auto& w : windows) {
            for (auto* buf : {qa, qb}) CUDA_CHECK(cudaMemset(buf, 0xCD, kLanes * kQ * 2));
            for (auto* buf : {ka, va, kb, vb}) {
                CUDA_CHECK(cudaMemset(buf, 0xCD, kLanes * kKv * 2));
            }
            kernels::launch_split_qkv_bf16(
                packed_d + static_cast<long long>(w[0]) * kStride,
                qa + static_cast<long long>(w[0]) * kQ,
                ka + static_cast<long long>(w[0]) * kKv,
                va + static_cast<long long>(w[0]) * kKv,
                w[1], kQ, kKv, s);
            const std::uint32_t win_h[2] = {
                static_cast<std::uint32_t>(w[0]),
                static_cast<std::uint32_t>(w[1])};
            CUDA_CHECK(cudaMemcpyAsync(win_d, win_h, 8,
                                       cudaMemcpyHostToDevice, s));
            kernels::launch_split_qkv_bf16_devwin(
                packed_d, qb, kb, vb, win_d, kLanes, kQ, kKv, s);
            CUDA_CHECK(cudaStreamSynchronize(s));
            auto cmp = [](const void* x, const void* y, std::size_t n) {
                std::vector<std::uint8_t> a(n), b(n);
                CUDA_CHECK(cudaMemcpy(a.data(), x, n, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(b.data(), y, n, cudaMemcpyDeviceToHost));
                return std::memcmp(a.data(), b.data(), n) == 0;
            };
            const bool eq = cmp(qa, qb, kLanes * kQ * 2) &&
                            cmp(ka, kb, kLanes * kKv * 2) &&
                            cmp(va, vb, kLanes * kKv * 2);
            std::printf("split win=(%d,%d): %s\n", w[0], w[1],
                        eq ? "eq" : "NE");
            ok = ok && eq;
        }
    }

    // ── Kernel 3: qk_rmsnorm_rope ────────────────────────────────────
    {
        constexpr int kQH = 4, kKH = 2, kD = 16;
        std::vector<std::uint16_t> q_h(kLanes * kQH * kD),
            kk_h(kLanes * kKH * kD), wq_h(kD), wk_h(kD);
        for (std::size_t i = 0; i < q_h.size(); ++i)
            q_h[i] = static_cast<std::uint16_t>(0x3b00 + i * 7);
        for (std::size_t i = 0; i < kk_h.size(); ++i)
            kk_h[i] = static_cast<std::uint16_t>(0x3a80 + i * 11);
        for (int i = 0; i < kD; ++i) {
            wq_h[i] = static_cast<std::uint16_t>(0x3f80 - i);
            wk_h[i] = static_cast<std::uint16_t>(0x3f00 + i);
        }
        std::vector<std::int32_t> pos_h(kLanes);
        for (int i = 0; i < kLanes; ++i) pos_h[i] = 3 + i * 13;
        std::uint16_t *qa{}, *ka{}, *qb{}, *kb{}, *wq{}, *wk{};
        std::int32_t* pos_d{};
        CUDA_CHECK(cudaMalloc(&qa, q_h.size() * 2));
        CUDA_CHECK(cudaMalloc(&qb, q_h.size() * 2));
        CUDA_CHECK(cudaMalloc(&ka, kk_h.size() * 2));
        CUDA_CHECK(cudaMalloc(&kb, kk_h.size() * 2));
        CUDA_CHECK(cudaMalloc(&wq, kD * 2));
        CUDA_CHECK(cudaMalloc(&wk, kD * 2));
        CUDA_CHECK(cudaMalloc(&pos_d, kLanes * 4));
        CUDA_CHECK(cudaMemcpy(wq, wq_h.data(), kD * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(wk, wk_h.data(), kD * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(pos_d, pos_h.data(), kLanes * 4,
                              cudaMemcpyHostToDevice));
        for (const auto& w : windows) {
            CUDA_CHECK(cudaMemcpy(qa, q_h.data(), q_h.size() * 2,
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(qb, q_h.data(), q_h.size() * 2,
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(ka, kk_h.data(), kk_h.size() * 2,
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(kb, kk_h.data(), kk_h.size() * 2,
                                  cudaMemcpyHostToDevice));
            kernels::launch_qk_rmsnorm_rope_bf16(
                qa + static_cast<long long>(w[0]) * kQH * kD,
                ka + static_cast<long long>(w[0]) * kKH * kD,
                wq, wk, pos_d + w[0], w[1], kQH, kKH, kD,
                /*theta=*/10000.f, /*eps=*/1e-6f, s);
            const std::uint32_t win_h[2] = {
                static_cast<std::uint32_t>(w[0]),
                static_cast<std::uint32_t>(w[1])};
            CUDA_CHECK(cudaMemcpyAsync(win_d, win_h, 8,
                                       cudaMemcpyHostToDevice, s));
            kernels::launch_qk_rmsnorm_rope_bf16_devwin(
                qb, kb, wq, wk, pos_d, win_d, kLanes, kQH, kKH, kD,
                10000.f, 1e-6f, s);
            CUDA_CHECK(cudaStreamSynchronize(s));
            std::vector<std::uint16_t> a(q_h.size()), b(q_h.size());
            CUDA_CHECK(cudaMemcpy(a.data(), qa, q_h.size() * 2,
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(b.data(), qb, q_h.size() * 2,
                                  cudaMemcpyDeviceToHost));
            bool eq = std::memcmp(a.data(), b.data(), q_h.size() * 2) == 0;
            std::vector<std::uint16_t> c(kk_h.size()), d2(kk_h.size());
            CUDA_CHECK(cudaMemcpy(c.data(), ka, kk_h.size() * 2,
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(d2.data(), kb, kk_h.size() * 2,
                                  cudaMemcpyDeviceToHost));
            eq = eq && std::memcmp(c.data(), d2.data(), kk_h.size() * 2) == 0;
            std::printf("qknormrope win=(%d,%d): %s\n", w[0], w[1],
                        eq ? "eq" : "NE");
            ok = ok && eq;
        }
    }

    std::printf("%s\n", ok ? "PEEL-WINDOW-KERNELS-OK" : "MISMATCH");
    return ok ? 0 : 1;
}

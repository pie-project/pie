// gather_tokens correctness + bandwidth test.
//
// Correctness: fill paged K/V so each bf16 element encodes its global index,
// run a gather plan through the kernel, and verify every destination element
// equals the SOURCE element it should have been packed from (the deterministic
// host reference). Covers within-page runs, multi-op packing, and the
// multi-layer variant. Bandwidth: a full-page gather over many pages, timed vs
// a `cudaMemcpy` of the same byte volume (exit gate: >= 80%).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/gather_tokens.hpp"

using pie_cuda_driver::kernels::GatherTokenOp;
using pie_cuda_driver::kernels::launch_gather_tokens_bf16;
using pie_cuda_driver::kernels::launch_gather_tokens_bf16_layers;

namespace {

int g_fail = 0;

void rt(cudaError_t e, const char* what, int line) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FATAL %s:%d: %s -> %s\n", __FILE__, line, what,
                     cudaGetErrorString(e));
        std::exit(2);
    }
}
#define RT(e) rt((e), #e, __LINE__)

// Element value at a global element index (distinct per (page,tok,head,dim)).
std::uint16_t enc(std::int64_t g) { return static_cast<std::uint16_t>((g * 2654435761u) & 0xFFFF); }

// One-layer correctness over `ops`. token_stride = num_kv_heads*head_dim.
void check_ops(const char* name, int num_pages, int page_size, int num_kv_heads,
               int head_dim, const std::vector<GatherTokenOp>& ops) {
    const std::int64_t ts = static_cast<std::int64_t>(num_kv_heads) * head_dim;
    const std::int64_t ps = ts * page_size;
    const std::int64_t n = static_cast<std::int64_t>(num_pages) * ps;

    std::vector<std::uint16_t> k(n), v(n);
    for (std::int64_t g = 0; g < n; ++g) { k[g] = enc(g); v[g] = enc(g ^ 0x5555); }

    std::uint16_t *dk, *dv; GatherTokenOp* dops;
    RT(cudaMalloc(&dk, n * 2)); RT(cudaMalloc(&dv, n * 2));
    RT(cudaMalloc(&dops, ops.size() * sizeof(GatherTokenOp)));
    RT(cudaMemcpy(dk, k.data(), n * 2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(dv, v.data(), n * 2, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(dops, ops.data(), ops.size() * sizeof(GatherTokenOp), cudaMemcpyHostToDevice));

    launch_gather_tokens_bf16(dk, dv, dops, static_cast<int>(ops.size()),
                              page_size, num_kv_heads, head_dim, nullptr);
    RT(cudaDeviceSynchronize());

    std::vector<std::uint16_t> gk(n), gv(n);
    RT(cudaMemcpy(gk.data(), dk, n * 2, cudaMemcpyDeviceToHost));
    RT(cudaMemcpy(gv.data(), dv, n * 2, cudaMemcpyDeviceToHost));

    // Host reference: apply the plan on the CPU, compare element-for-element.
    std::vector<std::uint16_t> rk = k, rv = v;
    for (const auto& o : ops) {
        for (std::uint32_t i = 0; i < o.len; ++i) {
            const std::int64_t s = static_cast<std::int64_t>(o.src_page) * ps +
                                   static_cast<std::int64_t>(o.src_off + i) * ts;
            const std::int64_t d = static_cast<std::int64_t>(o.dst_page) * ps +
                                   static_cast<std::int64_t>(o.dst_off + i) * ts;
            for (std::int64_t e = 0; e < ts; ++e) { rk[d + e] = k[s + e]; rv[d + e] = v[s + e]; }
        }
    }
    bool ok = true;
    for (std::int64_t g = 0; g < n && ok; ++g) ok = (gk[g] == rk[g]) && (gv[g] == rv[g]);
    std::printf("[%s] %s\n", ok ? " ok " : "FAIL", name);
    if (!ok) ++g_fail;

    cudaFree(dk); cudaFree(dv); cudaFree(dops);
}

double ms(cudaEvent_t a, cudaEvent_t b) { float m = 0.f; RT(cudaEventElapsedTime(&m, a, b)); return m; }

void bandwidth() {
    // Copy N full pages, realistic KV geometry.
    constexpr int PAGES = 4096, PS = 16, KVH = 8, HD = 128, N_OPS = 2048;
    const std::int64_t ts = static_cast<std::int64_t>(KVH) * HD;
    const std::int64_t ps = ts * PS;
    const std::int64_t n = static_cast<std::int64_t>(PAGES) * ps;

    std::vector<GatherTokenOp> ops(N_OPS);
    for (int i = 0; i < N_OPS; ++i)
        ops[i] = {static_cast<std::uint32_t>(i), 0,
                  static_cast<std::uint32_t>(N_OPS + i), 0,
                  static_cast<std::uint32_t>(PS)};  // src page i -> fresh page

    std::uint16_t *dk, *dv; GatherTokenOp* dops;
    RT(cudaMalloc(&dk, n * 2)); RT(cudaMalloc(&dv, n * 2));
    RT(cudaMalloc(&dops, ops.size() * sizeof(GatherTokenOp)));
    RT(cudaMemset(dk, 1, n * 2)); RT(cudaMemset(dv, 2, n * 2));
    RT(cudaMemcpy(dops, ops.data(), ops.size() * sizeof(GatherTokenOp), cudaMemcpyHostToDevice));

    // Bytes moved: N_OPS pages * page bytes * 2 (K+V) * 2 (read+write).
    const double moved = static_cast<double>(N_OPS) * ps * 2.0 * 2.0 * 2.0;

    cudaEvent_t e0, e1; RT(cudaEventCreate(&e0)); RT(cudaEventCreate(&e1));
    constexpr int WARM = 5, IT = 30;
    auto run = [&] { launch_gather_tokens_bf16(dk, dv, dops, N_OPS, PS, KVH, HD, nullptr); };
    for (int i = 0; i < WARM; ++i) run(); RT(cudaDeviceSynchronize());
    RT(cudaEventRecord(e0)); for (int i = 0; i < IT; ++i) run(); RT(cudaEventRecord(e1));
    RT(cudaEventSynchronize(e1));
    const double gbps_gather = moved / (ms(e0, e1) / IT / 1e3) / 1e9;

    // Reference: cudaMemcpy the same contiguous byte volume (K then V halves).
    const std::size_t half = static_cast<std::size_t>(N_OPS) * ps * 2;
    RT(cudaEventRecord(e0));
    for (int i = 0; i < IT; ++i) {
        RT(cudaMemcpy(dk + static_cast<std::int64_t>(N_OPS) * ps, dk, half, cudaMemcpyDeviceToDevice));
        RT(cudaMemcpy(dv + static_cast<std::int64_t>(N_OPS) * ps, dv, half, cudaMemcpyDeviceToDevice));
    }
    RT(cudaEventRecord(e1)); RT(cudaEventSynchronize(e1));
    const double gbps_memcpy = moved / (ms(e0, e1) / IT / 1e3) / 1e9;

    const double pct = gbps_gather / gbps_memcpy * 100.0;
    std::printf("\n[bandwidth] gather_tokens %.0f GB/s vs cudaMemcpy %.0f GB/s = %.0f%%\n",
                gbps_gather, gbps_memcpy, pct);
    if (pct < 80.0) { std::printf("  BELOW 80%% target\n"); ++g_fail; }

    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(dk); cudaFree(dv); cudaFree(dops);
}

}  // namespace

int main() {
    RT(cudaSetDevice(0));

    // Within-page run, offset src+dst.
    check_ops("single op, offset run", 8, 16, 8, 128,
              {{0, 3, 4, 0, 10}});
    // Multi-op dense packing into one dst page (compact's split emits these).
    check_ops("multi-op pack into one page", 8, 16, 8, 128,
              {{0, 0, 5, 0, 4}, {1, 2, 5, 4, 6}, {2, 8, 5, 10, 6}});
    // Full-page copies.
    check_ops("full-page copies", 8, 16, 8, 128,
              {{0, 0, 4, 0, 16}, {1, 0, 5, 0, 16}});
    // Non-8-aligned token stride (head_dim=7 ⇒ ts=7, u16 fallback path).
    check_ops("u16 fallback (unaligned stride)", 8, 4, 1, 7,
              {{0, 1, 2, 0, 2}});

    // Multi-layer variant: 3 layers, one op each.
    {
        constexpr int PAGES = 8, PS = 16, KVH = 8, HD = 128, LAYERS = 3;
        const std::int64_t ts = static_cast<std::int64_t>(KVH) * HD;
        const std::int64_t ps = ts * PS;
        const std::int64_t ls = static_cast<std::int64_t>(PAGES) * ps; // layer stride
        const std::int64_t n = LAYERS * ls;
        std::vector<std::uint16_t> k(n), v(n);
        for (std::int64_t g = 0; g < n; ++g) { k[g] = enc(g); v[g] = enc(g ^ 0x5555); }
        std::vector<GatherTokenOp> ops{{0, 0, 4, 0, 16}};
        std::uint16_t *dk, *dv; GatherTokenOp* dops;
        RT(cudaMalloc(&dk, n * 2)); RT(cudaMalloc(&dv, n * 2));
        RT(cudaMalloc(&dops, ops.size() * sizeof(GatherTokenOp)));
        RT(cudaMemcpy(dk, k.data(), n * 2, cudaMemcpyHostToDevice));
        RT(cudaMemcpy(dv, v.data(), n * 2, cudaMemcpyHostToDevice));
        RT(cudaMemcpy(dops, ops.data(), ops.size() * sizeof(GatherTokenOp), cudaMemcpyHostToDevice));
        launch_gather_tokens_bf16_layers(dk, dv, dops, 1, LAYERS, ls, PS, KVH, HD, nullptr);
        RT(cudaDeviceSynchronize());
        std::vector<std::uint16_t> gk(n);
        RT(cudaMemcpy(gk.data(), dk, n * 2, cudaMemcpyDeviceToHost));
        bool ok = true;
        for (int L = 0; L < LAYERS && ok; ++L)
            for (std::int64_t e = 0; e < 16 * ts; ++e) {
                const std::int64_t d = L * ls + 4 * ps + e;
                const std::int64_t s = L * ls + 0 * ps + e;
                ok = (gk[d] == k[s]);
            }
        std::printf("[%s] multi-layer variant (3 layers)\n", ok ? " ok " : "FAIL");
        if (!ok) ++g_fail;
        cudaFree(dk); cudaFree(dv); cudaFree(dops);
    }

    bandwidth();

    std::printf(g_fail ? "\nGATHER_TOKENS FAILED (%d)\n" : "\nALL PASS (0 failures)\n", g_fail);
    return g_fail ? 1 : 0;
}

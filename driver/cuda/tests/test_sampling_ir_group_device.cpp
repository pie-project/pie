// Device verify for the #10 cross-request gather/scatter helpers (group.cpp).
// Needs a GPU. Checks that gather compacts the right scattered rows and scatter
// writes a group's compact result back to the right original rows.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>

#include <cuda_runtime.h>

#include "sampling_ir/group.hpp"

using namespace pie_cuda_driver::sampling_ir;

namespace {
int g_failures = 0;
#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)
#define RT(call)                                                               \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,        \
                         cudaGetErrorString(_e));                              \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)
}  // namespace

int main() {
    const std::uint32_t V = 8;
    const int total_rows = 4;
    const std::vector<std::uint32_t> rows = {2, 0, 3};  // scattered group (non-contiguous)

    // ── gather: src [4,V] bf16-as-u16, row r = (r+1)*100 + j → compact [3,V]. ──
    std::vector<std::uint16_t> src(static_cast<std::size_t>(total_rows) * V);
    for (int r = 0; r < total_rows; ++r)
        for (std::uint32_t j = 0; j < V; ++j)
            src[r * V + j] = static_cast<std::uint16_t>((r + 1) * 100 + j);

    std::uint16_t* d_src = nullptr;
    std::uint16_t* d_dst = nullptr;
    RT(cudaMalloc(&d_src, src.size() * sizeof(std::uint16_t)));
    RT(cudaMalloc(&d_dst, rows.size() * V * sizeof(std::uint16_t)));
    RT(cudaMemcpy(d_src, src.data(), src.size() * sizeof(std::uint16_t),
                  cudaMemcpyHostToDevice));

    gather_logits_bf16(d_src, std::span<const std::uint32_t>(rows.data(), rows.size()), V,
                       d_dst, /*stream=*/nullptr);
    RT(cudaDeviceSynchronize());

    std::vector<std::uint16_t> dst(rows.size() * V);
    RT(cudaMemcpy(dst.data(), d_dst, dst.size() * sizeof(std::uint16_t),
                  cudaMemcpyDeviceToHost));
    for (std::size_t g = 0; g < rows.size(); ++g)
        for (std::uint32_t j = 0; j < V; ++j)
            CHECK(dst[g * V + j] == src[rows[g] * V + j]);  // compact row g == source row rows[g]

    // ── scatter: compact [3] tokens {10,20,30} → dst [4] at rows {2,0,3}. ──
    const std::vector<std::int32_t> comp = {10, 20, 30};
    std::int32_t* d_comp = nullptr;
    std::int32_t* d_out = nullptr;
    RT(cudaMalloc(&d_comp, comp.size() * sizeof(std::int32_t)));
    RT(cudaMalloc(&d_out, total_rows * sizeof(std::int32_t)));
    RT(cudaMemcpy(d_comp, comp.data(), comp.size() * sizeof(std::int32_t),
                  cudaMemcpyHostToDevice));
    RT(cudaMemset(d_out, 0xff, total_rows * sizeof(std::int32_t)));  // -1 sentinel

    scatter_tokens_i32(d_comp, std::span<const std::uint32_t>(rows.data(), rows.size()),
                       d_out, /*stream=*/nullptr);
    RT(cudaDeviceSynchronize());

    std::vector<std::int32_t> out(total_rows);
    RT(cudaMemcpy(out.data(), d_out, total_rows * sizeof(std::int32_t),
                  cudaMemcpyDeviceToHost));
    CHECK(out[2] == 10);  // rows[0]=2 ← comp[0]
    CHECK(out[0] == 20);  // rows[1]=0 ← comp[1]
    CHECK(out[3] == 30);  // rows[2]=3 ← comp[2]
    CHECK(out[1] == -1);  // row 1 not in the group → untouched

    RT(cudaFree(d_src));
    RT(cudaFree(d_dst));
    RT(cudaFree(d_comp));
    RT(cudaFree(d_out));

    if (g_failures == 0) {
        std::fprintf(stderr, "sampling_ir_group_device: OK\n");
        return 0;
    }
    std::fprintf(stderr, "sampling_ir_group_device: %d failure(s)\n", g_failures);
    return 1;
}

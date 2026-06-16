#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/mxfp4_marlin.hpp"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,   \
                         #cond);                                            \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t _err = (expr);                                           \
        if (_err != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA FAIL %s:%d: %s (%s)\n", __FILE__,    \
                         __LINE__, #expr, cudaGetErrorString(_err));        \
            std::exit(2);                                                    \
        }                                                                    \
    } while (0)

template <typename T>
T* device_from_host(const std::vector<T>& host) {
    T* device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, host.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(
        device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

template <typename T>
std::vector<T> host_from_device(const T* device, std::size_t count) {
    std::vector<T> host(count);
    CUDA_CHECK(cudaMemcpy(
        host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
}

int select_row(int row, pie_cuda_driver::kernels::Mxfp4RowSelect mode) {
    using pie_cuda_driver::kernels::Mxfp4RowSelect;
    switch (mode) {
        case Mxfp4RowSelect::Identity: return row;
        case Mxfp4RowSelect::Even: return 2 * row;
        case Mxfp4RowSelect::Odd: return 2 * row + 1;
    }
    return row;
}

void test_weight_even_row_offset_and_valid_padding() {
    using pie_cuda_driver::kernels::Mxfp4RowSelect;
    constexpr int source_rows = 8;
    constexpr int source_row_offset = 1;
    constexpr int selected_rows = 4;
    constexpr int valid_rows = 2;
    constexpr int source_stride_k = 64;
    constexpr int source_col_offset = 0;
    constexpr int source_k = 64;
    constexpr int target_k = 64;
    constexpr int source_packs_per_row = source_stride_k / 8;
    constexpr int target_packs = target_k / 8;

    std::vector<std::uint32_t> raw(
        source_rows * source_packs_per_row);
    for (int row = 0; row < source_rows; ++row) {
        for (int pack = 0; pack < source_packs_per_row; ++pack) {
            raw[row * source_packs_per_row + pack] =
                0xA0000000u | (static_cast<std::uint32_t>(row) << 8) |
                static_cast<std::uint32_t>(pack);
        }
    }

    auto* d_raw = device_from_host(raw);
    std::uint32_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(
        &d_out, target_packs * selected_rows * sizeof(std::uint32_t)));
    pie_cuda_driver::kernels::launch_mxfp4_weight_to_gptq_w4(
        d_raw, d_out, source_rows, source_row_offset, selected_rows,
        valid_rows, source_stride_k, source_col_offset, source_k, target_k,
        Mxfp4RowSelect::Even, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto out = host_from_device(d_out, target_packs * selected_rows);

    std::vector<std::uint32_t> expected(out.size(), 0);
    for (int pack = 0; pack < target_packs; ++pack) {
        for (int dst_row = 0; dst_row < selected_rows; ++dst_row) {
            const int idx = pack * selected_rows + dst_row;
            const int src_row =
                select_row(source_row_offset + dst_row, Mxfp4RowSelect::Even);
            if (dst_row < valid_rows && src_row < source_rows) {
                expected[idx] = raw[src_row * source_packs_per_row + pack];
            }
        }
    }
    CHECK(out == expected);
    CUDA_CHECK(cudaFree(d_raw));
    CUDA_CHECK(cudaFree(d_out));
}

void test_weight_column_offset_and_target_padding() {
    using pie_cuda_driver::kernels::Mxfp4RowSelect;
    constexpr int source_rows = 3;
    constexpr int selected_rows = 3;
    constexpr int valid_rows = 3;
    constexpr int source_stride_k = 128;
    constexpr int source_col_offset = 64;
    constexpr int source_k = 64;
    constexpr int target_k = 128;
    constexpr int source_packs_per_row = source_stride_k / 8;
    constexpr int source_pack_offset = source_col_offset / 8;
    constexpr int source_packs = source_k / 8;
    constexpr int target_packs = target_k / 8;

    std::vector<std::uint32_t> raw(source_rows * source_packs_per_row);
    for (int row = 0; row < source_rows; ++row) {
        for (int pack = 0; pack < source_packs_per_row; ++pack) {
            raw[row * source_packs_per_row + pack] =
                0xB0000000u | (static_cast<std::uint32_t>(row) << 8) |
                static_cast<std::uint32_t>(pack);
        }
    }

    auto* d_raw = device_from_host(raw);
    std::uint32_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(
        &d_out, target_packs * selected_rows * sizeof(std::uint32_t)));
    pie_cuda_driver::kernels::launch_mxfp4_weight_to_gptq_w4(
        d_raw, d_out, source_rows, 0, selected_rows, valid_rows,
        source_stride_k, source_col_offset, source_k, target_k,
        Mxfp4RowSelect::Identity, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto out = host_from_device(d_out, target_packs * selected_rows);

    std::vector<std::uint32_t> expected(out.size(), 0);
    for (int pack = 0; pack < target_packs; ++pack) {
        for (int row = 0; row < selected_rows; ++row) {
            const int idx = pack * selected_rows + row;
            if (pack < source_packs) {
                expected[idx] =
                    raw[row * source_packs_per_row + source_pack_offset + pack];
            }
        }
    }
    CHECK(out == expected);
    CUDA_CHECK(cudaFree(d_raw));
    CUDA_CHECK(cudaFree(d_out));
}

void test_scale_group_offset_permutation_and_valid_padding() {
    using pie_cuda_driver::kernels::Mxfp4RowSelect;
    constexpr int source_rows = 4;
    constexpr int source_row_offset = 1;
    constexpr int selected_rows = 8;
    constexpr int valid_rows = 2;
    constexpr int source_stride_groups = 8;
    constexpr int source_group_offset = 2;
    constexpr int source_groups = 4;
    constexpr int target_groups = 8;
    constexpr int total = selected_rows * target_groups;

    std::vector<std::uint8_t> raw(source_rows * source_stride_groups);
    for (int row = 0; row < source_rows; ++row) {
        for (int group = 0; group < source_stride_groups; ++group) {
            raw[row * source_stride_groups + group] =
                static_cast<std::uint8_t>(row * 16 + group);
        }
    }

    auto* d_raw = device_from_host(raw);
    std::uint8_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, total));
    pie_cuda_driver::kernels::launch_mxfp4_scales_to_marlin_e8m0(
        d_raw, d_out, source_rows, source_row_offset, selected_rows,
        valid_rows, source_stride_groups, source_group_offset, source_groups,
        target_groups, Mxfp4RowSelect::Identity, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto out = host_from_device(d_out, total);

    std::vector<std::uint8_t> expected(total, 0);
    for (int out_idx = 0; out_idx < total; ++out_idx) {
        const int lane4 = out_idx & 3;
        const int base4 = out_idx & ~3;
        const int pre_lane4 = (lane4 == 1) ? 2 : ((lane4 == 2) ? 1 : lane4);
        const int after_marlin = base4 + pre_lane4;
        const int block64 = after_marlin & ~63;
        const int tid64 = after_marlin & 63;
        const int src64 = (tid64 % 8) * 8 + (tid64 / 8);
        const int transposed_idx = block64 + src64;
        const int group = transposed_idx / selected_rows;
        const int dst_row = transposed_idx - group * selected_rows;
        const int src_row = source_row_offset + dst_row;
        if (dst_row < valid_rows && src_row < source_rows &&
            group < source_groups) {
            expected[out_idx] =
                raw[src_row * source_stride_groups + source_group_offset + group];
        }
    }
    CHECK(out == expected);
    CUDA_CHECK(cudaFree(d_raw));
    CUDA_CHECK(cudaFree(d_out));
}

void test_bf16_row_gather_odd_offset_and_valid_padding() {
    using pie_cuda_driver::kernels::Mxfp4RowSelect;
    constexpr int batch = 2;
    constexpr int source_rows = 8;
    constexpr int source_row_offset = 1;
    constexpr int selected_rows = 4;
    constexpr int valid_rows = 2;

    std::vector<std::uint16_t> raw(batch * source_rows);
    for (int b = 0; b < batch; ++b) {
        for (int row = 0; row < source_rows; ++row) {
            raw[b * source_rows + row] =
                static_cast<std::uint16_t>(b * 100 + row);
        }
    }

    auto* d_raw = device_from_host(raw);
    std::uint16_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, batch * selected_rows * sizeof(std::uint16_t)));
    pie_cuda_driver::kernels::launch_bf16_row_map_to_dense(
        d_raw, d_out, batch, source_rows, source_row_offset, selected_rows,
        valid_rows, Mxfp4RowSelect::Odd, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto out = host_from_device(d_out, batch * selected_rows);

    std::vector<std::uint16_t> expected(out.size(), 0);
    for (int b = 0; b < batch; ++b) {
        for (int dst_row = 0; dst_row < selected_rows; ++dst_row) {
            const int src_row =
                select_row(source_row_offset + dst_row, Mxfp4RowSelect::Odd);
            if (dst_row < valid_rows && src_row < source_rows) {
                expected[b * selected_rows + dst_row] =
                    raw[b * source_rows + src_row];
            }
        }
    }
    CHECK(out == expected);
    CUDA_CHECK(cudaFree(d_raw));
    CUDA_CHECK(cudaFree(d_out));
}

}  // namespace

int main() {
    test_weight_even_row_offset_and_valid_padding();
    test_weight_column_offset_and_target_padding();
    test_scale_group_offset_permutation_and_valid_padding();
    test_bf16_row_gather_odd_offset_and_valid_padding();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d MXFP4 Marlin test failures\n", g_failures);
        return 1;
    }
    std::puts("MXFP4 Marlin kernel golden tests passed");
    return 0;
}

#include "kernels/mxfp4_marlin.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace pie_cuda_driver::kernels {

namespace {

__device__ __forceinline__ int select_row(
    int row,
    Mxfp4RowSelect mode)
{
    switch (mode) {
        case Mxfp4RowSelect::Identity: return row;
        case Mxfp4RowSelect::Even:     return 2 * row;
        case Mxfp4RowSelect::Odd:      return 2 * row + 1;
    }
    return row;
}

__global__ void mxfp4_weight_to_gptq_w4_kernel(
    const std::uint8_t* __restrict__ raw,
    std::uint32_t*      __restrict__ out,
    int source_rows,
    int source_row_offset,
    int selected_rows,
    int valid_rows,
    int source_stride_k,
    int source_col_offset,
    int source_k,
    int target_k,
    Mxfp4RowSelect row_select)
{
    const int k_packs = target_k / 8;
    const int source_k_packs = source_k / 8;
    const std::size_t total =
        static_cast<std::size_t>(k_packs) * static_cast<std::size_t>(selected_rows);
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                            threadIdx.x;
    if (idx >= total) return;

    const int k_pack = static_cast<int>(idx / selected_rows);
    const int dst_row = static_cast<int>(idx - static_cast<std::size_t>(k_pack) *
                                               selected_rows);
    const int logical_row = source_row_offset + dst_row;
    const int src_row = select_row(logical_row, row_select);
    if (dst_row >= valid_rows || src_row < 0 || src_row >= source_rows ||
        k_pack >= source_k_packs) {
        out[idx] = 0;
        return;
    }

    const std::size_t row_stride_bytes =
        static_cast<std::size_t>(source_stride_k) / 2;
    const auto* src = reinterpret_cast<const std::uint32_t*>(
        raw + static_cast<std::size_t>(src_row) * row_stride_bytes +
        static_cast<std::size_t>(source_col_offset) / 2 +
        static_cast<std::size_t>(k_pack) * sizeof(std::uint32_t));
    out[idx] = *src;
}

__global__ void mxfp4_scales_to_marlin_e8m0_kernel(
    const std::uint8_t* __restrict__ raw,
    std::uint8_t*       __restrict__ out,
    int source_rows,
    int source_row_offset,
    int selected_rows,
    int valid_rows,
    int source_stride_groups,
    int source_group_offset,
    int source_groups,
    int target_groups,
    Mxfp4RowSelect row_select)
{
    const std::size_t total =
        static_cast<std::size_t>(target_groups) *
        static_cast<std::size_t>(selected_rows);
    const std::size_t out_idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                                threadIdx.x;
    if (out_idx >= total) return;

    // Inverse of mxfp4_marlin_process_scales:
    // view(-1, 4)[:, [0, 2, 1, 3]].
    const int lane4 = static_cast<int>(out_idx & 3);
    const std::size_t base4 = out_idx & ~std::size_t{3};
    const int pre_lane4 = (lane4 == 1) ? 2 : ((lane4 == 2) ? 1 : lane4);
    const std::size_t after_marlin = base4 + static_cast<std::size_t>(pre_lane4);

    // Inverse of marlin_permute_scales' 64-wide transpose.
    const std::size_t block64 = after_marlin & ~std::size_t{63};
    const int tid64 = static_cast<int>(after_marlin & 63);
    const int src64 = (tid64 % 8) * 8 + (tid64 / 8);
    const std::size_t transposed_idx = block64 + static_cast<std::size_t>(src64);

    const int group = static_cast<int>(transposed_idx / selected_rows);
    const int dst_row = static_cast<int>(
        transposed_idx - static_cast<std::size_t>(group) * selected_rows);
    if (group < 0 || group >= target_groups) return;
    const int logical_row = source_row_offset + dst_row;
    const int src_row = select_row(logical_row, row_select);
    if (dst_row >= valid_rows || src_row < 0 || src_row >= source_rows ||
        group >= source_groups) {
        out[out_idx] = 0;
        return;
    }

    out[out_idx] =
        raw[static_cast<std::size_t>(src_row) *
                static_cast<std::size_t>(source_stride_groups) +
            static_cast<std::size_t>(source_group_offset + group)];
}

__global__ void bf16_row_map_to_dense_kernel(
    const std::uint16_t* __restrict__ raw,
    std::uint16_t*       __restrict__ out,
    int batch,
    int source_rows,
    int source_row_offset,
    int selected_rows,
    int valid_rows,
    Mxfp4RowSelect row_select)
{
    const std::size_t total =
        static_cast<std::size_t>(batch) *
        static_cast<std::size_t>(selected_rows);
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                            threadIdx.x;
    if (idx >= total) return;

    const int b = static_cast<int>(idx / selected_rows);
    const int dst_row = static_cast<int>(
        idx - static_cast<std::size_t>(b) * selected_rows);
    const int logical_row = source_row_offset + dst_row;
    const int src_row = select_row(logical_row, row_select);
    if (dst_row >= valid_rows || src_row < 0 || src_row >= source_rows) {
        out[idx] = 0;
        return;
    }
    out[idx] =
        raw[static_cast<std::size_t>(b) * static_cast<std::size_t>(source_rows) +
            static_cast<std::size_t>(src_row)];
}

void validate_row_select(
    const char* op,
    int source_rows,
    int source_row_offset,
    int selected_rows,
    int valid_rows,
    Mxfp4RowSelect row_select)
{
    if (source_rows <= 0 || selected_rows <= 0 || valid_rows <= 0 ||
        valid_rows > selected_rows || source_row_offset < 0) {
        throw std::runtime_error(std::string(op) + ": row counts must be positive");
    }
    const long long logical_end =
        static_cast<long long>(source_row_offset) + valid_rows;
    const long long required = row_select == Mxfp4RowSelect::Identity
        ? logical_end
        : logical_end * 2;
    if (required > source_rows) {
        throw std::runtime_error(
            std::string(op) + ": row offset exceeds source row table");
    }
}

}  // namespace

void launch_mxfp4_weight_to_gptq_w4(
    const void* raw_mxfp4,
    void*       gptq_w4_out,
    int         source_rows,
    int         source_row_offset,
    int         selected_rows,
    int         valid_rows,
    int         source_stride_k,
    int         source_col_offset,
    int         source_k,
    int         target_k,
    Mxfp4RowSelect row_select,
    cudaStream_t stream)
{
    validate_row_select(
        "launch_mxfp4_weight_to_gptq_w4",
        source_rows, source_row_offset, selected_rows, valid_rows, row_select);
    if (source_k <= 0 || target_k <= 0 || source_stride_k <= 0 ||
        source_col_offset < 0 || source_k % 8 != 0 || target_k % 8 != 0 ||
        source_stride_k % 8 != 0 || source_col_offset % 8 != 0 ||
        target_k < source_k ||
        static_cast<long long>(source_col_offset) + source_k > source_stride_k) {
        throw std::runtime_error(
            "launch_mxfp4_weight_to_gptq_w4: source/target K, stride, "
            "and column offset must be divisible by 8; target K must cover "
            "source K; and the source slice must fit in the source stride");
    }
    const std::size_t total =
        static_cast<std::size_t>(target_k / 8) *
        static_cast<std::size_t>(selected_rows);
    constexpr int BLOCK = 256;
    const int grid = static_cast<int>((total + BLOCK - 1) / BLOCK);
    mxfp4_weight_to_gptq_w4_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const std::uint8_t*>(raw_mxfp4),
        static_cast<std::uint32_t*>(gptq_w4_out),
        source_rows, source_row_offset, selected_rows, valid_rows,
        source_stride_k, source_col_offset, source_k, target_k, row_select);
}

void launch_mxfp4_scales_to_marlin_e8m0(
    const void* raw_e8m0,
    void*       marlin_e8m0,
    int         source_rows,
    int         source_row_offset,
    int         selected_rows,
    int         valid_rows,
    int         source_stride_groups,
    int         source_group_offset,
    int         source_groups,
    int         target_groups,
    Mxfp4RowSelect row_select,
    cudaStream_t stream)
{
    validate_row_select(
        "launch_mxfp4_scales_to_marlin_e8m0",
        source_rows, source_row_offset, selected_rows, valid_rows, row_select);
    if (source_groups <= 0 || target_groups <= 0 ||
        source_stride_groups <= 0 || source_group_offset < 0 ||
        target_groups < source_groups ||
        static_cast<long long>(source_group_offset) + source_groups >
            source_stride_groups) {
        throw std::runtime_error(
            "launch_mxfp4_scales_to_marlin_e8m0: source/target groups, "
            "stride, and group offset must be positive; target groups must "
            "cover source groups; and the source slice must fit in stride");
    }
    const std::size_t total =
        static_cast<std::size_t>(target_groups) *
        static_cast<std::size_t>(selected_rows);
    if (total % 64 != 0) {
        throw std::runtime_error(
            "launch_mxfp4_scales_to_marlin_e8m0: scale layout requires total "
            "scale count divisible by 64");
    }
    constexpr int BLOCK = 256;
    const int grid = static_cast<int>((total + BLOCK - 1) / BLOCK);
    mxfp4_scales_to_marlin_e8m0_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const std::uint8_t*>(raw_e8m0),
        static_cast<std::uint8_t*>(marlin_e8m0),
        source_rows, source_row_offset, selected_rows, valid_rows,
        source_stride_groups, source_group_offset, source_groups,
        target_groups, row_select);
}

void launch_bf16_row_map_to_dense(
    const void* raw_bf16,
    void*       out_bf16,
    int         batch,
    int         source_rows,
    int         source_row_offset,
    int         selected_rows,
    int         valid_rows,
    Mxfp4RowSelect row_select,
    cudaStream_t stream)
{
    validate_row_select(
        "launch_bf16_row_map_to_dense",
        source_rows, source_row_offset, selected_rows, valid_rows, row_select);
    if (batch <= 0) {
        throw std::runtime_error(
            "launch_bf16_row_map_to_dense: batch must be positive");
    }
    const std::size_t total =
        static_cast<std::size_t>(batch) *
        static_cast<std::size_t>(selected_rows);
    constexpr int BLOCK = 256;
    const int grid = static_cast<int>((total + BLOCK - 1) / BLOCK);
    bf16_row_map_to_dense_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const std::uint16_t*>(raw_bf16),
        static_cast<std::uint16_t*>(out_bf16),
        batch, source_rows, source_row_offset, selected_rows, valid_rows,
        row_select);
}

}  // namespace pie_cuda_driver::kernels

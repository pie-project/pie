#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "entry_validation.hpp"
#include "model/precomputed_embeddings.hpp"

namespace {

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(
        stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
    return false;
}

}  // namespace

int main() {
    constexpr int rows = 4;
    constexpr int hidden = 3;
    std::uint16_t* device = nullptr;
    if (!check(
            cudaMalloc(&device, rows * hidden * sizeof(std::uint16_t)),
            "cudaMalloc") ||
        !check(
            cudaMemset(device, 0, rows * hidden * sizeof(std::uint16_t)),
            "cudaMemset")) {
        return 1;
    }

    const std::vector<std::uint16_t> embeddings = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };
    const std::uint32_t indptr[] = {
        0,
        hidden * sizeof(std::uint16_t),
        3 * hidden * sizeof(std::uint16_t),
    };
    const std::uint32_t shapes[] = {1, hidden, 2, hidden};
    const std::uint8_t dtypes[] = {2, 2};
    const std::uint32_t anchors[] = {1, 2};
    pie_cuda_driver::PrecomputedEmbeddingInputs inputs{
        .rows_h = reinterpret_cast<const std::uint8_t*>(embeddings.data()),
        .byte_indptr_h = indptr,
        .shapes_h = shapes,
        .dtypes_h = dtypes,
        .anchor_rows_h = anchors,
        .num_blocks = 2,
    };

    pie_cuda_driver::model::scatter_precomputed_embeddings(
        inputs, device, rows, hidden, nullptr);
    if (!check(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) return 1;

    std::vector<std::uint16_t> actual(rows * hidden);
    if (!check(
            cudaMemcpy(
                actual.data(), device, actual.size() * sizeof(std::uint16_t),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy")) {
        return 1;
    }
    const std::vector<std::uint16_t> expected = {
        0, 0, 0,
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };
    if (actual != expected) {
        std::fprintf(stderr, "precomputed embedding scatter mismatch\n");
        return 1;
    }

    const std::uint32_t bad_shapes[] = {1, hidden + 1};
    inputs.shapes_h = bad_shapes;
    inputs.num_blocks = 1;
    bool rejected = false;
    try {
        pie_cuda_driver::model::scatter_precomputed_embeddings(
            inputs, device, rows, hidden, nullptr);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    cudaFree(device);
    if (!rejected) {
        std::fprintf(stderr, "invalid embedding width was accepted\n");
        return 1;
    }

    constexpr std::size_t patch_bytes = 3 * 16 * 16 * sizeof(float);
    std::vector<std::uint8_t> pixels(9 * patch_bytes);
    const std::uint32_t pixel_indptr[] = {
        0, static_cast<std::uint32_t>(pixels.size())};
    std::vector<std::uint32_t> bad_positions(18, 3);
    const std::uint32_t image_grid[] = {1, 3, 3};
    const std::uint32_t image_anchor[] = {0};
    std::vector<std::uint8_t> encode_output(hidden * sizeof(std::uint16_t));
    std::uint32_t encode_output_indptr[] = {0, 0};
    PieEncodeDesc descriptor{};
    descriptor.abi_version = PIE_DRIVER_ABI_VERSION;
    descriptor.image_grids = {.ptr = image_grid, .len = 3};
    descriptor.image_pixels = {.ptr = pixels.data(), .len = pixels.size()};
    descriptor.image_pixel_indptr = {.ptr = pixel_indptr, .len = 2};
    descriptor.image_patch_positions = {
        .ptr = bad_positions.data(), .len = bad_positions.size()};
    descriptor.image_anchor_rows = {.ptr = image_anchor, .len = 1};
    descriptor.output_rows = {
        .ptr = encode_output.data(), .len = encode_output.size()};
    descriptor.output_row_indptr = {
        .ptr = encode_output_indptr, .len = 2};
    const pie_cuda_driver::abi::MultimodalLimits limits{
        .gemma4_pool_kernel = 3,
        .gemma4_position_table = 16,
    };
    if (pie_cuda_driver::abi::validate_encode_resources(
            descriptor, limits, hidden) != PIE_STATUS_INVALID_ARGUMENT) {
        std::fprintf(stderr, "out-of-range vision pool group was accepted\n");
        return 1;
    }
    return 0;
}

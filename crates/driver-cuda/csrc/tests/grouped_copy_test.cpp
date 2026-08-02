#include <cstdint>
#include <cstdio>
#include <vector>

#include "pipeline/grouped_copy.hpp"

int main() {
    constexpr std::uint32_t rows = 3;
    constexpr std::uint32_t logical_vocab = 5;
    constexpr std::uint32_t physical_stride = 8;
    std::vector<std::uint32_t> physical(
        static_cast<std::size_t>(rows) * physical_stride, UINT32_MAX);
    std::vector<std::uint32_t> expected(
        static_cast<std::size_t>(rows) * logical_vocab);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t column = 0; column < logical_vocab; ++column) {
            const std::uint32_t value = row * 100 + column;
            physical[static_cast<std::size_t>(row) * physical_stride + column] =
                value;
            expected[static_cast<std::size_t>(row) * logical_vocab + column] =
                value;
        }
    }
    std::vector<std::uint32_t> compact(expected.size());
    for (std::uint64_t index = 0; index < compact.size(); ++index) {
        compact[index] = physical[
            pie_cuda_driver::pipeline::grouped_row_strided_source_index(
                index, logical_vocab, physical_stride)];
    }
    if (compact != expected ||
        pie_cuda_driver::pipeline::grouped_row_strided_source_index(
            7, 0, 0) != 7) {
        std::fputs("grouped_copy_test: stride mismatch\n", stderr);
        return 1;
    }
    std::puts("grouped_copy_test: OK");
    return 0;
}

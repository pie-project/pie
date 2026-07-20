#pragma once

#include <cstdint>

namespace pie_cuda_driver {

struct PrecomputedEmbeddingInputs {
    const std::uint8_t* rows_h = nullptr;
    const std::uint32_t* byte_indptr_h = nullptr;
    const std::uint32_t* shapes_h = nullptr;
    const std::uint8_t* dtypes_h = nullptr;
    const std::uint32_t* anchor_rows_h = nullptr;
    int num_blocks = 0;
};

}  // namespace pie_cuda_driver

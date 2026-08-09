#pragma once
#include <cstddef>
namespace pie_cuda_driver {
std::size_t persistent_input_bytes(int N, int R, int max_page_refs,
                                   int max_custom_mask_bytes);
}  // namespace pie_cuda_driver

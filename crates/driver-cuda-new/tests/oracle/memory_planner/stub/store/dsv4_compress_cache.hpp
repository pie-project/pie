#pragma once
#include <cstddef>
namespace pie_cuda_driver {
struct HfConfig;
std::size_t dsv4_compress_bytes_per_token(const HfConfig& cfg);
}  // namespace pie_cuda_driver

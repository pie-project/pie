#pragma once
#include <cstddef>
namespace pie_cuda_driver {
struct HfConfig;
namespace model {
std::size_t qwen3_5_la_workspace_bytes(const HfConfig& cfg, int N, int tp);
}
}  // namespace pie_cuda_driver

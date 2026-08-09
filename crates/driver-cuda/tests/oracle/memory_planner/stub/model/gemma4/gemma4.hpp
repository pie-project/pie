#pragma once
#include <cstddef>
namespace pie_cuda_driver {
struct HfConfig;
namespace model {
std::size_t gemma4_moe_workspace_bytes(const HfConfig& cfg, int N);
}
}  // namespace pie_cuda_driver

#pragma once
#include <cstddef>
namespace pie_cuda_driver {
struct HfConfig;
namespace model {
std::size_t glm5_workspace_bytes(const HfConfig& cfg, int N, int R, int maxpos, int tp);
}
}  // namespace pie_cuda_driver

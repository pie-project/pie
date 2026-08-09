#pragma once
#include <cstddef>
#include <cuda_runtime.h>
namespace pie_cuda_driver {
struct HfConfig;
struct Config;
std::size_t attention_float_workspace_bytes(const HfConfig& hf,
                                            const Config& cfg,
                                            const cudaDeviceProp& prop,
                                            int N, int R,
                                            bool prefill_graph_capable);
}  // namespace pie_cuda_driver

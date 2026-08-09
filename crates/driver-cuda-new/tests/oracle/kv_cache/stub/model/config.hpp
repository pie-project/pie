#pragma once
// Stub for csrc/src/model/config.hpp. `kv_cache.cpp`'s two free functions read
// three integers from it.
#include <string>
namespace pie_cuda_driver {
struct HfConfig {
    int num_hidden_layers = 0;
    int num_key_value_heads = 0;
    int head_dim_kernel = 0;
};
}  // namespace pie_cuda_driver

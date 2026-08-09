#pragma once
// Stub for csrc/src/model/config.hpp.
//
// The three files under test read five integers and one vector from it.
// `head_dim` and `head_dim_kernel` are separate fields in the real config and
// the dsv4 cache reads `head_dim` while `kv_cache.cpp`'s free functions read
// `head_dim_kernel`; keeping both distinct here is what makes a swap between
// them show up in the transcript.
#include <string>
#include <vector>
namespace pie_cuda_driver {
struct HfConfig {
    int num_hidden_layers = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    int head_dim_kernel = 0;
    std::vector<int> dsv4_compress_ratios;
};
}  // namespace pie_cuda_driver

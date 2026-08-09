#pragma once
// Stub for csrc/src/model/config.hpp. `make_planner_profile_key` copies six
// fields out of HfConfig; the real header is 1,400 lines of every checkpoint
// family the tree supports and none of the rest is reachable from here.
#include <string>
namespace pie_cuda_driver {
struct HfConfig {
    std::string model_type;
    int hidden_size = 0;
    int num_hidden_layers = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim_kernel = 0;
};
}

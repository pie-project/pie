#pragma once
// Stub for csrc/src/model/config.hpp.
//
// `workspace.cpp` reads exactly these seven integers off it. The real header
// is 547 lines of checkpoint parsing that has nothing to do with the question
// the oracle asks, and pulling it in would drag `loaded_model.hpp` and the
// loader behind it.
//
// Seven, and not "the ones I remembered": they were taken by compiling
// `workspace.cpp` against an EMPTY struct and adding what the compiler asked
// for, so a field the shipping code reads cannot be missing here.
#include <string>

namespace pie_cuda_driver {

struct HfConfig {
    int hidden_size = 0;
    int intermediate_size = 0;
    int vocab_size = 0;
    int head_dim = 0;
    int head_dim_kernel = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
};

}  // namespace pie_cuda_driver

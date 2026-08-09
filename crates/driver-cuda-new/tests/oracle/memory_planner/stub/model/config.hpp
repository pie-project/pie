// The HfConfig subset the planner reads.
#pragma once
#include <string>

namespace pie_cuda_driver {

struct HfConfig {
    std::string model_type;
    int hidden_size = 0;
    int num_hidden_layers = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    int head_dim_kernel = 0;
    int max_position_embeddings = 0;
    int kv_lora_rank = 0;
    int qk_rope_head_dim = 0;
    bool gemma4_enable_moe = false;
    int linear_num_key_heads = 0;
    int linear_num_value_heads = 0;
    int linear_key_head_dim = 0;
    int linear_value_head_dim = 0;
    int linear_conv_kernel_dim = 0;
};

}  // namespace pie_cuda_driver

#ifndef PIE_VULKAN_MOE_PARAMS_GLSL
#define PIE_VULKAN_MOE_PARAMS_GLSL

struct RouterParams {
    uint n_experts;
    uint experts_per_token;
    uint softmax_over_all;
    uint logits_pitch;
};

struct ExpertCombineParams {
    uint width;
    uint experts_per_token;
    uint out_pitch;
};

struct MoeRouteParams {
    uint n;
    uint n_experts;
    uint experts_per_token;
    uint tile_rows;
    uint padded;
    uint width;
    uint x_pitch;
};

#endif  // PIE_VULKAN_MOE_PARAMS_GLSL

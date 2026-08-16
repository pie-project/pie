// The MoE parameter blocks, the Slang half of `moe/params.glsl`.
//
// A change of language and nothing else: the three structs below have the same
// members in the same order with the same types, because the driver packs a
// launch's scalars into a run and the module's block is what says where each
// word lands. Reordering a member here would move a number the plan already
// packed, and the shader would read a width where it wanted a pitch.

#ifndef PIE_VULKAN_MOE_PARAMS_SLANG
#define PIE_VULKAN_MOE_PARAMS_SLANG

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

#endif  // PIE_VULKAN_MOE_PARAMS_SLANG

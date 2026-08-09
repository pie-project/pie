// Shared host/shader ABI for row_gather.comp.
//
// This is the Metal `row_gather_params.h` moved unchanged into GLSL. Both
// fields are four-byte scalars, so std430 matches the C header member for
// member.

#ifndef PIE_VULKAN_ROW_GATHER_PARAMS_GLSL
#define PIE_VULKAN_ROW_GATHER_PARAMS_GLSL

struct RowGatherParams {
    uint width;
    uint count;
};

#endif  // PIE_VULKAN_ROW_GATHER_PARAMS_GLSL

// Shared host/shader ABI for row_gather.slang.
//
// The Metal `row_gather_params.h` again. Both fields are four-byte scalars, so
// the layout matches the C header member for member whichever language spells
// it.

#ifndef PIE_VULKAN_ROW_GATHER_PARAMS_SLANG
#define PIE_VULKAN_ROW_GATHER_PARAMS_SLANG

struct RowGatherParams {
    uint width;
    uint count;
};

#endif  // PIE_VULKAN_ROW_GATHER_PARAMS_SLANG

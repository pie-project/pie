// Shared host/shader ABI for the RMS-family kernels.
//
// The Metal sibling is `norm/rms_params.h` and these are the same three structs
// with the same fields in the same order, because the same host builds both:
// a struct that drifted between the two backends would be a layout the driver
// fills one way and the shader reads the other, and nothing reports it.
//
// std430 is the layout, and the fields are all four bytes, so the packing is
// the C one and the two headers agree member for member. That is a property of
// THESE structs, not of std430 -- a `vec3` or a nested struct would not, which
// is why nothing here has one.

#ifndef PIE_VULKAN_RMS_PARAMS_GLSL
#define PIE_VULKAN_RMS_PARAMS_GLSL

struct RmsParams {
    float eps;
    uint axis_size;
    uint w_stride;
    uint plus_one;
    float gain;
};

struct VNormParams {
    float eps;
    uint axis_size;
};

struct GatedRmsParams {
    float eps;
    uint vd;
};

struct LayerScalarParams {
    uint hidden;
};

struct PleCombineParams {
    float inv_sqrt2;
    uint n;
};

struct SoftcapParams {
    float cap;
    uint n;
};

#endif  // PIE_VULKAN_RMS_PARAMS_GLSL

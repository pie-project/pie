// Shared host/shader ABI for the GDN kernels.
//
// Ported directly from Metal's `gdn_params.h`; the field order is part of the
// launch contract because the host fills the same struct for both backends.

#ifndef PIE_VULKAN_GDN_PARAMS_GLSL
#define PIE_VULKAN_GDN_PARAMS_GLSL

struct GdnCoreParams {
    int Dk;
    int Dv;
    int Hk;
    int Hv;
    int conv_dim;
    int Kc;
    int q_off;
    int k_off;
    int v_off;
    float eps;
    float inv_sqrt_dk;
};

#endif  // PIE_VULKAN_GDN_PARAMS_GLSL

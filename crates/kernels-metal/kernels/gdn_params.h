#ifndef PIE_METAL_GDN_PARAMS_H
#define PIE_METAL_GDN_PARAMS_H

// Shared host/shader ABI for every GDN core variant.
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

#endif

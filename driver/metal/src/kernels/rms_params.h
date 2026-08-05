#ifndef PIE_METAL_RMS_PARAMS_H
#define PIE_METAL_RMS_PARAMS_H

// Shared host/shader ABI for the RMS-family kernels.
struct RmsParams {
  float eps;
  unsigned int axis_size;
  unsigned int w_stride;
  unsigned int plus_one;
  float gain;
};

struct VNormParams {
  float eps;
  unsigned int axis_size;
};

struct GatedRmsParams {
  float eps;
  unsigned int vd;
};

#endif

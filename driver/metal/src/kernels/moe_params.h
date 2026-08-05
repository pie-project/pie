#ifndef PIE_METAL_MOE_PARAMS_H
#define PIE_METAL_MOE_PARAMS_H

// Shared host/shader ABI for generic MoE routing kernels.
struct RouterParams {
  unsigned int n_experts;
  unsigned int experts_per_token;
};

struct ExpertCombineParams {
  unsigned int width;
  unsigned int experts_per_token;
};

struct MoeRouteParams {
  unsigned int n;
  unsigned int n_experts;
  unsigned int experts_per_token;
  unsigned int tile_rows;
  unsigned int padded;
  unsigned int width;
};

#endif

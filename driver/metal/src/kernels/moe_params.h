#ifndef PIE_METAL_MOE_PARAMS_H
#define PIE_METAL_MOE_PARAMS_H

// Shared host/shader ABI for generic MoE routing kernels.
struct RouterParams {
  unsigned int n_experts;
  unsigned int experts_per_token;
  /// 0: softmax the SELECTED logits, so the k weights sum to one. This is
  /// `norm_topk_prob: true` and what every family here shipped with.
  /// 1: softmax over ALL experts and then select, so the k weights sum to
  /// LESS than one and scale the routed FFN's contribution down with them.
  /// Zero is the old behaviour, so a site that does not set it keeps it.
  unsigned int softmax_over_all;
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

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
  /// Elements between one row of `logits` and the next, or 0 for `n_experts`.
  /// The router's output is packed `[rows, k]` either way; only its input
  /// carries the caller's layout. See `ExpertCombineParams::out_pitch`.
  unsigned int logits_pitch;
};

struct ExpertCombineParams {
  unsigned int width;
  unsigned int experts_per_token;
  /// Elements between one OUTPUT row and the next, or 0 for `width`.
  ///
  /// The mixture's output is token-major and lands in whatever layout the
  /// caller's activations are in. A batched decode's are packed; a prefill's
  /// are a uniform `scratch_widest_elems` apart, because a per-token DAG binds
  /// one offset for every value it touches. Nonzero is the second case.
  unsigned int out_pitch;
};

struct MoeRouteParams {
  unsigned int n;
  unsigned int n_experts;
  unsigned int experts_per_token;
  unsigned int tile_rows;
  unsigned int padded;
  unsigned int width;
  /// Elements between one INPUT row and the next for the gather, or 0 for
  /// `width`. The sorted stack it writes is always packed; only what it reads
  /// carries the caller's layout. See `ExpertCombineParams::out_pitch`.
  unsigned int x_pitch;
};

#endif

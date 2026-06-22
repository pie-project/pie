#pragma once

// Rotary position embedding (RoPE), NEOX-style.
//
// Unlike MLX's `fast::rope` (which takes a single contiguous start offset),
// the driver applies RoPE against an explicit per-token `positions` tensor,
// because the batched/paged forward flattens many requests with arbitrary
// positions into one call. The angle for token t, frequency pair i is
// `positions[t] * theta^(-2i/head_dim)`.

#include <optional>

#include "ops/tensor.hpp"

namespace pie_metal_driver::ops {

// Optional YaRN / long-context scaling parameters. Defaults reproduce plain
// RoPE (no scaling).
struct RopeParams {
    float theta = 10000.0f;       // rope base
    float scaling_factor = 1.0f;  // linear position scaling (1 = none)
    // YaRN NTK-by-parts (optional). When `yarn` is true the per-frequency
    // interpolation ramp is applied between low/high correction dims.
    bool  yarn = false;
    float yarn_orig_ctx = 0.0f;
    float yarn_beta_fast = 32.0f;
    float yarn_beta_slow = 1.0f;
    float yarn_mscale = 1.0f;
};

// Apply NEOX RoPE to the first `rope_dims` channels of each head.
//   x:         [head_dim, n_heads, n_tokens]  (feature-major)
//   positions: [n_tokens] int32 absolute positions
//   rope_dims: number of channels rotated (== head_dim for full RoPE,
//              < head_dim for partial-rotary archs).
// Returns a tensor with the same shape as `x`.
Tensor rope(const Tensor& x, const Tensor& positions, int rope_dims,
            const RopeParams& params = {});

}  // namespace pie_metal_driver::ops

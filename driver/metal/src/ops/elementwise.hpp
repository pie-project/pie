#pragma once

// Elementwise helpers. These are thin passthroughs to MLX so model graphs
// can stay inside the `ops::` namespace rather than reaching into
// `mlx::core` directly (keeps a single seam if we ever need to intercept).

#include "ops/tensor.hpp"

namespace pie_metal_driver::ops {

Tensor add(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);

// Scale by a host scalar.
Tensor scale(const Tensor& x, float s);

Tensor tanh(const Tensor& x);

// Fused residual: x + residual (alias for add, named for readability at the
// model-graph call sites).
Tensor residual_add(const Tensor& x, const Tensor& residual);

// Logit softcap: cap * tanh(x / cap). No-op when cap <= 0.
Tensor softcap(const Tensor& x, float cap);

}  // namespace pie_metal_driver::ops

#pragma once

// Normalization ops. Thin wrappers over MLX's fused `fast::rms_norm` /
// `fast::layer_norm`, plus the Gemma `(1 + weight)` variant.

#include "ops/tensor.hpp"

namespace pie_metal_driver::ops {

// RMSNorm over the feature axis (axis 0 of a feature-major `[features, n]`
// tensor — see gemm.hpp layout note). `weight` is the per-feature gain.
//
// `plus_one` selects the Gemma convention where the effective gain is
// `(1 + weight)` (the loader stores the raw weight; Gemma folds the +1 at
// runtime). When false, `weight` is used as-is (Llama/Qwen).
Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps,
                bool plus_one = false);

// Weightless RMSNorm (Gemma's value/query pre-norm with unit gain).
Tensor rms_norm(const Tensor& x, float eps);

// LayerNorm with optional weight/bias.
Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias,
                  float eps);

}  // namespace pie_metal_driver::ops

#pragma once

// Linear algebra ops for the Metal (MLX) driver.
//
// Layout convention (MLX-idiomatic, token-major): activations are stored
// **token-major** row-major, i.e. `x` has shape `[n_tokens, in]` with the
// contraction (feature) dim LAST. This matches MLX's fused ops
// (`fast::rms_norm`/`rope`/`sdpa` all act on the last axis) so no transposes
// are needed around them. Weights are stored `[out, in]` (HF row-major), so a
// linear projection is `x @ w.T -> [n_tokens, out]`.

#include "ops/tensor.hpp"

namespace pie_metal_driver::ops {

// y = x @ w.T, with x:[n, in], w:[out, in] -> y:[n, out]. No bias.
// (Call signature mirrors charlie's `linear(w, x)` — weight first.) Both
// operands keep their input dtype; MLX accumulates in fp32 for bf16/fp16.
Tensor linear(const Tensor& w, const Tensor& x);

// Plain matmul passthrough (a @ b) for callers that already have the
// operands in the orientation they want.
Tensor matmul(const Tensor& a, const Tensor& b);

// In-place-style bias add for projections that carry a bias (e.g. Qwen2
// QKV bias). `x`:[out, n], `bias`:[out] broadcast over the token axis.
Tensor add_bias(const Tensor& x, const Tensor& bias);

// Quantized linear: dequant-on-the-fly matmul. `w` is the packed quantized
// weight, `scales`/`biases` the per-group affine-quant metadata produced by
// the loader (delta). `group_size` and `bits` match MLX's affine quant
// scheme. Computes the same `w_deq @ x` as `linear`.
Tensor quantized_linear(const Tensor& w,
                        const Tensor& scales,
                        const Tensor& biases,
                        const Tensor& x,
                        int group_size = 64,
                        int bits = 4);

}  // namespace pie_metal_driver::ops

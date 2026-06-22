#pragma once

// Linear algebra ops for the Metal (MLX) driver.
//
// Layout convention (agreed with charlie's model graphs): activations are
// stored **feature-major**, i.e. `x` has shape `[in, n_tokens]` (mirrors the
// ggml/portable convention where the contraction dim is axis 0). Weights are
// stored `[out, in]` (HF row-major). So a linear projection is a plain
// `matmul(w, x) -> [out, n_tokens]` with no transpose.

#include "ops/tensor.hpp"

namespace pie_metal_driver::ops {

// y = w @ x, with w:[out, in], x:[in, n] -> y:[out, n]. No bias.
// Both operands keep their input dtype; MLX accumulates in fp32 for
// bf16/fp16 inputs.
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

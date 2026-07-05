#pragma once

// Activation functions and gated-MLP helpers.

#include "ops/tensor.hpp"

namespace pie_metal_driver::ops {

// SiLU / swish: x * sigmoid(x).
Tensor silu(const Tensor& x);

// GELU. `tanh_approx` selects the tanh approximation (used by Gemma);
// otherwise the exact erf form.
Tensor gelu(const Tensor& x, bool tanh_approx = false);

// SwiGLU gated MLP activation: silu(gate) * up. `gate` and `up` share shape.
Tensor swiglu(const Tensor& gate, const Tensor& up);

// GeGLU gated MLP activation: gelu(gate) * up.
Tensor geglu(const Tensor& gate, const Tensor& up, bool tanh_approx = false);

}  // namespace pie_metal_driver::ops

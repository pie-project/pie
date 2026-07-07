#include "ops/activation.hpp"

#include <cmath>

#include <mlx/mlx.h>

namespace pie_metal_driver::ops {

namespace mx = mlx::core;

Tensor silu(const Tensor& x) {
    // x * sigmoid(x)
    return mx::multiply(x, mx::sigmoid(x));
}

Tensor gelu(const Tensor& x, bool tanh_approx) {
    if (tanh_approx) {
        // 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
        const float k = 0.7978845608028654f;  // sqrt(2/pi)
        Tensor x3 = mx::multiply(mx::multiply(x, x), x);
        Tensor inner = mx::add(x, x3 * 0.044715f) * k;
        return mx::multiply(x * 0.5f, mx::tanh(inner) + 1.0f);
    }
    // exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    const float inv_sqrt2 = 0.7071067811865476f;
    return mx::multiply(x * 0.5f, mx::erf(x * inv_sqrt2) + 1.0f);
}

Tensor swiglu(const Tensor& gate, const Tensor& up) {
    return mx::multiply(silu(gate), up);
}

Tensor geglu(const Tensor& gate, const Tensor& up, bool tanh_approx) {
    return mx::multiply(gelu(gate, tanh_approx), up);
}

}  // namespace pie_metal_driver::ops

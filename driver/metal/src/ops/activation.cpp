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
        Tensor inner = mx::multiply(mx::add(x, mx::multiply(0.044715f, x3)), k);
        return mx::multiply(mx::multiply(0.5f, x), mx::add(1.0f, mx::tanh(inner)));
    }
    // exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    const float inv_sqrt2 = 0.7071067811865476f;
    return mx::multiply(mx::multiply(0.5f, x),
                        mx::add(1.0f, mx::erf(mx::multiply(x, inv_sqrt2))));
}

Tensor swiglu(const Tensor& gate, const Tensor& up) {
    return mx::multiply(silu(gate), up);
}

Tensor geglu(const Tensor& gate, const Tensor& up, bool tanh_approx) {
    return mx::multiply(gelu(gate, tanh_approx), up);
}

}  // namespace pie_metal_driver::ops

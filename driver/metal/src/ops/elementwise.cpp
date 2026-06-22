#include "ops/elementwise.hpp"

#include <mlx/mlx.h>

namespace pie_metal_driver::ops {

namespace mx = mlx::core;

Tensor add(const Tensor& a, const Tensor& b) { return mx::add(a, b); }
Tensor mul(const Tensor& a, const Tensor& b) { return mx::multiply(a, b); }
Tensor sub(const Tensor& a, const Tensor& b) { return mx::subtract(a, b); }

Tensor scale(const Tensor& x, float s) { return mx::multiply(x, s); }

Tensor tanh(const Tensor& x) { return mx::tanh(x); }

Tensor residual_add(const Tensor& x, const Tensor& residual) {
    return mx::add(x, residual);
}

Tensor softcap(const Tensor& x, float cap) {
    if (cap <= 0.0f) return x;
    // cap * tanh(x / cap)
    return mx::multiply(mx::tanh(mx::multiply(x, 1.0f / cap)), cap);
}

}  // namespace pie_metal_driver::ops

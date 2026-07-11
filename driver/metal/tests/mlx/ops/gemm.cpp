#include "ops/gemm.hpp"

#include <mlx/mlx.h>

namespace pie_metal_driver::ops {

namespace mx = mlx::core;

Tensor linear(const Tensor& w, const Tensor& x) {
    // x:[n, in] @ w.T (w:[out, in]) -> [n, out].
    return mx::matmul(x, mx::transpose(w));
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    return mx::matmul(a, b);
}

Tensor add_bias(const Tensor& x, const Tensor& bias) {
    // x:[n, out], bias:[out] broadcast over the token axis.
    return mx::add(x, bias);
}

Tensor quantized_linear(const Tensor& w,
                        const Tensor& scales,
                        const Tensor& biases,
                        const Tensor& x,
                        int group_size,
                        int bits) {
    // MLX quantized_matmul computes x @ w.T when transpose=true, with w the
    // affine-quantized weight [out, in/packed] and scales/biases the per-group
    // metadata. Matches the `linear` orientation.
    return mx::quantized_matmul(x, w, scales, biases,
                                /*transpose=*/true, group_size, bits);
}

}  // namespace pie_metal_driver::ops

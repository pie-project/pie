#include "ops/norm.hpp"

#include <optional>

#include <mlx/mlx.h>

namespace pie_metal_driver::ops {

namespace mx = mlx::core;

Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps,
                bool plus_one) {
    // MLX fast::rms_norm normalizes over the last axis and multiplies by
    // `weight`. For Gemma's `(1 + weight)` convention we fold the +1 here
    // (the loader stores the raw weight). When plus_one is false the weight
    // is used verbatim (Llama/Qwen).
    if (plus_one) {
        Tensor w1 = mx::add(weight, mx::array(1.0f));
        return mx::fast::rms_norm(x, std::optional<Tensor>(w1), eps);
    }
    return mx::fast::rms_norm(x, std::optional<Tensor>(weight), eps);
}

Tensor rms_norm(const Tensor& x, float eps) {
    // Weightless RMSNorm (unit gain).
    return mx::fast::rms_norm(x, std::nullopt, eps);
}

Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias,
                  float eps) {
    return mx::fast::layer_norm(x, std::optional<Tensor>(weight),
                                std::optional<Tensor>(bias), eps);
}

}  // namespace pie_metal_driver::ops

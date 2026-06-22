#include "ops/rope.hpp"

#include <cmath>
#include <vector>

#include <mlx/mlx.h>

namespace pie_metal_driver::ops {

namespace mx = mlx::core;

// NEOX-style RoPE applied against an explicit per-token `positions` tensor.
//
// MLX's `fast::rope` only accepts a single contiguous start offset, which
// doesn't fit the flattened multi-request batch (each token has its own
// absolute position). So we build cos/sin from `positions` and apply the
// rotate-half rotation with a handful of MLX ops. The fused path can be
// swapped in later for the pure-decode single-offset case as an optimization.
Tensor rope(const Tensor& x, const Tensor& positions, int rope_dims,
            const RopeParams& params) {
    const int ax = static_cast<int>(x.ndim()) - 1;  // head_dim axis
    const int head_dim = x.shape(ax);
    const int half = rope_dims / 2;

    // inv_freq[i] = theta^(-2i/rope_dims), i in [0, half).
    Tensor idx = mx::arange(0, half, mx::float32);
    Tensor inv_freq = mx::exp(
        idx * (-2.0f * std::log(params.theta) / static_cast<float>(rope_dims)));

    // positions (scaled) outer inv_freq -> angles [n_tokens, half].
    Tensor pos = mx::astype(positions, mx::float32);
    if (params.scaling_factor != 1.0f) {
        pos = pos * (1.0f / params.scaling_factor);
    }
    Tensor angles = mx::outer(pos, inv_freq);  // [n, half]

    // Broadcast over the heads axis: [n, 1, half].
    Tensor angles_b = mx::expand_dims(angles, 1);
    Tensor cos = mx::cos(angles_b);
    Tensor sin = mx::sin(angles_b);
    if (params.yarn && params.yarn_mscale != 1.0f) {
        cos = cos * params.yarn_mscale;
        sin = sin * params.yarn_mscale;
    }

    // Separate the rotated channels from any pass-through tail.
    Tensor x_rot = x;
    std::optional<Tensor> x_pass;
    bool has_pass = rope_dims < head_dim;
    if (has_pass) {
        auto parts = mx::split(x, mx::Shape{rope_dims}, ax);
        x_rot = parts[0];
        x_pass = parts[1];
    }

    // NEOX rotate-half: first/second halves of the rotated channels.
    auto halves = mx::split(x_rot, 2, ax);
    Tensor x1 = halves[0];
    Tensor x2 = halves[1];

    Tensor out1 = mx::subtract(mx::multiply(x1, cos), mx::multiply(x2, sin));
    Tensor out2 = mx::add(mx::multiply(x1, sin), mx::multiply(x2, cos));
    Tensor rotated = mx::concatenate({out1, out2}, ax);

    if (has_pass) {
        return mx::concatenate({rotated, *x_pass}, ax);
    }
    return rotated;
}

}  // namespace pie_metal_driver::ops

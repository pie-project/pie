#include "ops/rope.hpp"

#include <cmath>
#include <vector>

#include <mlx/mlx.h>

namespace pie_metal_driver::ops {

namespace mx = mlx::core;

// NEOX-style RoPE applied against an explicit per-token `positions` tensor.
//
// Composed reference path: build cos/sin from `positions` and apply the
// rotate-half rotation with a handful of MLX ops. Retained for the YaRN case
// (the fused kernel doesn't express the per-frequency interpolation ramp).
namespace {
Tensor rope_composed(const Tensor& x, const Tensor& positions, int rope_dims,
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
}  // namespace

// Apply NEOX RoPE to the first `rope_dims` channels of each head.
//
// Fast path (non-YaRN): MLX's `fast::rope` has an overload taking a per-element
// `offset` array. The forward flattens many requests with arbitrary positions
// into `[n_tokens, n_heads, head_dim]`; inserting a unit sequence axis
// (`[n_tokens, n_heads, 1, head_dim]`) and passing `offset = positions` makes
// each token rotate at its own absolute position in a single fused kernel —
// valid for both decode and prefill, with `dims` handling partial-rotary. This
// replaces ~15 primitive ops (and a per-call freq-table recompute) per Q/K with
// one kernel, the dominant batch=1 decode overhead. Verified equivalent to the
// composed path to float32 epsilon. The YaRN case keeps the composed path
// (the fused kernel doesn't express the per-frequency interpolation ramp).
Tensor rope(const Tensor& x, const Tensor& positions, int rope_dims,
            const RopeParams& params) {
    if (params.yarn) {
        return rope_composed(x, positions, rope_dims, params);
    }

    // Insert a unit sequence axis before head_dim: [..., head_dim] ->
    // [..., 1, head_dim], so each token is its own length-1 sequence rotated
    // at offset = its absolute position.
    const int ax = static_cast<int>(x.ndim()) - 1;
    mx::Shape with_seq = x.shape();
    with_seq.insert(with_seq.begin() + ax, 1);
    Tensor xs = mx::reshape(x, with_seq);

    const Tensor offset = mx::astype(positions, mx::int32);
    const float scale = (params.scaling_factor != 0.0f)
        ? 1.0f / params.scaling_factor
        : 1.0f;

    Tensor y = mx::fast::rope(xs, rope_dims, /*traditional=*/false,
                              std::optional<float>(params.theta), scale, offset);

    return mx::reshape(y, x.shape());
}

}  // namespace pie_metal_driver::ops
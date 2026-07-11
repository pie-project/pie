// Raw-Metal port of NEOX RoPE for Phase-0 decode (single token, in-place).
//
// Source semantics: pie_metal_driver::ops::rope (driver/metal/src/ops/rope.cpp)
//   non-YaRN fast path == MLX fast::rope(traditional=false, dims=rope_dims),
//   which dispatches mlx .../kernels/rope.metal `rope` (rope_impl, NEOX).
// Port notes (qwen3.6: head_dim=256, rope_dims=64 partial, theta=1e7):
//   * Single decode token (seq len 1) -> pos.y(seq)=0, one position per token.
//   * NEOX rotate-half over the first rope_dims channels: pair (i, i+half),
//     half=rope_dims/2. Channels [rope_dims, head_dim) are pass-through (untouched).
//   * position IS the per-token IO scalar -> read from a *buffer* (offset[0]) per
//     decode_abi I1, never setBytes; CB stays byte-identical.
//   * IN-PLACE: each thread owns a disjoint (i, i+half) pair, so reading then
//     writing the same buffer is hazard-free. Matches the qk-norm->rope fold
//     (rope runs on the resident q/k projection slot, no separate out copy).
//   * inv_freq[i] = exp2(-(i/half) * base), base = log2(theta); matches MLX.
//   * scale = 1/scaling_factor (1.0 for qwen3.6 default).
// Launch: dispatchThreads grid=(half, n_head, 1), tg=(half,1,1). bfloat native.

#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void rope_neox_decode(
    device T* x                       [[buffer(0)]],  // in-place [n_head, head_dim]
    const device int* position        [[buffer(1)]],  // IO scalar (I1): position[0]
    const constant float& scale       [[buffer(2)]],
    const constant float& base        [[buffer(3)]],  // log2(theta)
    const constant int& head_dim      [[buffer(4)]],
    uint2 pos  [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  const int i = int(pos.x);            // freq index, 0..half-1
  const int h = int(pos.y);            // head index
  const int half_rd = int(grid.x);        // rope_dims / 2

  float d = static_cast<float>(i) / static_cast<float>(half_rd);
  float inv_freq = exp2(-d * base);
  float L = scale * static_cast<float>(position[0]);
  float theta = L * inv_freq;
  float costheta = fast::cos(theta);
  float sintheta = fast::sin(theta);

  const int i1 = h * head_dim + i;
  const int i2 = i1 + half_rd;
  float x1 = static_cast<float>(x[i1]);
  float x2 = static_cast<float>(x[i2]);
  x[i1] = static_cast<T>(x1 * costheta - x2 * sintheta);
  x[i2] = static_cast<T>(x1 * sintheta + x2 * costheta);
}

#define instantiate_rope_neox(name, itype)                       \
  template [[host_name("rope_neox_decode_" #name)]]              \
  [[kernel]] void rope_neox_decode<itype>(                       \
      device itype*, const device int*, const constant float&,   \
      const constant float&, const constant int&, uint2, uint2);

instantiate_rope_neox(float32, float)
instantiate_rope_neox(float16, half)
instantiate_rope_neox(bfloat16, bfloat)

// ── M>1 batched NEOX RoPE (multi-batch lane). ────────────────────────────────
// Adds a token-row dim (pos.z = m); token m rotates by its own position[m] (per-row
// IO read — the M>1 relaxation of the sealed position[0]). x is token-major
// [N, n_head, head_dim]; per-token stride = n_head*head_dim = grid.y*head_dim.
// In-place + hazard-free (disjoint (m,h,i/i+half) pairs). Reduces to
// rope_neox_decode at N=1 (pos.z=0, position[0]). Launch grid=(half, n_head, N).
template <typename T>
[[kernel]] void rope_neox_mb(
    device T* x                       [[buffer(0)]],  // [N, n_head, head_dim]
    const device int* position        [[buffer(1)]],  // [N] per-token positions
    const constant float& scale       [[buffer(2)]],
    const constant float& base        [[buffer(3)]],
    const constant int& head_dim      [[buffer(4)]],
    uint3 pos  [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const int i = int(pos.x);            // freq index, 0..half-1
  const int h = int(pos.y);            // head index
  const int m = int(pos.z);            // token row
  const int half_rd = int(grid.x);
  const int n_head  = int(grid.y);

  float d = static_cast<float>(i) / static_cast<float>(half_rd);
  float inv_freq = exp2(-d * base);
  float L = scale * static_cast<float>(position[m]);
  float theta = L * inv_freq;
  float costheta = fast::cos(theta);
  float sintheta = fast::sin(theta);

  const int base_x = m * (n_head * head_dim);
  const int i1 = base_x + h * head_dim + i;
  const int i2 = i1 + half_rd;
  float x1 = static_cast<float>(x[i1]);
  float x2 = static_cast<float>(x[i2]);
  x[i1] = static_cast<T>(x1 * costheta - x2 * sintheta);
  x[i2] = static_cast<T>(x1 * sintheta + x2 * costheta);
}

#define instantiate_rope_neox_mb(name, itype)                    \
  template [[host_name("rope_neox_mb_" #name)]]                  \
  [[kernel]] void rope_neox_mb<itype>(                           \
      device itype*, const device int*, const constant float&,   \
      const constant float&, const constant int&, uint3, uint3);

instantiate_rope_neox_mb(float32, float)
instantiate_rope_neox_mb(float16, half)
instantiate_rope_neox_mb(bfloat16, bfloat)

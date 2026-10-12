#include <metal_stdlib>
using namespace metal;

template <typename T, bool SCALE_OUTPUT>
METAL_FUNC void rope_rotate_pair(
    device T* x, size_t i1, size_t i2, float theta, float output_scale) {
  const float costheta = fast::cos(theta);
  const float sintheta = fast::sin(theta);
  const float x1 = float(x[i1]);
  const float x2 = float(x[i2]);
  const float y1 = x1 * costheta - x2 * sintheta;
  const float y2 = x1 * sintheta + x2 * costheta;
  x[i1] = static_cast<T>(SCALE_OUTPUT ? output_scale * y1 : y1);
  x[i2] = static_cast<T>(SCALE_OUTPUT ? output_scale * y2 : y2);
}

template <typename T>
METAL_FUNC void rope_neox_freqs_body(
    device T* x, int position, float scale, const device float* inv_freq,
    int head_dim, int pair_half, float output_scale,
    int i, int head, size_t row_base) {
  const float theta = (scale * float(position)) * inv_freq[i];
  const size_t i1 = row_base + size_t(head * head_dim + i);
  rope_rotate_pair<T, true>(
      x, i1, i1 + size_t(pair_half), theta, output_scale);
}

// The first grid.x pairs of each head turn, pair i being (i, i + span) at
// theta^(-i/span): span is rotary/2 for a plain rotary, head_dim/2 for a
// proportional one.
template <typename T>
[[kernel]] void rope_neox_mb(
    device T* x                       [[buffer(0)]],
    const device int* position        [[buffer(1)]],
    const constant float& base        [[buffer(2)]],
    const constant int& head_dim      [[buffer(3)]],
    const constant int& span          [[buffer(4)]],
    uint3 pos  [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const int i = int(pos.x);
  const int h = int(pos.y);
  const int m = int(pos.z);
  const float d = float(i) / float(span);
  const float theta = float(position[m]) * exp2(-d * base);
  const size_t i1 = (size_t(m) * grid.y + h) * size_t(head_dim) + i;
  rope_rotate_pair<T, false>(x, i1, i1 + size_t(span), theta, 1.0f);
}

#define instantiate_rope_neox_mb(name, itype)                    \
  template [[host_name("neox_mb_" #name)]]                  \
  [[kernel]] void rope_neox_mb<itype>(                           \
      device itype*, const device int*, const constant float&,   \
      const constant int&, const constant int&, uint3, uint3);

instantiate_rope_neox_mb(bfloat16, bfloat)

template <typename T>
[[kernel]] void rope_neox_freqs_decode(
    device T* x                       [[buffer(0)]],
    const device int* position        [[buffer(1)]],
    const constant float& scale       [[buffer(2)]],
    const device float* inv_freq      [[buffer(3)]],
    const constant int& head_dim      [[buffer(4)]],

    const constant float& mscale      [[buffer(5)]],
    uint2 pos  [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  const int i = int(pos.x);
  const int h = int(pos.y);
  const int half_rd = int(grid.x);
  rope_neox_freqs_body(
      x, position[0], scale, inv_freq, head_dim, half_rd,
      mscale, i, h, 0);
}

template <typename T>
[[kernel]] void rope_neox_freqs_mb(
    device T* x                       [[buffer(0)]],
    const device int* position        [[buffer(1)]],
    const constant float& scale       [[buffer(2)]],
    const device float* inv_freq      [[buffer(3)]],
    const constant int& head_dim      [[buffer(4)]],
    const constant float& mscale      [[buffer(5)]],
    uint3 pos  [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const int i = int(pos.x);
  const int h = int(pos.y);
  const int row = int(pos.z);
  const int half_rd = int(grid.x);
  const int n_head = int(grid.y);
  rope_neox_freqs_body(
      x, position[row], scale, inv_freq, head_dim, half_rd,
      mscale, i, h, size_t(row) * size_t(n_head) * size_t(head_dim));
}

#define instantiate_rope_freqs(name, itype)                        \
  template [[host_name("neox_freqs_decode_" #name)]]          \
  [[kernel]] void rope_neox_freqs_decode<itype>(                   \
      device itype*, const device int*, const constant float&,     \
      const device float*, const constant int&, const constant float&, uint2, uint2); \
  template [[host_name("neox_freqs_mb_" #name)]]              \
  [[kernel]] void rope_neox_freqs_mb<itype>(                       \
      device itype*, const device int*, const constant float&,     \
      const device float*, const constant int&, const constant float&, \
      uint3, uint3);

instantiate_rope_freqs(bfloat16, bfloat)

template <typename T>
[[kernel]] void rope_neox_last_mb(
    device T* x                       [[buffer(0)]],
    const device int* position        [[buffer(1)]],
    const constant float& base        [[buffer(2)]],
    const constant int& head_dim      [[buffer(3)]],
    const constant int& interleaved   [[buffer(4)]],
    const constant float& sign        [[buffer(5)]],
    const constant float& yarn_factor [[buffer(6)]],
    const constant float& yarn_low    [[buffer(7)]],
    const constant float& yarn_high   [[buffer(8)]],
    uint3 pos  [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const int i = int(pos.x);
  const int h = int(pos.y);
  const int m = int(pos.z);
  const int n_head = int(grid.y);
  const int rope_half = int(grid.x);
  const int rotary = 2 * rope_half;
  const int offset = head_dim - rotary;

  const float d = 2.0f * static_cast<float>(i) / static_cast<float>(rotary);
  float inv_freq = exp2(-d * base);
  if (yarn_factor > 1.0f) {
    const float span = max(yarn_high - yarn_low, 0.001f);
    const float ramp = clamp((static_cast<float>(i) - yarn_low) / span, 0.0f, 1.0f);
    const float smooth = 1.0f - ramp;
    inv_freq = inv_freq / yarn_factor * (1.0f - smooth) + inv_freq * smooth;
  }
  const float theta = sign * static_cast<float>(position[m]) * inv_freq;
  const float costheta = fast::cos(theta);
  const float sintheta = fast::sin(theta);

  const int row_base = (m * n_head + h) * head_dim + offset;
  const int i1 = interleaved != 0 ? row_base + 2 * i : row_base + i;
  const int i2 = interleaved != 0 ? i1 + 1 : i1 + rope_half;
  const float x1 = static_cast<float>(x[i1]);
  const float x2 = static_cast<float>(x[i2]);
  x[i1] = static_cast<T>(x1 * costheta - x2 * sintheta);
  x[i2] = static_cast<T>(x1 * sintheta + x2 * costheta);
}

#define instantiate_rope_neox_last_mb(name, itype)               \
  template [[host_name("neox_last_mb_" #name)]]                  \
  [[kernel]] void rope_neox_last_mb<itype>(                      \
      device itype*, const device int*, const constant float&,   \
      const constant int&, const constant int&,                  \
      const constant float&, const constant float&,              \
      const constant float&, const constant float&, uint3, uint3);

instantiate_rope_neox_last_mb(bfloat16, bfloat)

template <typename T>
[[kernel]] void rope_neox_yarn_mb(
    device T* x                       [[buffer(0)]],
    const device int* position        [[buffer(1)]],
    const constant float& base        [[buffer(2)]],
    const constant int& head_dim      [[buffer(3)]],
    const constant float& factor      [[buffer(4)]],
    const constant float& low_dim     [[buffer(5)]],
    const constant float& high_dim    [[buffer(6)]],
    const constant float& mscale      [[buffer(7)]],
    const constant int& interleaved   [[buffer(8)]],
    uint3 pos  [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const int i = int(pos.x);
  const int h = int(pos.y);
  const int m = int(pos.z);
  const int n_head = int(grid.y);
  const int half_hd = int(grid.x);

  const float d = 2.0f * static_cast<float>(i) / static_cast<float>(head_dim);
  const float base_freq = exp2(-d * base);
  const float denom =
      high_dim == low_dim ? high_dim + 1e-3f - low_dim : high_dim - low_dim;
  const float ramp =
      metal::clamp((static_cast<float>(i) - low_dim) / denom, 0.0f, 1.0f);
  const float freq = base_freq * ((1.0f - ramp) + ramp / factor);
  const float theta = static_cast<float>(position[m]) * freq;
  const float costheta = fast::cos(theta) * mscale;
  const float sintheta = fast::sin(theta) * mscale;

  const int row_base = (m * n_head + h) * head_dim;
  const int i1 = interleaved != 0 ? row_base + 2 * i : row_base + i;
  const int i2 = interleaved != 0 ? i1 + 1 : i1 + half_hd;
  const float x1 = static_cast<float>(x[i1]);
  const float x2 = static_cast<float>(x[i2]);
  x[i1] = static_cast<T>(x1 * costheta - x2 * sintheta);
  x[i2] = static_cast<T>(x1 * sintheta + x2 * costheta);
}

#define instantiate_rope_neox_yarn_mb(name, itype)               \
  template [[host_name("neox_yarn_mb_" #name)]]                  \
  [[kernel]] void rope_neox_yarn_mb<itype>(                      \
      device itype*, const device int*, const constant float&,   \
      const constant int&, const constant float&,                \
      const constant float&, const constant float&,              \
      const constant float&, const constant int&, uint3, uint3);

instantiate_rope_neox_yarn_mb(bfloat16, bfloat)

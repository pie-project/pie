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
METAL_FUNC void rope_neox_geometric_body(
    device T* x, int position, float scale, float base, int head_dim,
    int pair_half, int i, int head, size_t row_base) {
  const float d = float(i) / float(pair_half);
  const float inv_freq = exp2(-d * base);
  const float theta = (scale * float(position)) * inv_freq;
  const size_t i1 = row_base + size_t(head * head_dim + i);
  rope_rotate_pair<T, false>(x, i1, i1 + size_t(pair_half), theta, 1.0f);
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

template <typename T>
[[kernel]] void rope_neox_decode(
    device T* x                       [[buffer(0)]],
    const device int* position        [[buffer(1)]],
    const constant float& scale       [[buffer(2)]],
    const constant float& base        [[buffer(3)]],
    const constant int& head_dim      [[buffer(4)]],
    uint2 pos  [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  const int i = int(pos.x);
  const int h = int(pos.y);
  rope_neox_geometric_body<T>(
      x, position[0], scale, base, head_dim, int(grid.x), i, h, 0);
}

#define instantiate_rope_neox(name, itype)                       \
  template [[host_name("neox_decode_" #name)]]              \
  [[kernel]] void rope_neox_decode<itype>(                       \
      device itype*, const device int*, const constant float&,   \
      const constant float&, const constant int&, uint2, uint2);

instantiate_rope_neox(bfloat16, bfloat)

template <typename T>
[[kernel]] void rope_neox_mb(
    device T* x                       [[buffer(0)]],
    const device int* position        [[buffer(1)]],
    const constant float& scale       [[buffer(2)]],
    const constant float& base        [[buffer(3)]],
    const constant int& head_dim      [[buffer(4)]],
    uint3 pos  [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const int i = int(pos.x);
  const int h = int(pos.y);
  const int m = int(pos.z);
  const int half_rd = int(grid.x);
  const int n_head  = int(grid.y);
  const size_t row_base =
      size_t(m) * size_t(n_head) * size_t(head_dim);
  rope_neox_geometric_body<T>(
      x, position[m], scale, base, head_dim, half_rd, i, h, row_base);
}

#define instantiate_rope_neox_mb(name, itype)                    \
  template [[host_name("neox_mb_" #name)]]                  \
  [[kernel]] void rope_neox_mb<itype>(                           \
      device itype*, const device int*, const constant float&,   \
      const constant float&, const constant int&, uint3, uint3);

instantiate_rope_neox_mb(bfloat16, bfloat)

template <typename T>
[[kernel]] void rope_neox_strided(
    device T* x                       [[buffer(0)]],
    const device int* position        [[buffer(1)]],
    const constant float& scale       [[buffer(2)]],
    const constant float& base        [[buffer(3)]],
    const constant int& head_dim      [[buffer(4)]],
    const constant int& row_pitch     [[buffer(5)]],
    uint3 pos  [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const int m = int(pos.z);
  rope_neox_geometric_body<T>(
      x, position[m], scale, base, head_dim, int(grid.x), int(pos.x), int(pos.y),
      size_t(m) * size_t(row_pitch));
}

#define instantiate_rope_neox_strided(name, itype)               \
  template [[host_name("neox_strided_" #name)]]             \
  [[kernel]] void rope_neox_strided<itype>(                      \
      device itype*, const device int*, const constant float&,   \
      const constant float&, const constant int&,                \
      const constant int&, uint3, uint3);

instantiate_rope_neox_strided(bfloat16, bfloat)

template <typename T>
[[kernel]] void rope_neox_prop_decode(
    device T* x                       [[buffer(0)]],
    const device int* position        [[buffer(1)]],
    const constant float& scale       [[buffer(2)]],
    const constant float& base        [[buffer(3)]],
    const constant int& head_dim      [[buffer(4)]],
    uint2 pos  [[thread_position_in_grid]]) {
  const int i = int(pos.x);
  const int h = int(pos.y);
  const int half_hd = head_dim / 2;

  float d = 2.0f * static_cast<float>(i) / static_cast<float>(head_dim);
  float inv_freq = exp2(-d * base);
  float theta = scale * static_cast<float>(position[0]) * inv_freq;
  float costheta = fast::cos(theta);
  float sintheta = fast::sin(theta);

  const int i1 = h * head_dim + i;
  const int i2 = i1 + half_hd;
  float x1 = static_cast<float>(x[i1]);
  float x2 = static_cast<float>(x[i2]);
  x[i1] = static_cast<T>(x1 * costheta - x2 * sintheta);
  x[i2] = static_cast<T>(x1 * sintheta + x2 * costheta);
}

#define instantiate_rope_prop(name, itype)                       \
  template [[host_name("neox_prop_decode_" #name)]]         \
  [[kernel]] void rope_neox_prop_decode<itype>(                  \
      device itype*, const device int*, const constant float&,   \
      const constant float&, const constant int&, uint2);

instantiate_rope_prop(bfloat16, bfloat)

template <typename T>
[[kernel]] void rope_neox_prop_mb(
    device T* x                       [[buffer(0)]],
    const device int* position        [[buffer(1)]],
    const constant float& scale       [[buffer(2)]],
    const constant float& base        [[buffer(3)]],
    const constant int& head_dim      [[buffer(4)]],
    uint3 pos  [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const int i = int(pos.x);
  const int h = int(pos.y);
  const int m = int(pos.z);
  const int n_head = int(grid.y);
  const int half_hd = head_dim / 2;

  float d = 2.0f * static_cast<float>(i) / static_cast<float>(head_dim);
  float inv_freq = exp2(-d * base);
  float theta = scale * static_cast<float>(position[m]) * inv_freq;
  float costheta = fast::cos(theta);
  float sintheta = fast::sin(theta);

  const int i1 = (m * n_head + h) * head_dim + i;
  const int i2 = i1 + half_hd;
  float x1 = static_cast<float>(x[i1]);
  float x2 = static_cast<float>(x[i2]);
  x[i1] = static_cast<T>(x1 * costheta - x2 * sintheta);
  x[i2] = static_cast<T>(x1 * sintheta + x2 * costheta);
}

#define instantiate_rope_prop_mb(name, itype)                    \
  template [[host_name("neox_prop_mb_" #name)]]             \
  [[kernel]] void rope_neox_prop_mb<itype>(                      \
      device itype*, const device int*, const constant float&,   \
      const constant float&, const constant int&, uint3, uint3);

instantiate_rope_prop_mb(bfloat16, bfloat)

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
  const float inv_freq = exp2(-d * base);
  const float theta = static_cast<float>(position[m]) * inv_freq;
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
      const constant int&, const constant int&, uint3, uint3);

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

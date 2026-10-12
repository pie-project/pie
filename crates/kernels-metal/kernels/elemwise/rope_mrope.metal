#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void rope_mrope_interleaved(
    device T* x                       [[buffer(0)]],
    const device int* positions       [[buffer(1)]],
    const constant float& base        [[buffer(2)]],
    const constant int& head_dim      [[buffer(3)]],
    const constant int& s0            [[buffer(4)]],
    const constant int& s1            [[buffer(5)]],
    const constant int& s2            [[buffer(6)]],
    uint3 pos  [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const int i = int(pos.x);
  const int h = int(pos.y);
  const int m = int(pos.z);
  const int n_head = int(grid.y);
  // One lane per pair of the rotated prefix, which turns as its own neox
  // head: pairs (i, i + rotary/2) at theta^(-2i/rotary).
  const int half_rot = int(grid.x);

  const int pos_t = positions[3 * m + 0];
  const int pos_h = positions[3 * m + 1];
  const int pos_w = positions[3 * m + 2];
  (void)s0;

  int axis_pos;
  const int r = i % 3;
  if (r == 1 && i < 3 * s1) {
    axis_pos = pos_h;
  } else if (r == 2 && i < 3 * s2) {
    axis_pos = pos_w;
  } else {
    axis_pos = pos_t;
  }

  const float d = 2.0f * static_cast<float>(i) / static_cast<float>(2 * half_rot);
  const float inv_freq = exp2(-d * base);
  const float theta = static_cast<float>(axis_pos) * inv_freq;
  const float costheta = fast::cos(theta);
  const float sintheta = fast::sin(theta);

  const size_t i1 =
      (size_t(m) * size_t(n_head) + size_t(h)) * size_t(head_dim) + size_t(i);
  const size_t i2 = i1 + size_t(half_rot);
  const float x1 = static_cast<float>(x[i1]);
  const float x2 = static_cast<float>(x[i2]);
  x[i1] = static_cast<T>(x1 * costheta - x2 * sintheta);
  x[i2] = static_cast<T>(x1 * sintheta + x2 * costheta);
}

#define instantiate_rope_mrope(name, itype)                                \
  template [[host_name("rope_mrope_interleaved_" #name)]]                  \
  [[kernel]] void rope_mrope_interleaved<itype>(                           \
      device itype*, const device int*, const constant float&,             \
      const constant int&, const constant int&, const constant int&,       \
      const constant int&, uint3, uint3);

instantiate_rope_mrope(bfloat16, bfloat)

template <typename T>
[[kernel]] void rope_mrope_blocked(
    device T* x                       [[buffer(0)]],
    const device int* positions       [[buffer(1)]],
    const constant float& base        [[buffer(2)]],
    const constant int& head_dim      [[buffer(3)]],
    const constant int& s0            [[buffer(4)]],
    const constant int& s1            [[buffer(5)]],
    const constant int& s2            [[buffer(6)]],
    uint3 pos  [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const int i = int(pos.x);
  const int h = int(pos.y);
  const int m = int(pos.z);
  const int n_head = int(grid.y);
  const int half_hd = head_dim / 2;
  const int total = s0 + s1 + s2;

  const int axis_of[3] = {positions[3 * m + 0],
                          positions[3 * m + 1],
                          positions[3 * m + 2]};

  int axis;
  int within;
  if (i < s0) {
    axis = 0;
    within = i;
  } else if (i < s0 + s1) {
    axis = 1;
    within = i - s0;
  } else {
    axis = 2;
    within = i - s0 - s1;
  }

  const float d = 2.0f * static_cast<float>(within) / static_cast<float>(total);
  const float inv_freq = exp2(-d * base);
  const float theta = static_cast<float>(axis_of[axis]) * inv_freq;
  const float costheta = fast::cos(theta);
  const float sintheta = fast::sin(theta);

  const size_t i1 =
      (size_t(m) * size_t(n_head) + size_t(h)) * size_t(head_dim) + size_t(i);
  const size_t i2 = i1 + size_t(half_hd);
  const float x1 = static_cast<float>(x[i1]);
  const float x2 = static_cast<float>(x[i2]);
  x[i1] = static_cast<T>(x1 * costheta - x2 * sintheta);
  x[i2] = static_cast<T>(x1 * sintheta + x2 * costheta);
}

#define instantiate_rope_mrope_blocked(name, itype)                        \
  template [[host_name("rope_mrope_blocked_" #name)]]                      \
  [[kernel]] void rope_mrope_blocked<itype>(                               \
      device itype*, const device int*, const constant float&,             \
      const constant int&, const constant int&, const constant int&,       \
      const constant int&, uint3, uint3);

instantiate_rope_mrope_blocked(bfloat16, bfloat)

template <typename T>
[[kernel]] void rope_mrope_split(
    device T* x                       [[buffer(0)]],
    const device int* positions       [[buffer(1)]],
    const constant float& base        [[buffer(2)]],
    const constant int& head_dim      [[buffer(3)]],
    const constant int& s0            [[buffer(4)]],
    const constant int& s1            [[buffer(5)]],
    const constant int& s2            [[buffer(6)]],
    uint3 pos  [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const int i = int(pos.x);
  const int h = int(pos.y);
  const int m = int(pos.z);
  const int n_head = int(grid.y);

  const int axis_of[3] = {positions[3 * m + 0],
                          positions[3 * m + 1],
                          positions[3 * m + 2]};

  int axis;
  int within;
  int before;
  int width;
  if (i < s0) {
    axis = 0; within = i; before = 0; width = s0;
  } else if (i < s0 + s1) {
    axis = 1; within = i - s0; before = s0; width = s1;
  } else {
    axis = 2; within = i - s0 - s1; before = s0 + s1; width = s2;
  }
  if (width <= 0) {
    return;
  }

  const float d = static_cast<float>(within) / static_cast<float>(width);
  const float inv_freq = exp2(-d * base);
  const float theta = static_cast<float>(axis_of[axis]) * inv_freq;
  const float costheta = fast::cos(theta);
  const float sintheta = fast::sin(theta);

  const size_t row = (size_t(m) * size_t(n_head) + size_t(h)) * size_t(head_dim);
  const size_t i1 = row + size_t(2 * before + within);
  const size_t i2 = i1 + size_t(width);
  const float x1 = static_cast<float>(x[i1]);
  const float x2 = static_cast<float>(x[i2]);
  x[i1] = static_cast<T>(x1 * costheta - x2 * sintheta);
  x[i2] = static_cast<T>(x1 * sintheta + x2 * costheta);
}

#define instantiate_rope_mrope_split(name, itype)                          \
  template [[host_name("rope_mrope_split_" #name)]]                        \
  [[kernel]] void rope_mrope_split<itype>(                                 \
      device itype*, const device int*, const constant float&,             \
      const constant int&, const constant int&, const constant int&,       \
      const constant int&, uint3, uint3);

instantiate_rope_mrope_split(bfloat16, bfloat)

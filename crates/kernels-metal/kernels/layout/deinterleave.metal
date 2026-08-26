#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void split_rows(
    const device T* src           [[buffer(0)]],
    device T* left                [[buffer(1)]],
    device T* right               [[buffer(2)]],
    const constant int& left_dim  [[buffer(3)]],
    const constant int& right_dim [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]) {
  const int total = left_dim + right_dim;
  const int c = int(tid.x);

  if (c >= total) {
    return;
  }
  const size_t row = size_t(tid.y);
  const T value = src[row * size_t(total) + size_t(c)];
  if (c < left_dim) {
    left[row * size_t(left_dim) + size_t(c)] = value;
  } else {
    right[row * size_t(right_dim) + size_t(c - left_dim)] = value;
  }
}

#define instantiate_split_rows(name, itype)                                 \
  template [[host_name("split_rows_" #name)]]                               \
  [[kernel]] void split_rows<itype>(                                        \
      const device itype*, device itype*, device itype*,                    \
      const constant int&, const constant int&, uint2);

instantiate_split_rows(bfloat16, bfloat)

template <typename T>
[[kernel]] void select_slice(
    const device T* table      [[buffer(0)]],
    device T* y                [[buffer(1)]],
    const constant int& stride [[buffer(2)]],
    const constant int& offset [[buffer(3)]],
    const constant int& width  [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]) {
  const int c = int(tid.x);
  if (c >= width) {
    return;
  }
  const size_t row = size_t(tid.y);
  y[row * size_t(width) + size_t(c)] =
      table[row * size_t(stride) + size_t(offset) + size_t(c)];
}

#define instantiate_select_slice(name, itype)                               \
  template [[host_name("select_slice_" #name)]]                             \
  [[kernel]] void select_slice<itype>(                                      \
      const device itype*, device itype*, const constant int&,              \
      const constant int&, const constant int&, uint2);

instantiate_select_slice(bfloat16, bfloat)

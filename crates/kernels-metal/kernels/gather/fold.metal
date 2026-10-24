#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void pool_rows(
    const device T* x         [[buffer(0)]],
    device T* y               [[buffer(1)]],
    const constant int& width [[buffer(2)]],
    const constant int& block [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]) {
  const size_t c = size_t(tid.x);
  const size_t out = size_t(tid.y);
  const size_t base = out * size_t(block) * size_t(width);

  float acc = 0.0f;
  for (int r = 0; r < block; ++r) {
    acc += float(x[base + size_t(r) * size_t(width) + c]);
  }
  y[out * size_t(width) + c] = T(acc / float(block));
}

#define instantiate_pool_rows(name, itype)                            \
  template [[host_name("pool_rows_" #name)]]                          \
  [[kernel]] void pool_rows<itype>(                                   \
      const device itype*, device itype*, const constant int&,        \
      const constant int&, uint2);

instantiate_pool_rows(bfloat16, bfloat)

template <typename T>
[[kernel]] void merge_rows(
    const device T* x          [[buffer(0)]],
    device T* y                [[buffer(1)]],
    const constant int& merged [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]) {
  const size_t i = size_t(tid.y) * size_t(merged) + size_t(tid.x);
  y[i] = x[i];
}

#define instantiate_merge_rows(name, itype)                           \
  template [[host_name("merge_rows_" #name)]]                         \
  [[kernel]] void merge_rows<itype>(                                  \
      const device itype*, device itype*, const constant int&, uint2);

instantiate_merge_rows(bfloat16, bfloat)

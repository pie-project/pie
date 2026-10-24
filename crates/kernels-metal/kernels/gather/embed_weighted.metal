#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void embed_weighted(
    const device int* ids       [[buffer(0)]],
    const device float* weights [[buffer(1)]],
    const device T* table       [[buffer(2)]],
    device T* y                 [[buffer(3)]],
    const constant int& hidden  [[buffer(4)]],
    const constant int& vocab   [[buffer(5)]],
    const constant int& taps    [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]) {
  const int c = int(tid.x);
  if (c >= hidden) {
    return;
  }
  const size_t n = size_t(tid.y);
  const device int* row_ids = ids + n * size_t(taps);
  const device float* row_w = weights + n * size_t(taps);

  float acc = 0.0f;
  for (int t = 0; t < taps; ++t) {
    const int raw = row_ids[t];
    const int at = (raw >= 0 && raw < vocab) ? raw : 0;
    acc += row_w[t] * float(table[size_t(at) * size_t(hidden) + size_t(c)]);
  }
  y[n * size_t(hidden) + size_t(c)] = T(acc);
}

#define instantiate_embed_weighted(name, itype)                            \
  template [[host_name("embed_weighted_" #name)]]                          \
  [[kernel]] void embed_weighted<itype>(                                   \
      const device int*, const device float*, const device itype*,         \
      device itype*, const constant int&, const constant int&,             \
      const constant int&, uint2);

instantiate_embed_weighted(bfloat16, bfloat)

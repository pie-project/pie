#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void embed(
    const device int* ids      [[buffer(0)]],
    const device T* table      [[buffer(1)]],
    device T* y                [[buffer(2)]],
    const constant int& hidden [[buffer(3)]],
    const constant int& vocab  [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]) {
  const int c = int(tid.x);

  if (c >= hidden) {
    return;
  }
  const size_t n = size_t(tid.y);
  const int raw = ids[n];
  const int row = (raw >= 0 && raw < vocab) ? raw : 0;
  y[n * size_t(hidden) + size_t(c)] =
      table[size_t(row) * size_t(hidden) + size_t(c)];
}

#define instantiate_embed_dense(name, itype)                                \
  template [[host_name("embed_" #name)]]                                    \
  [[kernel]] void embed<itype>(                                             \
      const device int*, const device itype*, device itype*,                \
      const constant int&, const constant int&, uint2);

instantiate_embed_dense(bfloat16, bfloat)

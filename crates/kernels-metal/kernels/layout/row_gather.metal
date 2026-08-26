#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void row_gather(
    const device T* in            [[buffer(0)]],
    device T* out                 [[buffer(1)]],
    const device uint* rows       [[buffer(2)]],
    const constant uint& width    [[buffer(3)]],
    const constant uint& count    [[buffer(4)]],
    uint2 tid                     [[thread_position_in_grid]]) {
  const uint c = tid.x;
  const uint i = tid.y;
  if (c >= width || i >= count) return;
  out[size_t(i) * size_t(width) + size_t(c)] =
      in[size_t(rows[i]) * size_t(width) + size_t(c)];
}

#define instantiate_row_gather(name, itype)                \
  template [[host_name("row_gather_" #name)]]              \
  [[kernel]] void row_gather<itype>(                       \
      const device itype*, device itype*,                  \
      const device uint*, const constant uint&,            \
      const constant uint&, uint2);

instantiate_row_gather(bfloat16, bfloat)

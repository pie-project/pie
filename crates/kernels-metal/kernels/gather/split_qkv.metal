#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void split_qkv(
    const device T* packed [[buffer(0)]],
    device T* q            [[buffer(1)]],
    device T* k            [[buffer(2)]],
    device T* v            [[buffer(3)]],
    const constant uint& q_width  [[buffer(4)]],
    const constant uint& kv_width [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]) {
  const uint packed_width = q_width + 2u * kv_width;
  const uint c = tid.x;

  if (c >= packed_width) {
    return;
  }
  const size_t row = size_t(tid.y);
  const T value = packed[row * size_t(packed_width) + size_t(c)];
  if (c < q_width) {
    q[row * size_t(q_width) + size_t(c)] = value;
  } else if (c < q_width + kv_width) {
    k[row * size_t(kv_width) + size_t(c - q_width)] = value;
  } else {
    v[row * size_t(kv_width) + size_t(c - q_width - kv_width)] = value;
  }
}

#define instantiate_split_qkv(name, itype)                                  \
  template [[host_name("split_qkv_" #name)]]                                \
  [[kernel]] void split_qkv<itype>(                                         \
      const device itype*, device itype*, device itype*, device itype*,     \
      const constant uint&, const constant uint&, uint2);

instantiate_split_qkv(bf16, bfloat)

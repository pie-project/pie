#include <metal_stdlib>
using namespace metal;


template <typename T>
[[kernel]] void binary_add(
    const device T* x  [[buffer(0)]],
    const device T* y  [[buffer(1)]],
    device T* o        [[buffer(2)]],
    uint gid           [[thread_position_in_grid]]) {
  o[gid] = static_cast<T>(float(x[gid]) + float(y[gid]));
}

template <typename T>
[[kernel]] void binary_mul(
    const device T* x  [[buffer(0)]],
    const device T* y  [[buffer(1)]],
    device T* o        [[buffer(2)]],
    uint gid           [[thread_position_in_grid]]) {
  o[gid] = static_cast<T>(float(x[gid]) * float(y[gid]));
}

#define instantiate_binary(op, name, itype)                          \
  template [[host_name(#op "_" #name)]]                              \
  [[kernel]] void op<itype>(                                         \
      const device itype*, const device itype*, device itype*, uint);

instantiate_binary(binary_add, bfloat16, bfloat)
instantiate_binary(binary_add, float32, float)
instantiate_binary(binary_mul, bfloat16, bfloat)
instantiate_binary(binary_mul, float32, float)

template <typename T>
[[kernel]] void act_silu(
    const device T* x  [[buffer(0)]],
    device T* o        [[buffer(1)]],
    uint gid           [[thread_position_in_grid]]) {
  const float v = float(x[gid]);
  o[gid] = static_cast<T>(v / (1.0f + precise::exp(-v)));
}

template <typename T>
[[kernel]] void act_tanh(
    const device T* x  [[buffer(0)]],
    device T* o        [[buffer(1)]],
    uint gid           [[thread_position_in_grid]]) {
  o[gid] = static_cast<T>(precise::tanh(float(x[gid])));
}

template <typename T>
[[kernel]] void act_gelu_tanh(
    const device T* x  [[buffer(0)]],
    device T* o        [[buffer(1)]],
    uint gid           [[thread_position_in_grid]]) {
  constexpr float k = 0.7978845608028654f;
  const float v = float(x[gid]);
  const float inner = k * (v + 0.044715f * v * v * v);
  o[gid] = static_cast<T>(0.5f * v * (1.0f + precise::tanh(inner)));
}

#define instantiate_activation(op, name, itype)                      \
  template [[host_name(#op "_" #name)]]                              \
  [[kernel]] void op<itype>(                                         \
      const device itype*, device itype*, uint);

instantiate_activation(act_silu, bfloat16, bfloat)
instantiate_activation(act_silu, float32, float)
instantiate_activation(act_tanh, bfloat16, bfloat)
instantiate_activation(act_tanh, float32, float)
instantiate_activation(act_gelu_tanh, bfloat16, bfloat)
instantiate_activation(act_gelu_tanh, float32, float)

// A blockwise-128 Hadamard: the row's last dim is cut into contiguous
// 128-vectors, and each is multiplied by the normalized 128x128 Sylvester
// Hadamard matrix H, whose entries are (-1)^popcount(i & j) / sqrt(128). H is
// symmetric and orthonormal, so H . H = I and the op is its own inverse. One
// threadgroup owns one 128-block; the block is staged in threadgroup memory
// before any lane writes, so the transform is safe in place.
template <typename T>
[[kernel]] void hadamard_block_128(
    device T* x  [[buffer(0)]],
    uint gid     [[threadgroup_position_in_grid]],
    uint lid     [[thread_position_in_threadgroup]]) {
  constexpr uint BLOCK = 128u;
  constexpr float INV_SQRT = 0.08838834764831845f;  // 1 / sqrt(128)
  threadgroup float tile[BLOCK];
  const size_t base = size_t(gid) * size_t(BLOCK);
  tile[lid] = float(x[base + lid]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float acc = 0.0f;
  for (uint j = 0u; j < BLOCK; ++j) {
    const float sign = (popcount(lid & j) & 1u) ? -1.0f : 1.0f;
    acc += sign * tile[j];
  }
  x[base + lid] = static_cast<T>(acc * INV_SQRT);
}

#define instantiate_hadamard_block_128(name, itype)                  \
  template [[host_name("hadamard_block_128_" #name)]]               \
  [[kernel]] void hadamard_block_128<itype>(                        \
      device itype*, uint, uint);

instantiate_hadamard_block_128(bfloat16, bfloat)
instantiate_hadamard_block_128(float32, float)

template <typename T>
[[kernel]] void clamp_bounds(
    device T* x                [[buffer(0)]],
    const constant float& lo   [[buffer(1)]],
    const constant float& hi   [[buffer(2)]],
    uint gid                   [[thread_position_in_grid]]) {
  const float v = float(x[gid]);
  const float c = v < lo ? lo : (v > hi ? hi : v);
  x[gid] = static_cast<T>(c);
}

template <typename T>
[[kernel]] void clamp_learned(
    device T* x                [[buffer(0)]],
    const device T* lo         [[buffer(1)]],
    const device T* hi         [[buffer(2)]],
    uint gid                   [[thread_position_in_grid]]) {
  const float lo_v = float(lo[0]);
  const float hi_v = float(hi[0]);
  const float v = float(x[gid]);
  const float c = v < lo_v ? lo_v : (v > hi_v ? hi_v : v);
  x[gid] = static_cast<T>(c);
}

#define instantiate_clamp_learned(name, itype)                       \
  template [[host_name("clamp_learned_" #name)]]                     \
  [[kernel]] void clamp_learned<itype>(                              \
      device itype*, const device itype*, const device itype*, uint);

instantiate_clamp_learned(bfloat16, bfloat)
instantiate_clamp_learned(float32, float)

#define instantiate_clamp(name, itype)                               \
  template [[host_name("clamp_bounds_" #name)]]                      \
  [[kernel]] void clamp_bounds<itype>(                               \
      device itype*, const constant float&, const constant float&, uint);

instantiate_clamp(bfloat16, bfloat)
instantiate_clamp(float32, float)

[[kernel]] void cast_f32_to_bf16(
    const device float* x  [[buffer(0)]],
    device bfloat* o       [[buffer(1)]],
    uint gid               [[thread_position_in_grid]]) {
  o[gid] = static_cast<bfloat>(x[gid]);
}

inline int relative_position_bucket(int d, bool bidirectional, int num_buckets, float log_ratio) {
  int bucket = 0;
  int n;
  if (bidirectional) {
    num_buckets /= 2;
    if (d > 0) bucket += num_buckets;
    n = d < 0 ? -d : d;
  } else {
    n = d < 0 ? -d : 0;
  }
  const int max_exact = num_buckets / 2;
  if (n < max_exact) return bucket + n;
  const float x = precise::log(float(n) / float(max_exact));
  const float scaled = x / log_ratio * float(num_buckets - max_exact);
  int large = max_exact + int(scaled);
  if (large > num_buckets - 1) large = num_buckets - 1;
  return bucket + large;
}

template <typename T>
[[kernel]] void relative_bucket_bias(
    const device T* embedding          [[buffer(0)]],
    device float* y                    [[buffer(1)]],
    const constant uint& heads         [[buffer(2)]],
    const constant uint& span          [[buffer(3)]],
    const constant uint& max_len       [[buffer(4)]],
    const constant uint& stride        [[buffer(5)]],
    const constant uint& num_buckets   [[buffer(6)]],
    const constant uint& bidirectional [[buffer(7)]],
    const constant float& log_ratio    [[buffer(8)]],
    uint2 tid                          [[thread_position_in_grid]]) {
  const uint c = tid.x;
  const uint h = tid.y;
  if (c >= span || h >= heads) return;
  const int d = int(c) - (int(max_len) - 1);
  const int b = relative_position_bucket(d, bidirectional != 0, int(num_buckets), log_ratio);
  y[size_t(h) * size_t(span) + size_t(c)] =
      float(embedding[size_t(b) * size_t(stride) + size_t(h)]);
}

#define instantiate_relative_bucket_bias(name, itype)                \
  template [[host_name("relative_bucket_bias_" #name)]]              \
  [[kernel]] void relative_bucket_bias<itype>(                       \
      const device itype*, device float*, const constant uint&,      \
      const constant uint&, const constant uint&, const constant uint&, \
      const constant uint&, const constant uint&, const constant float&, \
      uint2);

instantiate_relative_bucket_bias(bfloat16, bfloat)
instantiate_relative_bucket_bias(float32, float)

#include <metal_stdlib>

using namespace metal;

constant constexpr uint ANE_UNIT = 128;
constant constexpr float ANE_PEAK = 127.0f;
constant constexpr float ANE_SCALE = 128.0f;
constant constexpr float ANE_FLOOR = 0x1p-14f;

template <uint K>
inline void ane_rotate_block(thread float (&value)[K][4], uint lane, device const float *sign) {
  for (uint k = 0; k < K; ++k) {
    thread float (&v)[4] = value[k];
    for (uint half_span = 1; half_span < 4; half_span <<= 1)
      for (uint e = 0; e < 4; ++e)
        if (!(e & half_span)) {
          const float a = v[e], b = v[e + half_span];
          v[e] = a + b;
          v[e + half_span] = a - b;
        }
    for (uint mask = 1; mask < 32; mask <<= 1)
      for (uint e = 0; e < 4; ++e) {
        const float other = simd_shuffle_xor(v[e], mask);
        v[e] = (lane & mask) ? other - v[e] : v[e] + other;
      }
  }
  for (uint half_span = 1; half_span < K; half_span <<= 1)
    for (uint k = 0; k < K; ++k)
      if (!(k & half_span))
        for (uint e = 0; e < 4; ++e) {
          const float a = value[k][e], b = value[k + half_span][e];
          value[k][e] = a + b;
          value[k + half_span][e] = a - b;
        }
  const float norm = rsqrt(float(K * ANE_UNIT));
  for (uint k = 0; k < K; ++k)
    for (uint e = 0; e < 4; ++e) value[k][e] *= sign[k * ANE_UNIT + lane * 4 + e] * norm;
}

inline void ane_q4_values(thread float (&value)[4], device const uint *codes, device const bfloat *scales,
                          device const bfloat *biases, uint width, uint row, uint input) {
  const uint word = codes[ulong(row) * (width / 8) + input / 8];
  const ulong unit = ulong(row) * (width / 64) + input / 64;
  const float scale = float(scales[unit]), bias = float(biases[unit]);
  const uint shift = (input & 7) * 4;
  for (uint e = 0; e < 4; ++e) value[e] = float((word >> (shift + 4 * e)) & 15) * scale + bias;
}

kernel void ane_rotate(device const bfloat *input [[buffer(0)]],
                       device const float *sign [[buffer(1)]],
                       device half *rotated [[buffer(2)]],
                       device half *token_scale [[buffer(3)]],
                       constant uint &hidden [[buffer(4)]],
                       uint row [[threadgroup_position_in_grid]],
                       uint simd_group [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint simd_groups [[simdgroups_per_threadgroup]]) {
  threadgroup float peaks[32];
  const uint total = hidden / ANE_UNIT, blocks = (total - simd_group + simd_groups - 1) / simd_groups;
  float value[8][1][4];
  float peak = 0.0f;
  for (uint block = 0; block < blocks; ++block) {
    const ulong origin = ulong(row) * hidden + (block * simd_groups + simd_group) * ANE_UNIT + lane * 4;
    for (uint e = 0; e < 4; ++e) value[block][0][e] = float(input[origin + e]);
    ane_rotate_block<1>(value[block], lane, sign);
    for (uint e = 0; e < 4; ++e) peak = max(peak, fabs(value[block][0][e]));
  }
  peak = simd_max(peak);
  if (lane == 0) peaks[simd_group] = peak;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  peak = 0.0f;
  for (uint group = 0; group < simd_groups; ++group) peak = max(peak, peaks[group]);
  const float scale = max(peak, ANE_FLOOR) / ANE_PEAK, inverse = 1.0f / scale;
  for (uint block = 0; block < blocks; ++block) {
    const ulong origin = ulong(row) * hidden + (block * simd_groups + simd_group) * ANE_UNIT + lane * 4;
    for (uint e = 0; e < 4; ++e) rotated[origin + e] = half(value[block][0][e] * inverse);
  }
  if (simd_group == 0 && lane == 0) token_scale[row] = half(scale * ANE_SCALE);
}

kernel void ane_pack(device const half *rotated [[buffer(0)]],
                     device char *packed [[buffer(1)]],
                     constant uint &hidden [[buffer(2)]],
                     constant uint &channel [[buffer(3)]],
                     constant uint &stride [[buffer(4)]],
                     constant uint &rows [[buffer(5)]],
                     constant uint &stamp [[buffer(6)]],
                     uint2 tile [[threadgroup_position_in_grid]],
                     uint2 position [[thread_position_in_threadgroup]]) {
  threadgroup half staged[32][33];
  const uint row = tile.x * 32, base = tile.y * 32;
  for (uint j = position.y; j < 32; j += 8)
    staged[j][position.x] =
        row + j < rows ? rotated[ulong(row + j) * hidden + channel + base + position.x] : half(0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = position.y; j < 32; j += 8)
    packed[ulong(base + j) * stride + row + position.x] =
        char(clamp(rint(float(staged[position.x][j])), -ANE_PEAK, ANE_PEAK));
}

template <uint K>
kernel void ane_row_scale(device const uint *codes [[buffer(0)]],
                          device const bfloat *scales [[buffer(1)]],
                          device const bfloat *biases [[buffer(2)]],
                          device half *row_scale [[buffer(3)]],
                          device const float *sign [[buffer(4)]],
                          constant uint &width [[buffer(5)]],
                          constant uint &first [[buffer(6)]],
                          constant uint &input [[buffer(7)]],
                          constant uint &span [[buffer(8)]],
                          uint tile [[threadgroup_position_in_grid]],
                          uint simd_group [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tile * 8 + simd_group;
  float peak = 0.0f;
  for (uint block = 0; block < span; block += K * ANE_UNIT) {
    float value[K][4];
    for (uint k = 0; k < K; ++k)
      ane_q4_values(value[k], codes, scales, biases, width, first + row, input + block + k * ANE_UNIT + lane * 4);
    ane_rotate_block<K>(value, lane, sign);
    for (uint k = 0; k < K; ++k)
      for (uint e = 0; e < 4; ++e) peak = max(peak, fabs(value[k][e]));
  }
  peak = simd_max(peak);
  if (lane == 0) row_scale[row] = half(max(peak, ANE_FLOOR) / ANE_PEAK * ANE_SCALE);
}

template <uint K>
kernel void ane_weights(device const uint *codes [[buffer(0)]],
                        device const bfloat *scales [[buffer(1)]],
                        device const bfloat *biases [[buffer(2)]],
                        device const half *row_scale [[buffer(3)]],
                        device char *output [[buffer(4)]],
                        device half *scale [[buffer(5)]],
                        device const float *sign [[buffer(6)]],
                        constant uint &width [[buffer(7)]],
                        constant uint &first [[buffer(8)]],
                        constant uint &input [[buffer(9)]],
                        constant uint &stride [[buffer(10)]],
                        constant uint &scale_stride [[buffer(11)]],
                        uint2 tile [[threadgroup_position_in_grid]],
                        uint simd_group [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tile.x * 8 + simd_group, block = tile.y * K * ANE_UNIT;
  float value[K][4];
  for (uint k = 0; k < K; ++k)
    ane_q4_values(value[k], codes, scales, biases, width, first + row, input + block + k * ANE_UNIT + lane * 4);
  ane_rotate_block<K>(value, lane, sign);
  const float inverse = ANE_SCALE / float(row_scale[row]);
  for (uint k = 0; k < K; ++k)
    *(device char4 *)(output + ulong(row) * stride + block + k * ANE_UNIT + lane * 4) =
        char4(clamp(rint(float4(value[k][0], value[k][1], value[k][2], value[k][3]) * inverse), -ANE_PEAK, ANE_PEAK));
  if (tile.y == 0 && lane == 0) scale[ulong(row) * scale_stride] = row_scale[row];
}

using AneRowScale = void(device const uint *, device const bfloat *, device const bfloat *, device half *,
                         device const float *, constant uint &, constant uint &, constant uint &, constant uint &, uint,
                         uint, uint);
using AneWeights = void(device const uint *, device const bfloat *, device const bfloat *, device const half *,
                        device char *, device half *, device const float *, constant uint &, constant uint &,
                        constant uint &, constant uint &, constant uint &, uint2, uint, uint);
template [[host_name("ane_row_scale_inputs")]] kernel AneRowScale ane_row_scale<1>;
template [[host_name("ane_row_scale_intermediate")]] kernel AneRowScale ane_row_scale<4>;
template [[host_name("ane_weights_inputs")]] kernel AneWeights ane_weights<1>;
template [[host_name("ane_weights_intermediate")]] kernel AneWeights ane_weights<4>;

inline bool ane_not_finite(half value) { return (as_type<ushort>(value) & 0x7c00) == 0x7c00; }

kernel void ane_join(device bfloat *output [[buffer(0)]],
                     device const half *partial [[buffer(1)]],
                     device const half *token_scale [[buffer(2)]],
                     device atomic_uint *status [[buffer(3)]],
                     constant uint &hidden [[buffer(4)]],
                     constant uint &stride [[buffer(5)]],
                     constant uint &rows [[buffer(6)]],
                     constant uint &stamp [[buffer(7)]],
                     uint2 tile [[threadgroup_position_in_grid]],
                     uint2 position [[thread_position_in_threadgroup]]) {
  threadgroup float staged[32][33];
  threadgroup float scale[32];
  const uint row = tile.x * 32, channel = tile.y * 32, token = row + position.x;
  const bool chunk = token < rows;
  bool finite = true;
  for (uint j = position.y; j < 32; j += 8) {
    const half value = partial[ulong(channel + j) * stride + token];
    finite &= !(chunk && ane_not_finite(value));
    staged[j][position.x] = float(value);
  }
  if (position.y == 0) {
    const half intermediate = partial[ulong(hidden) * stride + token], input = token_scale[token];
    finite &= !(chunk && (ane_not_finite(intermediate) || ane_not_finite(input)));
    scale[position.x] = float(intermediate) * float(input);
  }
  if (!finite) atomic_store_explicit(status, 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = position.y; j < 32; j += 8) {
    if (row + j >= rows) continue;
    const ulong index = ulong(row + j) * hidden + channel + position.x;
    output[index] = bfloat(float(output[index]) + staged[position.x][j] * scale[j]);
  }
}

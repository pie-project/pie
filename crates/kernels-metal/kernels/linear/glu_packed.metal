#include <metal_stdlib>
#include "../prim/register_table.h"

using namespace metal;

template <typename T>
[[kernel]] void packed_swiglu(
    const device T* packed [[buffer(0)]],
    device T* out          [[buffer(1)]],
    const constant uint& intermediate [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint i = gid.x;
  const size_t row = size_t(gid.y) * size_t(intermediate);
  const size_t packed_row = row * 2;
  const float g = float(packed[packed_row + i]);
  const float u = float(packed[packed_row + intermediate + i]);
  out[row + i] = static_cast<T>((g / (1.0f + metal::exp(-g))) * u);
}

// packed_swiglu over a routed stack's sorted rows, skipping its padding rows
// (perm < 0), which no pass reads.
[[kernel]] void packed_swiglu_sorted(
    const device bfloat* packed [[buffer(0)]],
    device bfloat* out          [[buffer(1)]],
    const constant uint& intermediate [[buffer(2)]],
    const device int* perm      [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (perm[gid.y] < 0) return;
  const uint i = gid.x;
  const size_t row = size_t(gid.y) * size_t(intermediate);
  const size_t packed_row = row * 2;
  const float g = float(packed[packed_row + i]);
  const float u = float(packed[packed_row + intermediate + i]);
  out[row + i] = static_cast<bfloat>((g / (1.0f + metal::exp(-g))) * u);
}

#define instantiate_packed_swiglu(name, itype)                     \
  template [[host_name("packed_swiglu_" #name)]]                   \
  [[kernel]] void packed_swiglu<itype>(                            \
      const device itype*, device itype*,                          \
      const constant uint&, uint2);

instantiate_packed_swiglu(bfloat16, bfloat)

// packed_swiglu over at most eight rows that also lays the register tile's
// input table of the rows it writes, so the down projection runs no table
// pass: a threadgroup per row of the table's eight, the rows past `rows` laid
// as zeros, the first zeroing the split tile's `counters` arrival counters.
[[kernel]] void packed_swiglu_table(
    const device bfloat* packed [[buffer(0)]],
    device bfloat* out          [[buffer(1)]],
    const constant uint& intermediate [[buffer(2)]],
    device bfloat* table        [[buffer(3)]],
    const constant uint& rows   [[buffer(4)]],
    const constant uint& counters [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 lid2 [[thread_position_in_threadgroup]],
    uint2 threads2 [[threads_per_threadgroup]]) {
  threadgroup bfloat values[8192];
  const uint lid = lid2.x, threads = threads2.x;
  const uint width = intermediate, row = group.y;
  const bool live = row < rows;
  const size_t at = size_t(row) * size_t(width);
  for (uint k = lid; k < width; k += threads) {
    bfloat v = bfloat(0.0f);
    if (live) {
      const float g = float(packed[at * 2 + k]);
      const float u = float(packed[at * 2 + width + k]);
      v = static_cast<bfloat>((g / (1.0f + metal::exp(-g))) * u);
      out[at + k] = v;
    }
    table[register_table::slot(k, row)] = v;
    values[k] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const threadgroup bfloat* laid = values;
  register_table::lay_sums(laid, table, width, row, lid, threads);
  if (row == 0)
    register_table::lay_rest(table, width, 8, counters, lid, threads);
}

template <typename T>
[[kernel]] void packed_swiglu_clamp(
    const device T* packed [[buffer(0)]],
    device T* out          [[buffer(1)]],
    const constant uint& intermediate [[buffer(2)]],
    const constant float& limit       [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint i = gid.x;
  const size_t row = size_t(gid.y) * size_t(intermediate);
  const size_t packed_row = row * 2;
  float g = float(packed[packed_row + i]);
  float u = float(packed[packed_row + intermediate + i]);
  g = min(g, limit);
  u = clamp(u, -limit, limit);
  out[row + i] = static_cast<T>((g / (1.0f + metal::exp(-g))) * u);
}

#define instantiate_packed_swiglu_clamp(name, itype)               \
  template [[host_name("packed_swiglu_clamp_" #name)]]             \
  [[kernel]] void packed_swiglu_clamp<itype>(                      \
      const device itype*, device itype*,                          \
      const constant uint&, const constant float&, uint2);

instantiate_packed_swiglu_clamp(bfloat16, bfloat)

template <typename T>
[[kernel]] void packed_gptoss_swiglu(
    const device T* packed [[buffer(0)]],
    device T* out          [[buffer(1)]],
    const constant uint& intermediate [[buffer(2)]],
    const constant float& limit       [[buffer(3)]],
    const constant float& alpha       [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint i = gid.x;
  const size_t row = size_t(gid.y) * size_t(intermediate);
  const size_t packed_row = row * 2;
  float g = float(packed[packed_row + i]);
  float u = float(packed[packed_row + intermediate + i]);
  g = min(g, limit);
  u = clamp(u, -limit, limit);
  const float sig = 1.0f / (1.0f + fast::exp(-alpha * g));
  out[row + i] = static_cast<T>((g * sig) * (u + 1.0f));
}

#define instantiate_packed_gptoss_swiglu(name, itype)              \
  template [[host_name("packed_gptoss_swiglu_" #name)]]            \
  [[kernel]] void packed_gptoss_swiglu<itype>(                     \
      const device itype*, device itype*,                          \
      const constant uint&, const constant float&,                 \
      const constant float&, uint2);

instantiate_packed_gptoss_swiglu(bfloat16, bfloat)

// packed_gptoss_swiglu over a routed stack's sorted rows, skipping its padding rows.
[[kernel]] void packed_gptoss_swiglu_sorted(
    const device bfloat* packed [[buffer(0)]],
    device bfloat* out          [[buffer(1)]],
    const constant uint& intermediate [[buffer(2)]],
    const constant float& limit       [[buffer(3)]],
    const constant float& alpha       [[buffer(4)]],
    const device int* perm      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (perm[gid.y] < 0) return;
  const uint i = gid.x;
  const size_t row = size_t(gid.y) * size_t(intermediate);
  const size_t packed_row = row * 2;
  float g = float(packed[packed_row + i]);
  float u = float(packed[packed_row + intermediate + i]);
  g = min(g, limit);
  u = clamp(u, -limit, limit);
  const float sig = 1.0f / (1.0f + fast::exp(-alpha * g));
  out[row + i] = static_cast<bfloat>((g * sig) * (u + 1.0f));
}

template <typename T>
[[kernel]] void packed_geglu_tanh(
    const device T* packed [[buffer(0)]],
    device T* out          [[buffer(1)]],
    const constant uint& intermediate [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  constexpr float k = 0.7978845608028654f;
  const uint i = gid.x;
  const size_t row = size_t(gid.y) * size_t(intermediate);
  const size_t packed_row = row * 2;
  const float g = float(packed[packed_row + i]);
  const float u = float(packed[packed_row + intermediate + i]);
  const float inner = k * (g + 0.044715f * g * g * g);
  const float gelu = 0.5f * g * (1.0f + precise::tanh(inner));
  out[row + i] = static_cast<T>(gelu * u);
}

#define instantiate_packed_geglu_tanh(name, itype)                 \
  template [[host_name("packed_geglu_tanh_" #name)]]               \
  [[kernel]] void packed_geglu_tanh<itype>(                        \
      const device itype*, device itype*,                          \
      const constant uint&, uint2);

instantiate_packed_geglu_tanh(bfloat16, bfloat)

// packed_geglu_tanh over a routed stack's sorted rows, skipping its padding rows.
[[kernel]] void packed_geglu_tanh_sorted(
    const device bfloat* packed [[buffer(0)]],
    device bfloat* out          [[buffer(1)]],
    const constant uint& intermediate [[buffer(2)]],
    const device int* perm      [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (perm[gid.y] < 0) return;
  constexpr float k = 0.7978845608028654f;
  const uint i = gid.x;
  const size_t row = size_t(gid.y) * size_t(intermediate);
  const size_t packed_row = row * 2;
  const float g = float(packed[packed_row + i]);
  const float u = float(packed[packed_row + intermediate + i]);
  const float inner = k * (g + 0.044715f * g * g * g);
  const float gelu = 0.5f * g * (1.0f + precise::tanh(inner));
  out[row + i] = static_cast<bfloat>(gelu * u);
}

// packed_geglu_tanh_sorted, a threadgroup to a row and eight outputs to a lane, landing the
// row's 64-wide group sums (of the bfloat outputs) as `qmm_mpp_input_sums` lays them over
// `tile_rows`-row tiles, for the MPP routed matmul that reads the rows next.
[[kernel]] void packed_geglu_tanh_sorted_sums(
    const device bfloat* packed [[buffer(0)]],
    device bfloat* out          [[buffer(1)]],
    const constant uint& intermediate [[buffer(2)]],
    const device int* perm      [[buffer(3)]],
    device float* sums          [[buffer(4)]],
    const constant uint& tile_rows    [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 lid2 [[thread_position_in_threadgroup]],
    uint2 threads2 [[threads_per_threadgroup]]) {
  const uint row = tg.y, lid = lid2.x, threads = threads2.x;
  if (perm[row] < 0) return;
  constexpr float k = 0.7978845608028654f;
  const device bfloat* gate = packed + ulong(row) * intermediate * 2;
  const device bfloat* up = gate + intermediate;
  device bfloat* dst = out + ulong(row) * intermediate;
  const uint groups = intermediate / 64u;
  const ulong tile_base = ulong(row / tile_rows) * tile_rows * groups + row % tile_rows;
  for (uint at = lid * 8u; at < intermediate; at += threads * 8u) {
    float sum = 0.0f;
    bfloat h[8];
    for (uint j = 0; j < 8u; ++j) {
      const float g = float(gate[at + j]);
      const float u = float(up[at + j]);
      const float inner = k * (g + 0.044715f * g * g * g);
      h[j] = static_cast<bfloat>(0.5f * g * (1.0f + precise::tanh(inner)) * u);
      sum += float(h[j]);
    }
    *reinterpret_cast<device bfloat4*>(dst + at) = bfloat4(h[0], h[1], h[2], h[3]);
    *reinterpret_cast<device bfloat4*>(dst + at + 4u) = bfloat4(h[4], h[5], h[6], h[7]);
    sum += simd_shuffle_xor(sum, ushort(1));
    sum += simd_shuffle_xor(sum, ushort(2));
    sum += simd_shuffle_xor(sum, ushort(4));
    if (lid % 8u == 0u) sums[tile_base + ulong(at / 64u) * tile_rows] = sum;
  }
}

template <typename T>
[[kernel]] void packed_situ(
    const device T* packed [[buffer(0)]],
    device T* out          [[buffer(1)]],
    const constant uint& intermediate [[buffer(2)]],
    const constant float& beta        [[buffer(3)]],
    const constant float& up_cap      [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint i = gid.x;
  const size_t row = size_t(gid.y) * size_t(intermediate);
  const size_t packed_row = row * 2;
  const float g = float(packed[packed_row + i]);
  float u = float(packed[packed_row + intermediate + i]);
  const float s = beta * precise::tanh(g / beta) / (1.0f + metal::exp(-g));
  if (up_cap > 0.0f) {
    u = up_cap * precise::tanh(u / up_cap);
  }
  out[row + i] = static_cast<T>(s * u);
}

#define instantiate_packed_situ(name, itype)                       \
  template [[host_name("packed_situ_" #name)]]                     \
  [[kernel]] void packed_situ<itype>(                              \
      const device itype*, device itype*,                          \
      const constant uint&, const constant float&,                 \
      const constant float&, uint2);

instantiate_packed_situ(bfloat16, bfloat)

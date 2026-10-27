#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

constexpr constant int kMaxBlocks = 32;

inline float blend_block_sum(
    float acc, threadgroup float* partials, uint simd_lane, uint simd_group,
    uint simds) {
  acc = simd_sum(acc);
  if (simd_group == 0) partials[simd_lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane == 0) partials[simd_group] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  if (simd_group == 0) {
    total = simd_sum(simd_lane < simds ? partials[simd_lane] : 0.0f);
    if (simd_lane == 0) partials[0] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return partials[0];
}

template <typename T>
[[kernel]] void res_blend(
    const device T* prefix          [[buffer(0)]],
    const device T* blocks          [[buffer(1)]],
    const device T* norm_weight     [[buffer(2)]],
    const device T* proj_weight     [[buffer(3)]],
    device T* out                   [[buffer(4)]],
    const constant uint& n_blocks   [[buffer(5)]],
    const constant uint& hidden     [[buffer(6)]],
    const constant uint& block_rows [[buffer(7)]],
    const constant float& eps       [[buffer(8)]],
    uint gid                        [[threadgroup_position_in_grid]],
    uint lid                        [[thread_position_in_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_group                 [[simdgroup_index_in_threadgroup]],
    uint tg_size                    [[threads_per_threadgroup]]) {
  threadgroup float partials[32];
  threadgroup float prob_s[kMaxBlocks + 1];

  const uint row = gid;
  const size_t token_off = size_t(row) * size_t(hidden);
  const uint candidates = n_blocks + 1u;
  const uint simds = (tg_size + 31u) / 32u;

  for (uint j = 0; j < candidates; ++j) {
    const device T* v = j < n_blocks
        ? blocks + (size_t(j) * size_t(block_rows) + size_t(row)) * size_t(hidden)
        : prefix + token_off;

    float ss = 0.0f;
    for (uint h = lid; h < hidden; h += tg_size) {
      const float x = float(v[h]);
      ss = fma(x, x, ss);
    }
    ss = blend_block_sum(ss, partials, simd_lane, simd_group, simds);
    const float scale = precise::rsqrt(ss / float(hidden) + eps);

    float dot = 0.0f;
    for (uint h = lid; h < hidden; h += tg_size) {
      dot = fma(float(v[h]) * scale, float(norm_weight[h]) * float(proj_weight[h]), dot);
    }
    dot = blend_block_sum(dot, partials, simd_lane, simd_group, simds);
    if (lid == 0) prob_s[j] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lid == 0) {
    float m = prob_s[0];
    for (uint j = 1; j < candidates; ++j) m = max(m, prob_s[j]);
    float sum = 0.0f;
    for (uint j = 0; j < candidates; ++j) {
      prob_s[j] = precise::exp(prob_s[j] - m);
      sum += prob_s[j];
    }
    const float inv = 1.0f / sum;
    for (uint j = 0; j < candidates; ++j) prob_s[j] *= inv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint h = lid; h < hidden; h += tg_size) {
    float acc = 0.0f;
    for (uint j = 0; j < candidates; ++j) {
      const device T* v = j < n_blocks
          ? blocks + (size_t(j) * size_t(block_rows) + size_t(row)) * size_t(hidden)
          : prefix + token_off;
      acc = fma(prob_s[j], float(v[h]), acc);
    }
    out[token_off + h] = static_cast<T>(acc);
  }
}

#define instantiate_res_blend(name, itype)                                \
  template [[host_name("res_blend_" #name)]]                              \
  [[kernel]] void res_blend<itype>(                                       \
      const device itype*, const device itype*, const device itype*,      \
      const device itype*, device itype*, const constant uint&,           \
      const constant uint&, const constant uint&, const constant float&,  \
      uint, uint, uint, uint, uint);

instantiate_res_blend(bfloat16, bfloat)
instantiate_res_blend(float32, float)


// The candidate blocks where they lie (residual snapshots taken layers
// apart), each bound on its own: up to kScatteredBlocks of them.
constexpr constant int kScatteredBlocks = 24;

#define RES_BLOCK_ARGS(T)                                                         \
  const device T* b0 [[buffer(7)]], const device T* b1 [[buffer(8)]],             \
  const device T* b2 [[buffer(9)]], const device T* b3 [[buffer(10)]],            \
  const device T* b4 [[buffer(11)]], const device T* b5 [[buffer(12)]],           \
  const device T* b6 [[buffer(13)]], const device T* b7 [[buffer(14)]],           \
  const device T* b8 [[buffer(15)]], const device T* b9 [[buffer(16)]],           \
  const device T* b10 [[buffer(17)]], const device T* b11 [[buffer(18)]],         \
  const device T* b12 [[buffer(19)]], const device T* b13 [[buffer(20)]],         \
  const device T* b14 [[buffer(21)]], const device T* b15 [[buffer(22)]],         \
  const device T* b16 [[buffer(23)]], const device T* b17 [[buffer(24)]],         \
  const device T* b18 [[buffer(25)]], const device T* b19 [[buffer(26)]],         \
  const device T* b20 [[buffer(27)]], const device T* b21 [[buffer(28)]],         \
  const device T* b22 [[buffer(29)]], const device T* b23 [[buffer(30)]]

template <typename T>
[[kernel]] void res_blend_scattered(
    const device T* prefix          [[buffer(0)]],
    const device T* norm_weight     [[buffer(1)]],
    const device T* proj_weight     [[buffer(2)]],
    device T* out                   [[buffer(3)]],
    const constant uint& n_blocks   [[buffer(4)]],
    const constant uint& hidden     [[buffer(5)]],
    const constant float& eps       [[buffer(6)]],
    RES_BLOCK_ARGS(T),
    uint gid                        [[threadgroup_position_in_grid]],
    uint lid                        [[thread_position_in_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_group                 [[simdgroup_index_in_threadgroup]],
    uint tg_size                    [[threads_per_threadgroup]]) {
  threadgroup float partials[32];
  threadgroup float prob_s[kScatteredBlocks + 1];
  const device T* at[kScatteredBlocks] = {b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11,
                                          b12, b13, b14, b15, b16, b17, b18, b19, b20, b21,
                                          b22, b23};

  const uint row = gid;
  const size_t token_off = size_t(row) * size_t(hidden);
  const uint candidates = n_blocks + 1u;
  const uint simds = (tg_size + 31u) / 32u;

  for (uint j = 0; j < candidates; ++j) {
    const device T* v = (j < n_blocks ? at[j] : prefix) + token_off;

    float ss = 0.0f;
    for (uint h = lid; h < hidden; h += tg_size) {
      const float x = float(v[h]);
      ss = fma(x, x, ss);
    }
    ss = blend_block_sum(ss, partials, simd_lane, simd_group, simds);
    const float scale = precise::rsqrt(ss / float(hidden) + eps);

    float dot = 0.0f;
    for (uint h = lid; h < hidden; h += tg_size) {
      dot = fma(float(v[h]) * scale, float(norm_weight[h]) * float(proj_weight[h]), dot);
    }
    dot = blend_block_sum(dot, partials, simd_lane, simd_group, simds);
    if (lid == 0) prob_s[j] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lid == 0) {
    float m = prob_s[0];
    for (uint j = 1; j < candidates; ++j) m = max(m, prob_s[j]);
    float sum = 0.0f;
    for (uint j = 0; j < candidates; ++j) {
      prob_s[j] = precise::exp(prob_s[j] - m);
      sum += prob_s[j];
    }
    const float inv = 1.0f / sum;
    for (uint j = 0; j < candidates; ++j) prob_s[j] *= inv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint h = lid; h < hidden; h += tg_size) {
    float acc = 0.0f;
    for (uint j = 0; j < candidates; ++j) {
      const device T* v = (j < n_blocks ? at[j] : prefix) + token_off;
      acc = fma(prob_s[j], float(v[h]), acc);
    }
    out[token_off + h] = static_cast<T>(acc);
  }
}

#define instantiate_res_blend_scattered(name, itype)                      \
  template [[host_name("res_blend_scattered_" #name)]]                    \
  [[kernel]] void res_blend_scattered<itype>(                             \
      const device itype*, const device itype*, const device itype*,      \
      device itype*, const constant uint&, const constant uint&,          \
      const constant float&, RES_BLOCK_ARGS(itype),                       \
      uint, uint, uint, uint, uint);

instantiate_res_blend_scattered(bfloat16, bfloat)
instantiate_res_blend_scattered(float32, float)

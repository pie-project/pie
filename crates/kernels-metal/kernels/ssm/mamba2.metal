#include <metal_stdlib>
using namespace metal;

#ifndef PIE_RS_SNAPSHOTS
#define PIE_RS_SNAPSHOTS 4
#endif

// Mamba-2: a head's state `[head_dim = 64][state_dim = 128]` in f32, its
// rows read off one convolved `xbc` row (`x`, then the groups' `B`, then
// their `C`), its decay `exp(-exp(A_log) dt)` one scalar a step. A
// threadgroup a (lane, head), four threads a state row, each 32 of its
// columns in registers. Lands `(C S + D x) * silu(z)`; the group norm runs
// after. CUDA's `mamba2_step`, step for step.
constant constexpr int M2_P = 64;
constant constexpr int M2_N = 128;
constant constexpr int M2_PER = M2_N / 4;

template <typename T>
METAL_FUNC void mamba2_walk(
    const device T* xbc, const device T* dt, const device T* z,
    const device float* a_log, const device float* dt_bias, const device float* skip_d,
    device float* rstate, device T* y, int heads, int groups, int h,
    int begin, int end, uint slot, const device int* snap,
    threadgroup float* sb, threadgroup float* sc, uint tid) {
  const int p = int(tid) / 4;
  const int q = int(tid) % 4;
  const int inner = heads * M2_P;
  const size_t width = size_t(inner + 2 * groups * M2_N);
  const int g = h / (heads / groups);
  const size_t head = size_t(M2_P) * size_t(M2_N);
  const size_t cell = size_t(p) * M2_N + size_t(q) * M2_PER;
  device float* state = rstate + (size_t(slot) * size_t(heads) + size_t(h)) * head + cell;
  float s[M2_PER];
  for (int i = 0; i < M2_PER; ++i) s[i] = state[i];
  const float neg_a = -metal::exp(a_log[h]);
  const float bias = dt_bias[h];
  const float skip = skip_d[h];
  int snapped = 0;
  int snap_at = snap != nullptr && snap[0] > 0 ? begin + snap[0] : -1;

  for (int t = begin; t < end; ++t) {
    const size_t row = size_t(t) * width;
    for (int i = int(tid); i < M2_N; i += 256) {
      sb[i] = float(xbc[row + size_t(inner + g * M2_N + i)]);
      sc[i] = float(xbc[row + size_t(inner + (groups + g) * M2_N + i)]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float raw = float(dt[size_t(t) * size_t(heads) + size_t(h)]) + bias;
    const float step = raw > 20.0f ? raw : metal::log(1.0f + metal::exp(raw));
    const float decay = metal::exp(neg_a * step);
    const float xp = float(xbc[row + size_t(h * M2_P + p)]);
    const float dx = step * xp;
    float acc = 0.0f;
    for (int i = 0; i < M2_PER; ++i) {
      const int n = q * M2_PER + i;
      s[i] = s[i] * decay + dx * sb[n];
      acc += s[i] * sc[n];
    }
    acc += simd_shuffle_xor(acc, 1);
    acc += simd_shuffle_xor(acc, 2);
    if (q == 0) {
      const size_t o = size_t(t) * size_t(inner) + size_t(h * M2_P + p);
      const float zv = float(z[o]);
      y[o] = T((acc + skip * xp) * (zv / (1.0f + metal::exp(-zv))));
    }
    // The state as of a snapshot row lands in a slot of its own.
    if (t + 1 == snap_at) {
      device float* landed =
          rstate + (size_t(snap[PIE_RS_SNAPSHOTS + snapped]) * size_t(heads) + size_t(h)) * head + cell;
      for (int i = 0; i < M2_PER; ++i) landed[i] = s[i];
      ++snapped;
      snap_at = snapped < PIE_RS_SNAPSHOTS && snap[snapped] > 0 ? begin + snap[snapped] : -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (int i = 0; i < M2_PER; ++i) state[i] = s[i];
}

// One step a row, its slot the row's.
template <typename T>
[[kernel]] void mamba2_step(
    const device T* xbc            [[buffer(0)]],
    const device T* dt             [[buffer(1)]],
    const device T* z              [[buffer(2)]],
    const device float* a_log      [[buffer(3)]],
    const device float* dt_bias    [[buffer(4)]],
    const device float* skip_d     [[buffer(5)]],
    device float* rstate           [[buffer(6)]],
    const device uint* slots       [[buffer(7)]],
    device T* y                    [[buffer(8)]],
    const constant int& heads      [[buffer(9)]],
    const constant int& groups     [[buffer(10)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 lpos [[thread_position_in_threadgroup]]) {
  threadgroup float sb[M2_N];
  threadgroup float sc[M2_N];
  const int r = int(tg.z);
  mamba2_walk<T>(xbc, dt, z, a_log, dt_bias, skip_d, rstate, y, heads, groups, int(tg.y),
                 r, r + 1, slots[r], nullptr, sb, sc, lpos.x);
}

// A prompt's walk a request, its rows `indptr[r]..indptr[r + 1]`, its slot
// its first row's; the state as of each snapshot row lands in that row's slot.
template <typename T>
[[kernel]] void mamba2_chunked(
    const device T* xbc            [[buffer(0)]],
    const device int* indptr       [[buffer(1)]],
    const device T* dt             [[buffer(2)]],
    const device T* z              [[buffer(3)]],
    const device float* a_log      [[buffer(4)]],
    const device float* dt_bias    [[buffer(5)]],
    const device float* skip_d     [[buffer(6)]],
    device float* rstate           [[buffer(7)]],
    const device uint* slots       [[buffer(8)]],
    device T* y                    [[buffer(9)]],
    const constant int& heads      [[buffer(10)]],
    const constant int& groups     [[buffer(11)]],
    const device int* snaps        [[buffer(12)]],
    const constant int& snap_lane0 [[buffer(13)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 lpos [[thread_position_in_threadgroup]]) {
  threadgroup float sb[M2_N];
  threadgroup float sc[M2_N];
  const int r = int(tg.z);
  const int begin = indptr[r];
  const int end = indptr[r + 1];
  if (end <= begin) {
    return;
  }
  const device int* snap =
      snap_lane0 >= 0 ? snaps + size_t(snap_lane0 + r) * 2 * PIE_RS_SNAPSHOTS : nullptr;
  mamba2_walk<T>(xbc, dt, z, a_log, dt_bias, skip_d, rstate, y, heads, groups, int(tg.y),
                 begin, end, slots[begin], snap, sb, sc, lpos.x);
}

template [[host_name("mamba2_step_bfloat16")]] [[kernel]] void mamba2_step<bfloat>(
    const device bfloat*, const device bfloat*, const device bfloat*, const device float*,
    const device float*, const device float*, device float*, const device uint*,
    device bfloat*, const constant int&, const constant int&, uint3, uint3);

template [[host_name("mamba2_chunked_bfloat16")]] [[kernel]] void mamba2_chunked<bfloat>(
    const device bfloat*, const device int*, const device bfloat*, const device bfloat*,
    const device float*, const device float*, const device float*, device float*,
    const device uint*, device bfloat*, const constant int&, const constant int&,
    const device int*, const constant int&, uint3, uint3);

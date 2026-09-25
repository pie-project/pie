

#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;


template <int LANES>
inline float gdn_row_sum(float v) {
  v += simd_shuffle_xor(v, 1u);
  if (LANES >= 4) v += simd_shuffle_xor(v, 2u);
  if (LANES >= 8) v += simd_shuffle_xor(v, 4u);
  if (LANES >= 16) v += simd_shuffle_xor(v, 8u);
  if (LANES >= 32) v += simd_shuffle_xor(v, 16u);
  return v;
}

template <typename T, int LANES, int VROWS, int PER>
inline void gdn_scan_token(
    const device T* qkv,
    const device float* gates,
    device float* y,
    thread float (&st)[VROWS][PER],
    int at,
    int lane,
    int dv_base,
    int hv,
    int hk,
    int k_heads,
    int v_heads,
    int dk,
    int dv) {
  const size_t keys = size_t(k_heads) * size_t(dk);
  const size_t pitch = 2 * keys + size_t(v_heads) * size_t(dv);
  const float scale = 1.0f / metal::sqrt(float(dk));
  const size_t row = size_t(at) * pitch;
  const size_t qbase = row + size_t(hk) * size_t(dk);
  const size_t kbase = qbase + keys;
  const size_t vbase = row + 2 * keys + size_t(hv) * size_t(dv);

  float q[PER];
  float k[PER];
  float qs = 0.0f;
  float ks = 0.0f;
  for (int i = 0; i < PER; ++i) {
    const int d = PER * lane + i;
    const float qv = float(qkv[qbase + size_t(d)]);
    const float kv = float(qkv[kbase + size_t(d)]);
    q[i] = qv;
    k[i] = kv;
    qs += qv * qv;
    ks += kv * kv;
  }
  const float qinv = metal::rsqrt(gdn_row_sum<LANES>(qs) + 1e-6f) * scale;
  const float kinv = metal::rsqrt(gdn_row_sum<LANES>(ks) + 1e-6f);
  for (int i = 0; i < PER; ++i) {
    q[i] *= qinv;
    k[i] *= kinv;
  }

  const size_t fused = size_t(at) * 2 * size_t(v_heads) + size_t(hv);
  const float decay = metal::exp(gates[fused]);
  const float beta = gates[fused + size_t(v_heads)];
  const size_t out = (size_t(at) * size_t(v_heads) + size_t(hv)) * size_t(dv);

  float kv_mem[VROWS];
  for (int v = 0; v < VROWS; ++v) {
    float acc = 0.0f;
    for (int i = 0; i < PER; ++i) {
      st[v][i] *= decay;
      acc += st[v][i] * k[i];
    }
    kv_mem[v] = gdn_row_sum<LANES>(acc);
  }
  for (int v = 0; v < VROWS; ++v) {
    const float delta =
        (float(qkv[vbase + size_t(dv_base + v)]) - kv_mem[v]) * beta;
    float sum = 0.0f;
    for (int i = 0; i < PER; ++i) {
      st[v][i] += k[i] * delta;
      sum += st[v][i] * q[i];
    }
    sum = gdn_row_sum<LANES>(sum);
    if (lane == 0) {
      y[out + size_t(dv_base + v)] = sum;
    }
  }
}

template <int VROWS, int PER>
inline device float* gdn_cells(
    device float* rstate, int slot, int hv, int v_heads, int dv_base, int dk, int dv) {
  return rstate +
         ((size_t(slot) * size_t(v_heads) + size_t(hv)) * size_t(dv) +
          size_t(dv_base)) *
             size_t(dk);
}

template <int VROWS, int PER>
inline void gdn_load(thread float (&st)[VROWS][PER], const device float* cells, int lane, int dk) {
  for (int v = 0; v < VROWS; ++v) {
    for (int i = 0; i < PER; ++i) {
      st[v][i] = cells[size_t(v) * size_t(dk) + size_t(PER * lane + i)];
    }
  }
}

template <int VROWS, int PER>
inline void gdn_store(device float* cells, thread const float (&st)[VROWS][PER], int lane, int dk) {
  for (int v = 0; v < VROWS; ++v) {
    for (int i = 0; i < PER; ++i) {
      cells[size_t(v) * size_t(dk) + size_t(PER * lane + i)] = st[v][i];
    }
  }
}

template <typename T, int LANES, int VROWS, int PER>
[[kernel]] void gated_delta_scan(
    const device T* qkv         [[buffer(0)]],
    const device int* indptr    [[buffer(1)]],
    const device float* gates   [[buffer(2)]],
    device float* rstate        [[buffer(3)]],
    const device uint* slots    [[buffer(4)]],
    device float* y             [[buffer(5)]],
    const constant int& k_heads [[buffer(6)]],
    const constant int& v_heads [[buffer(7)]],
    const constant int& k_dim   [[buffer(8)]],
    const constant int& v_dim   [[buffer(9)]],
    uint3 pos [[thread_position_in_grid]]) {
  const int lane = int(pos.x) % LANES;
  const int dv_base = int(pos.y) * VROWS;
  const int hv = int(pos.z) % v_heads;
  const int n = int(pos.z) / v_heads;

  const int begin = indptr[n];
  const int end = indptr[n + 1];
  if (end <= begin) {
    return;
  }
  const int dk = k_dim;
  const int dv = v_dim;
  const int hk = hv / (v_heads / k_heads);

  device float* cells =
      gdn_cells<VROWS, PER>(rstate, int(slots[begin]), hv, v_heads, dv_base, dk, dv);
  float st[VROWS][PER];
  gdn_load<VROWS, PER>(st, cells, lane, dk);
  for (int t = begin; t < end; ++t) {
    gdn_scan_token<T, LANES, VROWS, PER>(
        qkv, gates, y, st, t, lane, dv_base, hv, hk, k_heads, v_heads, dk, dv);
  }
  gdn_store<VROWS, PER>(cells, st, lane, dk);
}

#define PIE_STAMP_gdn_scan(entry, lanes, vrows, per)                           \
  template [[host_name(entry)]]                                                \
  [[kernel]] void gated_delta_scan<bfloat, lanes, vrows, per>(                 \
      const device bfloat*, const device int*, const device float*,            \
      device float*, const device uint*, device float*,                        \
      const constant int&, const constant int&, const constant int&,           \
      const constant int&, uint3);

template <typename T, int LANES, int VROWS, int PER>
[[kernel]] void gated_delta_scan_step(
    const device T* qkv         [[buffer(0)]],
    const device float* gates   [[buffer(1)]],
    device float* rstate        [[buffer(2)]],
    const device uint* slots    [[buffer(3)]],
    device float* y             [[buffer(4)]],
    const constant int& k_heads [[buffer(5)]],
    const constant int& v_heads [[buffer(6)]],
    const constant int& k_dim   [[buffer(7)]],
    const constant int& v_dim   [[buffer(8)]],
    uint3 pos [[thread_position_in_grid]]) {
  const int lane = int(pos.x) % LANES;
  const int dv_base = int(pos.y) * VROWS;
  const int hv = int(pos.z) % v_heads;
  const int n = int(pos.z) / v_heads;
  const int dk = k_dim;
  const int dv = v_dim;
  const int hk = hv / (v_heads / k_heads);

  device float* cells =
      gdn_cells<VROWS, PER>(rstate, int(slots[n]), hv, v_heads, dv_base, dk, dv);
  float st[VROWS][PER];
  gdn_load<VROWS, PER>(st, cells, lane, dk);
  gdn_scan_token<T, LANES, VROWS, PER>(
      qkv, gates, y, st, n, lane, dv_base, hv, hk, k_heads, v_heads, dk, dv);
  gdn_store<VROWS, PER>(cells, st, lane, dk);
}

#define PIE_STAMP_gdn_scan_step(entry, lanes, vrows, per)                      \
  template [[host_name(entry)]]                                                \
  [[kernel]] void gated_delta_scan_step<bfloat, lanes, vrows, per>(            \
      const device bfloat*, const device float*, device float*,                \
      const device uint*, device float*,                                       \
      const constant int&, const constant int&, const constant int&,           \
      const constant int&, uint3);

template <typename T, int LANES, int VROWS, int PER>
[[kernel]] void gated_delta_scan_committed(
    const device T* qkv         [[buffer(0)]],
    const device int* indptr    [[buffer(1)]],
    const device int* replay    [[buffer(2)]],
    const device int* commit    [[buffer(3)]],
    const device int* slots     [[buffer(4)]],
    const constant int& lane0   [[buffer(5)]],
    const device float* gates   [[buffer(6)]],
    device float* rstate        [[buffer(7)]],
    device float* y             [[buffer(8)]],
    const constant int& k_heads [[buffer(9)]],
    const constant int& v_heads [[buffer(10)]],
    const constant int& k_dim   [[buffer(11)]],
    const constant int& v_dim   [[buffer(12)]],
    uint3 pos [[thread_position_in_grid]]) {
  const int lane = int(pos.x) % LANES;
  const int dv_base = int(pos.y) * VROWS;
  const int hv = int(pos.z) % v_heads;
  const int r = int(pos.z) / v_heads;

  int begin = indptr[r];
  for (int j = 0; j < r; ++j) {
    begin += replay[lane0 + j];
  }
  const int span = (indptr[r + 1] - indptr[r]) + replay[lane0 + r];
  if (span <= 0) {
    return;
  }
  const int slot = slots[lane0 + r];
  if (slot < 0) {
    return;
  }
  const int keep = min(commit[lane0 + r], span);
  const int dk = k_dim;
  const int dv = v_dim;
  const int hk = hv / (v_heads / k_heads);

  device float* cells = gdn_cells<VROWS, PER>(rstate, slot, hv, v_heads, dv_base, dk, dv);
  float st[VROWS][PER];
  gdn_load<VROWS, PER>(st, cells, lane, dk);
  for (int t = 0; t < span; ++t) {
    gdn_scan_token<T, LANES, VROWS, PER>(
        qkv, gates, y, st, begin + t, lane, dv_base, hv, hk, k_heads, v_heads, dk, dv);

    if (t + 1 == keep) {
      gdn_store<VROWS, PER>(cells, st, lane, dk);
    }
  }
}

#define PIE_STAMP_gdn_scan_committed(entry, lanes, vrows, per)                 \
  template [[host_name(entry)]]                                                \
  [[kernel]] void gated_delta_scan_committed<bfloat, lanes, vrows, per>(       \
      const device bfloat*, const device int*, const device int*,              \
      const device int*, const device int*, const constant int&,               \
      const device float*, device float*, device float*,                       \
      const constant int&, const constant int&, const constant int&,           \
      const constant int&, uint3);

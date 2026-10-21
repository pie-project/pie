

#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

#ifndef PIE_RS_SNAPSHOTS
#define PIE_RS_SNAPSHOTS 4
#endif


template <int LANES>
inline float gdn_row_sum(float v) {
  v += simd_shuffle_xor(v, 1u);
  if (LANES >= 4) v += simd_shuffle_xor(v, 2u);
  if (LANES >= 8) v += simd_shuffle_xor(v, 4u);
  if (LANES >= 16) v += simd_shuffle_xor(v, 8u);
  if (LANES >= 32) v += simd_shuffle_xor(v, 16u);
  return v;
}

inline float gdn_packed_dot(
    const thread float (&a)[16], const thread float (&b)[16]) {
  #pragma clang fp reassociate(off)
  float4 p = 0.0f;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    p.x = fma(a[i], b[i], p.x);
    p.y = fma(a[4 + i], b[4 + i], p.y);
    p.z = fma(a[8 + i], b[8 + i], p.z);
    p.w = fma(a[12 + i], b[12 + i], p.w);
  }
  return (p.x + p.y) + (p.z + p.w);
}

// Row `at`'s inputs to a scan: its query and key, each l2-normed over the
// head, the query scaled, its value rows, its decay and its beta.
template <typename T, int LANES, int VROWS, int PER>
inline void gdn_scan_row(
    const device T* qkv,
    const device float* gates,
    thread float (&q)[PER],
    thread float (&k)[PER],
    thread float (&vals)[VROWS],
    thread float& decay,
    thread float& beta,
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

  float qs = 0.0f;
  float ks = 0.0f;
  if constexpr (LANES == 8 && PER == 16) {
    const device vec<T,4>* qp = reinterpret_cast<const device vec<T,4>*>(qkv + qbase + PER*lane);
    const device vec<T,4>* kp = reinterpret_cast<const device vec<T,4>*>(qkv + kbase + PER*lane);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float4 qv = float4(qp[i]), kv = float4(kp[i]);
      q[4*i] = qv.x;
      q[4*i+1] = qv.y;
      q[4*i+2] = qv.z;
      q[4*i+3] = qv.w;
      k[4*i] = kv.x;
      k[4*i+1] = kv.y;
      k[4*i+2] = kv.z;
      k[4*i+3] = kv.w;
    }
    qs = gdn_packed_dot(q, q);
    ks = gdn_packed_dot(k, k);
  } else {
    for (int i = 0; i < PER; ++i) {
      const int d = PER * lane + i;
      const float qv = float(qkv[qbase + size_t(d)]);
      const float kv = float(qkv[kbase + size_t(d)]);
      q[i] = qv;
      k[i] = kv;
      qs += qv * qv;
      ks += kv * kv;
    }
  }
  const float qinv = metal::rsqrt(gdn_row_sum<LANES>(qs) + 1e-6f) * scale;
  const float kinv = metal::rsqrt(gdn_row_sum<LANES>(ks) + 1e-6f);
  for (int i = 0; i < PER; ++i) {
    q[i] *= qinv;
    k[i] *= kinv;
  }
  const size_t vbase = row + 2 * keys + size_t(hv) * size_t(dv);
  for (int v = 0; v < VROWS; ++v) {
    vals[v] = float(qkv[vbase + size_t(dv_base + v)]);
  }
  const size_t fused = size_t(at) * 2 * size_t(v_heads) + size_t(hv);
  decay = metal::exp(gates[fused]);
  beta = gates[fused + size_t(v_heads)];
}

// Decay the state and take each value row's memory, not yet reduced.
template <int LANES, int VROWS, int PER>
inline void gdn_scan_memory(
    thread float (&st)[VROWS][PER],
    const thread float (&k)[PER],
    float decay,
    thread float (&mem)[VROWS]) {
  for (int v = 0; v < VROWS; ++v) {
    float acc = 0.0f;
    for (int i = 0; i < PER; ++i) {
      st[v][i] *= decay;
      acc += st[v][i] * k[i];
    }
    if constexpr (LANES == 8 && PER == 16) acc = gdn_packed_dot(st[v], k);
    mem[v] = acc;
  }
}

// Fold the row in and take each value row's readout, not yet reduced; `mem`
// is reduced.
template <int LANES, int VROWS, int PER>
inline void gdn_scan_update(
    thread float (&st)[VROWS][PER],
    const thread float (&k)[PER],
    const thread float (&q)[PER],
    const thread float (&vals)[VROWS],
    const thread float (&mem)[VROWS],
    float beta,
    thread float (&out)[VROWS]) {
  for (int v = 0; v < VROWS; ++v) {
    const float delta = (vals[v] - mem[v]) * beta;
    float sum = 0.0f;
    for (int i = 0; i < PER; ++i) {
      st[v][i] += k[i] * delta;
      sum += st[v][i] * q[i];
    }
    if constexpr (LANES == 8 && PER == 16) sum = gdn_packed_dot(st[v], q);
    out[v] = sum;
  }
}

// One row of a scan, in the arithmetic every scan of a layout shares.
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
  float q[PER], k[PER], vals[VROWS], mem[VROWS], out[VROWS];
  float decay, beta;
  gdn_scan_row<T, LANES, VROWS, PER>(qkv, gates, q, k, vals, decay, beta, at, lane,
                                     dv_base, hv, hk, k_heads, v_heads, dk, dv);
  gdn_scan_memory<LANES, VROWS, PER>(st, k, decay, mem);
  for (int v = 0; v < VROWS; ++v) {
    mem[v] = gdn_row_sum<LANES>(mem[v]);
  }
  gdn_scan_update<LANES, VROWS, PER>(st, k, q, vals, mem, beta, out);
  const size_t row_out = (size_t(at) * size_t(v_heads) + size_t(hv)) * size_t(dv);
  for (int v = 0; v < VROWS; ++v) {
    out[v] = gdn_row_sum<LANES>(out[v]);
    if (lane == 0) {
      y[row_out + size_t(dv_base + v)] = out[v];
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
    const device int* snaps     [[buffer(10)]],
    const constant int& snap_lane0 [[buffer(11)]],
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
  // The rows whose state lands in a slot of its own, ascending; 0 ends them.
  const device int* snap =
      snap_lane0 >= 0 ? snaps + size_t(snap_lane0 + n) * 2 * PIE_RS_SNAPSHOTS : nullptr;
  int snapped = 0;
  int snap_at = snap != nullptr ? begin + snap[0] : -1;
  if constexpr (LANES != 32) {
    // A lane of 16 keys or more has no registers to spare for the next row.
    for (int t = begin; t < end; ++t) {
      gdn_scan_token<T, LANES, VROWS, PER>(
          qkv, gates, y, st, t, lane, dv_base, hv, hk, k_heads, v_heads, dk, dv);
      if (t + 1 == snap_at) {
        gdn_store<VROWS, PER>(
            gdn_cells<VROWS, PER>(rstate, snap[PIE_RS_SNAPSHOTS + snapped], hv, v_heads, dv_base,
                                  dk, dv),
            st, lane, dk);
        ++snapped;
        snap_at = snapped < PIE_RS_SNAPSHOTS && snap[snapped] > 0 ? begin + snap[snapped] : -1;
      }
    }
    gdn_store<VROWS, PER>(cells, st, lane, dk);
    return;
  }
  // Row t's readout and row t + 1's memory both read the state as row t
  // leaves it, so one pass takes both and their reductions overlap, each
  // with the arithmetic `gdn_scan_token` gives it; row t + 1's inputs load
  // while row t updates.
  float q[PER], k[PER], vals[VROWS], mem[VROWS], out[VROWS];
  float decay, beta;
  gdn_scan_row<T, LANES, VROWS, PER>(qkv, gates, q, k, vals, decay, beta, begin, lane,
                                     dv_base, hv, hk, k_heads, v_heads, dk, dv);
  gdn_scan_memory<LANES, VROWS, PER>(st, k, decay, mem);
  for (int v = 0; v < VROWS; ++v) {
    mem[v] = gdn_row_sum<LANES>(mem[v]);
  }
  for (int t = begin; t < end; ++t) {
    const bool more = t + 1 < end;
    float nq[PER], nk[PER], nvals[VROWS], ndecay, nbeta;
    gdn_scan_row<T, LANES, VROWS, PER>(qkv, gates, nq, nk, nvals, ndecay, nbeta,
                                       more ? t + 1 : t, lane, dv_base, hv, hk, k_heads,
                                       v_heads, dk, dv);
    gdn_scan_update<LANES, VROWS, PER>(st, k, q, vals, mem, beta, out);
    if (t + 1 == snap_at) {
      gdn_store<VROWS, PER>(
          gdn_cells<VROWS, PER>(rstate, snap[PIE_RS_SNAPSHOTS + snapped], hv, v_heads, dv_base,
                                dk, dv),
          st, lane, dk);
      ++snapped;
      snap_at = snapped < PIE_RS_SNAPSHOTS && snap[snapped] > 0 ? begin + snap[snapped] : -1;
    }
    if (more) {
      gdn_scan_memory<LANES, VROWS, PER>(st, nk, ndecay, mem);
    }
    const size_t row_out = (size_t(t) * size_t(v_heads) + size_t(hv)) * size_t(dv);
    for (int v = 0; v < VROWS; ++v) {
      if (more) {
        mem[v] = gdn_row_sum<LANES>(mem[v]);
      }
      out[v] = gdn_row_sum<LANES>(out[v]);
      if (lane == 0) {
        y[row_out + size_t(dv_base + v)] = out[v];
      }
    }
    for (int i = 0; i < PER; ++i) {
      q[i] = nq[i];
      k[i] = nk[i];
    }
    for (int v = 0; v < VROWS; ++v) {
      vals[v] = nvals[v];
    }
    decay = ndecay;
    beta = nbeta;
  }
  gdn_store<VROWS, PER>(cells, st, lane, dk);
}

#define PIE_STAMP_gdn_scan(entry, lanes, vrows, per)                           \
  template [[host_name(entry)]]                                                \
  [[kernel]] void gated_delta_scan<bfloat, lanes, vrows, per>(                 \
      const device bfloat*, const device int*, const device float*,            \
      device float*, const device uint*, device float*,                        \
      const constant int&, const constant int&, const constant int&,           \
      const constant int&, const device int*, const constant int&, uint3);

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
    const device int* snaps     [[buffer(13)]],
    const constant int& snap_lane0 [[buffer(14)]],
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
  // A lane that snapshots folds its own rows, none replayed: the state as of
  // each of its snapshot rows lands in that row's slot.
  const device int* snap =
      snap_lane0 >= 0 ? snaps + size_t(snap_lane0 + r) * 2 * PIE_RS_SNAPSHOTS : nullptr;
  int snapped = 0;
  int snap_at = snap != nullptr ? snap[0] : 0;
  for (int t = 0; t < span; ++t) {
    gdn_scan_token<T, LANES, VROWS, PER>(
        qkv, gates, y, st, begin + t, lane, dv_base, hv, hk, k_heads, v_heads, dk, dv);

    if (t + 1 == keep) {
      gdn_store<VROWS, PER>(cells, st, lane, dk);
    }
    if (t + 1 == snap_at) {
      gdn_store<VROWS, PER>(
          gdn_cells<VROWS, PER>(rstate, snap[PIE_RS_SNAPSHOTS + snapped], hv, v_heads, dv_base,
                                dk, dv),
          st, lane, dk);
      ++snapped;
      snap_at = snapped < PIE_RS_SNAPSHOTS ? snap[snapped] : 0;
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
      const constant int&, const device int*, const constant int&, uint3);

// The rows of a committed scan one pass stages in threadgroup memory.
constant constexpr int kVerifyChunk = 16;

// A committed scan over 128-wide heads whose threadgroup holds `SG` simdgroups
// of `32 / LANES` value-row groups, `VROWS` rows each. Each pass stages up to
// `kVerifyChunk` rows' normed q and k, the group's value rows, decay and beta
// in threadgroup memory, every row's loads issued side by side, so a row of
// the walk reads no device memory; `gated_delta_scan_committed` loaded each
// row's inputs behind the row before, and a verify walk of a dozen rows
// spent most of its time waiting on them. A replayed row only folds into the
// state: its readout was landed the fire that first ran it. A row of this
// fire's own lands its readout in `y` at its own row, where the window reads
// it. At 32 lanes the arithmetic is `gdn_scan_token`'s, so the bits are the
// committed scan's.
template <typename T, int LANES, int SG, int VROWS>
[[kernel]] void gated_delta_verify(
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
    const device int* snaps     [[buffer(11)]],
    const constant int& snap_lane0 [[buffer(12)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane_u [[thread_index_in_simdgroup]],
    uint sg_u [[simdgroup_index_in_threadgroup]]) {
  constexpr int D = 128;
  constexpr int PER = D / LANES;
  // The normed q and k are staged 4 columns a lane, as the 32-lane scan
  // holds them.
  constexpr int NORM_PER = 4;
  constexpr int SUBS = 32 / LANES;
  constexpr int GROUP_ROWS = SG * SUBS * VROWS;
  const int tid = int(sg_u) * 32 + int(lane_u);
  const int lane = int(lane_u) % LANES;
  const int hv = int(group.z) % v_heads;
  const int r = int(group.z) / v_heads;
  const int group_base = int(group.y) * GROUP_ROWS;
  const int my_rows = (int(sg_u) * SUBS + int(lane_u) / LANES) * VROWS;
  const int dv_base = group_base + my_rows;

  threadgroup float q_s[kVerifyChunk][D];
  threadgroup float k_s[kVerifyChunk][D];
  threadgroup float v_s[kVerifyChunk][GROUP_ROWS];
  threadgroup float decay_s[kVerifyChunk];
  threadgroup float beta_s[kVerifyChunk];

  int begin = indptr[r];
  for (int j = 0; j < r; ++j) {
    begin += replay[lane0 + j];
  }
  const int rep = replay[lane0 + r];
  const int span = (indptr[r + 1] - indptr[r]) + rep;
  const int slot = slots[lane0 + r];
  // Uniform over the threadgroup, so its barriers below are too.
  if (span <= 0 || slot < 0) {
    return;
  }
  const int own = indptr[r] - rep;
  const int keep = min(commit[lane0 + r], span);
  const int hk = hv / (v_heads / k_heads);
  const size_t keys = size_t(k_heads) * D;
  const size_t pitch = 2 * keys + size_t(v_heads) * D;
  const float scale = 1.0f / metal::sqrt(float(D));

  device float* cells = gdn_cells<VROWS, PER>(rstate, slot, hv, v_heads, dv_base, D, D);
  float st[VROWS][PER];
  gdn_load<VROWS, PER>(st, cells, lane, D);
  const device int* snap =
      snap_lane0 >= 0 ? snaps + size_t(snap_lane0 + r) * 2 * PIE_RS_SNAPSHOTS : nullptr;
  int snapped = 0;
  int snap_at = snap != nullptr ? snap[0] : 0;

  for (int c0 = 0; c0 < span; c0 += kVerifyChunk) {
    const int rows = min(kVerifyChunk, span - c0);
    // One simdgroup a row: its q and k, normed as `gdn_scan_row` norms them.
    for (int j = int(sg_u); j < rows; j += SG) {
      const size_t row = size_t(begin + c0 + j) * pitch;
      const size_t qbase = row + size_t(hk) * D;
      const size_t kbase = qbase + keys;
      float q[NORM_PER], k[NORM_PER];
      float qs = 0.0f;
      float ks = 0.0f;
      for (int i = 0; i < NORM_PER; ++i) {
        const int d = NORM_PER * int(lane_u) + i;
        const float qv = float(qkv[qbase + size_t(d)]);
        const float kv = float(qkv[kbase + size_t(d)]);
        q[i] = qv;
        k[i] = kv;
        qs += qv * qv;
        ks += kv * kv;
      }
      const float qinv = metal::rsqrt(gdn_row_sum<32>(qs) + 1e-6f) * scale;
      const float kinv = metal::rsqrt(gdn_row_sum<32>(ks) + 1e-6f);
      for (int i = 0; i < NORM_PER; ++i) {
        q_s[j][NORM_PER * int(lane_u) + i] = q[i] * qinv;
        k_s[j][NORM_PER * int(lane_u) + i] = k[i] * kinv;
      }
    }
    for (int at = tid; at < rows * GROUP_ROWS; at += SG * 32) {
      const int j = at / GROUP_ROWS;
      const int v = at % GROUP_ROWS;
      const size_t vbase = size_t(begin + c0 + j) * pitch + 2 * keys + size_t(hv) * D;
      v_s[j][v] = float(qkv[vbase + size_t(group_base + v)]);
    }
    if (tid < rows) {
      const size_t fused = size_t(begin + c0 + tid) * 2 * size_t(v_heads) + size_t(hv);
      decay_s[tid] = metal::exp(gates[fused]);
      beta_s[tid] = gates[fused + size_t(v_heads)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int j = 0; j < rows; ++j) {
      const int t = c0 + j;
      float q[PER], k[PER], vals[VROWS], mem[VROWS];
      for (int i = 0; i < PER; ++i) {
        q[i] = q_s[j][PER * lane + i];
        k[i] = k_s[j][PER * lane + i];
      }
      for (int v = 0; v < VROWS; ++v) {
        vals[v] = v_s[j][my_rows + v];
      }
      const float beta = beta_s[j];
      gdn_scan_memory<LANES, VROWS, PER>(st, k, decay_s[j], mem);
      for (int v = 0; v < VROWS; ++v) {
        mem[v] = gdn_row_sum<LANES>(mem[v]);
      }
      if (t < rep) {
        for (int v = 0; v < VROWS; ++v) {
          const float delta = (vals[v] - mem[v]) * beta;
          for (int i = 0; i < PER; ++i) {
            st[v][i] += k[i] * delta;
          }
        }
      } else {
        float out[VROWS];
        gdn_scan_update<LANES, VROWS, PER>(st, k, q, vals, mem, beta, out);
        const size_t row_out = (size_t(own + t) * size_t(v_heads) + size_t(hv)) * D;
        for (int v = 0; v < VROWS; ++v) {
          out[v] = gdn_row_sum<LANES>(out[v]);
          if (lane == 0) {
            y[row_out + size_t(dv_base + v)] = out[v];
          }
        }
      }
      if (t + 1 == keep) {
        gdn_store<VROWS, PER>(cells, st, lane, D);
      }
      if (t + 1 == snap_at) {
        gdn_store<VROWS, PER>(
            gdn_cells<VROWS, PER>(rstate, snap[PIE_RS_SNAPSHOTS + snapped], hv, v_heads, dv_base,
                                  D, D),
            st, lane, D);
        ++snapped;
        snap_at = snapped < PIE_RS_SNAPSHOTS ? snap[snapped] : 0;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

#define PIE_GDN_VERIFY(lanes, sg, vrows)                                       \
  template [[host_name("gated_delta_verify_bfloat16_l_" #lanes "_s_" #sg       \
                       "_v_" #vrows)]]                                         \
  [[kernel]] void gated_delta_verify<bfloat, lanes, sg, vrows>(                \
      const device bfloat*, const device int*, const device int*,              \
      const device int*, const device int*, const constant int&,               \
      const device float*, device float*, device float*,                       \
      const constant int&, const constant int&, const device int*,             \
      const constant int&, uint3, uint, uint);

PIE_GDN_VERIFY(32, 4, 4)
PIE_GDN_VERIFY(32, 8, 4)
PIE_GDN_VERIFY(32, 8, 2)
PIE_GDN_VERIFY(32, 4, 8)
PIE_GDN_VERIFY(8, 4, 1)
PIE_GDN_VERIFY(8, 4, 2)
PIE_GDN_VERIFY(8, 8, 1)
PIE_GDN_VERIFY(16, 4, 2)
PIE_GDN_VERIFY(16, 4, 1)

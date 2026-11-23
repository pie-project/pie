

#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

#include "../prim/state_map.h"
#include "../prim/kv_page.h"

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

template <int VROWS, int PER, typename I>
inline device float* gdn_cells(
    device float* rstate, const device ulong* map, const device I* ids, int hv, int v_heads,
    int dv_base, int dk, int dv) {
  (void)v_heads;
  return rstate + pie_state_atom(map, ids, uint(hv), ulong(dv) * ulong(dk)) +
         size_t(dv_base) * size_t(dk);
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
    const device ulong* map     [[buffer(12)]],
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
      gdn_cells<VROWS, PER>(rstate, map, slots + size_t(begin) * pie_state_pieces(map), hv,
                            v_heads, dv_base, dk, dv);
  float st[VROWS][PER];
  gdn_load<VROWS, PER>(st, cells, lane, dk);
  // The rows whose state lands in a slot of its own, ascending; 0 ends them.
  const device int* snap =
      snap_lane0 >= 0 ? snaps + size_t(snap_lane0 + n) * pie_state_snap_row(map) * PIE_RS_SNAPSHOTS : nullptr;
  int snapped = 0;
  int snap_at = snap != nullptr ? begin + snap[0] : -1;
  if constexpr (LANES != 32) {
    // A lane of 16 keys or more has no registers to spare for the next row.
    for (int t = begin; t < end; ++t) {
      gdn_scan_token<T, LANES, VROWS, PER>(
          qkv, gates, y, st, t, lane, dv_base, hv, hk, k_heads, v_heads, dk, dv);
      if (t + 1 == snap_at) {
        gdn_store<VROWS, PER>(
            gdn_cells<VROWS, PER>(rstate, map, snap + PIE_RS_SNAPSHOTS + snapped * int(pie_state_pieces(map)), hv, v_heads, dv_base,
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
          gdn_cells<VROWS, PER>(rstate, map, snap + PIE_RS_SNAPSHOTS + snapped * int(pie_state_pieces(map)), hv, v_heads, dv_base,
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
      const constant int&, const device int*, const constant int&,           \
      const device ulong*, uint3);

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
    const device ulong* map     [[buffer(9)]],
    uint3 pos [[thread_position_in_grid]]) {
  const int lane = int(pos.x) % LANES;
  const int dv_base = int(pos.y) * VROWS;
  const int hv = int(pos.z) % v_heads;
  const int n = int(pos.z) / v_heads;
  const int dk = k_dim;
  const int dv = v_dim;
  const int hk = hv / (v_heads / k_heads);

  device float* cells =
      gdn_cells<VROWS, PER>(rstate, map, slots + size_t(n) * pie_state_pieces(map), hv,
                            v_heads, dv_base, dk, dv);
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
      const constant int&, const device ulong*, uint3);

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
    const device ulong* map     [[buffer(15)]],
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
  const device int* ids = slots + size_t(lane0 + r) * pie_state_pieces(map);
  if (ids[0] < 0) {
    return;
  }
  const int keep = min(commit[lane0 + r], span);
  const int dk = k_dim;
  const int dv = v_dim;
  const int hk = hv / (v_heads / k_heads);

  device float* cells = gdn_cells<VROWS, PER>(rstate, map, ids, hv, v_heads, dv_base, dk, dv);
  float st[VROWS][PER];
  gdn_load<VROWS, PER>(st, cells, lane, dk);
  // A lane that snapshots folds its own rows, none replayed: the state as of
  // each of its snapshot rows lands in that row's slot.
  const device int* snap =
      snap_lane0 >= 0 ? snaps + size_t(snap_lane0 + r) * pie_state_snap_row(map) * PIE_RS_SNAPSHOTS : nullptr;
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
          gdn_cells<VROWS, PER>(rstate, map, snap + PIE_RS_SNAPSHOTS + snapped * int(pie_state_pieces(map)), hv, v_heads, dv_base,
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
      const constant int&, const device int*, const constant int&,             \
      const device ulong*, uint3);

// The rows of a committed scan one pass stages in threadgroup memory.
constant constexpr int kVerifyChunk = 16;

// A lane's row of a recurrent buffer's page table is the load's width in
// ints, which the dispatch passes (`paged`, or `page_ints`).
// Row `e` of a lane's extended rows in its buffer's plane, `slab` the
// plane's first row in the buffer's first slot: `pages` is the lane's row of
// the page table, the in-page token of its first extended row, then the
// slab slots of the pages its rows touch.
inline device uchar* gdn_buffered_row(
    device uchar* slab, const device int* pages, int e, uint page_tokens, uint page_bytes,
    uint row_bytes) {
  const uint at = uint(pages[0] + e);
  return slab + pie_page_at(uint(pages[1 + at / page_tokens]), page_bytes, 1) +
         size_t(at % page_tokens) * row_bytes;
}

// Where row `t` of a lane's extended rows lands a corrected plane: through
// the page table in its buffer page, or (no page table) at its own row
// `own_row` of the plane's scratch, which the fire then scatters into the
// buffer as it does the planes it gathers.
inline device uchar* gdn_corrected_row(
    device uchar* slab, const device int* pages, int t, int own_row, uint page_tokens,
    uint page_bytes, uint row_bytes) {
  return pages != nullptr ? gdn_buffered_row(slab, pages, t, page_tokens, page_bytes, row_bytes)
                          : slab + size_t(own_row) * row_bytes;
}

// `gdn_row_sum<32>` of each of `V` values, with fewer shuffles: each xor step
// of the first `log2(V)` halves the values a lane carries, keeping the half
// its lane bit names and taking its partner's of that half, so a lane ends
// with one value's sum after the steps every value's tree took. Each sum is
// the tree's, operand for operand. Lane `l` holds value `gdn_rows_index`'s.
template <int V>
inline float gdn_rows_sum(thread float (&x)[V], uint lane) {
  float a[V];
  for (int v = 0; v < V; ++v) {
    a[v] = x[v];
  }
  int count = V;
  ushort mask = 1;
  for (; count > 1; count /= 2, mask *= 2) {
    const bool upper = (lane & mask) != 0;
    const int split = count / 2;
    for (int i = 0; i < split; ++i) {
      const float keep = upper ? a[split + i] : a[i];
      const float send = upper ? a[i] : a[split + i];
      a[i] = keep + simd_shuffle_xor(send, mask);
    }
  }
  float sum = a[0];
  for (; mask < 32; mask *= 2) {
    sum += simd_shuffle_xor(sum, mask);
  }
  return sum;
}

// The value whose sum lane `lane` holds after `gdn_rows_sum<V>`, and a lane
// holding value `v`'s.
template <int V>
inline int gdn_rows_index(uint lane) {
  int index = 0;
  for (int split = V / 2, bit = 1; split > 0; split /= 2, bit *= 2) {
    index += (lane & uint(bit)) != 0 ? split : 0;
  }
  return index;
}

template <int V>
inline ushort gdn_rows_lane(int v) {
  ushort lane = 0;
  for (int split = V / 2, bit = 1; split > 0; split /= 2, bit *= 2) {
    if (v >= split) {
      lane |= ushort(bit);
      v -= split;
    }
  }
  return lane;
}

// A row's `[b | a]` gates of head `hv`, as `gdn_prep` derives them:
// `ba` the extended rows, or through a page table the window's own rows,
// the replayed rows read from the lane's buffer and the window's own landed
// there for the next fire.
template <typename T>
inline float2 gdn_verify_gates(
    const device T* ba, const device float* a_log, const device T* dt_bias, int hv, int v_heads,
    int t, int begin, int own, int rep, bool lands, device uchar* slab, const device int* pages,
    uint page_tokens, uint page_bytes, uint row_bytes) {
  const size_t vh = size_t(v_heads);
  float bv, av;
  if (pages == nullptr) {
    const size_t fused = size_t(begin + t) * 2 * vh;
    bv = float(ba[fused + size_t(hv)]);
    av = float(ba[fused + vh + size_t(hv)]);
  } else if (t < rep) {
    const device T* row = reinterpret_cast<const device T*>(
        gdn_buffered_row(slab, pages, t, page_tokens, page_bytes, row_bytes));
    bv = float(row[hv]);
    av = float(row[vh + size_t(hv)]);
  } else {
    const size_t fused = size_t(own + t) * 2 * vh;
    const T b = ba[fused + size_t(hv)];
    const T a = ba[fused + vh + size_t(hv)];
    bv = float(b);
    av = float(a);
    if (lands && pages[0] >= 0) {
      device T* row = reinterpret_cast<device T*>(
          gdn_buffered_row(slab, pages, t, page_tokens, page_bytes, row_bytes));
      row[hv] = b;
      row[vh + size_t(hv)] = a;
    }
  }
  const float z = av + float(dt_bias[hv]);
  const float sp = (z > 20.0f) ? z : metal::log(1.0f + metal::fast::exp(z));
  return float2(-metal::fast::exp(a_log[hv]) * sp, 1.0f / (1.0f + metal::fast::exp(-bv)));
}

// A committed scan over 128-wide heads, its gates derived from the `[b | a]`
// projection as it stages them, whose threadgroup holds `SG` simdgroups
// of `32 / LANES` value-row groups, `VROWS` rows each. Each pass stages up to
// `kVerifyChunk` rows' normed q and k, the group's value rows, decay and beta
// in threadgroup memory, every row's loads issued side by side, so a row of
// the walk reads no device memory; `gated_delta_scan_committed` loaded each
// row's inputs behind the row before, so a verify walk of a dozen rows
// waited on a dozen loads in turn. A replayed row only folds into the
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
    const device T* ba          [[buffer(6)]],
    const device float* a_log   [[buffer(7)]],
    const device T* dt_bias     [[buffer(8)]],
    device float* rstate        [[buffer(9)]],
    device float* y             [[buffer(10)]],
    device float* gates         [[buffer(11)]],
    const constant int& k_heads [[buffer(12)]],
    const constant int& v_heads [[buffer(13)]],
    const device int* snaps     [[buffer(14)]],
    const constant int& snap_lane0 [[buffer(15)]],
    device uchar* slab          [[buffer(16)]],
    const device int* page_table [[buffer(17)]],
    const constant uint& page_tokens [[buffer(18)]],
    const constant uint& page_bytes [[buffer(19)]],
    const constant uint& row_bytes [[buffer(20)]],
    const constant int& paged   [[buffer(21)]],
    const device ulong* map     [[buffer(22)]],
    // The corrected planes (`rs_corrected`): each own row's raw keys (bf16)
    // and its corrected value `u` (f32) land in the lane's buffer, where a
    // later decode step reads the row as a linear attention.
    device uchar* key_slab      [[buffer(23)]],
    const constant uint& key_row_bytes [[buffer(24)]],
    device uchar* u_slab        [[buffer(25)]],
    const constant uint& u_row_bytes [[buffer(26)]],
    const constant int& corrected [[buffer(27)]],
    const device int* ext_replay [[buffer(28)]],
    // Each own row's per-head scalars (f32): `v_heads` log-gates, then
    // `v_heads` key-norm inverses, as the scan below derives them.
    device uchar* scalar_slab   [[buffer(29)]],
    const constant uint& scalar_row_bytes [[buffer(30)]],
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
  // A lane's extended rows: its carried rows where it takes them
  // (`ext_replay`; a compact decode lane takes none, reading them from its
  // buffer), then its own. `begin + t` is row `t` of them.
  for (int j = 0; j < r; ++j) {
    begin += ext_replay[lane0 + j];
  }
  begin -= replay[lane0 + r] - ext_replay[lane0 + r];
  // A compact decode lane is the carried scan's, dispatched beside this
  // one: it re-steps nothing here. Uniform over the threadgroup.
  if (ext_replay[lane0 + r] < replay[lane0 + r]) {
    return;
  }
  const int rep = replay[lane0 + r];
  const int span = (indptr[r + 1] - indptr[r]) + rep;
  const device int* ids = slots + size_t(lane0 + r) * pie_state_pieces(map);
  // Uniform over the threadgroup, so its barriers below are too.
  if (span <= 0 || ids[0] < 0) {
    return;
  }
  const int own = indptr[r] - rep;
  const int keep = min(commit[lane0 + r], span);
  const device int* pages =
      paged != 0 ? page_table + size_t(lane0 + r) * uint(paged) : nullptr;
  const int hk = hv / (v_heads / k_heads);
  // Own rows land their corrected planes where the lane has a buffer page.
  const bool lands_corrected = corrected != 0 && (pages == nullptr || pages[0] >= 0);
  // A key head's raw keys land once: by its first value head's first group.
  const bool lands_keys =
      lands_corrected && group.y == 0 && hv % (v_heads / k_heads) == 0;
  const size_t keys = size_t(k_heads) * D;
  const size_t pitch = 2 * keys + size_t(v_heads) * D;
  const float scale = 1.0f / metal::sqrt(float(D));

  device float* cells = gdn_cells<VROWS, PER>(rstate, map, ids, hv, v_heads, dv_base, D, D);
  float st[VROWS][PER];
  gdn_load<VROWS, PER>(st, cells, lane, D);
  const device int* snap =
      snap_lane0 >= 0 ? snaps + size_t(snap_lane0 + r) * pie_state_snap_row(map) * PIE_RS_SNAPSHOTS : nullptr;
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
      if (lands_corrected && group.y == 0 && lane_u == 0 && c0 + j >= rep) {
        reinterpret_cast<device float*>(gdn_corrected_row(
            scalar_slab, pages, c0 + j, own + c0 + j, page_tokens, page_bytes,
            scalar_row_bytes))[size_t(v_heads) + size_t(hv)] = kinv;
      }
      if (lands_keys && c0 + j >= rep) {
        device T* row = reinterpret_cast<device T*>(gdn_corrected_row(
            key_slab, pages, c0 + j, own + c0 + j, page_tokens, page_bytes, key_row_bytes));
        for (int i = 0; i < NORM_PER; ++i) {
          const int d = NORM_PER * int(lane_u) + i;
          row[size_t(hk) * D + size_t(d)] = qkv[kbase + size_t(d)];
        }
      }
    }
    for (int at = tid; at < rows * GROUP_ROWS; at += SG * 32) {
      const int j = at / GROUP_ROWS;
      const int v = at % GROUP_ROWS;
      const size_t vbase = size_t(begin + c0 + j) * pitch + 2 * keys + size_t(hv) * D;
      v_s[j][v] = float(qkv[vbase + size_t(group_base + v)]);
    }
    // The row's gates, as `gdn_prep` derives them; the first
    // group of the head lands a row of the window's own in `gates`.
    if (tid < rows) {
      const int t = c0 + tid;
      const float2 g = gdn_verify_gates<T>(ba, a_log, dt_bias, hv, v_heads, t, begin, own, rep,
                                           group.y == 0, slab, pages, page_tokens, page_bytes,
                                           row_bytes);
      decay_s[tid] = metal::exp(g.x);
      beta_s[tid] = g.y;
      if (group.y == 0 && t >= rep) {
        const size_t at = size_t(own + t) * 2 * size_t(v_heads) + size_t(hv);
        gates[at] = g.x;
        gates[at + size_t(v_heads)] = g.y;
        if (lands_corrected) {
          reinterpret_cast<device float*>(gdn_corrected_row(
              scalar_slab, pages, t, own + t, page_tokens, page_bytes, scalar_row_bytes))[hv] = g.x;
        }
      }
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
      constexpr bool TRANSPOSED = LANES == 32 && VROWS > 1;
      if constexpr (TRANSPOSED) {
        const float sum = gdn_rows_sum<VROWS>(mem, lane_u);
        for (int v = 0; v < VROWS; ++v) {
          mem[v] = simd_shuffle(sum, gdn_rows_lane<VROWS>(v));
        }
      } else {
        for (int v = 0; v < VROWS; ++v) {
          mem[v] = gdn_row_sum<LANES>(mem[v]);
        }
      }
      if (t < rep) {
        for (int v = 0; v < VROWS; ++v) {
          const float delta = (vals[v] - mem[v]) * beta;
          for (int i = 0; i < PER; ++i) {
            st[v][i] += k[i] * delta;
          }
        }
      } else {
        if (lands_corrected && lane == 0) {
          // `u = beta (v - g S^T k)`: what the update below adds along `k`.
          device float* row = reinterpret_cast<device float*>(gdn_corrected_row(
              u_slab, pages, t, own + t, page_tokens, page_bytes, u_row_bytes));
          for (int v = 0; v < VROWS; ++v) {
            row[size_t(hv) * D + size_t(dv_base + v)] = (vals[v] - mem[v]) * beta;
          }
        }
        float out[VROWS];
        gdn_scan_update<LANES, VROWS, PER>(st, k, q, vals, mem, beta, out);
        const size_t row_out = (size_t(own + t) * size_t(v_heads) + size_t(hv)) * D;
        if constexpr (TRANSPOSED) {
          const float sum = gdn_rows_sum<VROWS>(out, lane_u);
          if (lane_u < uint(VROWS)) {
            y[row_out + size_t(dv_base + gdn_rows_index<VROWS>(lane_u))] = sum;
          }
        } else {
          for (int v = 0; v < VROWS; ++v) {
            out[v] = gdn_row_sum<LANES>(out[v]);
            if (lane == 0) {
              y[row_out + size_t(dv_base + v)] = out[v];
            }
          }
        }
      }
      if (t + 1 == keep) {
        gdn_store<VROWS, PER>(cells, st, lane, D);
      }
      if (t + 1 == snap_at) {
        gdn_store<VROWS, PER>(
            gdn_cells<VROWS, PER>(rstate, map, snap + PIE_RS_SNAPSHOTS + snapped * int(pie_state_pieces(map)), hv, v_heads, dv_base,
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
      const device bfloat*, const device float*, const device bfloat*,         \
      device float*, device float*, device float*,                             \
      const constant int&, const constant int&, const device int*,             \
      const constant int&, device uchar*, const device int*,                   \
      const constant uint&, const constant uint&, const constant uint&,        \
      const constant int&, const device ulong*, device uchar*,                 \
      const constant uint&, device uchar*, const constant uint&,               \
      const constant int&, const device int*, device uchar*, const constant uint&, uint3, uint, \
      uint);

PIE_GDN_VERIFY(32, 8, 2)
PIE_GDN_VERIFY(32, 4, 4)
PIE_GDN_VERIFY(32, 8, 4)

// A decode step over a lane's buffered rows read as a linear attention
// (new-memory/designs/lagged-fold.md): one row of the window's own a lane,
// its `rep` carried rows never re-stepped. With the committed state S0, the
// carried rows' normed keys k_s and corrected values u_s (their buffer's
// corrected planes) and their log-gates, the own row's
//   a = G0 S0^T k + sum_s G_s (k_s . k) u_s   (= g_t S_{t-1}^T k)
//   o = G0 S0^T q + sum_s G_s (k_s . q) u_s + (k . q) u,  u = beta (v - a)
// with G_s = exp(sum of the log-gates past row s, the own row's with them)
// in f32, G0 = exp(all of them). The fold is the rank-1 update of each
// committed row from its stored k and u (S <- g S + k u^T), then the own
// row's when it commits; the state is stored at the commit length only.
// The own row's raw keys, u and `[b | a]` land in its buffer pages. Its
// threadgroup is the verify scan's: `SG` simdgroups of `VROWS` value rows,
// 32 lanes of four key columns each.
// The per-head table the carried scan reads: for each lane's carried row
// `t` and key head `hk`, the row's key norm and its normed key's dots with
// the own row's normed q and k, as float4(kinv, k.q, k.k, 0) at
// `table[((lane) * k_heads + hk) * table_rows + t]`. One simdgroup a (lane,
// key head, row): every value head and row group of the scan reads a row's
// reductions made once.
template <typename T>
[[kernel]] void gated_delta_carried_dots(
    const device T* qkv         [[buffer(0)]],
    const device int* indptr    [[buffer(1)]],
    const device int* replay    [[buffer(2)]],
    const constant int& lane0   [[buffer(3)]],
    device uchar* key_slab      [[buffer(4)]],
    const device int* page_table [[buffer(5)]],
    const constant uint& page_tokens [[buffer(6)]],
    const constant uint& page_bytes [[buffer(7)]],
    const constant uint& key_row_bytes [[buffer(8)]],
    const constant uint& page_ints [[buffer(9)]],
    const constant int& k_heads [[buffer(10)]],
    const constant int& v_heads [[buffer(11)]],
    device float4* table        [[buffer(12)]],
    const constant uint& table_rows [[buffer(13)]],
    const device int* ext_replay [[buffer(14)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane_u [[thread_index_in_simdgroup]]) {
  constexpr int D = 128;
  constexpr int PER = D / 32;
  const int lane = int(lane_u);
  const int t = int(group.y);
  const int hk = int(group.z) % k_heads;
  const int r = int(group.z) / k_heads;
  const int rep = replay[lane0 + r];
  if (t >= rep || uint(t) >= table_rows || indptr[r + 1] - indptr[r] != 1) {
    return;
  }
  int begin = indptr[r];
  // A lane's extended rows: its carried rows where it takes them
  // (`ext_replay`; a compact decode lane takes none, reading them from its
  // buffer), then its own. `begin + t` is row `t` of them.
  for (int j = 0; j < r; ++j) {
    begin += ext_replay[lane0 + j];
  }
  begin -= replay[lane0 + r] - ext_replay[lane0 + r];
  const device int* pages = page_table + size_t(lane0 + r) * page_ints;
  const size_t keys = size_t(k_heads) * D;
  const size_t pitch = 2 * keys + size_t(v_heads) * D;
  const size_t qbase = size_t(begin + rep) * pitch + size_t(hk) * D;
  const size_t kbase = qbase + keys;
  const float scale = 1.0f / metal::sqrt(float(D));
  float qn[PER], kn[PER], kr[PER];
  float qs = 0.0f, ks = 0.0f, rs = 0.0f;
  const device T* krow = reinterpret_cast<const device T*>(
      gdn_buffered_row(key_slab, pages, t, page_tokens, page_bytes, key_row_bytes));
  for (int i = 0; i < PER; ++i) {
    const int d = PER * lane + i;
    qn[i] = float(qkv[qbase + size_t(d)]);
    kn[i] = float(qkv[kbase + size_t(d)]);
    kr[i] = float(krow[size_t(hk) * D + size_t(d)]);
    qs += qn[i] * qn[i];
    ks += kn[i] * kn[i];
    rs += kr[i] * kr[i];
  }
  const float qinv = metal::rsqrt(gdn_row_sum<32>(qs) + 1e-6f) * scale;
  const float kinv = metal::rsqrt(gdn_row_sum<32>(ks) + 1e-6f);
  const float rinv = metal::rsqrt(gdn_row_sum<32>(rs) + 1e-6f);
  float dq = 0.0f, dk = 0.0f;
  for (int i = 0; i < PER; ++i) {
    const float k = kr[i] * rinv;
    dq += k * qn[i] * qinv;
    dk += k * kn[i] * kinv;
  }
  dq = gdn_row_sum<32>(dq);
  dk = gdn_row_sum<32>(dk);
  if (lane == 0) {
    table[(size_t(lane0 + r) * size_t(k_heads) + size_t(hk)) * table_rows + size_t(t)] =
        float4(rinv, dq, dk, 0.0f);
  }
}

template [[host_name("gated_delta_carried_dots_bfloat16")]]
[[kernel]] void gated_delta_carried_dots<bfloat>(
    const device bfloat*, const device int*, const device int*, const constant int&,
    device uchar*, const device int*, const constant uint&, const constant uint&,
    const constant uint&, const constant uint&, const constant int&, const constant int&,
    device float4*, const constant uint&, const device int*, uint3, uint);

template <typename T, int SG, int VROWS>
[[kernel]] void gated_delta_carried(
    const device T* qkv         [[buffer(0)]],
    const device int* indptr    [[buffer(1)]],
    const device int* replay    [[buffer(2)]],
    const device int* commit    [[buffer(3)]],
    const device int* slots     [[buffer(4)]],
    const constant int& lane0   [[buffer(5)]],
    const device T* ba          [[buffer(6)]],
    const device float* a_log   [[buffer(7)]],
    const device T* dt_bias     [[buffer(8)]],
    device float* rstate        [[buffer(9)]],
    device float* y             [[buffer(10)]],
    device float* gates         [[buffer(11)]],
    const constant int& k_heads [[buffer(12)]],
    const constant int& v_heads [[buffer(13)]],
    device uchar* ba_slab       [[buffer(14)]],
    const device int* page_table [[buffer(15)]],
    const constant uint& page_tokens [[buffer(16)]],
    const constant uint& page_bytes [[buffer(17)]],
    const constant uint& ba_row_bytes [[buffer(18)]],
    device uchar* key_slab      [[buffer(19)]],
    const constant uint& key_row_bytes [[buffer(20)]],
    device uchar* u_slab        [[buffer(21)]],
    const constant uint& u_row_bytes [[buffer(22)]],
    const device ulong* map     [[buffer(23)]],
    const constant uint& page_ints [[buffer(24)]],
    const device float4* table  [[buffer(25)]],
    const constant uint& table_rows [[buffer(26)]],
    const device int* ext_replay [[buffer(27)]],
    // The rows' per-head scalars (f32): `v_heads` log-gates, then `v_heads`
    // key-norm inverses. The own row lands its; a short carry reads its
    // carried rows' in place of their `[b | a]` softplus and key norm.
    device uchar* scalar_slab   [[buffer(28)]],
    const constant uint& scalar_row_bytes [[buffer(29)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane_u [[thread_index_in_simdgroup]],
    uint sg_u [[simdgroup_index_in_threadgroup]]) {
  constexpr int D = 128;
  constexpr int PER = D / 32;
  constexpr int GROUP_ROWS = SG * VROWS;
  const int tid = int(sg_u) * 32 + int(lane_u);
  const int lane = int(lane_u);
  const int hv = int(group.z) % v_heads;
  const int r = int(group.z) / v_heads;
  const int group_base = int(group.y) * GROUP_ROWS;
  const int my_rows = int(sg_u) * VROWS;
  const int dv_base = group_base + my_rows;

  threadgroup float q_s[D];
  threadgroup float k_s[D];
  // The carried rows' log-gates and weights, a chunk at a time, so a lane
  // carries any number of rows (a beam in a parents pass carries every row
  // since its fold): `total_s` is every gate's sum (G, the state's weight's
  // log), `part_s` a simdgroup's share of it.
  threadgroup float glog_c[kVerifyChunk];
  threadgroup float weight_c[kVerifyChunk];
  threadgroup float part_s[SG];
  threadgroup float total_s;
  threadgroup float kn_s[kVerifyChunk][D];
  threadgroup float u_s[kVerifyChunk][GROUP_ROWS];
  threadgroup float dq_s[kVerifyChunk];
  threadgroup float kinv_c[kVerifyChunk];
  threadgroup float dk_s[kVerifyChunk];
  threadgroup float own_s[4];

  int begin = indptr[r];
  // A lane's extended rows: its carried rows where it takes them
  // (`ext_replay`; a compact decode lane takes none, reading them from its
  // buffer), then its own. `begin + t` is row `t` of them.
  for (int j = 0; j < r; ++j) {
    begin += ext_replay[lane0 + j];
  }
  begin -= replay[lane0 + r] - ext_replay[lane0 + r];
  const int rep = replay[lane0 + r];
  const device int* ids = slots + size_t(lane0 + r) * pie_state_pieces(map);
  // One own row a lane (the dispatch's rule); uniform over the threadgroup.
  if (indptr[r + 1] - indptr[r] != 1 || ids[0] < 0) {
    return;
  }
  const int own = indptr[r] - rep;
  const int t_own = rep;
  const int keep = min(commit[lane0 + r], rep + 1);
  const device int* pages = page_table + size_t(lane0 + r) * page_ints;
  // The own row lands in the buffer for the fires after this one, unless
  // this fire folds it: a fold the runtime states advances the set as it
  // prepares the fire, and one the device places is frozen ahead
  // (`frozen_ahead`), so no later fire replays it.
  const bool lands = pages[0] >= 0 && keep < rep + 1;
  const int hk = hv / (v_heads / k_heads);
  const size_t keys = size_t(k_heads) * D;
  const size_t pitch = 2 * keys + size_t(v_heads) * D;
  const float scale = 1.0f / metal::sqrt(float(D));
  const size_t vh = size_t(v_heads);

  // The own row: normed q and k (simdgroup 0), its gates (thread 32).
  const size_t row = size_t(begin + t_own) * pitch;
  const size_t qbase = row + size_t(hk) * D;
  const size_t kbase = qbase + keys;
  if (sg_u == 0) {
    float q[PER], k[PER];
    float qs = 0.0f, ks = 0.0f;
    // Four of a lane's values a load (PER is four).
    const float4 q4 = float4(*reinterpret_cast<const device vec<T, 4>*>(qkv + qbase + size_t(PER * lane)));
    const float4 k4 = float4(*reinterpret_cast<const device vec<T, 4>*>(qkv + kbase + size_t(PER * lane)));
    for (int i = 0; i < PER; ++i) {
      q[i] = q4[i];
      k[i] = k4[i];
      qs += q[i] * q[i];
      ks += k[i] * k[i];
    }
    const float qinv = metal::rsqrt(gdn_row_sum<32>(qs) + 1e-6f) * scale;
    const float kinv = metal::rsqrt(gdn_row_sum<32>(ks) + 1e-6f);
    float kq = 0.0f;
    for (int i = 0; i < PER; ++i) {
      q_s[PER * lane + i] = q[i] * qinv;
      k_s[PER * lane + i] = k[i] * kinv;
      kq += q[i] * qinv * k[i] * kinv;
    }
    kq = gdn_row_sum<32>(kq);
    if (lane == 0) {
      own_s[0] = kq;
      if (lands && group.y == 0) {
        reinterpret_cast<device float*>(gdn_buffered_row(
            scalar_slab, pages, t_own, page_tokens, page_bytes, scalar_row_bytes))[vh + size_t(hv)] =
            kinv;
      }
    }
    // A key head's raw keys land once: by its first value head's first group.
    if (lands && group.y == 0 && hv % (v_heads / k_heads) == 0) {
      device T* krow = reinterpret_cast<device T*>(
          gdn_buffered_row(key_slab, pages, t_own, page_tokens, page_bytes, key_row_bytes));
      for (int i = 0; i < PER; ++i) {
        const int d = PER * lane + i;
        krow[size_t(hk) * D + size_t(d)] = qkv[kbase + size_t(d)];
      }
    }
  } else if (tid == 32) {
    const float2 g = gdn_verify_gates<T>(ba, a_log, dt_bias, hv, v_heads, t_own, begin, own, rep,
                                         group.y == 0 && lands, ba_slab, pages, page_tokens,
                                         page_bytes, ba_row_bytes);
    own_s[1] = g.x;
    own_s[2] = g.y;
    if (group.y == 0) {
      const size_t at = size_t(own + t_own) * 2 * vh + size_t(hv);
      gates[at] = g.x;
      gates[at + vh] = g.y;
      if (lands) {
        reinterpret_cast<device float*>(gdn_buffered_row(
            scalar_slab, pages, t_own, page_tokens, page_bytes, scalar_row_bytes))[hv] = g.x;
      }
    }
  }
  // A carried row's log-gate, from its buffered `[b | a]`.
  const float a_scale = -metal::fast::exp(a_log[hv]);
  const float bias = float(dt_bias[hv]);
  auto log_gate = [&](int t) -> float {
    const device T* brow = reinterpret_cast<const device T*>(
        gdn_buffered_row(ba_slab, pages, t, page_tokens, page_bytes, ba_row_bytes));
    const float z = float(brow[vh + size_t(hv)]) + bias;
    const float sp = (z > 20.0f) ? z : metal::log(1.0f + metal::fast::exp(z));
    return a_scale * sp;
  };
  // Every carried row's log-gate summed (each thread its rows, then each
  // simdgroup, then thread 0 with the own row's): G.
  float part = 0.0f;
  for (int t = tid; t < rep; t += SG * 32) {
    part += log_gate(t);
  }
  part = gdn_row_sum<32>(part);
  if (lane == 0) {
    part_s[sg_u] = part;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float total = own_s[1];
    for (int g = 0; g < SG; ++g) {
      total += part_s[g];
    }
    total_s = total;
  }

  // The committed state's reads of the own q and k, before any fold.
  device float* cells = gdn_cells<VROWS, PER>(rstate, map, ids, hv, v_heads, dv_base, D, D);
  float st[VROWS][PER];
  for (int v = 0; v < VROWS; ++v) {
    const float4 s4 = *reinterpret_cast<const device float4*>(cells + size_t(v) * D + size_t(PER * lane));
    for (int i = 0; i < PER; ++i) {
      st[v][i] = s4[i];
    }
  }
  float qn[PER], kn[PER];
  for (int i = 0; i < PER; ++i) {
    qn[i] = q_s[PER * lane + i];
    kn[i] = k_s[PER * lane + i];
  }
  float sa[VROWS], so[VROWS];
  for (int v = 0; v < VROWS; ++v) {
    float pa = 0.0f, po = 0.0f;
    for (int i = 0; i < PER; ++i) {
      pa += st[v][i] * kn[i];
      po += st[v][i] * qn[i];
    }
    sa[v] = gdn_row_sum<32>(pa);
    so[v] = gdn_row_sum<32>(po);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // The state's weight: every gate's. A carried row's: the gates past it,
  // exp(G - P), P its own and every earlier row's, run on by thread 0.
  const float total = total_s;
  const float g0 = metal::exp(total);
  float run = 0.0f;
  float acc_a[VROWS], acc_o[VROWS];
  for (int v = 0; v < VROWS; ++v) {
    acc_a[v] = g0 * sa[v];
    acc_o[v] = g0 * so[v];
  }

  for (int c0 = 0; c0 < rep; c0 += kVerifyChunk) {
    const int rows = min(kVerifyChunk, rep - c0);
    if (tid < rows) {
      glog_c[tid] = log_gate(c0 + tid);
    }
    if (table_rows == 0u) {
      // A short carry (no table made): each row's reductions here, a
      // simdgroup a row side by side, from the own row's staged normed q
      // and k; a row that folds stages its normed key.
      for (int j = int(sg_u); j < rows; j += SG) {
        const device T* krow = reinterpret_cast<const device T*>(gdn_buffered_row(
            key_slab, pages, c0 + j, page_tokens, page_bytes, key_row_bytes));
        const float4 kr4 = float4(*reinterpret_cast<const device vec<T, 4>*>(
            krow + size_t(hk) * D + size_t(PER * lane)));
        float rs = 0.0f;
        for (int i = 0; i < PER; ++i) {
          rs += kr4[i] * kr4[i];
        }
        const float rinv = metal::rsqrt(gdn_row_sum<32>(rs) + 1e-6f);
        float dq = 0.0f, dk = 0.0f;
        for (int i = 0; i < PER; ++i) {
          const float k = kr4[i] * rinv;
          dq += k * q_s[PER * lane + i];
          dk += k * k_s[PER * lane + i];
          if (c0 + j < keep) {
            kn_s[j][PER * lane + i] = k;
          }
        }
        dq = gdn_row_sum<32>(dq);
        dk = gdn_row_sum<32>(dk);
        if (lane == 0) {
          kinv_c[j] = rinv;
          dq_s[j] = dq;
          dk_s[j] = dk;
        }
      }
    } else {
      // Each row's reductions from the per-head table (made once a lane
      // and key head); a row that folds stages its normed key.
      const device float4* dots = table + (size_t(lane0 + r) * size_t(k_heads) + size_t(hk)) * table_rows;
      if (tid < rows) {
        const float4 kd = dots[c0 + tid];
        kinv_c[tid] = kd.x;
        dq_s[tid] = kd.y;
        dk_s[tid] = kd.z;
      }
      for (int j = int(sg_u); j < rows && c0 + j < keep; j += SG) {
        const device T* krow = reinterpret_cast<const device T*>(gdn_buffered_row(
            key_slab, pages, c0 + j, page_tokens, page_bytes, key_row_bytes));
        const float kinv = dots[c0 + j].x;
        const float4 kr4 = float4(*reinterpret_cast<const device vec<T, 4>*>(
            krow + size_t(hk) * D + size_t(PER * lane)));
        for (int i = 0; i < PER; ++i) {
          kn_s[j][PER * lane + i] = kr4[i] * kinv;
        }
      }
    }
    // Four of a row's values a thread, one load (GROUP_ROWS is a multiple
    // of four).
    for (int at = tid; at < rows * (GROUP_ROWS / 4); at += SG * 32) {
      const int j = at / (GROUP_ROWS / 4);
      const int v = (at % (GROUP_ROWS / 4)) * 4;
      const device float* urow = reinterpret_cast<const device float*>(gdn_buffered_row(
          u_slab, pages, c0 + j, page_tokens, page_bytes, u_row_bytes));
      const float4 u4 = *reinterpret_cast<const device float4*>(urow + size_t(hv) * D +
                                                               size_t(group_base + v));
      for (int i = 0; i < 4; ++i) {
        u_s[j][v + i] = u4[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The chunk's weights in parallel, a row a lane of the first
    // simdgroup: each row's prefix of log-gates within the chunk on the
    // run before it, the run moved on by the chunk's sum.
    if (sg_u == 0) {
      const float g = lane < rows ? glog_c[lane] : 0.0f;
      const float prefix = metal::simd_prefix_inclusive_sum(g);
      if (lane < rows) {
        weight_c[lane] = metal::exp(total - (run + prefix));
      }
      run += metal::simd_sum(g);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = 0; j < rows; ++j) {
      const int t = c0 + j;
      const float w = weight_c[j];
      for (int v = 0; v < VROWS; ++v) {
        const float u = u_s[j][my_rows + v];
        acc_a[v] += w * dk_s[j] * u;
        acc_o[v] += w * dq_s[j] * u;
      }
      if (t < keep) {
        const float g = metal::exp(glog_c[j]);
        for (int v = 0; v < VROWS; ++v) {
          const float u = u_s[j][my_rows + v];
          for (int i = 0; i < PER; ++i) {
            st[v][i] = g * st[v][i] + kn_s[j][PER * lane + i] * u;
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // The own row: its corrected value, readout and landing.
  const float kq = own_s[0];
  const float glog = own_s[1];
  const float beta = own_s[2];
  const size_t vbase = row + 2 * keys + size_t(hv) * D;
  const size_t row_out = (size_t(own + t_own) * vh + size_t(hv)) * D;
  device float* urow = lands ? reinterpret_cast<device float*>(gdn_buffered_row(
                                   u_slab, pages, t_own, page_tokens, page_bytes, u_row_bytes))
                             : nullptr;
  for (int v = 0; v < VROWS; ++v) {
    const float val = float(qkv[vbase + size_t(dv_base + v)]);
    const float u = beta * (val - acc_a[v]);
    const float o = acc_o[v] + kq * u;
    if (lane == 0) {
      y[row_out + size_t(dv_base + v)] = o;
      if (urow != nullptr) {
        urow[size_t(hv) * D + size_t(dv_base + v)] = u;
      }
    }
    if (keep == rep + 1) {
      const float g = metal::exp(glog);
      for (int i = 0; i < PER; ++i) {
        st[v][i] = g * st[v][i] + kn[i] * u;
      }
    }
  }
  if (keep > 0) {
    for (int v = 0; v < VROWS; ++v) {
      *reinterpret_cast<device float4*>(cells + size_t(v) * D + size_t(PER * lane)) =
          float4(st[v][0], st[v][1], st[v][2], st[v][3]);
    }
  }
}

#define PIE_GDN_CARRIED(sg, vrows)                                             \
  template [[host_name("gated_delta_carried_bfloat16_s_" #sg "_v_" #vrows)]]   \
  [[kernel]] void gated_delta_carried<bfloat, sg, vrows>(                      \
      const device bfloat*, const device int*, const device int*,              \
      const device int*, const device int*, const constant int&,               \
      const device bfloat*, const device float*, const device bfloat*,         \
      device float*, device float*, device float*,                             \
      const constant int&, const constant int&, device uchar*,                 \
      const device int*, const constant uint&, const constant uint&,           \
      const constant uint&, device uchar*, const constant uint&,               \
      device uchar*, const constant uint&, const device ulong*,                \
      const constant uint&, const device float4*, const constant uint&,        \
      const device int*, device uchar*, const constant uint&, uint3, uint, uint);

PIE_GDN_CARRIED(8, 2)
PIE_GDN_CARRIED(8, 16)

// A beam group's gated-delta fold (BEAM-PAGE-LISTS §11.3): before the
// layer's step, each group's state takes the `n` rows every live beam
// shares, from its first lane's buffered keys, `u` and `[b | a]`: the
// carried scan's commit, S <- g S + k u^T a row in turn, with each key
// normed and each gate made as that scan makes them. That lane's page table
// row starts `n` rows past them, so row `t` is its extended row `t - n`.
// `folds` holds `[lane, n]` a group: the folding groups lead it and the
// first entry of no rows ends it, each threadgroup folding every group in
// turn. One simdgroup of `VROWS` value rows of one value head a
// threadgroup.
template <typename T, int VROWS>
[[kernel]] void gated_delta_group_fold(
    const device int* folds     [[buffer(0)]],
    const constant int& lane0   [[buffer(1)]],
    const constant int& lanes   [[buffer(2)]],
    const device int* slots     [[buffer(3)]],
    const device float* a_log   [[buffer(4)]],
    const device T* dt_bias     [[buffer(5)]],
    device float* rstate        [[buffer(6)]],
    const constant int& k_heads [[buffer(7)]],
    const constant int& v_heads [[buffer(8)]],
    device uchar* ba_slab       [[buffer(9)]],
    const device int* page_table [[buffer(10)]],
    const constant uint& page_tokens [[buffer(11)]],
    const constant uint& page_bytes [[buffer(12)]],
    const constant uint& ba_row_bytes [[buffer(13)]],
    device uchar* key_slab      [[buffer(14)]],
    const constant uint& key_row_bytes [[buffer(15)]],
    device uchar* u_slab        [[buffer(16)]],
    const constant uint& u_row_bytes [[buffer(17)]],
    const device ulong* map     [[buffer(18)]],
    const constant uint& page_ints [[buffer(19)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane_u [[thread_index_in_simdgroup]]) {
  constexpr int D = 128;
  constexpr int PER = D / 32;
  const int lane = int(lane_u);
  const int hv = int(group.z);
  for (int f = 0; folds[2 * f + 1] > 0; ++f) {
  const int at_lane = folds[2 * f];
  const int n = folds[2 * f + 1];
  if (at_lane < lane0 || at_lane >= lane0 + lanes) {
    continue;
  }
  const device int* ids = slots + size_t(at_lane) * pie_state_pieces(map);
  if (ids[0] < 0) {
    continue;
  }
  const device int* pages = page_table + size_t(at_lane) * page_ints;
  const int hk = hv / (v_heads / k_heads);
  const int dv_base = int(group.y) * VROWS;
  const size_t vh = size_t(v_heads);
  const float a_scale = -metal::fast::exp(a_log[hv]);
  const float bias = float(dt_bias[hv]);
  device float* cells = gdn_cells<VROWS, PER>(rstate, map, ids, hv, v_heads, dv_base, D, D);
  float st[VROWS][PER];
  for (int v = 0; v < VROWS; ++v) {
    const float4 s4 = *reinterpret_cast<const device float4*>(cells + size_t(v) * D + size_t(PER * lane));
    for (int i = 0; i < PER; ++i) {
      st[v][i] = s4[i];
    }
  }
  for (int t = 0; t < n; ++t) {
    const int e = t - n;
    const device T* krow = reinterpret_cast<const device T*>(
        gdn_buffered_row(key_slab, pages, e, page_tokens, page_bytes, key_row_bytes));
    float kr[PER];
    float rs = 0.0f;
    for (int i = 0; i < PER; ++i) {
      kr[i] = float(krow[size_t(hk) * D + size_t(PER * lane + i)]);
      rs += kr[i] * kr[i];
    }
    const float kinv = metal::rsqrt(gdn_row_sum<32>(rs) + 1e-6f);
    const device T* brow = reinterpret_cast<const device T*>(
        gdn_buffered_row(ba_slab, pages, e, page_tokens, page_bytes, ba_row_bytes));
    const float z = float(brow[vh + size_t(hv)]) + bias;
    const float sp = (z > 20.0f) ? z : metal::log(1.0f + metal::fast::exp(z));
    const float g = metal::exp(a_scale * sp);
    const device float* urow = reinterpret_cast<const device float*>(
        gdn_buffered_row(u_slab, pages, e, page_tokens, page_bytes, u_row_bytes));
    for (int v = 0; v < VROWS; ++v) {
      const float u = urow[size_t(hv) * D + size_t(dv_base + v)];
      for (int i = 0; i < PER; ++i) {
        st[v][i] = g * st[v][i] + kr[i] * kinv * u;
      }
    }
  }
  for (int v = 0; v < VROWS; ++v) {
    *reinterpret_cast<device float4*>(cells + size_t(v) * D + size_t(PER * lane)) =
        float4(st[v][0], st[v][1], st[v][2], st[v][3]);
  }
  }
}

template [[host_name("gated_delta_group_fold_bfloat16")]]
[[kernel]] void gated_delta_group_fold<bfloat, 8>(
    const device int*, const constant int&, const constant int&, const device int*,
    const device float*, const device bfloat*, device float*, const constant int&,
    const constant int&, device uchar*, const device int*, const constant uint&,
    const constant uint&, const constant uint&, device uchar*, const constant uint&,
    device uchar*, const constant uint&, const device ulong*, const constant uint&, uint3,
    uint);

// `gated_delta_carried` for a carry of one chunk (up to `kVerifyChunk`
// rows, a decode at its run-ahead lag): the fused step's tiling, a
// simdgroup holding 16 value rows of the head's state in registers, loaded
// and stored once (new-memory/designs/carried-kernel.md). Each carried
// row's scalars (its key's norm, the normed key's dots with the own row's
// q and k, its log-gate, where its `u` lies) are made once a threadgroup,
// row s on simdgroup s mod SG, and shared through threadgroup memory behind
// one barrier, with the normed key of each row the lane folds; each
// simdgroup then reads its own `u` columns of each row from the buffer.
// A head's 128 value rows lie over `128 / (16 SG)` threadgroups.
template <typename T, int SG, int VROWS>
[[kernel]] void gated_delta_carried_short(
    const device T* qkv         [[buffer(0)]],
    const device int* indptr    [[buffer(1)]],
    const device int* replay    [[buffer(2)]],
    const device int* commit    [[buffer(3)]],
    const device int* slots     [[buffer(4)]],
    const constant int& lane0   [[buffer(5)]],
    const device T* ba          [[buffer(6)]],
    const device float* a_log   [[buffer(7)]],
    const device T* dt_bias     [[buffer(8)]],
    device float* rstate        [[buffer(9)]],
    device float* y             [[buffer(10)]],
    device float* gates         [[buffer(11)]],
    const constant int& k_heads [[buffer(12)]],
    const constant int& v_heads [[buffer(13)]],
    device uchar* ba_slab       [[buffer(14)]],
    const device int* page_table [[buffer(15)]],
    const constant uint& page_tokens [[buffer(16)]],
    const constant uint& page_bytes [[buffer(17)]],
    const constant uint& ba_row_bytes [[buffer(18)]],
    device uchar* key_slab      [[buffer(19)]],
    const constant uint& key_row_bytes [[buffer(20)]],
    device uchar* u_slab        [[buffer(21)]],
    const constant uint& u_row_bytes [[buffer(22)]],
    const device ulong* map     [[buffer(23)]],
    const constant uint& page_ints [[buffer(24)]],
    const device float4* table  [[buffer(25)]],
    const constant uint& table_rows [[buffer(26)]],
    const device int* ext_replay [[buffer(27)]],
    // The rows' per-head scalars (f32): `v_heads` log-gates, then `v_heads`
    // key-norm inverses. The own row lands its; a short carry reads its
    // carried rows' in place of their `[b | a]` softplus and key norm.
    device uchar* scalar_slab   [[buffer(28)]],
    const constant uint& scalar_row_bytes [[buffer(29)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane_u [[thread_index_in_simdgroup]],
    uint sg_u [[simdgroup_index_in_threadgroup]]) {
  constexpr int D = 128;
  constexpr int PER = D / 32;
  (void)table;
  (void)table_rows;
  const int lane = int(lane_u);
  const int sg = int(sg_u);
  const int tid = sg * 32 + lane;
  const int hv = int(group.z) % v_heads;
  const int r = int(group.z) / v_heads;
  const int dv_base = int(group.y) * SG * VROWS + sg * VROWS;

  threadgroup float glog_s[kVerifyChunk];
  threadgroup float dq_s[kVerifyChunk];
  threadgroup float dk_s[kVerifyChunk];
  threadgroup ulong u_at_s[kVerifyChunk];
  threadgroup float kn_s[kVerifyChunk][D];

  int begin = indptr[r];
  for (int j = 0; j < r; ++j) {
    begin += ext_replay[lane0 + j];
  }
  begin -= replay[lane0 + r] - ext_replay[lane0 + r];
  const int rep = min(replay[lane0 + r], kVerifyChunk);
  const device int* ids = slots + size_t(lane0 + r) * pie_state_pieces(map);
  if (indptr[r + 1] - indptr[r] != 1 || ids[0] < 0) {
    return;
  }
  const int own = indptr[r] - rep;
  const int t_own = rep;
  const int keep = min(commit[lane0 + r], rep + 1);
  const device int* pages = page_table + size_t(lane0 + r) * page_ints;
  // The own row lands for later fires unless this one folds it.
  const bool lands = pages[0] >= 0 && keep < rep + 1;
  const int hk = hv / (v_heads / k_heads);
  const size_t keys = size_t(k_heads) * D;
  const size_t pitch = 2 * keys + size_t(v_heads) * D;
  const float scale = 1.0f / metal::sqrt(float(D));
  const size_t vh = size_t(v_heads);

  // The own row's normed q and k, every simdgroup its own.
  const size_t row = size_t(begin + t_own) * pitch;
  const size_t qbase = row + size_t(hk) * D;
  const size_t kbase = qbase + keys;
  float qn[PER], kn[PER];
  float own_kinv;
  {
    const float4 q4 = float4(*reinterpret_cast<const device vec<T, 4>*>(qkv + qbase + size_t(PER * lane)));
    const float4 k4 = float4(*reinterpret_cast<const device vec<T, 4>*>(qkv + kbase + size_t(PER * lane)));
    float qs = 0.0f, ks = 0.0f;
    for (int i = 0; i < PER; ++i) {
      qs += q4[i] * q4[i];
      ks += k4[i] * k4[i];
    }
    const float qinv = metal::rsqrt(gdn_row_sum<32>(qs) + 1e-6f) * scale;
    const float kinv = metal::rsqrt(gdn_row_sum<32>(ks) + 1e-6f);
    own_kinv = kinv;
    for (int i = 0; i < PER; ++i) {
      qn[i] = q4[i] * qinv;
      kn[i] = k4[i] * kinv;
    }
  }
  float kq = 0.0f;
  for (int i = 0; i < PER; ++i) {
    kq += qn[i] * kn[i];
  }
  kq = gdn_row_sum<32>(kq);
  if (lands && group.y == 0 && tid == 0) {
    reinterpret_cast<device float*>(gdn_buffered_row(
        scalar_slab, pages, t_own, page_tokens, page_bytes, scalar_row_bytes))[vh + size_t(hv)] =
        own_kinv;
  }
  if (lands && group.y == 0 && sg == 0 && hv % (v_heads / k_heads) == 0) {
    device T* krow = reinterpret_cast<device T*>(
        gdn_buffered_row(key_slab, pages, t_own, page_tokens, page_bytes, key_row_bytes));
    for (int i = 0; i < PER; ++i) {
      const int d = PER * lane + i;
      krow[size_t(hk) * D + size_t(d)] = qkv[kbase + size_t(d)];
    }
  }
  const float2 own_g = gdn_verify_gates<T>(ba, a_log, dt_bias, hv, v_heads, t_own, begin, own, rep,
                                          lands && group.y == 0 && tid == 0, ba_slab, pages,
                                          page_tokens, page_bytes, ba_row_bytes);
  if (group.y == 0 && tid == 0) {
    const size_t at = size_t(own + t_own) * 2 * vh + size_t(hv);
    gates[at] = own_g.x;
    gates[at + vh] = own_g.y;
    if (lands) {
      reinterpret_cast<device float*>(gdn_buffered_row(
          scalar_slab, pages, t_own, page_tokens, page_bytes, scalar_row_bytes))[hv] = own_g.x;
    }
  }

  // Each carried row's scalars, once a threadgroup: row s on simdgroup
  // s mod SG. Its log-gate and key-norm inverse are the ones its landing
  // stored; its keys' dots with the own row's q and k are made here.
  for (int s = sg; s < rep; s += SG) {
    const device T* krow = reinterpret_cast<const device T*>(
        gdn_buffered_row(key_slab, pages, s, page_tokens, page_bytes, key_row_bytes));
    const device float* srow = reinterpret_cast<const device float*>(
        gdn_buffered_row(scalar_slab, pages, s, page_tokens, page_bytes, scalar_row_bytes));
    const float4 kr4 = float4(*reinterpret_cast<const device vec<T, 4>*>(krow + size_t(hk) * D + size_t(PER * lane)));
    const float rinv = srow[vh + size_t(hv)];
    float dq = 0.0f, dk = 0.0f;
    for (int i = 0; i < PER; ++i) {
      const float k = kr4[i] * rinv;
      dq += k * qn[i];
      dk += k * kn[i];
      if (s < keep) {
        kn_s[s][PER * lane + i] = k;
      }
    }
    dq = gdn_row_sum<32>(dq);
    dk = gdn_row_sum<32>(dk);
    if (lane == 0) {
      dq_s[s] = dq;
      dk_s[s] = dk;
      glog_s[s] = srow[hv];
      u_at_s[s] = ulong(gdn_buffered_row(u_slab, pages, s, page_tokens, page_bytes, u_row_bytes) - u_slab);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float total = own_g.x;
  for (int t = 0; t < rep; ++t) {
    total += glog_s[t];
  }
  device float* cells = gdn_cells<VROWS, PER>(rstate, map, ids, hv, v_heads, dv_base, D, D);
  float st[VROWS][PER];
  for (int v = 0; v < VROWS; ++v) {
    const float4 s4 = *reinterpret_cast<const device float4*>(cells + size_t(v) * D + size_t(PER * lane));
    for (int i = 0; i < PER; ++i) {
      st[v][i] = s4[i];
    }
  }
  const float g0 = metal::exp(total);
  float acc_a[VROWS], acc_o[VROWS];
  for (int v = 0; v < VROWS; ++v) {
    float pa = 0.0f, po = 0.0f;
    for (int i = 0; i < PER; ++i) {
      pa += st[v][i] * kn[i];
      po += st[v][i] * qn[i];
    }
    acc_a[v] = g0 * gdn_row_sum<32>(pa);
    acc_o[v] = g0 * gdn_row_sum<32>(po);
  }
  float run = 0.0f;
  for (int t = 0; t < rep; ++t) {
    const float glog = glog_s[t];
    run += glog;
    const float w = metal::exp(total - run);
    const float wq = w * dq_s[t], wk = w * dk_s[t];
    const device float* urow = reinterpret_cast<const device float*>(u_slab + u_at_s[t]);
    const float g = metal::exp(glog);
    for (int v = 0; v < VROWS; ++v) {
      const float u = urow[size_t(hv) * D + size_t(dv_base + v)];
      acc_a[v] += wk * u;
      acc_o[v] += wq * u;
      if (t < keep) {
        for (int i = 0; i < PER; ++i) {
          st[v][i] = g * st[v][i] + kn_s[t][PER * lane + i] * u;
        }
      }
    }
  }

  // The own row: its corrected value, readout and landing.
  const float beta = own_g.y;
  const size_t vbase = row + 2 * keys + size_t(hv) * D;
  const size_t row_out = (size_t(own + t_own) * vh + size_t(hv)) * D;
  device float* urow = lands ? reinterpret_cast<device float*>(gdn_buffered_row(
                                   u_slab, pages, t_own, page_tokens, page_bytes, u_row_bytes))
                             : nullptr;
  const float g_own = metal::exp(own_g.x);
  for (int v = 0; v < VROWS; ++v) {
    const float val = float(qkv[vbase + size_t(dv_base + v)]);
    const float u = beta * (val - acc_a[v]);
    const float o = acc_o[v] + kq * u;
    if (lane == 0) {
      y[row_out + size_t(dv_base + v)] = o;
      if (urow != nullptr) {
        urow[size_t(hv) * D + size_t(dv_base + v)] = u;
      }
    }
    if (keep == rep + 1) {
      for (int i = 0; i < PER; ++i) {
        st[v][i] = g_own * st[v][i] + kn[i] * u;
      }
    }
  }
  if (keep > 0) {
    for (int v = 0; v < VROWS; ++v) {
      *reinterpret_cast<device float4*>(cells + size_t(v) * D + size_t(PER * lane)) =
          float4(st[v][0], st[v][1], st[v][2], st[v][3]);
    }
  }
}

#define PIE_GDN_CARRIED_SHORT(sg, vrows)                                      \
  template [[host_name("gated_delta_carried_short_bfloat16_s_" #sg "_v_" #vrows)]] \
  [[kernel]] void gated_delta_carried_short<bfloat, sg, vrows>(               \
      const device bfloat*, const device int*, const device int*,             \
      const device int*, const device int*, const constant int&,              \
      const device bfloat*, const device float*, const device bfloat*,        \
      device float*, device float*, device float*,                            \
      const constant int&, const constant int&, device uchar*,                \
      const device int*, const constant uint&, const constant uint&,          \
      const constant uint&, device uchar*, const constant uint&,              \
      device uchar*, const constant uint&, const device ulong*,               \
      const constant uint&, const device float4*, const constant uint&,       \
      const device int*, device uchar*, const constant uint&, uint3, uint, uint);

PIE_GDN_CARRIED_SHORT(8, 4)

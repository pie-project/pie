#include <metal_stdlib>
using namespace metal;

template <typename T, int WIDTH, int KMAX>
[[kernel]] void gated_delta(
    const device T* qkv         [[buffer(0)]],
    const device float* gates   [[buffer(1)]],
    device float* rstate        [[buffer(2)]],
    const device uint* slots    [[buffer(3)]],
    device float* y             [[buffer(4)]],
    const constant int& k_heads [[buffer(5)]],
    const constant int& v_heads [[buffer(6)]],
    const constant int& k_dim   [[buffer(7)]],
    const constant int& v_dim   [[buffer(8)]],
    const constant int& splits  [[buffer(9)]],
    uint3 gpos [[threadgroup_position_in_grid]],
    uint3 lpos [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
  // ONE TOKEN THROUGH THE DELTA RULE, a simdgroup per value column.
  //
  // The state is `[v_dim][k_dim]` per head, `k_dim` contiguous. The
  // previous shape gave each THREAD one column and walked its `k_dim`
  // cells serially, twice (decay-and-read, then update-and-write): across
  // the threadgroup that is 128 lanes striding `k_dim` floats apart —
  // every load its own cache line — and four passes over the state. Here
  // a SIMDGROUP takes a column: its 32 lanes hold the column's cells
  // `PER` apart in registers (coalesced, one read and one write per cell),
  // the two dot products the rule needs fold with `simd_sum`, and the
  // column count is split across `splits` threadgroups down z so a
  // one-row fire still spreads a head over the device.
  constexpr int PER = KMAX / 32;
  threadgroup float sq[KMAX];
  threadgroup float sk[KMAX];
  threadgroup float2 fold[WIDTH];

  const int tid = int(lpos.x);
  const int hv = int(gpos.y);
  const int n = int(gpos.z) / splits;
  const int part = int(gpos.z) % splits;
  const int dk = k_dim;
  const int dv = v_dim;
  const int hk = hv / (v_heads / k_heads);

  const size_t keys = size_t(k_heads) * size_t(dk);
  const size_t row =
      size_t(n) * (2 * keys + size_t(v_heads) * size_t(dv));
  const size_t qbase = row + size_t(hk) * size_t(dk);
  const size_t kbase = qbase + keys;
  const size_t vbase = row + 2 * keys + size_t(hv) * size_t(dv);

  float2 sums = float2(0.0f, 0.0f);
  for (int i = tid; i < dk; i += WIDTH) {
    const float qv = float(qkv[qbase + size_t(i)]);
    const float kv = float(qkv[kbase + size_t(i)]);
    sq[i] = qv;
    sk[i] = kv;
    sums += float2(qv * qv, kv * kv);
  }
  fold[tid] = sums;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int off = WIDTH / 2; off > 0; off >>= 1) {
    if (tid < off) {
      fold[tid] += fold[tid + off];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float scale = 1.0f / metal::sqrt(float(dk));
  const float qinv = metal::rsqrt(fold[0].x + 1e-6f) * scale;
  const float kinv = metal::rsqrt(fold[0].y + 1e-6f);
  for (int i = tid; i < dk; i += WIDTH) {
    sq[i] *= qinv;
    sk[i] *= kinv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const size_t fused = size_t(n) * 2 * size_t(v_heads) + size_t(hv);
  const float decay = metal::exp(gates[fused]);
  const float beta = gates[fused + size_t(v_heads)];

  device float* state =
      rstate + (size_t(slots[n]) * size_t(v_heads) + size_t(hv)) *
                   size_t(dv) * size_t(dk);
  const size_t out = (size_t(n) * size_t(v_heads) + size_t(hv)) * size_t(dv);

  // This threadgroup's columns, a simdgroup at a time.
  const int per_part = dv / splits;
  const int c0 = part * per_part;
  constexpr int simdgroups = WIDTH / 32;
  const int lane = int(simd_lane);
  for (int c = c0 + int(simd_gid); c < c0 + per_part; c += simdgroups) {
    device float* cell = state + size_t(c) * size_t(dk);
    float s[PER];
    float kv_mem = 0.0f;
    for (int j = 0; j < PER; ++j) {
      const int i = lane + 32 * j;
      if (i < dk) {
        s[j] = cell[i] * decay;
        kv_mem += s[j] * sk[i];
      } else {
        s[j] = 0.0f;
      }
    }
    kv_mem = simd_sum(kv_mem);
    const float delta = (float(qkv[vbase + size_t(c)]) - kv_mem) * beta;
    float acc = 0.0f;
    for (int j = 0; j < PER; ++j) {
      const int i = lane + 32 * j;
      if (i < dk) {
        const float v = s[j] + sk[i] * delta;
        cell[i] = v;
        acc += v * sq[i];
      }
    }
    acc = simd_sum(acc);
    if (lane == 0) {
      y[out + size_t(c)] = acc;
    }
  }
}

template <typename T, int WIDTH, int KMAX>
[[kernel]] void gated_delta_chunked(
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
    uint3 pos [[thread_position_in_grid]],
    uint3 lpos [[thread_position_in_threadgroup]]) {
  threadgroup float sq[KMAX];
  threadgroup float sk[KMAX];
  threadgroup float2 fold[WIDTH];

  const int tid = int(lpos.x);
  const int hv = int(pos.y);
  const int r = int(pos.z);
  const int begin = indptr[r];
  const int end = indptr[r + 1];

  if (end <= begin) {
    return;
  }
  const int dk = k_dim;
  const int dv = v_dim;
  const int hk = hv / (v_heads / k_heads);

  const size_t keys = size_t(k_heads) * size_t(dk);
  const size_t pitch = 2 * keys + size_t(v_heads) * size_t(dv);
  const float scale = 1.0f / metal::sqrt(float(dk));
  device float* state =
      rstate + (size_t(slots[begin]) * size_t(v_heads) + size_t(hv)) *
                   size_t(dv) * size_t(dk);

  for (int t = begin; t < end; ++t) {
    const size_t row = size_t(t) * pitch;
    const size_t qbase = row + size_t(hk) * size_t(dk);
    const size_t kbase = qbase + keys;
    const size_t vbase = row + 2 * keys + size_t(hv) * size_t(dv);

    float2 sums = float2(0.0f, 0.0f);
    for (int i = tid; i < dk; i += WIDTH) {
      const float qv = float(qkv[qbase + size_t(i)]);
      const float kv = float(qkv[kbase + size_t(i)]);
      sq[i] = qv;
      sk[i] = kv;
      sums += float2(qv * qv, kv * kv);
    }
    fold[tid] = sums;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int off = WIDTH / 2; off > 0; off >>= 1) {
      if (tid < off) {
        fold[tid] += fold[tid + off];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float qinv = metal::rsqrt(fold[0].x + 1e-6f) * scale;
    const float kinv = metal::rsqrt(fold[0].y + 1e-6f);
    for (int i = tid; i < dk; i += WIDTH) {
      sq[i] *= qinv;
      sk[i] *= kinv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const size_t fused = size_t(t) * 2 * size_t(v_heads) + size_t(hv);
    const float decay = metal::exp(gates[fused]);
    const float beta = gates[fused + size_t(v_heads)];
    const size_t out = (size_t(t) * size_t(v_heads) + size_t(hv)) * size_t(dv);

    for (int c = tid; c < dv; c += WIDTH) {
      device float* cell = state + size_t(c) * size_t(dk);
      float kv_mem = 0.0f;
      for (int i = 0; i < dk; ++i) {
        const float s = cell[i] * decay;
        cell[i] = s;
        kv_mem += s * sk[i];
      }
      const float delta = (float(qkv[vbase + size_t(c)]) - kv_mem) * beta;
      float acc = 0.0f;
      for (int i = 0; i < dk; ++i) {
        const float s = cell[i] + sk[i] * delta;
        cell[i] = s;
        acc += s * sq[i];
      }
      y[out + size_t(c)] = acc;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

#define instantiate_gated_delta(name, itype, width, kmax)                \
  template [[host_name("gated_delta_" #name)]]                           \
  [[kernel]] void gated_delta<itype, width, kmax>(                       \
      const device itype*, const device float*, device float*,           \
      const device uint*, device float*,                                 \
      const constant int&, const constant int&, const constant int&,     \
      const constant int&, const constant int&, uint3, uint3, uint, uint);

#define instantiate_gated_delta_chunked(name, itype, width, kmax)        \
  template [[host_name("gated_delta_chunked_" #name)]]                   \
  [[kernel]] void gated_delta_chunked<itype, width, kmax>(               \
      const device itype*, const device int*, const device float*,       \
      device float*, const device uint*, device float*,                  \
      const constant int&, const constant int&, const constant int&,     \
      const constant int&, uint3, uint3);

instantiate_gated_delta(bfloat16, bfloat, 128, 256)

instantiate_gated_delta_chunked(bfloat16, bfloat, 128, 256)

// ── the committed form ───────────────────────────────────────────────────────
//
// `gated_delta_chunked` over the extended row run `causal_conv1d_committed`
// describes, with one difference that is the whole point: the recurrence runs
// on a WORK copy of the bank and the bank itself is written only as of row
// `commit[lane0 + r] - 1`. The rows past the commit are computed (their
// outputs are the speculative window's logits) from the state they should see
// and leave nothing behind. Each thread copies, scans and commits its own
// columns, so no barrier orders the bank write. `work` is one bank per fire
// lane, `[lane][v_heads][v_dim][k_dim]`.
template <typename T, int WIDTH, int KMAX>
[[kernel]] void gated_delta_committed(
    const device T* qkv         [[buffer(0)]],
    const device int* indptr    [[buffer(1)]],
    const device int* replay    [[buffer(2)]],
    const device int* commit    [[buffer(3)]],
    const device int* slots     [[buffer(4)]],
    const constant int& lane0   [[buffer(5)]],
    const device float* gates   [[buffer(6)]],
    device float* rstate        [[buffer(7)]],
    device float* work          [[buffer(8)]],
    device float* y             [[buffer(9)]],
    const constant int& k_heads [[buffer(10)]],
    const constant int& v_heads [[buffer(11)]],
    const constant int& k_dim   [[buffer(12)]],
    const constant int& v_dim   [[buffer(13)]],
    const constant int& splits  [[buffer(14)]],
    uint3 gpos [[threadgroup_position_in_grid]],
    uint3 lpos [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
  // The shape of `gated_delta` (a simdgroup per value column, the column's
  // cells in registers, the columns split across `splits` threadgroups
  // down z) over the extended run: the run's rows are staged `SPAN` at a
  // time, and a column's state stays in registers across every row of the
  // chunk — the bank is read once, and written once, as of `keep`. Only a
  // run longer than one chunk goes through `work` between chunks.
  // SPAN rows a chunk, sized for a depth-one window and its replay (fold
  // ≤ 2 + 2 own rows) in ONE chunk while keeping the staging under 10 KB —
  // an 8-row chunk with a fold table a row was 24 KB, which on this device
  // (32 KB a core) parks a single threadgroup per core and made the arm 6×
  // the decode kernel's per-row cost. The norm fold is `gated_delta`'s
  // WIDTH-wide tree, one row at a time through one table, so the banks it
  // leaves are bit for bit the unbuffered fold's.
  constexpr int PER = KMAX / 32;
  constexpr int SPAN = 4;
  constexpr int simdgroups = WIDTH / 32;
  threadgroup float sq[SPAN][KMAX];
  threadgroup float sk[SPAN][KMAX];
  threadgroup float2 fold[WIDTH];

  const int tid = int(lpos.x);
  const int hv = int(gpos.y);
  const int r = int(gpos.z) / splits;
  const int part = int(gpos.z) % splits;
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
  int keep = commit[lane0 + r];
  if (keep > span) {
    keep = span;
  }

  const int dk = k_dim;
  const int dv = v_dim;
  const int hk = hv / (v_heads / k_heads);

  const size_t keys = size_t(k_heads) * size_t(dk);
  const size_t pitch = 2 * keys + size_t(v_heads) * size_t(dv);
  const float scale = 1.0f / metal::sqrt(float(dk));
  const size_t head = size_t(dv) * size_t(dk);
  device float* bank =
      rstate + (size_t(slot) * size_t(v_heads) + size_t(hv)) * head;
  device float* carry =
      work + (size_t(lane0 + r) * size_t(v_heads) + size_t(hv)) * head;

  const int per_part = dv / splits;
  const int c0 = part * per_part;
  const int lane = int(simd_lane);

  for (int t0 = 0; t0 < span; t0 += SPAN) {
    const int rows = min(SPAN, span - t0);

    // Stage this chunk's normalized q and k rows, a row at a time.
    for (int t = 0; t < rows; ++t) {
      const size_t row = size_t(begin + t0 + t) * pitch;
      const size_t qbase = row + size_t(hk) * size_t(dk);
      const size_t kbase = qbase + keys;
      float2 sums = float2(0.0f, 0.0f);
      for (int i = tid; i < dk; i += WIDTH) {
        const float qv = float(qkv[qbase + size_t(i)]);
        const float kv = float(qkv[kbase + size_t(i)]);
        sq[t][i] = qv;
        sk[t][i] = kv;
        sums += float2(qv * qv, kv * kv);
      }
      fold[tid] = sums;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int off = WIDTH / 2; off > 0; off >>= 1) {
        if (tid < off) {
          fold[tid] += fold[tid + off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      const float qinv = metal::rsqrt(fold[0].x + 1e-6f) * scale;
      const float kinv = metal::rsqrt(fold[0].y + 1e-6f);
      for (int i = tid; i < dk; i += WIDTH) {
        sq[t][i] *= qinv;
        sk[t][i] *= kinv;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Every column of this threadgroup, the chunk's rows in registers.
    for (int c = c0 + int(simd_gid); c < c0 + per_part; c += simdgroups) {
      const device float* from = (t0 == 0 ? bank : carry) + size_t(c) * size_t(dk);
      float s[PER];
      for (int j = 0; j < PER; ++j) {
        const int i = lane + 32 * j;
        s[j] = (i < dk) ? from[i] : 0.0f;
      }
      for (int t = 0; t < rows; ++t) {
        const int at = begin + t0 + t;
        const size_t fused = size_t(at) * 2 * size_t(v_heads) + size_t(hv);
        const float decay = metal::exp(gates[fused]);
        const float beta = gates[fused + size_t(v_heads)];
        const size_t vbase = size_t(at) * pitch + 2 * keys + size_t(hv) * size_t(dv);
        float kv_mem = 0.0f;
        for (int j = 0; j < PER; ++j) {
          const int i = lane + 32 * j;
          if (i < dk) {
            s[j] *= decay;
            kv_mem += s[j] * sk[t][i];
          }
        }
        kv_mem = simd_sum(kv_mem);
        const float delta = (float(qkv[vbase + size_t(c)]) - kv_mem) * beta;
        float acc = 0.0f;
        for (int j = 0; j < PER; ++j) {
          const int i = lane + 32 * j;
          if (i < dk) {
            s[j] += sk[t][i] * delta;
            acc += s[j] * sq[t][i];
          }
        }
        acc = simd_sum(acc);
        if (lane == 0) {
          y[(size_t(at) * size_t(v_heads) + size_t(hv)) * size_t(dv) + size_t(c)] = acc;
        }
        if (t0 + t + 1 == keep) {
          device float* durable = bank + size_t(c) * size_t(dk);
          for (int j = 0; j < PER; ++j) {
            const int i = lane + 32 * j;
            if (i < dk) {
              durable[i] = s[j];
            }
          }
        }
      }
      if (t0 + rows < span) {
        device float* to = carry + size_t(c) * size_t(dk);
        for (int j = 0; j < PER; ++j) {
          const int i = lane + 32 * j;
          if (i < dk) {
            to[i] = s[j];
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

#define instantiate_gated_delta_committed(name, itype, width, kmax)     \
  template [[host_name("gated_delta_committed_" #name)]]               \
  [[kernel]] void gated_delta_committed<itype, width, kmax>(           \
      const device itype*, const device int*, const device int*,       \
      const device int*, const device int*, const constant int&,       \
      const device float*, device float*, device float*, device float*,\
      const constant int&, const constant int&, const constant int&,   \
      const constant int&, const constant int&, uint3, uint3, uint, uint);

instantiate_gated_delta_committed(bfloat16, bfloat, 128, 256)

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
    uint3 pos [[thread_position_in_grid]],
    uint3 lpos [[thread_position_in_threadgroup]]) {
  threadgroup float sq[KMAX];
  threadgroup float sk[KMAX];
  threadgroup float2 fold[WIDTH];

  const int tid = int(lpos.x);
  const int hv = int(pos.y);
  const int n = int(pos.z);
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
      const constant int&, uint3, uint3);

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
    uint3 pos [[thread_position_in_grid]],
    uint3 lpos [[thread_position_in_threadgroup]]) {
  threadgroup float sq[KMAX];
  threadgroup float sk[KMAX];
  threadgroup float2 fold[WIDTH];

  const int tid = int(lpos.x);
  const int hv = int(pos.y);
  const int r = int(pos.z);
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
  device float* state =
      work + (size_t(lane0 + r) * size_t(v_heads) + size_t(hv)) * head;

  for (int c = tid; c < dv; c += WIDTH) {
    for (int i = 0; i < dk; ++i) {
      state[size_t(c) * size_t(dk) + size_t(i)] = bank[size_t(c) * size_t(dk) + size_t(i)];
    }
  }

  for (int t = 0; t < span; ++t) {
    const size_t row = size_t(begin + t) * pitch;
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

    const size_t fused = size_t(begin + t) * 2 * size_t(v_heads) + size_t(hv);
    const float decay = metal::exp(gates[fused]);
    const float beta = gates[fused + size_t(v_heads)];
    const size_t out = (size_t(begin + t) * size_t(v_heads) + size_t(hv)) * size_t(dv);

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
      if (t + 1 == keep) {
        device float* durable = bank + size_t(c) * size_t(dk);
        for (int i = 0; i < dk; ++i) {
          durable[i] = cell[i];
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
      const constant int&, uint3, uint3);

instantiate_gated_delta_committed(bfloat16, bfloat, 128, 256)

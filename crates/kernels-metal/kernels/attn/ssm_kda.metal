#include <metal_stdlib>
using namespace metal;

template <typename T, int WIDTH, int DMAX>
[[kernel]] void kda_step(
    const device T* mixed          [[buffer(0)]],
    const device T* f              [[buffer(1)]],
    const device T* b              [[buffer(2)]],
    const device float* dt_bias    [[buffer(3)]],
    const device float* a_log      [[buffer(4)]],
    device float* rstate           [[buffer(5)]],
    const device uint* slots       [[buffer(6)]],
    device float* y                [[buffer(7)]],
    const constant int& heads      [[buffer(8)]],
    const constant int& head_dim   [[buffer(9)]],
    const constant float& norm_eps [[buffer(10)]],
    const constant float& gate_floor [[buffer(11)]],
    uint3 pos [[thread_position_in_grid]],
    uint3 lpos [[thread_position_in_threadgroup]]) {
  threadgroup float sq[DMAX];
  threadgroup float sk[DMAX];
  threadgroup float sg[DMAX];
  threadgroup float2 fold[WIDTH];

  const int tid = int(lpos.x);
  const int h = int(pos.y);
  const int n = int(pos.z);
  const int d = head_dim;
  const int plane = heads * head_dim;

  const size_t wide = size_t(plane);
  const size_t row = size_t(n) * 3 * wide;
  const size_t head = size_t(h) * size_t(d);

  float2 sums = float2(0.0f, 0.0f);
  // q and k are L2-normed PER HEAD (the reference's `l2norm(q, dim=-1)`).
  for (int i = tid; i < d; i += WIDTH) {
    const float qv = float(mixed[row + head + size_t(i)]);
    const float kv = float(mixed[row + wide + head + size_t(i)]);
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
  const float qinv = metal::rsqrt(fold[0].x + norm_eps);
  const float kinv = metal::rsqrt(fold[0].y + norm_eps);

  const float alpha = metal::exp(a_log[h]);
  // The reference recurrence's `scale`: q carries head_dim^-1/2.
  const float qscale = metal::rsqrt(float(d));
  for (int i = tid; i < d; i += WIDTH) {
    const size_t at = head + size_t(i);
    sq[i] = float(mixed[row + at]) * qinv * qscale;
    sk[i] = float(mixed[row + wide + at]) * kinv;
    const float z = float(f[size_t(n) * wide + at]) + dt_bias[at];
    if (gate_floor != 0.0f) {
      // A floored decay (`gate_lower_bound`): log-gate = floor * sigmoid(alpha * z).
      sg[i] = metal::exp(gate_floor / (1.0f + metal::exp(-alpha * z)));
    } else {
      const float sp = (z > 20.0f) ? z : metal::log(1.0f + metal::exp(z));
      sg[i] = metal::exp(-alpha * sp);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float beta =
      1.0f / (1.0f + metal::exp(-float(b[size_t(n) * size_t(heads) + size_t(h)])));
  device float* state =
      rstate + (size_t(slots[n]) * size_t(heads) + size_t(h)) * size_t(d) *
                   size_t(d);
  const size_t out = size_t(n) * wide + head;
  const size_t vbase = row + 2 * wide + head;

  for (int vi = tid; vi < d; vi += WIDTH) {
    device float* cell = state + size_t(vi) * size_t(d);
    float mem = 0.0f;
    for (int i = 0; i < d; ++i) {
      const float s = cell[i] * sg[i];
      cell[i] = s;
      mem += s * sk[i];
    }
    const float delta = (float(mixed[vbase + size_t(vi)]) - mem) * beta;
    float acc = 0.0f;
    for (int i = 0; i < d; ++i) {
      const float s = cell[i] + sk[i] * delta;
      cell[i] = s;
      acc += s * sq[i];
    }
    y[out + size_t(vi)] = acc;
  }
}

template <typename T, int WIDTH, int DMAX>
[[kernel]] void kda_chunked(
    const device T* mixed          [[buffer(0)]],
    const device int* indptr       [[buffer(1)]],
    const device T* f              [[buffer(2)]],
    const device T* b              [[buffer(3)]],
    const device float* dt_bias    [[buffer(4)]],
    const device float* a_log      [[buffer(5)]],
    device float* rstate           [[buffer(6)]],
    const device uint* slots       [[buffer(7)]],
    device float* y                [[buffer(8)]],
    const constant int& heads      [[buffer(9)]],
    const constant int& head_dim   [[buffer(10)]],
    const constant float& norm_eps [[buffer(11)]],
    const constant float& gate_floor [[buffer(12)]],
    uint3 pos [[thread_position_in_grid]],
    uint3 lpos [[thread_position_in_threadgroup]]) {
  threadgroup float sq[DMAX];
  threadgroup float sk[DMAX];
  threadgroup float sg[DMAX];
  threadgroup float2 fold[WIDTH];

  const int tid = int(lpos.x);
  const int h = int(pos.y);
  const int r = int(pos.z);
  const int begin = indptr[r];
  const int end = indptr[r + 1];

  if (end <= begin) {
    return;
  }
  const int d = head_dim;
  const int plane = heads * head_dim;

  const size_t wide = size_t(plane);
  const size_t head = size_t(h) * size_t(d);
  const float alpha = metal::exp(a_log[h]);
  // The reference recurrence's `scale`: q carries head_dim^-1/2.
  const float qscale = metal::rsqrt(float(d));
  device float* state =
      rstate + (size_t(slots[begin]) * size_t(heads) + size_t(h)) * size_t(d) *
                   size_t(d);

  for (int t = begin; t < end; ++t) {
    const size_t row = size_t(t) * 3 * wide;

    float2 sums = float2(0.0f, 0.0f);
    // q and k are L2-normed PER HEAD (the reference's `l2norm(q, dim=-1)`).
    for (int i = tid; i < d; i += WIDTH) {
      const float qv = float(mixed[row + head + size_t(i)]);
      const float kv = float(mixed[row + wide + head + size_t(i)]);
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
    const float qinv = metal::rsqrt(fold[0].x + norm_eps);
    const float kinv = metal::rsqrt(fold[0].y + norm_eps);

    for (int i = tid; i < d; i += WIDTH) {
      const size_t at = head + size_t(i);
      sq[i] = float(mixed[row + at]) * qinv * qscale;
      sk[i] = float(mixed[row + wide + at]) * kinv;
      const float z = float(f[size_t(t) * wide + at]) + dt_bias[at];
      if (gate_floor != 0.0f) {
        // A floored decay (`gate_lower_bound`): log-gate = floor * sigmoid(alpha * z).
        sg[i] = metal::exp(gate_floor / (1.0f + metal::exp(-alpha * z)));
      } else {
        const float sp = (z > 20.0f) ? z : metal::log(1.0f + metal::exp(z));
        sg[i] = metal::exp(-alpha * sp);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float beta = 1.0f /
        (1.0f + metal::exp(-float(b[size_t(t) * size_t(heads) + size_t(h)])));
    const size_t out = size_t(t) * wide + head;
    const size_t vbase = row + 2 * wide + head;

    for (int vi = tid; vi < d; vi += WIDTH) {
      device float* cell = state + size_t(vi) * size_t(d);
      float mem = 0.0f;
      for (int i = 0; i < d; ++i) {
        const float s = cell[i] * sg[i];
        cell[i] = s;
        mem += s * sk[i];
      }
      const float delta = (float(mixed[vbase + size_t(vi)]) - mem) * beta;
      float acc = 0.0f;
      for (int i = 0; i < d; ++i) {
        const float s = cell[i] + sk[i] * delta;
        cell[i] = s;
        acc += s * sq[i];
      }
      y[out + size_t(vi)] = acc;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

#define instantiate_kda_step(name, itype, width, dmax)                   \
  template [[host_name("kda_step_" #name)]]                              \
  [[kernel]] void kda_step<itype, width, dmax>(                          \
      const device itype*, const device itype*, const device itype*,     \
      const device float*, const device float*, device float*,           \
      const device uint*, device float*,                                 \
      const constant int&, const constant int&, const constant float&,   \
      const constant float&, uint3, uint3);

#define instantiate_kda_chunked(name, itype, width, dmax)                \
  template [[host_name("kda_chunked_" #name)]]                           \
  [[kernel]] void kda_chunked<itype, width, dmax>(                       \
      const device itype*, const device int*, const device itype*,       \
      const device itype*, const device float*, const device float*,     \
      device float*, const device uint*, device float*,                  \
      const constant int&, const constant int&, const constant float&,   \
      const constant float&, uint3, uint3);

instantiate_kda_step(bfloat16, bfloat, 128, 256)

instantiate_kda_chunked(bfloat16, bfloat, 128, 256)

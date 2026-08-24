// Raw-Metal Kimi Delta Attention: the step, and the window.
//
// Two entrypoints, one per point, with the per-token arithmetic written out
// twice for `gated_delta.metal`'s reason -- the two differ in which axis the
// threadgroup owns and where the token barrier sits.
//
// ── THE OPERANDS ARE PACKED, AND THE KERNEL IS WHAT CUTS THEM ──────────────
//
// `ssm.kda_step` declares ONE post-convolution operand:
//
//   mixed [N, 3 * H * D]   `[q | k | v]`, in that order, each `H * D` wide
//   f     [N, H * D]       the raw forget projection
//   b     [N, H]           the raw beta projection
//
// The three planes of `mixed` are cut here rather than upstream, so nothing
// claims a row stride of `H * D` for bytes whose stride is three times that.
//
// ── THE ARITHMETIC ─────────────────────────────────────────────────────────
//
// Transcribed from `pie::ssm::kda_qkv_prep` + `pie::ssm::kda_gate_beta` +
// `pie::ssm::kda_recurrent_step_batched` in `kernels-cuda/kernels/ssm/kda.cuh`,
// which is where the numeric contract was measured.
//
//   q = l2norm(mixed[.., 0 .. W])          eps = norm_eps, THE STATEMENT'S
//   k = l2norm(mixed[.., W .. 2W])         eps = norm_eps
//   v =        mixed[.., 2W .. 3W]         (widened only)
//   g[h, d] = -exp(a_log[h]) * softplus(f[.., h*D + d] + dt_bias[h*D + d])
//   beta[h] = sigmoid(b[.., h])
//
// THE NORM IS OVER THE WHOLE `H * D` PLANE, one l2 per token, which is what
// `kda_qkv_prep` does and what the legacy kimi forward fired. Every published
// KDA normalises per HEAD instead, and so does this tree's own gated-delta
// prologue; the two differ, one of them is wrong, and the cuda header records
// that neither can be settled without a checkpoint. This kernel reproduces the
// cuda body, so the seam stays in one place rather than two. A threadgroup
// owns one (token, head) and folds the whole row for it, which costs `H` reads
// of a row the recurrence then spends `D * D` multiplies on.
//
// Then, per (token, head, value channel `vi`) -- note the decay is PER `k`
// CHANNEL here, where gated delta's is one scalar per head:
//
//   S[vi, k] := S[vi, k] * exp(g[h, k])
//   mem      := sum_k S[vi, k] * k[k]
//   delta    := (v[vi] - mem) * beta
//   S[vi, k] := S[vi, k] + k[k] * delta
//   y[vi]    := sum_k S[vi, k] * q[k]
//
// softplus is `z > 20 ? z : log(1 + exp(z))`, the guarded spelling every
// kernel in this family shares, so all of them agree.
//
// ── THE STATE SLAB ─────────────────────────────────────────────────────────
//
// `[slots, H, D, D]` fp32, the k channel fastest -- `gdn_core.metal`'s
// `[slots, Hv, Dv, Dk]` at `Hv = H` and `Dv = Dk = D`, which is the product
// `driver-metal`'s `layout::recurrent::Shape::state_bytes_per_slot`
// allocates. `slots` is one seat per TOKEN; the window reads its first row's.
//
// Launch: dispatchThreads grid=(WIDTH, H, R), tg=(WIDTH, 1, 1) for both.

#include <metal_stdlib>
using namespace metal;

// One token of the recurrence, on the threadgroup that owns (token, head).
// `DMAX` bounds the head width the shared q/k/decay rows can hold; the claim
// body refuses a wider one by name.
template <typename T, int WIDTH, int DMAX>
[[kernel]] void kda_step(
    const device T* mixed          [[buffer(0)]],  // [N, 3*H*D] `[q | k | v]`
    const device T* f              [[buffer(1)]],  // [N, H*D]
    const device T* b              [[buffer(2)]],  // [N, H]
    const device float* dt_bias    [[buffer(3)]],  // [H*D]
    const device float* a_log      [[buffer(4)]],  // [H]
    device float* rstate           [[buffer(5)]],  // [slots, H, D, D]
    const device uint* slots       [[buffer(6)]],  // [N], one seat per row
    device float* y                [[buffer(7)]],  // [N, H*D]
    const constant int& heads      [[buffer(8)]],
    const constant int& head_dim   [[buffer(9)]],
    const constant float& norm_eps [[buffer(10)]],
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

  // The l2 folds are over the WHOLE plane, not over this head's slice.
  float2 sums = float2(0.0f, 0.0f);
  for (int i = tid; i < plane; i += WIDTH) {
    const float qv = float(mixed[row + size_t(i)]);
    const float kv = float(mixed[row + wide + size_t(i)]);
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
  for (int i = tid; i < d; i += WIDTH) {
    const size_t at = head + size_t(i);
    sq[i] = float(mixed[row + at]) * qinv;
    sk[i] = float(mixed[row + wide + at]) * kinv;
    const float z = float(f[size_t(n) * wide + at]) + dt_bias[at];
    const float sp = (z > 20.0f) ? z : metal::log(1.0f + metal::exp(z));
    sg[i] = metal::exp(-alpha * sp);
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

// The window, on the threadgroup that owns (request, head).
template <typename T, int WIDTH, int DMAX>
[[kernel]] void kda_chunked(
    const device T* mixed          [[buffer(0)]],  // [N, 3*H*D] `[q | k | v]`
    const device int* indptr       [[buffer(1)]],  // [R + 1]
    const device T* f              [[buffer(2)]],  // [N, H*D]
    const device T* b              [[buffer(3)]],  // [N, H]
    const device float* dt_bias    [[buffer(4)]],  // [H*D]
    const device float* a_log      [[buffer(5)]],  // [H]
    device float* rstate           [[buffer(6)]],  // [slots, H, D, D]
    const device uint* slots       [[buffer(7)]],  // [N], one seat per token
    device float* y                [[buffer(8)]],  // [N, H*D]
    const constant int& heads      [[buffer(9)]],
    const constant int& head_dim   [[buffer(10)]],
    const constant float& norm_eps [[buffer(11)]],
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
  // Threadgroup-uniform, so the whole threadgroup leaves together.
  if (end <= begin) {
    return;
  }
  const int d = head_dim;
  const int plane = heads * head_dim;

  const size_t wide = size_t(plane);
  const size_t head = size_t(h) * size_t(d);
  const float alpha = metal::exp(a_log[h]);
  device float* state =
      rstate + (size_t(slots[begin]) * size_t(heads) + size_t(h)) * size_t(d) *
                   size_t(d);

  for (int t = begin; t < end; ++t) {
    const size_t row = size_t(t) * 3 * wide;

    float2 sums = float2(0.0f, 0.0f);
    for (int i = tid; i < plane; i += WIDTH) {
      const float qv = float(mixed[row + size_t(i)]);
      const float kv = float(mixed[row + wide + size_t(i)]);
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
      sq[i] = float(mixed[row + at]) * qinv;
      sk[i] = float(mixed[row + wide + at]) * kinv;
      const float z = float(f[size_t(t) * wide + at]) + dt_bias[at];
      const float sp = (z > 20.0f) ? z : metal::log(1.0f + metal::exp(z));
      sg[i] = metal::exp(-alpha * sp);
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
    // The next token overwrites the three shared rows while this one's
    // readers may still be in the loop above.
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
      uint3, uint3);

#define instantiate_kda_chunked(name, itype, width, dmax)                \
  template [[host_name("kda_chunked_" #name)]]                           \
  [[kernel]] void kda_chunked<itype, width, dmax>(                       \
      const device itype*, const device int*, const device itype*,       \
      const device itype*, const device float*, const device float*,     \
      device float*, const device uint*, device float*,                  \
      const constant int&, const constant int&, const constant float&,   \
      uint3, uint3);

instantiate_kda_step(bfloat16, bfloat, 128, 256)

instantiate_kda_chunked(bfloat16, bfloat, 128, 256)

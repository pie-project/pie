// Raw-Metal gated DeltaNet recurrence: the step, and the window.
//
// Two entrypoints, one per point. `gated_delta` advances one token per
// request; `gated_delta_chunked` walks a CSR window of tokens in order,
// because the recurrence is sequential in `t` by definition and the only
// parallelism left is across heads and value channels. The two scans are
// written out twice on purpose -- they differ in which axis the threadgroup
// owns and where the token barrier sits, and a fold would put both on a
// runtime branch inside the innermost loop.
//
// ── THE OPERANDS ARE PACKED, AND THE KERNEL IS WHAT CUTS THEM ──────────────
//
// `ssm.gated_delta` declares ONE post-convolution operand and ONE gate row,
// and both are packed rectangles:
//
//   qkv   [N, 2*Kh*Dk + Hv*Dv]   `[q | k | v]`, in that order
//   gates [N, 2*Hv]              `[g_log | beta]`, in that order
//
// A caller that split either with pointer arithmetic would be claiming a row
// stride of `Kh*Dk` (or of `Hv`) for bytes whose stride is the whole packed
// row -- true at one token and false at two. That is a defect this family has
// shipped: a shim once wrote `gates` as two compact halves while the kernel
// read it packed, so every window longer than one token was wrong. So the
// executor hands over the rectangles it was given, the kernel is TOLD the
// packing through the four head numbers the point declares, and the cut
// happens at the load. `ssm.gdn_prep` writes `gates` in exactly this shape.
//
// ── THE ARITHMETIC ─────────────────────────────────────────────────────────
//
// Transcribed from `pie::ssm::qwen_gdn_qk_norm` + `pie::ssm::qwen_gdn_v_gates`
// + `pie::ssm::recurrent_step_batched_gqa` in `kernels-cuda/kernels/ssm/`,
// which is where the numeric contract was measured. Per (token, value head):
//
//   q = l2norm(qkv[.., q slice of key head hk]) / sqrt(Dk)
//   k = l2norm(qkv[.., k slice of key head hk])
//   v = qkv[.., v slice of value head hv]                       (widened only)
//   g = exp(g_log[hv]),  beta = beta[hv]
//
//   S      := S * g                       (in place, per (dv, dk) cell)
//   kv_mem := sum_dk S[dv, dk] * k[dk]
//   delta  := (v[dv] - kv_mem) * beta
//   S      := S + k (x) delta             (in place)
//   y[dv]  := sum_dk S[dv, dk] * q[dk]
//
// The l2 norms are per KEY HEAD over `Dk` channels with a hard `1e-6`, which
// is `qwen_gdn_qk_norm`'s epsilon -- this point declares none, so a number of
// its own would be a number nothing states. `Hv / Kh` is the GQA repeat: value
// head `hv` reads key head `hv / repeat`, exactly as the cuda kernel's
// `h_k = h / repeat`.
//
// ── THE STATE SLAB ─────────────────────────────────────────────────────────
//
// `[slots, Hv, Dv, Dk]` fp32, `Dk` fastest -- the layout `gdn_core.metal`
// already indexes (`rstate + ((slot * Hv + hv) * Dv + dv) * Dk`) and the
// product `driver-metal`'s `layout::recurrent::Shape::state_bytes_per_slot`
// allocates. The cuda plane keeps its own slab the other way round
// (`state_offset<false>` is `k * V_d + v`); a slab is the plane's, and this is
// this plane's.
//
// The `slots` table is one seat PER TOKEN, not per request, which is what
// `driver-metal`'s `bind::tables` stages and what the legacy slotted GDN
// kernels read. The step indexes it by row; the window indexes it by its first
// row, since every token of a request sits in the same seat.
//
// Launch: dispatchThreads grid=(WIDTH, Hv, R), tg=(WIDTH, 1, 1) for both --
// one threadgroup per (request, value head), `WIDTH` threads striding the
// value channels. `grid.x` is exactly `WIDTH`, so every threadgroup is full
// and the barriers below are reached by all of it.

#include <metal_stdlib>
using namespace metal;

// One token of the recurrence, on the threadgroup that owns (token, value
// head). `KMAX` bounds the key head width the shared q/k rows can hold; the
// claim body refuses a wider one by name.
template <typename T, int WIDTH, int KMAX>
[[kernel]] void gated_delta(
    const device T* qkv         [[buffer(0)]],  // [N, 2*Kh*Dk + Hv*Dv]
    const device float* gates   [[buffer(1)]],  // [N, 2*Hv] `[g_log | beta]`
    device float* rstate        [[buffer(2)]],  // [slots, Hv, Dv, Dk]
    const device uint* slots    [[buffer(3)]],  // [N], one seat per row
    device float* y             [[buffer(4)]],  // [N, Hv*Dv]
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

  // Stage the head's q and k rows and fold their squares in the same pass.
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

  // `[g_log | beta]`, cut where the packing says and nowhere else.
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

// The window, on the threadgroup that owns (request, value head). Same
// per-token arithmetic; the token loop is the point.
template <typename T, int WIDTH, int KMAX>
[[kernel]] void gated_delta_chunked(
    const device T* qkv         [[buffer(0)]],  // [N, 2*Kh*Dk + Hv*Dv]
    const device int* indptr    [[buffer(1)]],  // [R + 1]
    const device float* gates   [[buffer(2)]],  // [N, 2*Hv] `[g_log | beta]`
    device float* rstate        [[buffer(3)]],  // [slots, Hv, Dv, Dk]
    const device uint* slots    [[buffer(4)]],  // [N], one seat per token
    device float* y             [[buffer(5)]],  // [N, Hv*Dv]
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
  // Threadgroup-uniform: `r` is the same for every thread here, so the whole
  // threadgroup leaves together and no barrier below is reached by a subset.
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
    // The next token overwrites `sq`/`sk` while this one's readers may still
    // be in the loop above. A value channel is owned by the same thread at
    // every token, so the state cells need no such fence.
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

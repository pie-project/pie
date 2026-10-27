#include <metal_stdlib>

using namespace metal;


constant constexpr int HC_MAX_MULT = 8;

constant constexpr int HC_GATES_CHUNK = 256;


template <typename T>
[[kernel]] void hc_expand(
    const device T* input      [[buffer(0)]],
    device T* output           [[buffer(1)]],
    const constant int& n_rows [[buffer(2)]],
    const constant int& M      [[buffer(3)]],
    const constant int& H      [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int h = int(gid.x);
  const int n = int(gid.y);
  if (h >= H || n >= n_rows) return;
  const T val = input[size_t(n) * size_t(H) + size_t(h)];
  device T* out = output + size_t(n) * size_t(M) * size_t(H) + size_t(h);
  for (int m = 0; m < M; ++m) {
    out[size_t(m) * size_t(H)] = val;
  }
}

#define instantiate_hc_expand(name, itype)                        \
  template [[host_name("hc_expand_" #name)]]                      \
  [[kernel]] void hc_expand<itype>(                               \
      const device itype*, device itype*,                         \
      const constant int&, const constant int&, const constant int&, \
      uint2);

instantiate_hc_expand(bfloat16, bfloat)


template <typename T, int BLOCK>
[[kernel]] void hc_rmsnorm_f32(
    const device T* input      [[buffer(0)]],
    device float* output       [[buffer(1)]],
    const constant int& dim    [[buffer(2)]],
    const constant float& eps  [[buffer(3)]],
    uint gid        [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint tg_size    [[threads_per_threadgroup]]) {
  threadgroup float partials[BLOCK / 32];
  threadgroup float inv_rms[1];

  const size_t base = size_t(gid) * size_t(dim);
  const device T* row = input + base;
  device float* out = output + base;

  float local = 0.0f;
  for (uint d = lid; d < uint(dim); d += tg_size) {
    const float v = float(row[d]);
    local += v * v;
  }
  local = simd_sum(local);
  if (simd_group == 0) partials[simd_lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane == 0) partials[simd_group] = local;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    const float total = simd_sum(partials[simd_lane]);
    if (simd_lane == 0) inv_rms[0] = precise::rsqrt(total / float(dim) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float s = inv_rms[0];
  for (uint d = lid; d < uint(dim); d += tg_size) {
    out[d] = float(row[d]) * s;
  }
}

#define instantiate_hc_rmsnorm_f32(name, itype, block)            \
  template [[host_name("hc_rmsnorm_f32_" #name)]]                 \
  [[kernel]] void hc_rmsnorm_f32<itype, block>(                   \
      const device itype*, device float*,                         \
      const constant int&, const constant float&,                 \
      uint, uint, uint, uint, uint);

instantiate_hc_rmsnorm_f32(bfloat16, bfloat, 256)


template <int BLOCK>
[[kernel]] void hc_project(
    const device float* normed  [[buffer(0)]],
    const device float* hc_fn   [[buffer(1)]],
    device float* mixes         [[buffer(2)]],
    const constant int& fan_in  [[buffer(3)]],
    const constant int& mix_hc  [[buffer(4)]],
    uint gid        [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint tg_size    [[threads_per_threadgroup]]) {
  threadgroup float partials[BLOCK / 32];

  const int o = int(gid) % mix_hc;
  const int n = int(gid) / mix_hc;

  const device float* row = normed + size_t(n) * size_t(fan_in);
  const device float* w = hc_fn + size_t(o) * size_t(fan_in);

  float local = 0.0f;
  for (uint d = lid; d < uint(fan_in); d += tg_size) {
    local += row[d] * w[d];
  }
  local = simd_sum(local);

  constexpr int GROUPS = BLOCK / 32;
  if (lid < uint(GROUPS)) partials[lid] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane == 0) partials[simd_group] = local;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    const float held = simd_lane < uint(GROUPS) ? partials[simd_lane] : 0.0f;
    const float total = simd_sum(held);
    if (simd_lane == 0) {
      mixes[size_t(n) * size_t(mix_hc) + size_t(o)] = total;
    }
  }
}

#define instantiate_hc_project(block)                             \
  template [[host_name("hc_project")]]                            \
  [[kernel]] void hc_project<block>(                              \
      const device float*, const device float*, device float*,    \
      const constant int&, const constant int&,                   \
      uint, uint, uint, uint, uint);

instantiate_hc_project(256)


template <typename T, int BLOCK>
[[kernel]] void hc_gates(
    const device float* mixes    [[buffer(0)]],
    const device float* scale    [[buffer(1)]],
    const device float* base     [[buffer(2)]],
    const device T* residual     [[buffer(3)]],
    device float* post_mix       [[buffer(4)]],
    device float* comb_mix       [[buffer(5)]],
    device T* layer_input        [[buffer(6)]],
    const constant int& M        [[buffer(7)]],
    const constant int& H        [[buffer(8)]],
    const constant float& hc_eps [[buffer(9)]],
    const constant float& hc_post_alpha [[buffer(10)]],
    const constant int& sinkhorn_iters  [[buffer(11)]],
    const constant int& combined [[buffer(12)]],
    uint2 gid    [[threadgroup_position_in_grid]],
    uint2 lid2   [[thread_position_in_threadgroup]],
    uint2 tg2    [[threads_per_threadgroup]]) {
  const uint lid = lid2.x;
  const uint tg_size = tg2.x;

  const int n = int(gid.x);
  const int tid = int(lid);

  // Without a combine (the independent form) the row is the pre and post
  // logits alone, the post gate carries the eps the pre gate does, and the
  // combine is the identity.
  const int mix_hc = combined ? M * 2 + M * M : M * 2;
  const device float* row = mixes + size_t(n) * size_t(mix_hc);

  threadgroup float pre[HC_MAX_MULT];
  threadgroup float post[HC_MAX_MULT];
  threadgroup float comb[HC_MAX_MULT * HC_MAX_MULT];

  if (tid < M) {
    const float logit = row[tid] * scale[0] + base[tid];
    pre[tid] = 1.0f / (1.0f + precise::exp(-logit)) + hc_eps;
  }
  if (tid < M) {
    const float logit = row[M + tid] * scale[1] + base[M + tid];
    post[tid] = 1.0f / (1.0f + precise::exp(-logit)) * hc_post_alpha + (combined ? 0.0f : hc_eps);
    if (gid.y == 0) post_mix[size_t(n) * size_t(M) + size_t(tid)] = post[tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (!combined) {
    if (tid < M * M && gid.y == 0) {
      comb_mix[size_t(n) * size_t(M) * size_t(M) + size_t(tid)] = (tid / M == tid % M) ? 1.0f : 0.0f;
    }
    const device T* res_n = residual + size_t(n) * size_t(M) * size_t(H);
    device T* out = layer_input + size_t(n) * size_t(H);
    const uint h_begin = gid.y * uint(HC_GATES_CHUNK);
    const uint h_end = min(uint(H), h_begin + uint(HC_GATES_CHUNK));
    for (uint h = h_begin + lid; h < h_end; h += tg_size) {
      float acc = 0.0f;
      for (int i = 0; i < M; ++i) {
        acc += pre[i] * float(res_n[size_t(i) * size_t(H) + size_t(h)]);
      }
      out[h] = T(acc);
    }
    return;
  }

  if (tid < M * M) {
    comb[tid] = row[2 * M + tid] * scale[2] + base[2 * M + tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < M) {
    float max_v = -INFINITY;
    for (int j = 0; j < M; ++j) max_v = max(max_v, comb[tid * M + j]);
    float sum = 0.0f;
    for (int j = 0; j < M; ++j) {
      comb[tid * M + j] = precise::exp(comb[tid * M + j] - max_v);
      sum += comb[tid * M + j];
    }
    for (int j = 0; j < M; ++j) comb[tid * M + j] = comb[tid * M + j] / sum + hc_eps;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < M) {
    float col_sum = 0.0f;
    for (int i = 0; i < M; ++i) col_sum += comb[i * M + tid];
    col_sum += hc_eps;
    for (int i = 0; i < M; ++i) comb[i * M + tid] = comb[i * M + tid] / col_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int iter = 0; iter < sinkhorn_iters - 1; ++iter) {
    if (tid < M) {
      float row_sum = 0.0f;
      for (int j = 0; j < M; ++j) row_sum += comb[tid * M + j];
      row_sum += hc_eps;
      for (int j = 0; j < M; ++j) comb[tid * M + j] = comb[tid * M + j] / row_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < M) {
      float col_sum = 0.0f;
      for (int i = 0; i < M; ++i) col_sum += comb[i * M + tid];
      col_sum += hc_eps;
      for (int i = 0; i < M; ++i) comb[i * M + tid] = comb[i * M + tid] / col_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid < M * M && gid.y == 0) {
    comb_mix[size_t(n) * size_t(M) * size_t(M) + size_t(tid)] = comb[tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device T* res_n = residual + size_t(n) * size_t(M) * size_t(H);
  device T* out = layer_input + size_t(n) * size_t(H);

  const uint h_begin = gid.y * uint(HC_GATES_CHUNK);
  const uint h_end = min(uint(H), h_begin + uint(HC_GATES_CHUNK));
  for (uint h = h_begin + lid; h < h_end; h += tg_size) {
    float acc = 0.0f;
    for (int i = 0; i < M; ++i) {
      acc += pre[i] * float(res_n[size_t(i) * size_t(H) + size_t(h)]);
    }
    out[h] = T(acc);
  }
}

#define instantiate_hc_gates(name, itype, block)                  \
  template [[host_name("hc_gates_" #name)]]                       \
  [[kernel]] void hc_gates<itype, block>(                         \
      const device float*, const device float*, const device float*, \
      const device itype*, device float*, device float*, device itype*, \
      const constant int&, const constant int&, const constant float&, \
      const constant float&, const constant int&, const constant int&, \
      uint2, uint2, uint2);

instantiate_hc_gates(bfloat16, bfloat, 256)


template <typename T, int BLOCK>
[[kernel]] void hc_collapse(
    const device float* mixes    [[buffer(0)]],
    const device float* scale    [[buffer(1)]],
    const device float* base     [[buffer(2)]],
    const device T* residual     [[buffer(3)]],
    device T* out                [[buffer(4)]],
    const constant int& M        [[buffer(5)]],
    const constant int& H        [[buffer(6)]],
    const constant float& hc_eps [[buffer(7)]],
    uint gid     [[threadgroup_position_in_grid]],
    uint lid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
  const int n = int(gid);
  const int tid = int(lid);

  threadgroup float gates[HC_MAX_MULT];
  if (tid < M) {
    const float logit = mixes[size_t(n) * size_t(M) + size_t(tid)] * scale[0] + base[tid];
    gates[tid] = 1.0f / (1.0f + precise::exp(-logit)) + hc_eps;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const device T* res_n = residual + size_t(n) * size_t(M) * size_t(H);
  device T* out_n = out + size_t(n) * size_t(H);
  for (uint h = lid; h < uint(H); h += tg_size) {
    float acc = 0.0f;
    for (int i = 0; i < M; ++i) {
      acc += gates[i] * float(res_n[size_t(i) * size_t(H) + size_t(h)]);
    }
    out_n[h] = T(acc);
  }
}

#define instantiate_hc_collapse(name, itype, block)               \
  template [[host_name("hc_collapse_" #name)]]                    \
  [[kernel]] void hc_collapse<itype, block>(                      \
      const device float*, const device float*, const device float*, \
      const device itype*, device itype*,                         \
      const constant int&, const constant int&, const constant float&, \
      uint, uint, uint);

instantiate_hc_collapse(bfloat16, bfloat, 256)


template <typename T>
[[kernel]] void hc_fold(
    const device T* x            [[buffer(0)]],
    const device T* residual     [[buffer(1)]],
    const device float* post_mix [[buffer(2)]],
    const device float* comb_mix [[buffer(3)]],
    device T* out                [[buffer(4)]],
    const constant int& n_rows   [[buffer(5)]],
    const constant int& M        [[buffer(6)]],
    const constant int& H        [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (M > HC_MAX_MULT) return;
  const int h = int(gid.x);
  const int n = int(gid.y);
  if (h >= H || n >= n_rows) return;

  const device float* comb_n = comb_mix + size_t(n) * size_t(M) * size_t(M);
  const device float* post_n = post_mix + size_t(n) * size_t(M);
  const float x_h = float(x[size_t(n) * size_t(H) + size_t(h)]);
  const device T* res_n = residual + size_t(n) * size_t(M) * size_t(H) + size_t(h);

  float r[HC_MAX_MULT];
  for (int i = 0; i < M; ++i) {
    r[i] = float(res_n[size_t(i) * size_t(H)]);
  }

  device T* out_n = out + size_t(n) * size_t(M) * size_t(H) + size_t(h);
  for (int j = 0; j < M; ++j) {
    float acc = post_n[j] * x_h;
    for (int i = 0; i < M; ++i) {
      acc += comb_n[i * M + j] * r[i];
    }
    out_n[size_t(j) * size_t(H)] = T(acc);
  }
}

#define instantiate_hc_fold(name, itype)                          \
  template [[host_name("hc_fold_" #name)]]                        \
  [[kernel]] void hc_fold<itype>(                                 \
      const device itype*, const device itype*,                   \
      const device float*, const device float*, device itype*,    \
      const constant int&, const constant int&, const constant int&, \
      uint2);

instantiate_hc_fold(bfloat16, bfloat)


template <typename T>
[[kernel]] void hc_mix(
    const device T* gates      [[buffer(0)]],
    const device T* normed     [[buffer(1)]],
    device T* y                [[buffer(2)]],
    const constant int& n_rows [[buffer(3)]],
    const constant int& M      [[buffer(4)]],
    const constant int& H      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int h = int(gid.x);
  const int n = int(gid.y);
  if (h >= H || n >= n_rows) return;

  const size_t base = size_t(n) * size_t(M) * size_t(H) + size_t(h);
  const device T* gr = gates + base;
  const device T* nr = normed + base;
  float acc = 0.0f;
  for (int s = 0; s < M; ++s) {
    const float g = float(gr[size_t(s) * size_t(H)]);
    const float v = float(nr[size_t(s) * size_t(H)]);
    acc += v / (1.0f + precise::exp(-g));
  }
  y[size_t(n) * size_t(H) + size_t(h)] = T(acc / float(M));
}

#define instantiate_hc_mix(name, itype)                           \
  template [[host_name("hc_mix_" #name)]]                         \
  [[kernel]] void hc_mix<itype>(                                  \
      const device itype*, const device itype*, device itype*,    \
      const constant int&, const constant int&, const constant int&, \
      uint2);

instantiate_hc_mix(bfloat16, bfloat)

template <typename T>
[[kernel]] void hc_inject(
    const device T* o          [[buffer(0)]],
    const device T* gates      [[buffer(1)]],
    device T* hyper            [[buffer(2)]],
    const constant int& n_rows [[buffer(3)]],
    const constant int& M      [[buffer(4)]],
    const constant int& H      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (M > HC_MAX_MULT) return;
  const int h = int(gid.x);
  const int n = int(gid.y);
  if (h >= H || n >= n_rows) return;

  const device T* gr = gates + size_t(n) * size_t(M);
  const float ov = float(o[size_t(n) * size_t(H) + size_t(h)]);
  device T* hr = hyper + size_t(n) * size_t(M) * size_t(H) + size_t(h);
  for (int s = 0; s < M; ++s) {
    const float logit = float(gr[s]) / float(M);
    const float g = 2.0f / (1.0f + precise::exp(-logit));
    hr[size_t(s) * size_t(H)] = T(float(hr[size_t(s) * size_t(H)]) + g * ov);
  }
}

#define instantiate_hc_inject(name, itype)                        \
  template [[host_name("hc_inject_" #name)]]                      \
  [[kernel]] void hc_inject<itype>(                               \
      const device itype*, const device itype*, device itype*,    \
      const constant int&, const constant int&, const constant int&, \
      uint2);

instantiate_hc_inject(bfloat16, bfloat)

template <typename T, int BLOCK>
[[kernel]] void ple_gate(
    const device T* key        [[buffer(0)]],
    const device T* query      [[buffer(1)]],
    const device T* value      [[buffer(2)]],
    device T* y                [[buffer(3)]],
    const constant int& M      [[buffer(4)]],
    const constant int& H      [[buffer(5)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint tg_size    [[threads_per_threadgroup]]) {
  const int n = int(gid) / M;
  const int s = int(gid) - n * M;

  const size_t stream = (size_t(n) * size_t(M) + size_t(s)) * size_t(H);
  const device T* kr = key + stream;
  const device T* qr = query + stream;
  const device T* vr = value + size_t(n) * size_t(H);
  device T* yr = y + stream;

  float local = 0.0f;
  for (uint i = lid; i < uint(H); i += tg_size) {
    local += float(kr[i]) * float(qr[i]);
  }

  threadgroup float partials[32];
  threadgroup float shared_gate[1];
  local = simd_sum(local);
  if (simd_group == 0) partials[simd_lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane == 0) partials[simd_group] = local;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    const float dot = simd_sum(partials[simd_lane]) * precise::rsqrt(float(H));
    if (simd_lane == 0) {

      float damped = precise::sqrt(fmax(fabs(dot), 1e-6f));
      damped = dot > 0.0f ? damped : (dot < 0.0f ? -damped : 0.0f);
      shared_gate[0] = 1.0f / (1.0f + precise::exp(-damped));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float gate = shared_gate[0];

  for (uint i = lid; i < uint(H); i += tg_size) {
    yr[i] = T(gate * float(vr[i]));
  }
}

#define instantiate_ple_gate(name, itype, block)                  \
  template [[host_name("ple_gate_" #name)]]                       \
  [[kernel]] void ple_gate<itype, block>(                         \
      const device itype*, const device itype*, const device itype*, \
      device itype*, const constant int&, const constant int&,    \
      uint, uint, uint, uint, uint);

instantiate_ple_gate(bfloat16, bfloat, 256)

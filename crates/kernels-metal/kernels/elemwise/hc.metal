#include <metal_stdlib>

using namespace metal;

// `Hc`: hyper-connections — the residual stream carried as `M` parallel
// copies, mixed into one layer input by learned gates, and folded back after
// the sublayer under a DOUBLY STOCHASTIC combiner. Ported organ-for-organ
// from `kernels-cuda/kernels/elemwise/hc.cuh`, whose four points these four
// entrypoints answer one for one.
//
// The mixing matrix is projected onto the Birkhoff polytope by Sinkhorn-Knopp:
// softmax the `M x M` logits along the row to seed, ONE column normalization,
// then `sinkhorn_iters - 1` alternating row/column sweeps. That off-by-one is
// the whole subtlety — 20 stated iterations are one seed column-norm plus 19
// sweeps, exactly as the MLX reference (`v4mlx/hc.py`) spells it — and it is
// the reason the loop below counts to `sinkhorn - 1` and not to `sinkhorn`.
//
// **THE SINKHORN RUNS IN FP32, ALWAYS.** The residual planes are bf16 and the
// gate planes are f32 by the op's own signature; the sums here are small and
// the doubly-stochastic property is what a bf16 accumulation would lose
// first. Nothing in this file accumulates in the storage type.
//
// The two gate curves are NOT the same function: the pre gate is
// `sigmoid(.) + eps` (a width weight, ~0..1) and the post gate is
// `alpha * sigmoid(.)` with the model's `alpha = 2` (a depth weight, 0..2).

// The largest stream fan the mixers unroll into threadgroup and register
// arrays — `MAX_HC_MULT` in the CUDA twin, and a hard refusal at the host
// entry rather than a shape check here.
constant constexpr int HC_MAX_MULT = 8;
// Hidden columns one `hc_gates` threadgroup collapses (`hc.rs` launches
// `ceil(H / HC_GATES_CHUNK)` of them per token).
constant constexpr int HC_GATES_CHUNK = 256;

// ---- expand ---------------------------------------------------------------

// Tiles one `H`-wide row across `M` residual streams: the `[N, H]` embedding
// becomes the `[N, M, H]` hyper stream every layer then rides. One thread per
// INPUT element, each writing its own `M` outputs.
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

// ---- rmsnorm, widened to f32 ----------------------------------------------

// RMS-normalises the WIDE stream row (`M * H` across, weightless) and lands it
// in f32: the mix coefficients read off it downstream are too sensitive for a
// bf16 round trip. One threadgroup per row.
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

// ---- the mix projection ---------------------------------------------------

// `mixes[n, o] = dot(normed[n, :], hc_fn[o, :])` — the per-token mix row the
// gates below split, projected out of the weightless-RMS-normed stream row by
// the layer's own `{attn,ffn}_hc.fn` plane (`[2M + M*M, M*H]`).
//
// **THIS IS A GEMM AND IT IS NOT `linear.matmul`**, for one reason: both
// operands are f32 and the dense gemm on this plane instantiates bf16 only.
// The rectangle is also the smallest a GEMM in this tree ever sees — `2M +
// M*M` is TWENTY-FOUR columns at the model's `hc_mult 4` — so a tiled point
// would launch one tile and a vector point would launch twenty-four rows of
// one; one threadgroup per `(row, column)` reducing `M*H` is the shape this
// arithmetic actually has. And the sinkhorn downstream is fp32 by this
// file's own header, which a bf16 detour to reach a bf16 gemm would undo.
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

  // **THE POINT IS ONE-DIMENSIONAL AND THE PAIR IS DERIVED**, not because a
  // `(column, row)` grid would be wrong but because this plane's position
  // attributes must all be scalars or all be vectors of one width, and the
  // three lane indices below are scalars.
  const int o = int(gid) % mix_hc;
  const int n = int(gid) / mix_hc;

  const device float* row = normed + size_t(n) * size_t(fan_in);
  const device float* w = hc_fn + size_t(o) * size_t(fan_in);

  float local = 0.0f;
  for (uint d = lid; d < uint(fan_in); d += tg_size) {
    local += row[d] * w[d];
  }
  local = simd_sum(local);
  // The cross-simdgroup fold, written so no lane addresses a slot the array
  // does not have: `partials` is `BLOCK / 32` long and a simdgroup is 32
  // lanes wide, so the lanes past the group count read a zero they supply
  // themselves rather than a threadgroup cell out of bounds.
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

// ---- gates ----------------------------------------------------------------

// Splits the per-token mix row into its three planes, Sinkhorn-normalises the
// combiner, publishes both gate matrices, and collapses the `M` streams into
// the sublayer's input under the pre gate. One threadgroup per token.
//
// **THE MIX ROW'S STRIDE IS `2M + M*M`**, exactly as the CUDA twin reads it,
// and it is now also the WIDTH of what the op hands over: `hc_project` above
// fires `rmsnorm(stream) @ hc_fn` — the `{attn,ffn}_hc.fn` plane the model
// text used to intern — and lands the `[N, 2M + M*M]` row the reference
// splits. This entry did not change to accept it; it never had to. What
// changed is that the leading `2M + M*M` floats of its operand are the mix
// row rather than the first columns of the `[N, M*H]` normed buffer that
// stood in for one while no plane produced it.
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
    uint2 gid    [[threadgroup_position_in_grid]],
    uint2 lid2   [[thread_position_in_threadgroup]],
    uint2 tg2    [[threads_per_threadgroup]]) {
  const uint lid = lid2.x;
  const uint tg_size = tg2.x;
  // `gid.y` is which `HC_GATES_CHUNK`-wide slice of the hidden width this
  // threadgroup collapses; every slice recomputes the (tiny) gate matrices,
  // and slice 0 alone publishes them. One threadgroup per token used to do
  // the whole row and left the device 15/16ths idle at decode.
  const int n = int(gid.x);
  const int tid = int(lid);

  const int mix_hc = M * 2 + M * M;
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
    post[tid] = 1.0f / (1.0f + precise::exp(-logit)) * hc_post_alpha;
    if (gid.y == 0) post_mix[size_t(n) * size_t(M) + size_t(tid)] = post[tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < M * M) {
    comb[tid] = row[2 * M + tid] * scale[2] + base[2 * M + tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // The seed: a softmax along each row, then `+ eps` so no entry is exactly
  // zero going into the alternating normalization.
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

  // The FIRST column normalization, outside the loop — this is the half-sweep
  // that makes `sinkhorn_iters` iterations cost `sinkhorn_iters - 1` passes
  // below.
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
      const constant float&, const constant int&,                 \
      uint2, uint2, uint2);

instantiate_hc_gates(bfloat16, bfloat, 256)

// ---- the trunk collapse ---------------------------------------------------

// `y[n, h] = sum_i g_i * streams[n, i*H + h]`, `g_i = sigmoid(mixes[n, i] *
// scale[0] + base[i]) + hc_eps` — the model's final fold of its `M` residual
// streams into the one row the final norm reads (`hc_head`). `M` gates and no
// post, no combiner, no Sinkhorn: the trunk's mix row is `M` wide, which is
// what `hc_project` lands off the `[M, M*H]` `hc_head.fn` plane. One
// threadgroup per token; `hc.cuh`'s `hc_head_postprocess`, transcribed.
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

// ---- fold -----------------------------------------------------------------

// Mixes the sublayer's output back into the `M` streams under the gate
// matrices the same token's `hc_gates` published:
//
//   y[j][h] = post[j] * x[h] + sum_i comb[i][j] * residual[i][h]
//
// One thread per `(token, h)` column, which is what lets the fold be read
// before it is written even when `y` and `residual` alias.
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

// ---- the GATED-RESIDUAL flavor (qwen4) ------------------------------------
//
// Same residual-stream algebra as the sinkhorn family above, a different gate:
// a low-rank GEMM chain produces per-element logits and a sigmoid of them is
// the whole mixing rule. There is no Birkhoff projection here and no f32 gate
// plane — the gates arrive in the activation's own dtype, off ordinary linear
// nodes. Ported organ for organ from `hc.cuh`'s `hc_mix`, `hc_inject` and
// `ple_gate`.
//
// The `win` staged-geometry seat the CUDA twins carry has no counterpart on
// this plane: nothing here reads a live-rows word, because this shell hands
// every one of these ops a grid already carved to the fire's rows.

// `y[h] = mean_s( sigmoid(gates[s*H + h]) * normed[s*H + h] )`.
//
// One thread per `(row, h)`: the reduction is over the STREAM fan, which is
// four, so it is a register loop and not a threadgroup one.
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

// `hyper[s*H + h] += 2 * sigmoid(gates[s] / M) * o[h]`, in place.
//
// One thread per `(row, h)`, which owns that column of every stream — so the
// `M` gate logits are read `H` times rather than staged, and at a stream fan
// of four that is cheaper than a barrier.
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

// The PLE gate: one threadgroup per `(row, stream)`, flattened the way the
// grouped norms flatten — group `b` is row `b / M`, stream `b % M`. The dot
// over `H` is a threadgroup reduction; the gate is the sigmoid of its SIGNED
// square root, and `sign(0)` is zero and not the clamp floor.
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
      // The reference's own damping: the square root of the clamped
      // magnitude, carrying the dot's SIGN.
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

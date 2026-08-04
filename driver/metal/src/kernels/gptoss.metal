// GPT-OSS's own kernels: the router's top-k, its SwiGLU, and the expert sum.
//
// Everything else this family needs is either shared (rms, kv_append, embed,
// argmax) or a variant living beside the kernel it varies (the routed matvec in
// quantized_qmv.metal, the sink attention in sdpa_sliding.metal, the YaRN rope
// in rope.metal).

#include <metal_stdlib>
using namespace metal;

struct RouterParams {
  uint n_experts;      // 32 on the 20B
  uint experts_per_token;  // 4
};

// top-k over the router's logits, then a softmax over ONLY the k that survive.
//
// One threadgroup per token row, one lane per expert. The selection is k rounds
// of "take the max, record it, mask it out" -- cheaper than any sort at these
// k, and it reproduces argpartition+take exactly, ties breaking toward the
// lower index.
//
// The reduction is THREADGROUP-wide, not simdgroup-wide. It used to be the
// latter, which is the same thing only while `n_experts <= 32`: gpt-oss has 32,
// so one simdgroup held the whole distribution and nothing was wrong. Qwen3-MoE
// has 128. At that width `simd_max` reduces four disjoint quarters, and every
// one of the four `simd_lid == 0` lanes then writes `expert_ids[r]` -- a race
// whose winner is a quarter's local maximum rather than the row's. So each
// simdgroup now posts its (max, lowest-winning-lane) pair and one thread picks
// across them.
//
// Emits both halves of the routing decision: the ids the routed matvecs index
// their weight stack with, and the normalized weights `ExpertCombine` sums by.
// mlx-lm softmaxes the top-k VALUES, not the full logit vector, so the weights
// sum to 1 over the chosen experts.
//
// `experts_per_token` is bounded by `kRouterMaxTopK` because the chosen values
// have to live somewhere between the selection and the softmax. The host
// refuses a larger k rather than letting this overflow.
constant constexpr uint kRouterMaxTopK = 16;
constant constexpr uint kRouterMaxSimdgroups = 32;  // 1024 threads / 32

// SCALED: gemma 4's router applies a LEARNED per-expert gain after its softmax,
// indexed by the selection. Two instantiations rather than an optional buffer:
// an unbound buffer is not a null pointer here, so a kernel that read it "only
// when scaled" would still have to be bound by llama and gpt-oss, which have no
// such tensor to bind.
template <typename T, bool SCALED>
[[kernel]] void router_topk(
    const device T* logits     [[buffer(0)]],
    device int* expert_ids     [[buffer(1)]],
    device T* expert_weights   [[buffer(2)]],
    constant RouterParams& p   [[buffer(3)]],
    const device T* per_expert_scale [[buffer(4)]],
    // uint3 rather than uint: Metal requires every position input to have the
    // same dimensionality, and the threadgroup position below is 3D.
    uint3 lid3 [[thread_position_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint3 tgsize [[threads_per_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  const uint lid = lid3.x;
  const uint n = p.n_experts;
  const uint k = min(p.experts_per_token, kRouterMaxTopK);
  const uint n_simd = min((tgsize.x + 31u) / 32u, kRouterMaxSimdgroups);
  constexpr float NEG_INF = -3.0e38f;

  // One threadgroup per token row. Every row routes independently -- which is
  // the whole reason a batched MoE cannot be one wider matmul.
  const uint row = tgid.y;
  logits += size_t(row) * size_t(n);
  expert_ids += size_t(row) * size_t(k);
  expert_weights += size_t(row) * size_t(k);

  float v = (lid < n) ? float(logits[lid]) : NEG_INF;

  threadgroup float part_v[kRouterMaxSimdgroups];
  threadgroup uint  part_i[kRouterMaxSimdgroups];
  threadgroup float chosen[kRouterMaxTopK];
  threadgroup uint  winner_of_round;

  for (uint r = 0; r < k; ++r) {
    // Per simdgroup: its max, and the lowest lane still holding it. Resolving
    // the tie by lane here and by index below keeps the choice deterministic
    // rather than dependent on which lane the hardware reduces last.
    const float m = simd_max(v);
    const uint w = simd_min((v == m) ? lid : 0xFFFFFFFFu);
    if (simd_lid == 0) {
      part_v[simd_gid] = m;
      part_i[simd_gid] = w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
      float best = NEG_INF;
      uint best_i = 0xFFFFFFFFu;
      for (uint sg = 0; sg < n_simd; ++sg) {
        // `>` keeps the FIRST simdgroup that attains the max, so the lowest
        // expert index wins a tie across simdgroups as well as within one.
        if (part_v[sg] > best) {
          best = part_v[sg];
          best_i = part_i[sg];
        }
      }
      expert_ids[r] = int(best_i);
      chosen[r] = best;
      winner_of_round = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == winner_of_round) v = NEG_INF;
    // Before the next round overwrites `part_*`, and after every thread has
    // read `winner_of_round`.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Softmax over the k selected logits only.
  if (lid == 0) {
    float mx = NEG_INF;
    for (uint r = 0; r < k; ++r) mx = max(mx, chosen[r]);
    float sum = 0;
    for (uint r = 0; r < k; ++r) {
      chosen[r] = fast::exp(chosen[r] - mx);
      sum += chosen[r];
    }
    for (uint r = 0; r < k; ++r) {
      float w = chosen[r] / sum;
      if (SCALED) w *= float(per_expert_scale[uint(expert_ids[r])]);
      expert_weights[r] = static_cast<T>(w);
    }
  }
}

#define instantiate_router_topk(name, itype)                       \
  template [[host_name("router_topk_" #name)]]                     \
  [[kernel]] void router_topk<itype, false>(                       \
      const device itype*, device int*, device itype*,             \
      constant RouterParams&, const device itype*,                 \
      uint3, uint, uint, uint3, uint3);                            \
  template [[host_name("router_topk_scaled_" #name)]]              \
  [[kernel]] void router_topk<itype, true>(                        \
      const device itype*, device int*, device itype*,             \
      constant RouterParams&, const device itype*,                 \
      uint3, uint, uint, uint3, uint3);

instantiate_router_topk(bfloat16, bfloat)

struct GptOssSwiGluParams {
  uint n;        // experts_per_token * intermediate
  float limit;   // 7.0
  float alpha;   // 1.702
};

// gpt-oss's SwiGLU, which is not anyone else's.
//
//   gate = min(gate, limit)              -- clamped ABOVE only
//   up   = clamp(up, -limit, limit)      -- clamped both ways
//   out  = gate * sigmoid(alpha*gate) * (up + 1)
//
// The `+1` on the linear branch and the asymmetric clamp are why `silu_mul`
// cannot serve: dropping either produces a model that runs and is wrong.
template <typename T>
[[kernel]] void gptoss_swiglu(
    const device T* gate            [[buffer(0)]],
    const device T* up              [[buffer(1)]],
    device T* out                   [[buffer(2)]],
    constant GptOssSwiGluParams& p  [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n) return;
  float g = float(gate[gid]);
  float u = float(up[gid]);
  g = min(g, p.limit);
  u = clamp(u, -p.limit, p.limit);
  const float sig = 1.0f / (1.0f + fast::exp(-p.alpha * g));
  out[gid] = static_cast<T>((g * sig) * (u + 1.0f));
}

#define instantiate_gptoss_swiglu(name, itype)                     \
  template [[host_name("gptoss_swiglu_" #name)]]                   \
  [[kernel]] void gptoss_swiglu<itype>(                            \
      const device itype*, const device itype*, device itype*,     \
      constant GptOssSwiGluParams&, uint);

instantiate_gptoss_swiglu(bfloat16, bfloat)

struct ExpertCombineParams {
  uint width;              // hidden
  uint experts_per_token;  // 4
};

// Sum the k experts' outputs, weighted by the router's softmax.
//
// One thread per hidden channel. `y` is [k, width] -- the routed down-projection
// wrote each expert's slot -- and the weights are the k the router emitted.
template <typename T>
[[kernel]] void expert_combine(
    const device T* y                [[buffer(0)]],
    const device T* expert_weights   [[buffer(1)]],
    device T* out                    [[buffer(2)]],
    constant ExpertCombineParams& p  [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x;
  if (c >= p.width) return;
  // `gid.y` is the token row. `y` is [rows, k, width] -- the routed
  // down-projection wrote each (row, slot) -- and the weights are [rows, k].
  const uint row = gid.y;
  y += size_t(row) * size_t(p.experts_per_token) * size_t(p.width);
  expert_weights += size_t(row) * size_t(p.experts_per_token);
  out += size_t(row) * size_t(p.width);
  float acc = 0;
  for (uint e = 0; e < p.experts_per_token; ++e) {
    acc += float(expert_weights[e]) * float(y[e * p.width + c]);
  }
  out[c] = static_cast<T>(acc);
}

#define instantiate_expert_combine(name, itype)                    \
  template [[host_name("expert_combine_" #name)]]                  \
  [[kernel]] void expert_combine<itype>(                           \
      const device itype*, const device itype*, device itype*,     \
      constant ExpertCombineParams&, uint2);

instantiate_expert_combine(bfloat16, bfloat)

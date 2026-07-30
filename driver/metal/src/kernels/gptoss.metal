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
// One threadgroup, one lane per expert. `n_experts` is 32 on this family, so a
// single simdgroup holds the whole distribution and the selection is k rounds of
// simd_max -- no sort, no scratch, and no dependence on the expert count beyond
// "fits in a threadgroup".
//
// Emits both halves of the routing decision: the ids the routed matvecs index
// their weight stack with, and the normalized weights `ExpertCombine` sums by.
// mlx-lm softmaxes the top-k VALUES, not the full logit vector, so the weights
// sum to 1 over the chosen experts.
template <typename T>
[[kernel]] void router_topk(
    const device T* logits     [[buffer(0)]],
    device int* expert_ids     [[buffer(1)]],
    device T* expert_weights   [[buffer(2)]],
    constant RouterParams& p   [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const uint n = p.n_experts;
  const uint k = p.experts_per_token;
  constexpr float NEG_INF = -3.0e38f;

  float v = (lid < n) ? float(logits[lid]) : NEG_INF;

  threadgroup float chosen[16];
  // k rounds: take the max, record it, mask it out. k is 4 here, so this is
  // cheaper than any sort and exactly reproduces argpartition+take for the
  // values -- ties break toward the lower index, as `simd_max` then the
  // lowest-lane test does.
  for (uint r = 0; r < k; ++r) {
    const float m = simd_max(v);
    // The lowest lane still holding the max wins, so a tie is resolved the same
    // way every time rather than by whichever lane the hardware reduces last.
    const uint winner = simd_min((v == m) ? lid : 0xFFFFFFFFu);
    if (simd_lid == 0) {
      expert_ids[r] = int(winner);
      chosen[r] = m;
    }
    if (lid == winner) v = NEG_INF;
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Softmax over the k selected logits only.
  if (simd_lid == 0) {
    float mx = NEG_INF;
    for (uint r = 0; r < k; ++r) mx = max(mx, chosen[r]);
    float sum = 0;
    for (uint r = 0; r < k; ++r) {
      chosen[r] = fast::exp(chosen[r] - mx);
      sum += chosen[r];
    }
    for (uint r = 0; r < k; ++r) {
      expert_weights[r] = static_cast<T>(chosen[r] / sum);
    }
  }
}

#define instantiate_router_topk(name, itype)                       \
  template [[host_name("router_topk_" #name)]]                     \
  [[kernel]] void router_topk<itype>(                              \
      const device itype*, device int*, device itype*,             \
      constant RouterParams&, uint, uint);

instantiate_router_topk(float32, float)
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

instantiate_gptoss_swiglu(float32, float)
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
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.width) return;
  float acc = 0;
  for (uint e = 0; e < p.experts_per_token; ++e) {
    acc += float(expert_weights[e]) * float(y[e * p.width + gid]);
  }
  out[gid] = static_cast<T>(acc);
}

#define instantiate_expert_combine(name, itype)                    \
  template [[host_name("expert_combine_" #name)]]                  \
  [[kernel]] void expert_combine<itype>(                           \
      const device itype*, const device itype*, device itype*,     \
      constant ExpertCombineParams&, uint);

instantiate_expert_combine(float32, float)
instantiate_expert_combine(bfloat16, bfloat)

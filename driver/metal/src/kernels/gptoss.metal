// GPT-OSS's model-specific SwiGLU. Generic routing lives in moe_route.metal.

#include <metal_stdlib>
using namespace metal;

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

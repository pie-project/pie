#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void attn_sink_rescale(
    const device T* o_in     [[buffer(0)]],
    device T* o_out          [[buffer(1)]],
    const device float* lse  [[buffer(2)]],
    const device T* sinks    [[buffer(3)]],
    uint3 tid  [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  constexpr float kLn2 = 0.69314718055994530942f;

  const uint head_dim = grid.x;
  const uint heads = grid.y;
  const uint d = tid.x;
  const uint h = tid.y;
  const uint t = tid.z;

  const float lse_val = lse[size_t(t) * size_t(heads) + size_t(h)];
  float r;
  if (!isfinite(lse_val)) {

    r = 1.0f;
  } else {
    const float diff = lse_val * kLn2 - static_cast<float>(sinks[h]);
    r = 1.0f / (1.0f + precise::exp(-diff));
  }

  const size_t i =
      (size_t(t) * size_t(heads) + size_t(h)) * size_t(head_dim) + size_t(d);
  o_out[i] = static_cast<T>(static_cast<float>(o_in[i]) * r);
}

#define instantiate_attn_sink_rescale(name, itype)                      \
  template [[host_name("attn_sink_rescale_" #name)]]                    \
  [[kernel]] void attn_sink_rescale<itype>(                             \
      const device itype*, device itype*, const device float*,          \
      const device itype*, uint3, uint3);

instantiate_attn_sink_rescale(bfloat16, bfloat)

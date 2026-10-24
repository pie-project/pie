#include <metal_stdlib>
using namespace metal;

// The gated-delta gates from the fused `[b | a]` projection: per value head,
// `g = -exp(a_log) * softplus(a + dt_bias)` and `beta = sigmoid(b)`, landed
// as one `[g | beta]` f32 row.
template <typename T>
[[kernel]] void gdn_prep(
    const device T* ba          [[buffer(0)]],
    const device float* a_log   [[buffer(1)]],
    const device T* dt_bias     [[buffer(2)]],
    device float* gates         [[buffer(3)]],
    const constant int& v_heads [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]]) {
  const uint h = pos.x;
  const uint t = pos.y;
  const size_t vh = size_t(v_heads);
  const size_t row = size_t(t) * 2 * vh;

  const float bv = float(ba[row + size_t(h)]);
  const float av = float(ba[row + vh + size_t(h)]);

  const float z = av + float(dt_bias[h]);
  const float sp = (z > 20.0f) ? z : metal::log(1.0f + metal::fast::exp(z));

  gates[row + size_t(h)] = -metal::fast::exp(a_log[h]) * sp;
  gates[row + vh + size_t(h)] = 1.0f / (1.0f + metal::fast::exp(-bv));
}

#define instantiate_gdn_prep(name, itype)                            \
  template [[host_name("gdn_prep_" #name)]]                          \
  [[kernel]] void gdn_prep<itype>(                                   \
      const device itype*, const device float*, const device itype*,          \
      device float*, const constant int&, uint2);

instantiate_gdn_prep(bfloat16, bfloat)

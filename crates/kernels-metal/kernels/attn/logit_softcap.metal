#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void logit_softcap(
    const device T* logits      [[buffer(0)]],
    device T* out               [[buffer(1)]],
    const constant float& cap   [[buffer(2)]],
    uint gid                    [[thread_position_in_grid]]) {
  const float x = static_cast<float>(logits[gid]);
  out[gid] = static_cast<T>(cap * precise::tanh(x / cap));
}

#define instantiate_softcap(name, itype)                               \
  template [[host_name("logit_softcap_" #name)]]                       \
  [[kernel]] void logit_softcap<itype>(                                \
      const device itype*, device itype*, const constant float&, uint);

instantiate_softcap(bfloat16, bfloat)

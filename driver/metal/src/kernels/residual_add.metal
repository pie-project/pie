// Raw-Metal residual add for Phase-0 decode (golden tags `attn_resid`, `layer_out`).
//
// out = x + residual, elementwise over the hidden dim. MLX `residual_add` is a
// plain mx::add; bf16+bf16 accumulates in float and rounds to bf16 per element.
// Used twice per layer (post-attention + post-MLP) for all 24 layers.
//
// Launch: dispatchThreads grid=(hidden, 1, 1), tg=(256, 1, 1).

#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void residual_add(
    const device T* x        [[buffer(0)]],   // [hidden]
    const device T* residual [[buffer(1)]],   // [hidden]
    device T* out            [[buffer(2)]],   // [hidden] (may alias x)
    const constant int& width [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  (void)width;  // flat token-major rows are exactly the dispatched extent
  out[tid] = T(float(x[tid]) + float(residual[tid]));
}

#define instantiate_residual_add(name, itype)                     \
  template [[host_name("residual_add_" #name)]]                   \
  [[kernel]] void residual_add<itype>(                            \
      const device itype*, const device itype*, device itype*, const constant int&, uint);

instantiate_residual_add(float32, float)
instantiate_residual_add(float16, half)
instantiate_residual_add(bfloat16, bfloat)

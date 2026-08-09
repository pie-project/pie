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
    uint tid [[thread_position_in_grid]]) {
  out[tid] = T(float(x[tid]) + float(residual[tid]));
}

// Prefill variant: rows are a uniform `row_pitch` elements apart, so the whole
// prompt runs as one dispatch instead of one per token. `tid.y` selects the
// row; the arithmetic is identical, which is what makes this safe to swap in.
// Mirrors `silu_mul_strided` exactly, including where the pitch sits.
template <typename T>
[[kernel]] void residual_add_strided(
    const device T* x        [[buffer(0)]],
    const device T* residual [[buffer(1)]],
    device T* out            [[buffer(2)]],   // may alias x
    const constant int& row_pitch [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]) {
  const size_t i = size_t(tid.y) * size_t(row_pitch) + size_t(tid.x);
  out[i] = T(float(x[i]) + float(residual[i]));
}

#define instantiate_residual_add_strided(name, itype)             \
  template [[host_name("residual_add_strided_" #name)]]           \
  [[kernel]] void residual_add_strided<itype>(                    \
      const device itype*, const device itype*, device itype*,    \
      const constant int&, uint2);

instantiate_residual_add_strided(bfloat16, bfloat)

#define instantiate_residual_add(name, itype)                     \
  template [[host_name("residual_add_" #name)]]                   \
  [[kernel]] void residual_add<itype>(                            \
      const device itype*, const device itype*, device itype*, uint);

instantiate_residual_add(bfloat16, bfloat)

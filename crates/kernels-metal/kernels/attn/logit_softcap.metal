// Raw-Metal gemma4 final logit softcap (decode M=1).
//
//   out[i] = cap * tanh(logits[i] / cap)        (cap = 30 on gemma4)
//
// gemma4 applies a tanh softcap to the final logits before sampling. Elementwise
// over the vocab (262144); one thread per element; float compute. Can run
// in-place (out == logits). Mirrors MLX `ops::softcap`. bfloat native.
// bind::Softcap = { Logits=0, Out=1 }; the cap is buffer 2, a scalar of its own.

#include <metal_stdlib>
using namespace metal;

// THE CAP IS A SCALAR, NOT A STRUCT.
//
// This took `constant SoftcapParams& p [[buffer(2)]]` with `{ float cap; uint
// unused; }` in it -- MLX's layout, and the shape `kernels-vulkan` and
// `kernels-wgpu` then copied. The second word was already dead:
//
//   The bound read `if (gid >= p.n) return;` with `p.n` stated as the
//   vocabulary -- ONE ROW -- while `LaunchRule::Elementwise` dispatches
//   `width * rows`. Every row after the first returned immediately and kept
//   whatever the arena held, which is a previous statement's output at that
//   offset and therefore different in fires of different shapes. THE GRID IS
//   THE EXTENT, so the bound went; the field stayed only to keep the struct's
//   size and layout.
//
// With the cap stated as `Const<f32>` there is no struct to keep the size of,
// so the field is gone too. The routine binds one `setBytes` where it bound a
// staged block, and word 0 of the statement's run is the same number it always
// was.

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

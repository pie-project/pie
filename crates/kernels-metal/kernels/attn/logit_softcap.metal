// Raw-Metal gemma4 final logit softcap (decode M=1).
//
//   out[i] = cap * tanh(logits[i] / cap)        (cap = 30 on gemma4)
//
// gemma4 applies a tanh softcap to the final logits before sampling. Elementwise
// over the vocab (262144); one thread per element; float compute. Can run
// in-place (out == logits). Mirrors MLX `ops::softcap`. bfloat native.
// bind::Softcap = { Logits=0, Out=1 }; Cap/N static geometry (SoftcapParams).

#include <metal_stdlib>
using namespace metal;

struct SoftcapParams {
  float cap;  // 30.0
  uint  unused;  // was the vocab; see below
};

// THE GRID IS THE EXTENT, so this bound is gone.
//
// This read `if (gid >= p.n) return;` with `p.n` stated as the vocabulary -- ONE
// ROW -- while `LaunchRule::Elementwise` dispatches `width * rows`. Every row
// after the first returned immediately and kept whatever the arena held,
// which is a previous statement's output at that offset and therefore
// different in fires of different shapes.
//
// The same defect `mlp/gated.metal` records at length: a per-row number
// cannot bound a whole-tensor dispatch, the text cannot state the whole
// count because `Tokens` is not known until a fire lowers, and the driver
// already spends the knowledge it does have on the grid. The field stays so
// the params struct keeps its size and layout.

template <typename T>
[[kernel]] void logit_softcap(
    const device T* logits      [[buffer(0)]],
    device T* out               [[buffer(1)]],
    constant SoftcapParams& p   [[buffer(2)]],
    uint gid                    [[thread_position_in_grid]]) {
  const float x = static_cast<float>(logits[gid]);
  out[gid] = static_cast<T>(p.cap * precise::tanh(x / p.cap));
}

#define instantiate_softcap(name, itype)                               \
  template [[host_name("logit_softcap_" #name)]]                       \
  [[kernel]] void logit_softcap<itype>(                                \
      const device itype*, device itype*, constant SoftcapParams&, uint);

instantiate_softcap(bfloat16, bfloat)

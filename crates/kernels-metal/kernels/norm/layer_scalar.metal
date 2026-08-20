// Raw-Metal gemma4 per-layer scalar multiply (decode M=1).
//
//   out[i] = x[i] * scalar[0]        (broadcast the learned [1] layer_scalar)
//
// gemma4 scales each decoder layer's output by a learned per-layer scalar
// (`layer_scalar`, shape [1]) broadcast over the hidden width. The scalar is read
// from a device buffer (I1: never setBytes) so it stays a stable resident slot.
// Elementwise over hidden; one thread per element; float compute, bfloat native.
// bind::LayerScalar = { X=0, Scalar=1, Out=2 }; N is the grid, and nothing else is bound.

#include <metal_stdlib>
using namespace metal;

// THE GRID IS THE EXTENT, so this bound is gone -- and now so is the struct
// that carried it.
//
// This read `if (gid >= p.n) return;` with `p.n` stated as the hidden width -- ONE
// ROW -- while `LaunchRule::Elementwise` dispatches `width * rows`. Every row
// after the first returned immediately and kept whatever the arena held,
// which is a previous statement's output at that offset and therefore
// different in fires of different shapes.
//
// The same defect `mlp/gated.metal` records at length: a per-row number
// cannot bound a whole-tensor dispatch, the text cannot state the whole
// count because `Tokens` is not known until a fire lowers, and the driver
// already spends the knowledge it does have on the grid.
//
// The field then STAYED, so that `LayerScalarParams` would keep its size and
// layout -- a struct held alive by nothing but its own ABI. With the routine
// stating no mark for it, there is no layout left to keep: the entrypoint
// takes three buffers, `layer_scalar_mul` binds three, and the plane stages no
// block for this symbol at all. `norm/layer_scalar.wgsl` and
// `norm/layer_scalar.slang` dropped theirs in the same change.

template <typename T>
[[kernel]] void layer_scalar_mul(
    const device T* x                [[buffer(0)]],
    const device T* scalar           [[buffer(1)]],  // [1]
    device T* out                    [[buffer(2)]],
    uint gid                         [[thread_position_in_grid]]) {
  const float s = static_cast<float>(scalar[0]);
  out[gid] = static_cast<T>(static_cast<float>(x[gid]) * s);
}

#define instantiate_layer_scalar(name, itype)                          \
  template [[host_name("layer_scalar_mul_" #name)]]                    \
  [[kernel]] void layer_scalar_mul<itype>(                             \
      const device itype*, const device itype*, device itype*, uint);

instantiate_layer_scalar(bfloat16, bfloat)

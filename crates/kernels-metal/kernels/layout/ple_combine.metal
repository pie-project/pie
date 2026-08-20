// Raw-Metal gemma4 PLE combine (decode M=1).
//
//   out[i] = (proj[i] + token[i]) * (1/sqrt(2))
//
// Final step of the gemma4 Per-Layer-Embedding precompute: the projected main
// embedding (post per-256-row RMSNorm) and the per-layer token embedding are
// summed and scaled by 1/sqrt(2), producing the [n_layers*ple_dim] PLE signal
// each decoder layer slices its [ple_dim] column from. Elementwise; one thread
// per element; float compute, bfloat native.
// bind::PleCombine = { Proj=0, Token=1, Out=2 }; the scale is buffer 3, a
// scalar of its own.

#include <metal_stdlib>
using namespace metal;

// THE SCALE IS A SCALAR, NOT A STRUCT.
//
// This took `constant PleCombineParams& p [[buffer(3)]]` with
// `{ float inv_sqrt2; uint unused; }` in it -- MLX's layout, and the shape
// `kernels-vulkan` and `kernels-wgpu` then copied. The second word was already
// dead:
//
//   The bound read `if (gid >= p.n) return;` with `p.n` stated as one row's
//   element count -- ONE ROW -- while `LaunchRule::Elementwise` dispatches
//   `width * rows`. Every row after the first returned immediately and kept
//   whatever the arena held, which is a previous statement's output at that
//   offset and therefore different in fires of different shapes. It is the
//   same defect `mlp/gated.metal` records at length: a per-row number cannot
//   bound a whole-tensor dispatch, the text cannot state the whole count
//   because `Tokens` is not known until a fire lowers, and the driver already
//   spends the knowledge it does have on the grid. THE GRID IS THE EXTENT, so
//   the bound went; the field stayed only to keep the struct's size and
//   layout.
//
// With the scale stated as `Const<f32>` there is no struct to keep the size of,
// so the field is gone too. The routine binds one `setBytes` where it bound a
// staged block, and word 0 of the statement's run is the same number it always
// was.

template <typename T>
[[kernel]] void ple_combine(
    const device T* proj            [[buffer(0)]],
    const device T* token           [[buffer(1)]],
    device T* out                   [[buffer(2)]],
    const constant float& inv_sqrt2 [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]]) {
  const float v = (static_cast<float>(proj[gid]) + static_cast<float>(token[gid])) *
                  inv_sqrt2;
  out[gid] = static_cast<T>(v);
}

#define instantiate_ple_combine(name, itype)                           \
  template [[host_name("ple_combine_" #name)]]                         \
  [[kernel]] void ple_combine<itype>(                                  \
      const device itype*, const device itype*, device itype*,         \
      const constant float&, uint);

instantiate_ple_combine(bfloat16, bfloat)

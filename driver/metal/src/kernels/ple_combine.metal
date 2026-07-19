// Raw-Metal gemma4 PLE combine (decode M=1).
//
//   out[i] = (proj[i] + token[i]) * (1/sqrt(2))
//
// Final step of the gemma4 Per-Layer-Embedding precompute: the projected main
// embedding (post per-256-row RMSNorm) and the per-layer token embedding are
// summed and scaled by 1/sqrt(2), producing the [n_layers*ple_dim] PLE signal
// each decoder layer slices its [ple_dim] column from. Elementwise; one thread
// per element; float compute, bfloat native.
// bind::PleCombine = { Proj=0, Token=1, Out=2 }; InvSqrt2/N static (PleCombineParams).

#include <metal_stdlib>
using namespace metal;

struct PleCombineParams {
  float inv_sqrt2;  // 0.7071067811865476
  uint  n;          // element count (n_layers * ple_dim)
};

template <typename T>
[[kernel]] void ple_combine(
    const device T* proj          [[buffer(0)]],
    const device T* token         [[buffer(1)]],
    device T* out                 [[buffer(2)]],
    constant PleCombineParams& p  [[buffer(3)]],
    uint gid                      [[thread_position_in_grid]]) {
  if (gid >= p.n) return;
  const float v = (static_cast<float>(proj[gid]) + static_cast<float>(token[gid])) *
                  p.inv_sqrt2;
  out[gid] = static_cast<T>(v);
}

#define instantiate_ple_combine(name, itype)                           \
  template [[host_name("ple_combine_" #name)]]                         \
  [[kernel]] void ple_combine<itype>(                                  \
      const device itype*, const device itype*, device itype*,         \
      constant PleCombineParams&, uint);

instantiate_ple_combine(float32, float)
instantiate_ple_combine(float16, half)
instantiate_ple_combine(bfloat16, bfloat)

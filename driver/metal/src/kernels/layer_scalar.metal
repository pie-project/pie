// Raw-Metal gemma4 per-layer scalar multiply (decode M=1).
//
//   out[i] = x[i] * scalar[0]        (broadcast the learned [1] layer_scalar)
//
// gemma4 scales each decoder layer's output by a learned per-layer scalar
// (`layer_scalar`, shape [1]) broadcast over the hidden width. The scalar is read
// from a device buffer (I1: never setBytes) so it stays a stable resident slot.
// Elementwise over hidden; one thread per element; float compute, bfloat native.
// bind::LayerScalar = { X=0, Scalar=1, Out=2 }; N static geometry (LayerScalarParams).

#include <metal_stdlib>
using namespace metal;

struct LayerScalarParams {
  uint n;  // hidden width
};

template <typename T>
[[kernel]] void layer_scalar_mul(
    const device T* x                [[buffer(0)]],
    const device T* scalar           [[buffer(1)]],  // [1]
    device T* out                    [[buffer(2)]],
    constant LayerScalarParams& p    [[buffer(3)]],
    uint gid                         [[thread_position_in_grid]]) {
  if (gid >= p.n) return;
  const float s = static_cast<float>(scalar[0]);
  out[gid] = static_cast<T>(static_cast<float>(x[gid]) * s);
}

#define instantiate_layer_scalar(name, itype)                          \
  template [[host_name("layer_scalar_mul_" #name)]]                    \
  [[kernel]] void layer_scalar_mul<itype>(                             \
      const device itype*, const device itype*, device itype*,         \
      constant LayerScalarParams&, uint);

instantiate_layer_scalar(float32, float)
instantiate_layer_scalar(float16, half)
instantiate_layer_scalar(bfloat16, bfloat)

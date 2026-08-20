// Deinterleave the packed QKV projection into three buffers.
//
// The projection writes `[rows, q_width + 2*kv_width]` contiguously; the rope,
// the KV append and the attention each want their own tensor. This is the copy
// that separates them.
//
// WHY A COPY AND NOT A VIEW. The handwritten Metal path split QKV by binding
// offsets — no dispatch at all — and that is the better shape. Expressing it
// needs a slice construct the DSL does not have, and adding one changes the
// shared lowering, so it would land on CUDA too. Until then the honest fix is
// the kernel the text already names, rather than a driver that knows what a
// QKV split is and binds three offsets on its own account.
//
// THE WIDTHS ARE OPERANDS, NOT LITERALS. `q_width` and `kv_width` arrive as
// dispatch constants because the TEXT states them. Baking them in is the
// defect `_d_256` was: a literal that fits one checkpoint and reads past the
// end of every head on the next.
//
// THEY ARE TWO SCALARS, NOT ONE STRUCT. This took `constant SplitQkvParams& p
// [[buffer(4)]]` holding `{ q_width, kv_width }` -- MLX's layout, and the
// shape `kernels-vulkan` and `kernels-wgpu` then copied from here. The
// argument for it was that a statement's params in stated order ARE the
// struct, so the driver could bind the address of the run it staged and let
// the layout follow from the order; the cost was that no signature could name
// either width, so nothing between the text and the shader could say which
// word was which. With both stated as `Const<u32>` marks the routine names
// them, metal binds each as its own `setBytes` at buffers 4 and 5, and words
// 0 and 1 of the statement's run are the same two numbers they always were.
//
// The ORDER is still the statement's and still load-bearing: read swapped,
// both boundaries below land inside a neighbouring projection rather than out
// of bounds, so nothing faults and every head is wrong.
//
// Launch: dispatchThreads grid=(q_width + 2*kv_width, rows, 1), tg=(256, 1, 1)
// — `LaunchRule::ElementwiseRows`, one thread per packed element.

#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void split_qkv(
    const device T* packed [[buffer(0)]],   // [rows, q_width + 2*kv_width]
    device T* q            [[buffer(1)]],   // [rows, q_width]
    device T* k            [[buffer(2)]],   // [rows, kv_width]
    device T* v            [[buffer(3)]],   // [rows, kv_width]
    const constant uint& q_width  [[buffer(4)]],
    const constant uint& kv_width [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]) {
  const uint packed_width = q_width + 2u * kv_width;
  const uint c = tid.x;
  // The grid is rounded up to whole threadgroups, so the tail runs over the
  // end of a row. Metal does not report that; it reads the next row.
  if (c >= packed_width) {
    return;
  }
  const size_t row = size_t(tid.y);
  const T value = packed[row * size_t(packed_width) + size_t(c)];
  if (c < q_width) {
    q[row * size_t(q_width) + size_t(c)] = value;
  } else if (c < q_width + kv_width) {
    k[row * size_t(kv_width) + size_t(c - q_width)] = value;
  } else {
    v[row * size_t(kv_width) + size_t(c - q_width - kv_width)] = value;
  }
}

#define instantiate_split_qkv(name, itype)                                  \
  template [[host_name("split_qkv_" #name)]]                                \
  [[kernel]] void split_qkv<itype>(                                         \
      const device itype*, device itype*, device itype*, device itype*,     \
      const constant uint&, const constant uint&, uint2);

instantiate_split_qkv(bf16, bfloat)

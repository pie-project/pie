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
// dispatch constants because the TEXT states them (`OpKind::Launch::params`).
// Baking them in is the defect `_d_256` was: a literal that fits one
// checkpoint and reads past the end of every head on the next.
//
// They arrive as ONE struct at one buffer, which is the tree's convention --
// `moe/route.metal` takes `constant RouterParams&` and so on. A statement's
// params in stated order ARE the struct, so the driver binds the address of
// the run it staged and the layout follows from the order.
//
// Launch: dispatchThreads grid=(q_width + 2*kv_width, rows, 1), tg=(256, 1, 1)
// — `LaunchRule::ElementwiseRows`, one thread per packed element.

#include <metal_stdlib>
using namespace metal;

/// The two widths, in the order the statement states them.
struct SplitQkvParams {
  unsigned int q_width;
  unsigned int kv_width;
};

template <typename T>
[[kernel]] void split_qkv(
    const device T* packed [[buffer(0)]],   // [rows, q_width + 2*kv_width]
    device T* q            [[buffer(1)]],   // [rows, q_width]
    device T* k            [[buffer(2)]],   // [rows, kv_width]
    device T* v            [[buffer(3)]],   // [rows, kv_width]
    const constant SplitQkvParams& p [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]) {
  const uint q_width = p.q_width;
  const uint kv_width = p.kv_width;
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
      const constant SplitQkvParams&, uint2);

instantiate_split_qkv(bf16, bfloat)

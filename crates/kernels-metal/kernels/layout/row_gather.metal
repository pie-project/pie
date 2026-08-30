#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void row_gather(
    const device T* in            [[buffer(0)]],
    device T* out                 [[buffer(1)]],
    const device uint* rows       [[buffer(2)]],
    const constant uint& width    [[buffer(3)]],
    const constant uint& count    [[buffer(4)]],
    uint2 tid                     [[thread_position_in_grid]]) {
  const uint c = tid.x;
  const uint i = tid.y;
  if (c >= width || i >= count) return;
  out[size_t(i) * size_t(width) + size_t(c)] =
      in[size_t(rows[i]) * size_t(width) + size_t(c)];
}

#define instantiate_row_gather(name, itype)                \
  template [[host_name("row_gather_" #name)]]              \
  [[kernel]] void row_gather<itype>(                       \
      const device itype*, device itype*,                  \
      const device uint*, const constant uint&,            \
      const constant uint&, uint2);

instantiate_row_gather(bfloat16, bfloat)
// **THE SECOND ELEMENT IS THE LOG-SUM-EXP PLANE'S.** `Fallback::Copy` moves
// every row-shaped rectangle a copied region reads or writes, and an
// attention region's `lse` column is f32 where its activations are bf16 — so
// one instantiation would refuse the very window the copy exists for. The
// CUDA twin never had to say this: it moves BYTES and is blind to the
// element. Here the element is the template argument, so the pair of them is
// two lines rather than none.
instantiate_row_gather(float32, float)

/// **SCATTER: the answers put back where their rows came from.**
///
/// `row_gather`'s body read the other way, and it lives in this file for the
/// reason its CUDA twin keeps both halves in one function: the two are a
/// permutation and its inverse, so a pair that could drift apart about what
/// `rows` MEANS is a pair that will. Row `i` of `in` lands at fire row
/// `rows[i]` of `out`; the rows the map does not name are not written, which
/// is what keeps a copy one consumer's slow path rather than a fact about
/// the arena.
template <typename T>
[[kernel]] void row_scatter(
    const device T* in            [[buffer(0)]],
    device T* out                 [[buffer(1)]],
    const device uint* rows       [[buffer(2)]],
    const constant uint& width    [[buffer(3)]],
    const constant uint& count    [[buffer(4)]],
    uint2 tid                     [[thread_position_in_grid]]) {
  const uint c = tid.x;
  const uint i = tid.y;
  if (c >= width || i >= count) return;
  out[size_t(rows[i]) * size_t(width) + size_t(c)] =
      in[size_t(i) * size_t(width) + size_t(c)];
}

#define instantiate_row_scatter(name, itype)               \
  template [[host_name("row_scatter_" #name)]]             \
  [[kernel]] void row_scatter<itype>(                      \
      const device itype*, device itype*,                  \
      const device uint*, const constant uint&,            \
      const constant uint&, uint2);

instantiate_row_scatter(bfloat16, bfloat)
instantiate_row_scatter(float32, float)

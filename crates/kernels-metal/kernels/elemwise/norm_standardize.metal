#include <metal_stdlib>
using namespace metal;

/// **THE VISION TOWER'S OUTPUT STANDARDIZATION** — the Metal mirror of
/// `kernels-cuda`'s `::pie::elemwise::standardize` (`.wiki/alto/multimodal.md`
/// §21.3).
///
/// `y = (x - bias) * scale`, per COLUMN, both planes `[width]` of the
/// activation's element, in place. `Gemma4VisionModel.forward` ends with
/// exactly that line, after the pooler's `sqrt(hidden)` and before the
/// multimodal embedder's projection.
///
/// **THE PLANES ARE THE COLUMN'S AND NOT THE ROW'S.** `tid.x` is the column
/// and `tid.y` the row, which is `add_bias`'s own grid one file over — so
/// both reads are `[tid.x]` and every row reads the same pair. A kernel that
/// indexed by `tid.y` would answer a different rectangle and, at any width
/// above the row count, would read past the plane on the second row.
///
/// **AND IT ROUNDS ONCE, AT THE STORE.** The difference and the product are
/// both taken in `float`; the only cast back to `T` is the assignment. This
/// is the whole reason the op is a row rather than `add_bias` with a negated
/// plane followed by a per-column multiply: the pooler has just run the rows
/// up by `sqrt(1152) ~= 33.9` and this is what brings them back, so where
/// `x` nearly cancels `bias` the surviving difference is many ulps of a bf16
/// quantum at `|x|` and a composed spelling would have rounded it away
/// between its two launches.
template <typename T>
[[kernel]] void standardize(
    device T* out             [[buffer(0)]],
    const device T* bias      [[buffer(1)]],
    const device T* scale     [[buffer(2)]],
    const constant int& width [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]) {
  const size_t i = size_t(tid.y) * size_t(width) + size_t(tid.x);
  out[i] = T((float(out[i]) - float(bias[tid.x])) * float(scale[tid.x]));
}

#define instantiate_standardize(name, itype)                          \
  template [[host_name("standardize_" #name)]]                        \
  [[kernel]] void standardize<itype>(                                 \
      device itype*, const device itype*, const device itype*,        \
      const constant int&, uint2);

instantiate_standardize(bfloat16, bfloat)

#include <metal_stdlib>
using namespace metal;

/// **THE ROW GATHER, AND THE TWO ANSWERS AN UNADDRESSABLE ID GETS.**
///
/// `y[n] = table[ids[n]]`, one thread per output element — and the id is
/// checked against the stated row count before it is an address, because a
/// gather that trusted its index would read the checkpoint at whatever offset
/// a corrupted stream named.
///
/// **WHAT THE GUARD WRITES IS THE OP'S, NOT THE SHADER'S**, and the two ops
/// this body serves answer differently — which is why `ZERO_OOB` is a stamp
/// and not a branch on data:
///
/// - `layout.embed` **CLAMPS TO ROW ZERO**, which is what
///   `kernels-cuda`'s `::pie::layout::embed` does (`layout.cuh`, `tid_raw >= 0
///   && tid_raw < vocab ? tid_raw : 0`) and what `embed_weighted.metal` does
///   one file over: a defined row rather than an address.
/// - `layout.embed_concat` **WRITES ZERO**, which is what
///   `kernels-cuda`'s `::pie::layout::embed_concat` does
///   (`embed_concat.cuh`, `y[at] = (id < 0 || id >= vocab) ? 0 : ...`). The
///   PLE's sixteen hashed ids per token are a HASH's output and not a
///   tokenizer's, so a head that hashed out of the table contributes nothing
///   to the concatenated row instead of contributing row zero sixteen times.
///
/// Neither answer is this plane's invention; both are the twin's, per op.
template <typename T, bool ZERO_OOB>
[[kernel]] void embed(
    const device int* ids      [[buffer(0)]],
    const device T* table      [[buffer(1)]],
    device T* y                [[buffer(2)]],
    const constant int& hidden [[buffer(3)]],
    const constant int& vocab  [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]) {
  const int c = int(tid.x);

  if (c >= hidden) {
    return;
  }
  const size_t n = size_t(tid.y);
  const int raw = ids[n];
  const bool addressable = (raw >= 0 && raw < vocab);
  const size_t at = n * size_t(hidden) + size_t(c);

  if (ZERO_OOB && !addressable) {
    y[at] = T(0);
    return;
  }
  const int row = addressable ? raw : 0;
  y[at] = table[size_t(row) * size_t(hidden) + size_t(c)];
}

#define instantiate_embed_dense(name, itype)                                \
  template [[host_name("embed_" #name)]]                                    \
  [[kernel]] void embed<itype, false>(                                      \
      const device int*, const device itype*, device itype*,                \
      const constant int&, const constant int&, uint2);                     \
  template [[host_name("embed_concat_" #name)]]                             \
  [[kernel]] void embed<itype, true>(                                       \
      const device int*, const device itype*, device itype*,                \
      const constant int&, const constant int&, uint2);

instantiate_embed_dense(bfloat16, bfloat)

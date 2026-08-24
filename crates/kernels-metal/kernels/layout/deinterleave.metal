// The two column cuts: one packed row out to two, and one layer's slice of a
// laid-out relay.
//
// `kernels-cuda/kernels/layout/deinterleave.cuh` holds the same pair (plus the
// five its own callers need) and this is the metal reading of those two.
//
// # `select_slice` and not `select`
//
// `select` is a Metal Standard Library function (`metal::select`), and this
// file says `using namespace metal;` like every other shader in the tree. An
// entry point of that name would shadow the builtin inside its own definition
// and be a name collision for any later reader of this file. The POINT is
// still `layout.select`; only the entrypoint is spelled apart from it.
//
// # Why a copy and not a binding offset
//
// `select` is a base and an offset — the whole arithmetic — so a plane could
// one day answer it at BINDING with a view and never launch at all, which the
// declaration's own doc says. Until the binder can say that, a copy is what
// says it: a slice this kernel wrote is a rectangle the arena owns, with no
// aliasing rule for a later pass to discover. `split_rows` is the same shape
// of statement with both halves kept.
//
// `stride` AND `width` are both operands of `select_slice` and neither is
// derivable from the other: the packed relay's row is `layers * width` and the
// result's is `width`, and the statement's param says WHICH layer, not how
// many. `offset = layer * width` is computed on the host, where the layer is
// known, and the host is also where the "does this row reach that slice"
// bound check lives.
//
// Launch (both): dispatchThreads grid=(row width, rows, 1), tg=(256, 1, 1) —
// `elementwise_rows`, one thread per element the kernel reads.
//
// # UNVERIFIED
//
// Written without a Metal toolchain or an Apple device. Never compiled, never
// run, no number compared against anything.

#include <metal_stdlib>
using namespace metal;

// `[rows, left_dim + right_dim]` in, `[rows, left_dim]` and
// `[rows, right_dim]` out. The grid walks the SOURCE row, so each element is
// read once and lands in exactly one of the two results.
template <typename T>
[[kernel]] void split_rows(
    const device T* src           [[buffer(0)]],  // [rows, left + right]
    device T* left                [[buffer(1)]],  // [rows, left]
    device T* right               [[buffer(2)]],  // [rows, right]
    const constant int& left_dim  [[buffer(3)]],
    const constant int& right_dim [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]) {
  const int total = left_dim + right_dim;
  const int c = int(tid.x);
  // The grid is rounded up to whole threadgroups, so the tail runs over the
  // end of a row. Metal does not report that; it reads the next row.
  if (c >= total) {
    return;
  }
  const size_t row = size_t(tid.y);
  const T value = src[row * size_t(total) + size_t(c)];
  if (c < left_dim) {
    left[row * size_t(left_dim) + size_t(c)] = value;
  } else {
    right[row * size_t(right_dim) + size_t(c - left_dim)] = value;
  }
}

#define instantiate_split_rows(name, itype)                                 \
  template [[host_name("split_rows_" #name)]]                               \
  [[kernel]] void split_rows<itype>(                                        \
      const device itype*, device itype*, device itype*,                    \
      const constant int&, const constant int&, uint2);

instantiate_split_rows(bfloat16, bfloat)

// `[rows, stride]` in, `[rows, width]` out, taken at column `offset`. This IS
// `split_rows` keeping one of `layers` parts instead of both of two, and the
// grid walks the RESULT row rather than the source's because the source row is
// wider than what is kept.
template <typename T>
[[kernel]] void select_slice(
    const device T* table      [[buffer(0)]],  // [rows, stride]
    device T* y                [[buffer(1)]],  // [rows, width]
    const constant int& stride [[buffer(2)]],
    const constant int& offset [[buffer(3)]],
    const constant int& width  [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]) {
  const int c = int(tid.x);
  if (c >= width) {
    return;
  }
  const size_t row = size_t(tid.y);
  y[row * size_t(width) + size_t(c)] =
      table[row * size_t(stride) + size_t(offset) + size_t(c)];
}

#define instantiate_select_slice(name, itype)                               \
  template [[host_name("select_slice_" #name)]]                             \
  [[kernel]] void select_slice<itype>(                                      \
      const device itype*, device itype*, const constant int&,              \
      const constant int&, const constant int&, uint2);

instantiate_select_slice(bfloat16, bfloat)

// pie_half2.cuh -- the honest name for the packed-half arithmetic, forwarding
// to the file that implements it.
//
// WHY THIS FILE STILL EXISTS, AND WHY IT IS TWO LINES
//
// `new-horizon.md` §13.4 draws a line between two kinds of shim:
//
//   * a shim named after NVIDIA's header EXACTLY, because the includer is
//     upstream source we do not own -- NVRTC matches `includeNames[]` against
//     the literal string in the directive, so carrying `cuda_fp16.h` under
//     that spelling leaves FlashInfer unmodified and makes the resolution
//     ours;
//   * a shim with an honest name, because the includer is one of OUR `.cu`
//     files, which is being rewritten anyway and has no business pretending
//     to include a vendor header it does not need.
//
// Both callers exist, so both names must resolve. What must NOT exist is two
// implementations. `cuda_fp16.h` was written as a strict superset of this
// file precisely so the merge would be one `#include`, and this is it.
//
// The alternative -- keeping both bodies and relying on no translation unit
// including both -- is a one-definition-rule violation waiting for the first
// kernel that needs a packed multiply and an fp16 attention variant in the
// same compile. `build.rs` now walks `csrc/src` and carries every file in it,
// so both ARE in every header set; "no TU includes both today" stopped being
// a property anyone could check.
//
// The bit-parity evidence lives with the implementation: 35 of 35 rows
// bit-identical against nvcc over 32,945,058 comparisons, native and with
// every architecture fallback forced on.

#pragma once

#include "cuda_fp16.h"

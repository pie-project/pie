//===-- math_constants.h - one constant, and it is a bit pattern ---------===//
//
// `#include <math_constants.h>`, answered. The toolkit header is 150-odd
// `#define`s of float and double limits; the device text in this tree takes
// exactly one of them out.
//
// # Measured, not assumed
//
// `kernels/attn/attention_mla_naive.cuh` opens it at `:18` and uses
// `CUDART_INF_F` as the online-softmax running maximum's identity — seven
// sites, `:130`, `:183`, `:187`, `:455`, `:603`, `:608`, `:622`, `:624`, all
// of the form `float m = -CUDART_INF_F;` or a comparison against it. Without
// this file:
//
// ```text
// probe: NVRTC 13.0, sm_89
//   attention_mla_naive.cuh(13): catastrophic error:
//     cannot open source file "math_constants.h"
// ```
//
// The definition below is the toolkit's, character for character —
// `__int_as_float(0x7f800000)` is IEEE-754 positive infinity spelled as the
// bit pattern rather than as `HUGE_VALF`, and `__int_as_float` is an NVRTC
// builtin needing no shim. The equivalence is covered by the same measurement
// `csrc/shim/cuda_pipeline.h` records: the device half of
// `attention_mla_naive.cuh` compiled through this file and through
// `/usr/local/cuda/include` produced **byte-identical PTX**, 117 621 bytes.
//
// # Why not the whole header
//
// Because a shim that answers more than its callers ask is a shim whose extra
// rows are unmeasured. `csrc/shim/cstdint`'s argument in full; this file is
// the sharpest instance of it in the directory, at one constant out of a
// hundred and fifty. Adding a row here is cheap and should be done the same
// way: because a real include reached for it and a probe showed the failure.
//
//===----------------------------------------------------------------------===//
#pragma once

#define CUDART_INF_F __int_as_float(0x7f800000)

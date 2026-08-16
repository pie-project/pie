// pie_fp8.cuh -- the honest name for the fp8 conversions, and the include
// order they need, in one file.
//
// WHY THIS FILE IS NOT JUST `#include "cuda_fp8.h"`
//
// `new-horizon.md` §13.4 says a shim impersonates a vendor header only when
// the includer is source we cannot edit. FlashInfer writes
// `#include <cuda_fp8.h>`, so that spelling must resolve; our own
// `attn/kv_paged.cu` and `quant/dequant_fp8.cu` write `#include "prelude/fp8.cuh"`,
// so this one must too. Two names, one implementation -- keeping two bodies
// would be an ODR violation waiting for the first translation unit that
// quantises a KV page and runs an fp8 attention variant together, and since
// `src/source.rs` lists every file under `kernels/`, both spellings are
// in every header set, so nothing prevents that TU from existing.
//
// But the forward is TWO lines, not one, and the second one is the point.
// `cuda_fp8.h` deliberately includes nothing: a header in the set that reached
// for another would create a diamond its includer never asked for, and every
// FlashInfer file that includes it already includes `<cuda_fp16.h>` first. So
// it takes `__half` and `__half_raw` on faith, guarded on
// `__CUDA_FP16_TYPES_EXIST__`, and a translation unit that includes it alone
// gets a name error.
//
// That is right for the vendor spelling and wrong for this one. Our `.cu`
// files are not FlashInfer; they include one header and expect it to work.
// Encapsulating the vendor's ordering requirement is exactly what the honest
// name is FOR -- and it is why this file survived becoming a forwarder rather
// than being deleted.
//
// The bit-parity evidence lives with the implementations: 28 of 28 checks
// bit-identical against nvcc over every fp8 byte pattern, and 35 of 35 for the
// half types over 32,945,058 comparisons.

#pragma once

// First: the half types `cuda_fp8.h` names but does not define.
#include "cuda_fp16.h"
// Then the conversions themselves.
#include "cuda_fp8.h"

//===-- cuda_runtime_api.h - the C half of a header we already carry ------===//
//
// This is what `#include <cuda_runtime_api.h>` resolves to when the compiler
// is NVRTC and the include path is a header set carried in the binary rather
// than a directory on a disk.
//
// # Measured, not assumed
//
// `cute/util/debug.hpp:38` includes it, and CuTe reaches that file from
// `cute/tensor.hpp`, which is the first line of every CUTLASS unit. Without
// this file the diagnostic is
//
//     cute/util/debug.hpp(38): catastrophic error:
//     cannot open source file "cuda_runtime_api.h"
//
// and the compile stops before a single template has been parsed -- so the
// failure arrives with nothing useful said about anything else, which is the
// same shape `cstddef` had. Found with a `libnvrtc` probe (`G3`) against the
// real header text, with `/usr/local/cuda/include` REMOVED from the include
// list: the toolkit has this header, the carried set did not, and every
// CUTLASS probe before `G3` had the toolkit on the path and so never asked.
//
// # Why it forwards rather than declares
//
// Upstream, `cuda_runtime.h` is the C++ header and `cuda_runtime_api.h` is
// the C one it includes; the split exists so a `.c` file can have the runtime
// API without the C++ overloads. Under NVRTC there is no `.c` file and no
// second compilation model, so the split has nothing to separate. Carrying
// two copies of the same declarations would give a translation unit two
// places to disagree, and the shim's whole argument is that a translation
// unit should have exactly one of anything.
//
// So this forwards, and `csrc/shim/cuda_runtime.h` remains the single place
// the runtime surface is written down -- including the two programmatic
// dependent launch intrinsics added for the CUTLASS mainloop, which
// `KernelModule::fire_ex`'s launch attributes pair with.
//
// # What CuTe actually takes from it
//
// Nothing that is not already there. `debug.hpp` names `cudaError_t`,
// `cudaGetLastError` and `cudaGetErrorString` inside `CUTE_CHECK_LAST`, all
// of which are guarded by `#if !defined(__CUDACC_RTC__)` -- so under NVRTC
// the include is required to RESOLVE and its contents are never used. That is
// worth stating because it is the argument against declaring anything new
// here: a row added to satisfy a resolve, whose names are then unreachable,
// is text that can drift without any compile noticing.
//
// Added by `moe-cutlass` for the FlashInfer MoE unit. Coordination table is
// in `csrc/shim/README.md`.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda_runtime.h>

#pragma once

// A stand-in for <cuda_runtime.h>, for the launch-ABI proof.
//
// Unlike this crate's other stubs, nothing here is a RECORDING stub: the ABI
// proof compiles and never runs, so no behaviour has to be reproduced. All it
// needs is for `rope.hpp` to parse, and the only thing `rope.hpp` takes from
// the CUDA runtime is `cudaStream_t`.
//
// A stub rather than the real header because the job that runs this documents
// that it needs neither a CUDA toolkit nor a GPU, and because a proof whose
// answer depends on which CUDA happens to be installed is not one. Found ahead
// of any real header: the stub directory is the only include path the ABI test
// adds besides `csrc/src`.
//
// `cudaStream_t` is spelled EXACTLY as the real header spells it -- a pointer
// to an incomplete `CUstream_st` -- and not as the `void*` this crate's
// recording stubs use. The check being made is a function-pointer
// initialisation, which admits no parameter conversions, so the type on both
// sides of it has to be the one the shipping build would resolve. `void*`
// would still typecheck (the typedef is consistent within the translation
// unit) but it would quietly make the stream operand the one operand this
// proof does not really examine.

struct CUstream_st;
using cudaStream_t = CUstream_st*;

#pragma once

// A stand-in for <cublas_v2.h>, for the launch-ABI proof.
//
// The same bargain `cuda_runtime.h` next door makes, for the same reason: the
// proof compiles and never runs, so nothing here has to behave. It exists
// because two of `attn`'s rows -- the MLA absorb pair -- are declared in
// `gemm/gemm.hpp`, which takes a cuBLAS handle, and a family is not proven
// while two of its rows sit outside the shim.
//
// `cublasHandle_t` is spelled EXACTLY as `cublas_api.h` spells it: a pointer
// to an incomplete `cublasContext`. Not `void*`. The check is a
// function-pointer initialisation, which admits no conversions, so a
// convenient spelling here would silently make the handle the one operand the
// proof does not examine -- the same trap the stream stub documents.

struct cublasContext;
using cublasHandle_t = cublasContext*;

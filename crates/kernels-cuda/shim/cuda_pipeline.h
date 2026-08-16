//===-- cuda_pipeline.h - the three cp.async primitives, as PTX ----------===//
//
// `#include <cuda_pipeline.h>`, answered. The toolkit header is a host-side
// C++ file that reaches `<cuda_pipeline_helpers.h>`, `<cstddef>` and a
// `nvcuda::experimental` namespace NVRTC cannot open; what the device text
// actually TAKES OUT of it is three functions, and all three are one inline
// PTX instruction each.
//
// # Measured, not assumed — and this one is measured to the byte
//
// `kernels/attn/attention_mla_naive.cuh`'s tensor-core kernel
// (`mla_mma_paged_kernel`) stages its KV tiles through shared memory with
// `cp.async` and opens the header for it at `:16`. Three call sites, and
// nothing else in the file touches it:
//
// ```text
// attention_mla_naive.cuh:534   __pipeline_memcpy_async(dst, src, 16);
// attention_mla_naive.cuh:539   __pipeline_commit();
// attention_mla_naive.cuh:556   __pipeline_commit();
// attention_mla_naive.cuh:558   __pipeline_wait_prior(kStages - 1);
// ```
//
// Without this file the whole compile stops on the first line:
//
// ```text
// probe: NVRTC 13.0, sm_89, --fmad=false --prec-div=true --prec-sqrt=true
//   attention_mla_naive.cuh(12): catastrophic error:
//     cannot open source file "cuda_pipeline.h"
//   1 catastrophic error detected
// ```
//
// **THE EQUIVALENCE IS PROVEN, NOT ARGUED.** A hand-written intrinsic shim is
// exactly the kind of thing that compiles and then quietly emits a different
// instruction — a `ca` cache policy where the toolkit emits `cg`, a missing
// `src-size` operand that changes the zero-fill behaviour. So the device half
// of `attention_mla_naive.cuh` was compiled twice through the same NVRTC, at
// the same architecture and the same numerics flags, differing only in whether
// this file or `/usr/local/cuda/include/cuda_pipeline.h` answered the include:
//
// ```text
//   this shim      117 621 bytes of PTX, 2 .entry
//   toolkit        117 621 bytes of PTX, 2 .entry
//   cmp            IDENTICAL
// ```
//
// Byte-identical, register allocation included. The `16, 16` below is why: a
// first draft emitted the three-operand `cp.async.cg.shared.global [d], [s],
// 16;` and the toolkit emitted the four-operand `…, 16, 16;`. PTX defines
// `src-size` to default to `cp-size`, so the two forms mean the same thing and
// the first draft was CORRECT — but "means the same thing" is a claim a reader
// has to check against the ISA, and "the bytes are the same" is a claim
// anybody can re-run. The four-operand form is written because it costs
// nothing and turns the second claim into the first.
//
// # Why not `#ifdef __CUDACC_RTC__` with the toolkit header under `#else`
//
// That is `supergraph-nvrtc`'s pattern (`new-horizon.md` §62, and the right
// answer where a builtin has to be hand-declared for NVRTC and left alone for
// nvcc). It does not apply here for a mechanical reason: this file is only
// ever REACHED by NVRTC. `src/source.rs` lists `csrc/` and hands the result to
// `nvrtcCreateProgram` as `includeNames[]`; no C++ compiler has `csrc/shim` on
// an include path, so the `#else` arm would be text nothing can select. The
// pattern earns its keep when one file is compiled by both, which is true of
// a vendored header and false of every file in this directory.
//
// # The 8- and 4-byte arms are unreached and are still written
//
// Every call site in this tree passes 16. The toolkit's `__pipeline_memcpy_async`
// takes a size and dispatches on it, and a shim that accepted only the size it
// happened to see would fail the next caller with "no matching function"
// rather than with a wrong answer — but it would fail at a call site that is
// correct against the real header, which is the diagnosis that sends a reader
// to the wrong file. The cache policies match the toolkit's: `cg` (bypass L1)
// for a 16-byte copy, `ca` below it, which is what the PTX ISA recommends and
// what the toolkit emits.
//
//===----------------------------------------------------------------------===//
#pragma once

// `dst` is a generic pointer into shared memory; `cp.async` wants the shared
// window's 32-bit address, which is what `__cvta_generic_to_shared` produces.
// NVRTC supplies that as a builtin, so it needs no shim of its own.
__device__ __forceinline__ void __pipeline_memcpy_async(void* dst, const void* src,
                                                        unsigned long n) {
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    if (n == 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 16;" ::"r"(s), "l"(src));
    } else if (n == 8) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8, 8;" ::"r"(s), "l"(src));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4, 4;" ::"r"(s), "l"(src));
    }
}

__device__ __forceinline__ void __pipeline_commit() {
    asm volatile("cp.async.commit_group;" ::);
}

// `cp.async.wait_group` takes an IMMEDIATE, which is why this is a template
// over the count and a macro over the template rather than a function taking
// an `int`. The toolkit reaches the same place by a different route; the call
// site spells `__pipeline_wait_prior(kStages - 1)` either way, and `kStages`
// is a `constexpr int`, so the argument is a constant expression at every
// caller in this tree. A non-constant argument is a compile error here and is
// a compile error against the toolkit header too — the "n" constraint is the
// same one it uses.
template <int N>
__device__ __forceinline__ void pie_pipeline_wait_prior() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}
#define __pipeline_wait_prior(N) pie_pipeline_wait_prior<(N)>()

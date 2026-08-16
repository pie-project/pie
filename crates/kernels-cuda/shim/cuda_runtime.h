//===-- cuda_runtime.h - two typedefs, standing in for a host API -------===//
//
// `ushort` and `uchar`. This is what `#include <cuda_runtime.h>` resolves to
// when the compiler is NVRTC and the include path is a header set carried in
// the binary rather than a directory on a disk.
//
// # Why a header this small answers a header that large
//
// Thirteen of the twenty-eight files in FlashInfer's attention closure
// `#include <cuda_runtime.h>`, and the real one is the CUDA runtime API --
// `cudaMalloc`, `cudaStream_t`, `cudaError_t`, thousands of host declarations
// that NVRTC cannot compile and a device cannot call. Every FlashInfer use of
// those names is in a HOST function, and NVRTC refuses host functions outright
// ("a function explicitly marked as a __host__ function is not allowed in JIT
// mode"), so the vendored copies guard those functions away and the names go
// with them.
//
// What does NOT go with them was measured, and it is two words wide.
// `math.cuh`'s `ptx_exp2` and `tanh` wrappers declare `ushort y_u16;` as the
// operand of an `ex2.approx.f16` / `tanh.approx.f16` asm, inside `__device__`
// functions we very much want. `ushort` is a `vector_types.h` typedef, and
// `vector_types.h` is exactly what the real `cuda_fp16.h` declines to include
// under `__CUDACC_RTC__`:
//
// ```text
// /usr/local/cuda/include/cuda_fp16.h:129   #if !defined(__CUDACC_RTC__)
//                                    131   #include "vector_types.h"
// ```
//
// NVRTC's own preamble predefines the vector TYPES -- `uint4`, `float4`,
// `uchar4`, `dim3`, `uint3`, seventeen of seventeen probed -- and neither of
// the two scalar aliases. So this file is the difference between those two
// lists, and nothing else.
//
// # Why carrying beats guarding, here
//
// The alternative was thirteen `#ifndef __CUDACC_RTC__` guards in vendored
// source plus a hand-written `typedef unsigned short ushort;` spliced into
// `math.cuh` -- which is no longer a guard, and a diff that stops reading as
// "N guards added". Carrying the name leaves thirteen upstream files closer to
// upstream, and it is the rule `csrc/shim/cooperative_groups.h` already states:
// when the includer is source we do not own, impersonate the header under
// NVIDIA's own spelling and make the resolution ours.
//
// # What is deliberately NOT here
//
// `cudaError_t`, `cudaStream_t`, `dim3`, `cudaSuccess`. Not one of them,
// deliberately: a shim that supplied them would let a host dispatch function
// compile far enough to fail somewhere else, and the point of the guards in
// the vendored tree is that the host half of FlashInfer -- the launcher, the
// scheduler, the error macros -- is precisely the half this crate replaces
// with `cuLaunchKernel` from Rust. Refusing to fake the launch API is what
// keeps that boundary from blurring.

#ifndef PIE_NVRTC_CUDA_RUNTIME_H_
#define PIE_NVRTC_CUDA_RUNTIME_H_

// The fixed-width integers, because the real `cuda_runtime.h` makes them
// visible too -- through `crt/common_functions.h` and `<stdint.h>` -- and
// `mma.cuh` leans on exactly that: it includes `<cuda_runtime.h>` and
// `<type_traits>`, never `<cstdint>`, and then declares `uint32_t* R` in
// twenty `ldmatrix`/`mma` wrappers. Reproducing the transitive include is
// what keeps that file needing no guard at all.
#include <cstdint>

// `vector_types.h`'s two scalar aliases, spelled as it spells them.
typedef unsigned short ushort;
typedef unsigned char uchar;

// # The two programmatic-dependent-launch intrinsics
//
// `cudaGridDependencySynchronize` and `cudaTriggerProgrammaticLaunchCompletion`
// are declared in the toolkit's `cuda_device_runtime_api.h`, which the real
// `cuda_runtime.h` pulls in, and NVRTC supplies NEITHER header: a compile that
// reaches either name answers *"identifier is undefined"*, measured.
//
// # Why these belong when `cudaStream_t` does not
//
// The paragraph above refuses the launch API on a boundary argument, and these
// do not cross it. `cudaError_t` and `cudaStream_t` are HOST types whose only
// use is in a host launcher this crate replaces; supplying them would let a
// launcher compile far enough to fail elsewhere. These two are `__device__`
// builtins with no host form at all -- they are single PTX instructions, they
// are callable only from inside a `__global__`, and the code that calls them is
// device text we carry verbatim. Eleven of CUTLASS's nineteen fused-MoE
// `__global__`s open with `cudaGridDependencySynchronize()` and close with
// `cudaTriggerProgrammaticLaunchCompletion()`; refusing them would not keep a
// boundary sharp, it would make eleven kernels uncompilable for the sake of a
// distinction they are on the correct side of.
//
// # Why inline PTX and not a builtin
//
// nvcc lowers both to one instruction each and NVRTC has no `__nv_` builtin
// for either, so the honest implementation is the instruction. Measured rc=0
// under NVRTC 13.0 at `compute_89` and `compute_90a`; the probe is preserved as
// `nvrtc-probes/cutlass_moe_c9_griddepcontrol.py`.
//
// The `wait` form carries a `"memory"` clobber and the `launch_dependents` form
// does not, and that asymmetry is the semantics rather than an oversight:
// `griddepcontrol.wait` is the point after which the prior grid's writes are
// visible, so the compiler must not hoist a load across it, while
// `launch_dependents` only releases a successor and orders nothing this thread
// goes on to read. nvcc's own lowering makes the same distinction.
//
// `sm_90` is where PDL was introduced, and every call site in the carried
// device text is inside upstream's own `#if (defined(__CUDA_ARCH__) &&
// (__CUDA_ARCH__ >= 900))` guard -- so on this sm_89 box these bodies are not
// reached and the instruction is never assembled. That guard, not this file,
// is what makes the text correct below sm_90; these definitions must exist
// regardless, because a name is looked up whether or not its `#if` is taken
// on some other arch. Whether `griddepcontrol` assembles at all on `sm_89` is
// UNMEASURED and deliberately not relied on.
// Under `--relocatable-device-code=true` NVRTC's builtin header DEFINES both,
// so defining them again is a redefinition. See `cooperative_groups.h` for
// the other half of the same switch.
#if !defined(__CUDACC_RDC__)
__device__ __forceinline__ void cudaGridDependencySynchronize() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void cudaTriggerProgrammaticLaunchCompletion() {
  asm volatile("griddepcontrol.launch_dependents;");
}
#endif

#endif  // PIE_NVRTC_CUDA_RUNTIME_H_

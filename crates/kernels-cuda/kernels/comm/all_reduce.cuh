//===-- all_reduce.cuh --------------------------------------------*- CUDA -*-===//
//
// The custom P2P all-reduce's NVRTC root: two upstream headers and one
// element type.
//
// # What this file is
//
// `#include` of `flashinfer/comm/vllm_custom_all_reduce.cuh` and
// `flashinfer/comm/trtllm_allreduce_fusion.cuh`, plus a `using`. It holds no
// `__global__` of ours, no launcher, no `<<<>>>` and no host function. The
// five `__global__`s it exists to instantiate are upstream's:
//
//     vllm::cross_device_reduce_1stage<T, ngpus>
//     vllm::cross_device_reduce_2stage<T, ngpus>
//     flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_oneshot_lamport<
//         Pattern, T, NRanks, Fp32Acc, TriggerCompletionAtEnd>
//     flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_twoshot_sync<
//         Pattern, T, NRanks, Fp32Acc>
//
// (the one-shot counts twice, once per `TriggerCompletionAtEnd`). The host
// arithmetic that used to surround them -- `vllm::CustomAllreduce::allreduce`
// and `allreduce_fusion_kernel_launcher` -- is `kernels_cuda::comm` in Rust,
// and both `// PIE: REMOVED` markers in the vendored headers say so at the
// site.
//
// `cascade/merge_states.cuh` beside this file is the same shape for the same
// reason and its header carries the long form of the argument. The one thing
// this file does that that one does not is include TWO upstream headers into
// one translation unit, which is a decision rather than a convenience: see
// below.
//
// # Why one root and not two
//
// The plain reduction and the fused landing are different upstream files with
// no include edge between them, so two roots would compile. One root is
// cheaper by exactly the thing a JIT pays for -- `Root::key` folds the root
// text and the whole header set into a cache key, so two roots over the same
// header set are two cache entries and two ~0.5 s first-call compiles instead
// of one -- and it costs nothing, because the two headers share no name.
// `vllm::` and `flashinfer::trtllm_allreduce_fusion::` are disjoint
// namespaces, and both were compiled together here before this sentence was
// written.
//
// The one collision worth naming is that BOTH files reach `<cuda_bf16.h>` and
// `<cuda_fp16.h>`, and both of those are `shim/` files that alias the
// prelude's `bf16` and `f16`. That is the property that makes
// this work rather than a hazard: there is exactly one bf16 in the process,
// so a `RankData` written by the vllm kernel and a `vec_t<__nv_bfloat16, 8>`
// read by the fused one are the same sixteen bits.
//
// # The internalised copy, which is the whole point
//
// The two includes below resolve against the CARRIED set --
// `kernels/flashinfer/comm/`, listed in `src/source.rs` and handed to
// NVRTC as `includeNames[]`. It is not the flashinfer wheel's copy: no `-I`
// anywhere in this repository puts a directory in front of NVRTC, which has
// no search list at all.
//
// The root therefore demands `Headers::LibraryAndUpstream`
// (`src/comm/mod.rs`'s `ROOT`), exactly as `cascade/merge_states.cuh` does
// and for the same reason.
//
// # `--device-as-default-execution-space` is NOT passed, and that is measured
//
// `cascade/merge_states.cuh` needs it, because `cascade.cuh`'s helpers carry
// no execution-space annotation and NVRTC would parse them as host functions.
// Neither comm header has that problem: every surviving function in both is
// `__device__`, `__global__` or `DINLINE`, which is what removing the host
// halves left behind. Compiled without the flag at sm_89, sm_90a and sm_100a:
// rc=0 on all three, and all five template-ids lower.

#include <cuda_bf16.h>
#include <cstdint>

#include "flashinfer/comm/vllm_custom_all_reduce.cuh"
#include "flashinfer/comm/trtllm_allreduce_fusion.cuh"

namespace pie::comm {

// The element type. pie's activations are bf16 everywhere a collective can
// see them, so there is no dtype axis in this lattice and `T` is fixed at
// both call sites.
//
// It is a `using` and not a template parameter for `cascade/merge_states.cuh`'s
// reason: the buffers these kernels reduce are the ones a linear layer wrote,
// and those are `__nv_bfloat16` -- which `shim/cuda_bf16.h` aliases to
// `bf16`, the one canonical sixteen-bit float in this process.
//
// **Nothing below names it**, and that is deliberate rather than an
// oversight. Both name expressions in `src/comm/mod.rs` spell
// `__nv_bfloat16` directly, because that is the spelling upstream's own
// signatures are written in and a second name for one type is a second thing
// to keep in step. This alias exists so that a reader who greps this root for
// "what dtype is this lattice" finds an answer, and so that the day a second
// element type appears there is a line to change.
using DType = __nv_bfloat16;

}  // namespace pie::comm

//===-- supergraph.cuh - arming a conditional handle from inside a graph -===//
//
// Two `__global__`s, one line each, and the only device text in this tree
// whose argument is not a tensor. `cudaGraphSetConditional` writes a
// CONDITIONAL HANDLE -- a shell object -- so these kernels belong to no
// family named after a value, which is why `graph/` exists as a directory
// rather than these landing in `layout/`. Not beside `graph_pad.cuh`
// either: that file's "graph" is the model's lattice and this one's is
// CUDA's, and a shared word is the worst reason to share a directory.
// `driver-cuda/src/device/graph.rs`'s header makes the same argument from the
// other side, and made it first: the handle is the shell's, the tensor
// families are `kernels-cuda`'s, and a kernel over a handle is neither's
// until somebody says where it goes. Here.
//
// # Why the arming is a kernel at all
//
// Arming a conditional handle from INSIDE a graph is the whole mechanism that
// lets a replay take a fire's arms with no host round-trip: the kernel reads
// a slot out of the device-resident predicate word and writes the handle, and
// the conditional node downstream of it reads what was written. A host that
// armed the handle would have to be told the predicate first, which is the
// round-trip the supergraph exists to remove.
//
// # This was `driver-cuda/csrc/supergraph.cu`, and it needed nvcc. IT DOES NOT
//
// That file's own first line said the supergraph "cannot express in Rust"
// this one device function, and `driver-cuda/build.rs` gave it its own nvcc
// archive because "this needs nvcc". The second claim was measured on this
// box (L40S, sm_89, CUDA 13.0, `libnvrtc.so.13` / `libcuda.so.1`) and is
// FALSE. What was run, and what came back:
//
//   | step                                              | result             |
//   |---------------------------------------------------|--------------------|
//   | NVRTC compiles a kernel calling                    | rc = 0             |
//   |   `cudaGraphSetConditional`, real header           |                    |
//   |   `<cuda_device_runtime_api.h>`, `-I .../include`   |                    |
//   | the emitted PTX                                    | `.extern .func`    |
//   |                                                    | + `call.uni` -- an |
//   |                                                    | unresolved         |
//   |                                                    | external, not an   |
//   |                                                    | inlined intrinsic  |
//   | `nvrtcGetCUBIN` at `--gpu-architecture=sm_89`      | 3,624 B; `nm` says |
//   |                                                    | `U cudaGraphSet-`  |
//   |                                                    | `Conditional`;     |
//   |                                                    | SASS `CALL.ABS.`   |
//   |                                                    | `NOINC`            |
//   | `cuModuleLoadData` on that PTX                     | OK -- the DRIVER   |
//   |                                                    | resolves the       |
//   |                                                    | symbol at load     |
//   | `cuModuleGetFunction`                              | OK                 |
//   | capture it into a graph                            | OK, 1 node         |
//   |   (`cuStreamBeginCaptureToGraph` + `cuLaunchKernel`)|                   |
//   | `cuGraphAddNode` of a `CU_GRAPH_NODE_TYPE_`        | OK                 |
//   |   `CONDITIONAL` IF bound to the same handle        |                    |
//
// **What was NOT proved, stated so that nobody reads more out of the table
// than is in it: a conditional graph was never executed end to end.** That
// probe stalled populating the IF *body* -- `cuGraphAddKernelNode_v2` and
// `cuStreamBeginCaptureToGraph` both answered `invalid argument` while the
// parent graph was mid-capture -- and that is a property of the probe's
// ctypes plumbing, not of NVRTC. The sequence that works is the one
// `driver-cuda/src/device/graph.rs` already implements and
// `driver-cuda/tests/gpu_supergraph.rs` already runs.
//
// The argument that closes the gap is not in the table, and it is the reason
// the table is enough. `cudaGraphSetConditional` is declared `extern
// __device__ __cudart_builtin__` with **no definition in any toolkit header**
// and **it is not in `libcudadevrt.a`** (the archive was extracted and
// searched: no match). A call to an extern device function with no definition
// in any linkable library MUST be resolved by the driver at module load, no
// matter which frontend emitted the call -- and NVRTC and nvcc share the same
// device frontend, `cicc`. So nvcc's PTX for this call is the same
// `.extern .func`, and there is no nvcc-only lowering to lose.
//
// # The declaration is hand-written for NVRTC, and taken from the toolkit for
//   nvcc
//
// Both routes were measured and both compile to identical PTX. This file
// takes the hand-declared one under `__CUDACC_RTC__` because the real header
// is unreachable from here: this crate's NVRTC compiles hand a header SET
// carried in the binary (`source::DEVICE_HEADERS`) and pass NO `-I` at all --
// `runtime::nvrtc::options` is the architecture, `-std=c++17` and three float
// flags, and nothing else. Reaching `<cuda_device_runtime_api.h>` would mean
// putting a toolkit include path on the JIT for one declaration, which is a
// disk dependency on the launch path and a second way for a header to be
// found. It would also be invisible to
// `tests/layers.rs::every_include_reachable_from_a_unit_resolves`, which
// walks QUOTED includes: an angle include of an uncarried header passes that
// test and fails at the first fire on a machine with a GPU.
//
// Under nvcc the toolkit header is already there and force-included, so the
// `#else` takes it rather than redeclaring it -- `moe/moe_dispatch.cuh`'s
// split, for its reason.
//
// # The two launches this file was called from, verbatim
//
// The `.cu` is deleted and the geometry is stated by
// `driver-cuda/src/fire/supergraph.rs` now. A citation has to resolve to text
// that still exists, so `csrc/supergraph.cu:61` and `:74` are recorded here:
//
// ```text
//     pie_supergraph_set_cond_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
//         static_cast<cudaGraphConditionalHandle>(handle),
//         reinterpret_cast<const std::uint8_t*>(preds),
//         slot);
//
//     pie_supergraph_set_switch_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
//         static_cast<cudaGraphConditionalHandle>(handle),
//         reinterpret_cast<const std::uint8_t*>(preds),
//         slot);
// ```
//
// One block of one thread, no dynamic shared memory: one thread writes one
// handle, and a second thread writing the same handle is the racing call the
// CUDA documentation calls undefined behaviour. No `LaunchRule` states
// `<<<1, 1>>>` and none should for two rows (`new-horizon.md` §10.5), so both
// rows are `LaunchRule::Unstated` and the driver builds the `Launch`.
//
// # The handle crosses as `u64`, and that is `Ty::Usize` exactly
//
// `cudaGraphConditionalHandle` is `unsigned long long` (`driver_types.h`),
// which on LP64 is a DIFFERENT type from `size_t` with the same width -- so
// the parameter below is the prelude's `u64` (`decltype(sizeof(0))`, i.e.
// `size_t`), the row says `Ty::Usize`, and the one conversion happens at the
// one call, cast explicitly. Spelling the parameter `unsigned long long` and
// the row `Ty::Usize` would compile here and fail
// `abi::emit_device_typecheck`'s function-pointer initialisation, which is
// the check that exists to catch exactly this.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "pie_device.cuh"

#ifdef __CUDACC_RTC__
// NVRTC: hand-declared, because there is no header here to take it from. The
// signature is `cuda_device_runtime_api.h:480`'s, with the two macros
// expanded -- `CUDARTAPI` is empty off Windows (`crt/host_defines.h:101`) and
// `__cudart_builtin__` is `__location__(cudart_builtin)`, an nvcc placement
// annotation with no meaning to NVRTC. `extern "C"` is load-bearing and is
// the toolkit's own: the real declaration sits inside the `extern "C" {` that
// opens at `cuda_device_runtime_api.h:184`, which is why the emitted PTX
// names `cudaGraphSetConditional` unmangled and why the driver can resolve it
// at module load.
extern "C" __device__ void cudaGraphSetConditional(unsigned long long handle,
                                                   unsigned int value);
#else
// nvcc: the toolkit's own declaration, through the header that carries it.
// Angle-bracketed, so `-iquote` cannot answer with one of the shims wearing
// NVIDIA's filenames.
#include <cuda_runtime.h>
#endif

namespace pie_cuda_driver::kernels::graph::device {

// The scalar layer is the PRELUDE's, named here so the kernels below read as
// they did when they were `std::uint8_t` and `unsigned long long`.
using ::pie_cuda_driver::kernels::device::u64;
using ::pie_cuda_driver::kernels::device::u8;

// Arms `handle` from `preds[slot]` as a BOOLEAN: the conditional node
// downstream takes its IF branch when the byte is non-zero.
__global__ void supergraph_set_cond(u64 handle, const u8* __restrict__ preds, int slot) {
    cudaGraphSetConditional(static_cast<unsigned long long>(handle), preds[slot]);
}

// The SWITCH arming kernel (`.wiki/driver/graph.md` §6.1), and the whole
// device-side difference from the IF form above: `cudaGraphSetConditional`
// takes an unsigned value, an IF reads it as 0/1, and a SWITCH reads it as a
// body INDEX. So writing the slot's byte through unchanged is the entire
// change -- the predicate word is already a byte per slot, and a slot that
// holds a kernel index instead of a boolean needs no new storage.
//
// An out-of-range index selects no body, which is CUDA's rule and the one
// this kernel deliberately does NOT clamp: a fire whose predicate says "arm
// 4" of a three-arm switch has a lowering/driver disagreement, and running
// arm 0 instead would answer with the wrong program rather than with nothing.
__global__ void supergraph_set_switch(u64 handle, const u8* __restrict__ preds, int slot) {
    cudaGraphSetConditional(static_cast<unsigned long long>(handle),
                            static_cast<unsigned int>(preds[slot]));
}

}  // namespace pie_cuda_driver::kernels::graph::device

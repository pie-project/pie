// The one device function the supergraph cannot express in Rust.
//
// `cudaGraphSetConditional` is a `__device__` function -- it is absent from
// cudarc's bindings, and correctly so. Arming a conditional handle from
// inside a graph is the whole mechanism that lets a replay take a fire's
// arms with NO host round-trip: the kernel reads a slot out of the
// device-resident predicate word and writes the handle, and the conditional
// node downstream of it reads what was written.
//
// It lives here rather than in `kernels-cuda` for the reason
// `src/cuda/graph.rs` states: its argument is a conditional handle, which is
// a SHELL object, not a tensor. `kernels-cuda` owns kernels over tensors.
//
// This is the Rust port of `driver-cuda/csrc/src/batch/supergraph.cu`'s
// `supergraph_set_cond_kernel` and its launcher.

#include <cstdint>

#include <cuda_runtime.h>

namespace {

__global__ void pie_supergraph_set_cond_kernel(cudaGraphConditionalHandle h,
                                               const std::uint8_t* preds,
                                               int slot) {
    cudaGraphSetConditional(h, preds[slot]);
}

// The SWITCH arming kernel (`.wiki/driver/graph.md` §6.1), and the whole
// device-side difference from the IF form above: `cudaGraphSetConditional`
// takes an unsigned value, an IF reads it as 0/1, and a SWITCH reads it as
// a body INDEX. So writing the slot's byte through unchanged is the entire
// change -- the predicate word is already a byte per slot, and a slot that
// holds a kernel index instead of a boolean needs no new storage.
//
// An out-of-range index selects no body, which is CUDA's rule and the one
// this clamp deliberately does NOT paper over: a fire whose predicate says
// "arm 4" of a three-arm switch has a lowering/driver disagreement, and
// running arm 0 instead would answer with the wrong program rather than
// with nothing.
__global__ void pie_supergraph_set_switch_kernel(cudaGraphConditionalHandle h,
                                                 const std::uint8_t* preds,
                                                 int slot) {
    cudaGraphSetConditional(h, static_cast<unsigned int>(preds[slot]));
}

}  // namespace

extern "C" {

// Arms `handle` from `preds[slot]` on `stream`, which must be capturing --
// the launch becomes the conditional node's upstream dependency.
//
// Returns a `cudaError_t` as an int; the Rust side turns a non-zero into its
// own `Error`. The C++ original throws out of `CUDA_CHECK`, which is not a
// shape that crosses `extern "C"`.
int pie_supergraph_set_cond(unsigned long long handle,
                            const unsigned char* preds,
                            int slot,
                            void* stream) {
    pie_supergraph_set_cond_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<cudaGraphConditionalHandle>(handle),
        reinterpret_cast<const std::uint8_t*>(preds),
        slot);
    return static_cast<int>(cudaGetLastError());
}

// Arms `handle` from `preds[slot]` as a body INDEX rather than a boolean.
// Same contract as `pie_supergraph_set_cond`: `stream` must be capturing.
int pie_supergraph_set_switch(unsigned long long handle,
                              const unsigned char* preds,
                              int slot,
                              void* stream) {
    pie_supergraph_set_switch_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<cudaGraphConditionalHandle>(handle),
        reinterpret_cast<const std::uint8_t*>(preds),
        slot);
    return static_cast<int>(cudaGetLastError());
}

}  // extern "C"

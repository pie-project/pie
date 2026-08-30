#pragma once

/// **THE ONE DEVICE FUNCTION A CUDA GRAPH'S CONTROL FLOW IS WRITTEN IN.**
///
/// `cudaGraphSetConditional` is the only way a graph's `IF`/`SWITCH` node
/// learns whether to take its body: the predicate is not a host branch and
/// not a node parameter, it is a store a KERNEL makes into a handle the
/// driver minted while the parent graph was being recorded. That is the whole
/// reason a conditional is worth more than the `cudaGraphNodeSetEnabled` the
/// fold reaches for — the decision rides inside the graph, so a replay under a
/// composition the recording fire never saw still decides correctly.
///
/// # The symbol, and the link stage it does NOT need
///
/// The toolkit declares it `extern __device__ __cudart_builtin__ void
/// cudaGraphSetConditional(unsigned long long, unsigned int)` in
/// `cuda_device_runtime_api.h` and defines it NOWHERE — not in any header, and
/// not in `libcudadevrt.a` (the archive was extracted and searched;
/// `.wiki/driver/new-horizon.md` §62.3). A call to an extern device function
/// with no definition in any linkable library is resolved by the DRIVER at
/// module load, whichever frontend emitted the call, and NVRTC and nvcc share
/// `cicc`. So this unit compiles like every other one here: whole-program, no
/// `--relocatable-device-code`, no `cuLink` against the device runtime. The
/// declaration is restated below rather than `#include`d because the toolkit
/// header is not in this plane's carried closure and the signature is two
/// integers.
extern "C" __device__ void cudaGraphSetConditional(
    unsigned long long handle,
    unsigned int value);

namespace pie::graph {

/// **SET ONE CONDITIONAL HANDLE FROM A ROW COUNT THE DEVICE ALREADY HOLDS.**
///
/// `indptr` is a window's rebased row CSR — `[lanes + 1]` `i32`, the vector
/// `engine_cuda::window::Window` stages once per distinct window per fire — so
/// `indptr[lanes]` is that window's row count and `indptr[lanes] != 0` is
/// exactly the zero-row rule `model_exec::fire::walk` applies on the host
/// (decision #3). One thread, one load, one store: the predicate is the
/// artifact's own semantics read from the device rather than restated.
///
/// `arm` is what makes this kernel WARMABLE. A `cudaGraphSetConditional` call
/// outside a conditional graph's launch has no handle to store into, so the
/// eager pass that compiles this unit and loads its module — which must happen
/// BEFORE the capture, since a JIT inside `cudaStreamBeginCapture` is host
/// work the thread-local mode refuses — fires it with `arm = 0` and reaches
/// the early return. The captured launch fires it with `arm = 1`.
///
/// A null `indptr` is the window this fire has no vector for, and it reads as
/// `absent`: the caller states what an absent window means rather than having
/// this kernel guess, because "no rows" and "no table" are not the same
/// sentence.
__global__ void set_conditional(
    unsigned long long handle,
    const int* __restrict__ indptr,
    int lanes,
    unsigned int absent,
    int arm)
{
    if (arm == 0) return;
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned int value = absent;
    if (indptr != nullptr && lanes >= 0) {
        value = (indptr[lanes] != 0) ? 1u : 0u;
    }
    cudaGraphSetConditional(handle, value);
}

/// The same store, from a value the CALLER states rather than one the device
/// reads — the synthetic form a gate drives both arms of, and the form a
/// predicate that is not a row count would take.
///
/// `live` is a device byte: non-zero takes the body. It is a POINTER and not
/// an immediate on purpose — an immediate would be a node parameter, frozen at
/// capture, which is the host branch a conditional exists to replace.
__global__ void set_conditional_byte(
    unsigned long long handle,
    const unsigned char* __restrict__ live,
    unsigned int absent,
    int arm)
{
    if (arm == 0) return;
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const unsigned int value = (live == nullptr) ? absent : ((*live != 0) ? 1u : 0u);
    cudaGraphSetConditional(handle, value);
}

}

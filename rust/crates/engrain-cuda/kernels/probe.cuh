// The kernel that exists to prove the path: source -> nvcc -> fatbin ->
// embedded in the cdylib -> loaded by the driver API -> launched from Python
// on PyTorch's own stream -> recorded inside a CUDA graph.
//
// Everything in this file is deliberately trivial. It is the build system
// under test, not the algorithm, and until this is boring nothing else should
// be written.

#include <stdint.h>

extern "C" __global__ void en_probe_identity(int32_t* out, int32_t count, int32_t bias) {
    int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        out[index] = index + bias;
    }
}

// Accumulates rather than assigns, so that replaying a captured graph N times
// is visible as N. A graph that silently did not replay would otherwise look
// exactly like one that did.
extern "C" __global__ void en_probe_accumulate(int32_t* out, int32_t count, int32_t by) {
    int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        out[index] += by;
    }
}

// Silent CUDA implementations for the prepare oracle.
//
// `attention_workspace.cpp` runs for real here (the prepare hook plans
// into real workspaces), but its pin/event traffic is the attn_ws oracle's
// subject, not this one's — the transcript this oracle pins is the PLANNER
// call sequence. So the six entry points allocate and succeed, silently.
#include "cuda_runtime.h"

#include <cstdlib>

cudaError_t cudaMallocHost(void** ptr, std::size_t size) {
    *ptr = std::malloc(size == 0 ? 1 : size);
    return cudaSuccess;
}
cudaError_t cudaFreeHost(void* ptr) {
    std::free(ptr);
    return cudaSuccess;
}
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned) {
    *event = static_cast<cudaEvent_t>(std::malloc(1));
    return cudaSuccess;
}
cudaError_t cudaEventDestroy(cudaEvent_t event) {
    std::free(event);
    return cudaSuccess;
}
cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { return cudaSuccess; }
cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }

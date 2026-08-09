#pragma once

// A recording stand-in for <cuda_runtime.h>, sized to `attention_workspace`.
//
// The class's observable behaviour IS its sequence of pin/unpin/event calls:
// which slot pins lazily and when, which event fences which reuse, what the
// destructor must sync before it frees. None of that survives a real call --
// a slot reused before its upload retires corrupts a plan the GPU is still
// reading, and returns success. So the six entry points the TU uses become
// recorders, and the calls themselves become the transcript.
//
// Found ahead of the real header because it is copied into $WORK and $WORK is
// first on the include path. Only the surface `attention_workspace.cpp`
// actually uses is declared -- anything else should fail to compile rather
// than silently no-op.

#include <cstddef>
#include <string>
#include <vector>

using cudaError_t = int;
using cudaStream_t = void*;
using cudaEvent_t = void*;

inline constexpr cudaError_t cudaSuccess = 0;
inline constexpr unsigned int cudaEventDisableTiming = 2;

cudaError_t cudaMallocHost(void** ptr, std::size_t size);
cudaError_t cudaFreeHost(void* ptr);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);

// ── the recorder's control surface (oracle.cpp only) ────────────────────────
namespace oracle_cuda {

/// Clear the log and all symbol tables; restart pin/event ordinals at 0.
void reset_case();
/// Every recorded row since the last reset, in call order.
const std::vector<std::string>& log();
/// Append a row of the oracle's own (calls, views, catches).
void note(const std::string& line);
/// Name a fabricated stream pointer for the transcript.
void name_stream(cudaStream_t s, const std::string& name);
/// The symbolic name of a pinned block, or "null"/"unknown".
std::string pin_name(const void* ptr);
/// Make the NEXT cudaMallocHost fail (one-shot).
void fail_next_malloc_host();
/// Make the NEXT cudaEventCreateWithFlags fail (one-shot).
void fail_next_event_create();

}  // namespace oracle_cuda

#pragma once
// Minimal CUDA surface for the store/ oracle. cudaMemcpyAsync really copies,
// so the extracted swap routines move real bytes and the transcript can hash
// the result -- verifying semantics, not just the offsets they computed.
#include <cstddef>
#include <cstring>

using cudaStream_t = struct CUstream_st*;
enum cudaError_t { cudaSuccess = 0, cudaErrorUnknown = 1 };
enum cudaMemcpyKind { cudaMemcpyDefault = 4, cudaMemcpyDeviceToDevice = 3 };
struct cudaDeviceProp { char name[256] = {0}; int major = 0; int minor = 0; int multiProcessorCount = 0; };

void oracle_note_copy(void* dst, const void* src, std::size_t n);

inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t n,
                                   cudaMemcpyKind, cudaStream_t) {
    oracle_note_copy(dst, src, n);
    std::memcpy(dst, src, n);
    return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t) { return "stub"; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }

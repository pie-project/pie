// Stub <cuda_runtime.h> for the llama_like oracle.
//
// Declarations only, wide enough for every driver header the TU includes.
// The oracle links with --gc-sections, so a function is only ever needed at
// runtime if a DRIVEN path reaches it — those get recorder definitions in
// the oracle; everything else is discarded with the code that named it.
#pragma once
#include <cstddef>
#include <cstring>
// The real <cuda_runtime.h> drags <cmath> in transitively, and the TU under
// test spells `std::sqrt` without including it — reproduce that reach.
#include <cmath>

using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;

using cudaStream_t = struct CUstream_st*;
using cudaEvent_t = struct CUevent_st*;
using cudaGraph_t = struct CUgraph_st*;
using cudaGraphExec_t = struct CUgraphExec_st*;

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
};

enum cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed = 2,
};

enum cudaStreamCaptureStatus {
    cudaStreamCaptureStatusNone = 0,
    cudaStreamCaptureStatusActive = 1,
};

struct cudaDeviceProp {
    char name[256];
    int major;
    int minor;
    int multiProcessorCount;
    std::size_t totalGlobalMem;
};

cudaError_t cudaGetDevice(int* dev);
cudaError_t cudaSetDevice(int dev);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int dev);
cudaError_t cudaMemGetInfo(std::size_t* free_bytes, std::size_t* total_bytes);
const char* cudaGetErrorString(cudaError_t e);
cudaError_t cudaGetLastError();

cudaError_t cudaMalloc(void** p, std::size_t bytes);
cudaError_t cudaFree(void* p);
cudaError_t cudaMallocHost(void** p, std::size_t bytes);
cudaError_t cudaFreeHost(void* p);
cudaError_t cudaHostAlloc(void** p, std::size_t bytes, unsigned flags);
constexpr unsigned cudaHostAllocDefault = 0;
cudaError_t cudaMemset(void* p, int v, std::size_t bytes);
cudaError_t cudaMemsetAsync(void* p, int v, std::size_t bytes, cudaStream_t s);
cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t bytes,
                       cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t bytes,
                            cudaMemcpyKind kind, cudaStream_t s);

cudaError_t cudaStreamCreate(cudaStream_t* s);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned flags);
constexpr unsigned cudaStreamNonBlocking = 1;
cudaError_t cudaStreamDestroy(cudaStream_t s);
cudaError_t cudaStreamSynchronize(cudaStream_t s);
cudaError_t cudaStreamIsCapturing(cudaStream_t s, cudaStreamCaptureStatus* st);
cudaError_t cudaStreamBeginCapture(cudaStream_t s, cudaStreamCaptureMode m);
cudaError_t cudaStreamEndCapture(cudaStream_t s, cudaGraph_t* g);
cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e,
                                unsigned flags = 0);

cudaError_t cudaEventCreate(cudaEvent_t* e);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned flags);
constexpr unsigned cudaEventDisableTiming = 2;
cudaError_t cudaEventDestroy(cudaEvent_t e);
cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s = nullptr);
cudaError_t cudaEventSynchronize(cudaEvent_t e);

cudaError_t cudaGraphDestroy(cudaGraph_t g);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t g);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b);
cudaError_t cudaDeviceSynchronize();

#include "cuda_check.hpp"

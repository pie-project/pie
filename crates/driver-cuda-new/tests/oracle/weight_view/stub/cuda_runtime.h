// Stub <cuda_runtime.h> for the weight-view oracle.
//
// The real `tensor.cpp` is compiled — transcribing `DeviceTensor::view`'s
// eleven lines into the oracle would make this a test of the transcription —
// and it touches CUDA in exactly two places, `cudaMalloc` and `cudaFree`,
// neither of which any tensor here reaches: every one is a non-owning
// `DeviceTensor::view`. `cuda_check.hpp` pulls in three more names for the
// graph-capture RAII it also declares, which nothing in this TU instantiates.
#pragma once
#include <cstddef>

using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;

using cudaStream_t = struct CUstream_st*;
using cudaGraph_t = struct CUgraph_st*;
using cudaGraphExec_t = struct CUgraphExec_st*;

cudaError_t cudaMalloc(void** ptr, std::size_t bytes);
cudaError_t cudaFree(void* ptr);
using cudaStreamCaptureMode = int;
constexpr cudaStreamCaptureMode cudaStreamCaptureModeRelaxed = 0;

cudaError_t cudaStreamBeginCapture(
    cudaStream_t stream, cudaStreamCaptureMode mode);
cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* graph);
cudaError_t cudaGraphDestroy(cudaGraph_t graph);
cudaError_t cudaGetLastError();
const char* cudaGetErrorString(cudaError_t err);

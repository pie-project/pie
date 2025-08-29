#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "flashinfer/vec_dtypes.cuh"

// Elementwise add residual (in-place on x)
template <typename T>
__global__ void add_residual_kernel(T* x, const T* residual, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = x[idx] + residual[idx];
    }
}

__device__ __forceinline__ float silu_act(const float &val) {
    return val / (1.0f + __expf(-val));
}

template <typename T, float (*Activation)(const float&)>
__global__ void act_and_mul_kernel(
    T* __restrict__ out,
    const T* __restrict__ input1,
    const T* __restrict__ input2,
    const int d
) {
    constexpr uint32_t vec_size = 16 / sizeof(T);
    const int64_t token_idx = blockIdx.x;
    const int64_t thread_idx = threadIdx.x;
    const int64_t stride = blockDim.x;
    const int64_t token_offset = token_idx * d;
    #pragma unroll 1
    for (uint32_t idx = thread_idx; idx < d / vec_size; idx += stride) {
        flashinfer::vec_t<float, vec_size> x_vec, y_vec, out_vec;
        x_vec.cast_load(input1 + token_offset + idx * vec_size);
        y_vec.cast_load(input2 + token_offset + idx * vec_size);
        #pragma unroll
        for (uint32_t i = 0; i < vec_size; ++i) {
            out_vec[i] = Activation(x_vec[i]) * y_vec[i];
        }
        out_vec.cast_store(out + token_offset + idx * vec_size);
    }
    const int64_t remaining_offset = (d / vec_size) * vec_size;
    #pragma unroll 1
    for (int64_t idx = thread_idx + remaining_offset; idx < d; idx += stride) {
        float x = static_cast<float>(__ldg(input1 + token_offset + idx));
        float y = static_cast<float>(__ldg(input2 + token_offset + idx));
        out[token_offset + idx] = static_cast<T>(Activation(x) * y);
    }
}

template <typename T>
inline void silu_and_mul(
    T* out_ptr,
    const T* in1_ptr,
    const T* in2_ptr,
    int num_tokens,
    int d,
    cudaStream_t stream
) {
    constexpr uint32_t vec_size = 16 / sizeof(T);
    dim3 grid_dim(num_tokens);
    uint32_t block_dim = std::min(static_cast<uint32_t>(d / vec_size), 256U);
    if (block_dim == 0) {
        block_dim = std::min(static_cast<uint32_t>(d), 256U);
    }
    act_and_mul_kernel<T, silu_act><<<grid_dim, block_dim, 0, stream>>>(
        out_ptr, in1_ptr, in2_ptr, d
    );
}

template <typename T>
inline void add_residual(T* x, const T* residual, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_residual_kernel<T><<<blocks, threads, 0, stream>>>(x, residual, n);
}

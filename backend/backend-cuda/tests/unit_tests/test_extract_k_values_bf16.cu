#include <cassert>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <limits>

#include "../../src/common.cuh"

int main() {
    using T = __nv_bfloat16;
    const int M = 2;
    const int N = 10;
    const int k = 5;

    std::vector<float> host_vals(M * N, -INFINITY);
    // Row 0 finite at 0,3,4,6 -> only 4 finite (<k)
    host_vals[0*N + 0] = 0.5f;
    host_vals[0*N + 3] = -1.25f;
    host_vals[0*N + 4] = 2.75f;
    host_vals[0*N + 6] = 7.0f;
    // Row 1 finite at 2..6 (five contiguous) -> exactly k
    host_vals[1*N + 2] = -0.5f;
    host_vals[1*N + 3] = 1.0f;
    host_vals[1*N + 4] = 1.5f;
    host_vals[1*N + 5] = 2.0f;
    host_vals[1*N + 6] = 5.5f;

    std::vector<T> h_A(host_vals.size());
    for (size_t i = 0; i < host_vals.size(); ++i) h_A[i] = __float2bfloat16(host_vals[i]);

    T *d_A = nullptr, *d_V = nullptr; int32_t *d_I = nullptr; cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(&d_A, h_A.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_V, M * k * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_I, M * k * sizeof(int32_t)));

    extract_k_values<T>(d_A, d_V, d_I, M, N, k, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<T> h_V(M * k); std::vector<int32_t> h_I(M * k, -1);
    CUDA_CHECK(cudaMemcpy(h_V.data(), d_V, h_V.size() * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_I.data(), d_I, h_I.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));

    auto bf = [](T v){ return __bfloat162float(v); };

    // Row 0: expect first 4 only (indices 0,3,4,6) and last slot untouched (implementation only writes when found)
    std::vector<int> exp_idx0 = {0,3,4,6};
    std::vector<float> exp_val0 = {0.5f,-1.25f,2.75f,7.0f};
    for (size_t j = 0; j < exp_idx0.size(); ++j) {
        int idx = 0 * k + (int)j;
        if (h_I[idx] != exp_idx0[j] || std::fabs(bf(h_V[idx]) - exp_val0[j]) > 3e-2f) {
            std::cerr << "Row0 mismatch slot " << j << " got (" << h_I[idx] << "," << bf(h_V[idx]) << ")\n";
            return 1;
        }
    }
    // Row1: expect 5 entries (2..6)
    std::vector<int> exp_idx1 = {2,3,4,5,6};
    std::vector<float> exp_val1 = {-0.5f,1.0f,1.5f,2.0f,5.5f};
    for (size_t j = 0; j < exp_idx1.size(); ++j) {
        int idx = 1 * k + (int)j;
        if (h_I[idx] != exp_idx1[j] || std::fabs(bf(h_V[idx]) - exp_val1[j]) > 3e-2f) {
            std::cerr << "Row1 mismatch slot " << j << " got (" << h_I[idx] << "," << bf(h_V[idx]) << ")\n";
            return 2;
        }
    }

    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_I)); CUDA_CHECK(cudaStreamDestroy(stream));
    std::cout << "PASS: test_extract_k_values_bf16" << std::endl; return 0;
}

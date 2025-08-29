#include <cassert>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <limits>

#include "../../src/common.cuh"

int main() {
    using T = float;

    const int M = 3;
    const int N = 8;
    const int k = 3;

    std::vector<T> h_A(M * N, -INFINITY);
    // Row 0: finite at cols 1,3,6,7 -> expect first 3 -> (1,3,6)
    h_A[0 * N + 1] = 0.1f;
    h_A[0 * N + 3] = 0.3f;
    h_A[0 * N + 6] = 0.6f;
    h_A[0 * N + 7] = 0.7f;
    // Row 1: finite at cols 0,4 -> expect (0,4, then nothing but kernel writes only when found)
    h_A[1 * N + 0] = 1.0f;
    h_A[1 * N + 4] = 1.4f;
    // Row 2: finite at cols 2,5,6 -> expect (2,5,6)
    h_A[2 * N + 2] = 2.2f;
    h_A[2 * N + 5] = 2.5f;
    h_A[2 * N + 6] = 2.6f;

    T *d_A = nullptr, *d_V = nullptr;
    int32_t *d_I = nullptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMalloc(&d_A, h_A.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(T), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_V, M * k * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_I, M * k * sizeof(int32_t)));

    extract_k_values<T>(d_A, d_V, d_I, M, N, k, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<T> h_V(M * k, 0);
    std::vector<int32_t> h_I(M * k, -1);
    CUDA_CHECK(cudaMemcpy(h_V.data(), d_V, h_V.size() * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_I.data(), d_I, h_I.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));

    auto check_row = [&](int row, std::vector<int> exp_cols, std::vector<T> exp_vals) {
        for (int j = 0; j < (int)exp_cols.size(); ++j) {
            int idx = row * k + j;
            if (h_I[idx] != exp_cols[j] || std::fabs(h_V[idx] - exp_vals[j]) > 1e-6f) {
                std::cerr << "Row " << row << ": j=" << j << " got (" << h_I[idx] << ", " << h_V[idx]
                          << ") expected (" << exp_cols[j] << ", " << exp_vals[j] << ")\n";
                return false;
            }
        }
        return true;
    };

    if (!check_row(0, {1,3,6}, {0.1f,0.3f,0.6f})) return 1;
    if (!check_row(1, {0,4}, {1.0f,1.4f})) return 1; // only two finite -> kernel only fills first two
    if (!check_row(2, {2,5,6}, {2.2f,2.5f,2.6f})) return 1;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_I));
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::cout << "PASS: test_extract_k_values" << std::endl;
    return 0;
}
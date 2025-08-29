#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Include the implementation unit so the template kernel is visible
#include "../../src/l4ma.cu"

int main() {
    using T = float;
    const int n = 1024;

    T *d_x = nullptr, *d_r = nullptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_r, n * sizeof(T)));

    std::vector<T> hx(n), hr(n);
    for (int i = 0; i < n; ++i) { hx[i] = static_cast<T>(i * 0.25f); hr[i] = static_cast<T>(i * -0.5f); }
    CUDA_CHECK(cudaMemcpy(d_x, hx.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r, hr.data(), n * sizeof(T), cudaMemcpyHostToDevice));

    add_residual_kernel<T><<<(n + 255) / 256, 256, 0, stream>>>(d_x, d_r, n);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<T> out(n);
    CUDA_CHECK(cudaMemcpy(out.data(), d_x, n * sizeof(T), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; ++i) {
        T expected = hx[i] + hr[i];
        if (std::fabs(out[i] - expected) > 1e-6f) {
            std::cerr << "Mismatch at " << i << ": got " << out[i] << ", exp " << expected << std::endl;
            return 1;
        }
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::cout << "PASS: test_add_residual_full" << std::endl;
    return 0;
}
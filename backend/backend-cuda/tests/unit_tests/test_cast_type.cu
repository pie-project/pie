#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../../src/common.cuh"

int main() {
    try {
        const size_t n = 2048;
        std::vector<float> h_input(n);
        for (size_t i = 0; i < n; ++i) {
            h_input[i] = static_cast<float>(std::sin(0.01 * i) * 100.0 + 0.25 * i);
        }

        float* d_in = nullptr;
        __nv_bfloat16* d_bf16 = nullptr;
        float* d_round = nullptr;
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bf16, n * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_round, n * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_in, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        cast_type<float, __nv_bfloat16>(d_in, d_bf16, n, stream);
        cast_type<__nv_bfloat16, float>(d_bf16, d_round, n, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<float> h_round(n);
        CUDA_CHECK(cudaMemcpy(h_round.data(), d_round, n * sizeof(float), cudaMemcpyDeviceToHost));

        // bfloat16 has ~7-bit mantissa => expect some quantization. Use relative tolerance.
        int num_bad = 0;
        for (size_t i = 0; i < n; ++i) {
            float a = h_input[i];
            float b = h_round[i];
            float denom = std::max(1.0f, std::fabs(a));
            if (std::fabs(a - b) / denom > 1e-2f) {
                ++num_bad;
                if (num_bad < 10) {
                    std::cerr << "Mismatch at " << i << ": a=" << a << ", b=" << b << std::endl;
                }
            }
        }
        if (num_bad > 0) {
            std::cerr << "Total mismatches: " << num_bad << std::endl;
            return 1;
        }

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_bf16));
        CUDA_CHECK(cudaFree(d_round));
        CUDA_CHECK(cudaStreamDestroy(stream));

        std::cout << "PASS: test_cast_type (float <-> bf16 round-trip)" << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return 2;
    }
}
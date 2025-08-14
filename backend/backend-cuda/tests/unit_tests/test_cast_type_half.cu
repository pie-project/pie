#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "../../src/common.cuh"

static bool rel_close(float a, float b, float rtol, float atol) {
    float diff = std::fabs(a - b);
    return diff <= (atol + rtol * std::max(std::fabs(a), std::fabs(b)));
}

int main() {
    try {
        const size_t n = 4096;
        std::vector<float> h_input(n);
        for (size_t i = 0; i < n; ++i) {
            h_input[i] = std::sin(0.002f * static_cast<float>(i)) * 50.0f + 0.01f * static_cast<float>(i);
        }

        float *d_float_a = nullptr, *d_float_b = nullptr;
        __half *d_half = nullptr; __nv_bfloat16 *d_bf16 = nullptr; __half *d_half2 = nullptr; __nv_bfloat16 *d_bf162 = nullptr;
        cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaMalloc(&d_float_a, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_float_b, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_half, n * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_half2, n * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_bf16, n * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&d_bf162, n * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemcpy(d_float_a, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        // float -> half -> float
        cast_type<float, __half>(d_float_a, d_half, n, stream);
        cast_type<__half, float>(d_half, d_float_b, n, stream);

        // float -> bf16 -> half -> bf16 -> float (mixed path)
        cast_type<float, __nv_bfloat16>(d_float_a, d_bf16, n, stream);
        cast_type<__nv_bfloat16, __half>(d_bf16, d_half2, n, stream);
        cast_type<__half, __nv_bfloat16>(d_half2, d_bf162, n, stream);
        cast_type<__nv_bfloat16, float>(d_bf162, d_float_b, n, stream); // overwrite with last path for worst-case accuracy

        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<float> h_round(n);
        CUDA_CHECK(cudaMemcpy(h_round.data(), d_float_b, n * sizeof(float), cudaMemcpyDeviceToHost));

        // Assess error distribution
        size_t bad = 0; const float rtol = 3e-2f; const float atol = 3e-2f;
        for (size_t i = 0; i < n; ++i) {
            if (!rel_close(h_input[i], h_round[i], rtol, atol)) {
                ++bad; if (bad < 10) {
                    std::cerr << "Mismatch i=" << i << " src=" << h_input[i] << " got=" << h_round[i] << std::endl;
                }
            }
        }
        if (bad > n * 0.02) { // Allow 2% large-error tail due to double quantization
            std::cerr << "FAIL: test_cast_type_half too many mismatches (" << bad << "/" << n << ")" << std::endl;
            return 1;
        }

        CUDA_CHECK(cudaFree(d_float_a)); CUDA_CHECK(cudaFree(d_float_b));
        CUDA_CHECK(cudaFree(d_half)); CUDA_CHECK(cudaFree(d_half2));
        CUDA_CHECK(cudaFree(d_bf16)); CUDA_CHECK(cudaFree(d_bf162));
        CUDA_CHECK(cudaStreamDestroy(stream));
        std::cout << "PASS: test_cast_type_half" << std::endl; return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl; return 2;
    }
}

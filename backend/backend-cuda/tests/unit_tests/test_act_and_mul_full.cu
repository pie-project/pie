#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "../../src/l4ma.cu"

static float silu_ref(float x) { return x / (1.0f + std::exp(-x)); }

int main() {
    using T = float;
    const int num_tokens = 3;
    const int d = 32;

    T *d_out = nullptr, *d_in1 = nullptr, *d_in2 = nullptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(&d_out, num_tokens * d * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_in1, num_tokens * d * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_in2, num_tokens * d * sizeof(T)));

    std::vector<T> h1(num_tokens * d), h2(num_tokens * d);
    for (int t = 0; t < num_tokens; ++t) {
        for (int i = 0; i < d; ++i) {
            h1[t*d + i] = 0.1f * (i - 8);
            h2[t*d + i] = 0.2f * (i + 3);
        }
    }
    CUDA_CHECK(cudaMemcpy(d_in1, h1.data(), h1.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in2, h2.data(), h2.size() * sizeof(T), cudaMemcpyHostToDevice));

    silu_and_mul<T>(d_out, d_in1, d_in2, num_tokens, d, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<T> out(num_tokens * d);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, out.size() * sizeof(T), cudaMemcpyDeviceToHost));

    for (int t = 0; t < num_tokens; ++t) {
        for (int i = 0; i < d; ++i) {
            float expected = silu_ref(h1[t*d + i]) * h2[t*d + i];
            if (std::fabs(out[t*d + i] - expected) > 1e-5f) {
                std::cerr << "Mismatch at token " << t << ", i=" << i << ": got " << out[t*d + i]
                          << ", exp " << expected << std::endl;
                return 1;
            }
        }
    }

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_in1));
    CUDA_CHECK(cudaFree(d_in2));
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::cout << "PASS: test_act_and_mul_full" << std::endl;
    return 0;
}
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../../src/tensor.hpp"

static float host_mean(const std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) s += x;
    return static_cast<float>(s / v.size());
}

int main() {
    try {
        // float path
        {
            const size_t n = 1024;
            Tensor<float> t(n);

            std::vector<float> h(n);
            for (size_t i = 0; i < n; ++i) {
                h[i] = static_cast<float>(i + 1);
            }
            t.from_vector(h);

            float m = t.mean();
            float expected = (1.0f + static_cast<float>(n)) / 2.0f;
            float tol = 1e-3f;
            if (std::fabs(m - expected) > tol) {
                std::cerr << "Mean mismatch (float): got " << m << ", expected " << expected << std::endl;
                return 1;
            }
        }

        // bf16 path
        {
            const size_t n = 2048;
            Tensor<__nv_bfloat16> t(n);
            std::vector<float> h(n);
            for (size_t i = 0; i < n; ++i) h[i] = std::sin(0.01f * i) + 0.1f * i;

            // Copy to device as bf16
            std::vector<__nv_bfloat16> h_bf16(n);
            for (size_t i = 0; i < n; ++i) h_bf16[i] = __float2bfloat16(h[i]);
            cudaMemcpy(t.data(), h_bf16.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

            float m = t.mean();
            float expected = host_mean(h);
            if (std::fabs(m - expected) / std::max(1.0f, std::fabs(expected)) > 1e-2f) {
                std::cerr << "Mean mismatch (bf16): got " << m << ", expected ~" << expected << std::endl;
                return 2;
            }
        }

        std::cout << "PASS: test_sum_reduction (float & bf16)" << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return 3;
    }
}
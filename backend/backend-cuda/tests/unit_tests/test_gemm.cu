#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <cublas_v2.h>

#include "../../src/common.cuh"

static bool nearly_equal(float a, float b, float tol = 1e-3f) {
    return std::fabs(a - b) <= tol * std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
}

int main() {
    try {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // ---- Test gemm_cublasLt<float> ----
        {
            cublasLtHandle_t ltHandle;
            CUBLAS_CHECK(cublasLtCreate(&ltHandle));

            const int m = 2, n = 4, k = 3;
            std::vector<float> hA = {
                1, 2, 3,
                4, 5, 6
            }; // 2x3
            std::vector<float> hB = {
                1,  0,  2,  1,
                0,  1, -1,  0,
                3, -1,  0,  2
            }; // 3x4
            std::vector<float> hC(m * n, 0);

            float *dA = nullptr, *dB = nullptr, *dC = nullptr;
            CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

            void* d_workspace = nullptr;
            size_t workspace_size = 1 << 20; // 1MB
            CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));

            gemm_cublasLt<float>(ltHandle, stream,
                                  dA, dB, /*bias*/nullptr, dC,
                                  m, n, k,
                                  d_workspace, workspace_size,
                                  /*transa*/false, /*transb*/false);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));

            // Expected C = A @ B
            std::vector<float> exp = {
                // row 0
                1*1 + 2*0 + 3*3, 1*0 + 2*1 + 3*(-1), 1*2 + 2*(-1) + 3*0, 1*1 + 2*0 + 3*2,
                // row 1
                4*1 + 5*0 + 6*3, 4*0 + 5*1 + 6*(-1), 4*2 + 5*(-1) + 6*0, 4*1 + 5*0 + 6*2
            };
            for (size_t i = 0; i < exp.size(); ++i) {
                if (!nearly_equal(hC[i], exp[i])) {
                    std::cerr << "gemm<float> mismatch at " << i << ": got " << hC[i] << ", exp " << exp[i] << std::endl;
                    return 1;
                }
            }

            CUDA_CHECK(cudaFree(dA));
            CUDA_CHECK(cudaFree(dB));
            CUDA_CHECK(cudaFree(dC));
            CUDA_CHECK(cudaFree(d_workspace));
            CUBLAS_CHECK(cublasLtDestroy(ltHandle));
        }

        // ---- Test multiply_bf16_cublas ----
        {
            cublasHandle_t handle;
            CUBLAS_CHECK(cublasCreate(&handle));

            const int m = 2, n = 3, k = 4;
            std::vector<float> fA = {
                1, 2, 3, 4,
                5, 6, 7, 8
            }; // 2x4
            std::vector<float> fB = {
                1, 0, 2,
                0, 1, -1,
                2, -1, 0,
                3,  0, 1
            }; // 4x3

            std::vector<__nv_bfloat16> hA(fA.size()), hB(fB.size()), hC(m * n);
            for (size_t i = 0; i < fA.size(); ++i) hA[i] = __float2bfloat16(fA[i]);
            for (size_t i = 0; i < fB.size(); ++i) hB[i] = __float2bfloat16(fB[i]);

            __nv_bfloat16 *dA = nullptr, *dB = nullptr, *dC = nullptr;
            CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(__nv_bfloat16)));
            CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(__nv_bfloat16)));
            CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(__nv_bfloat16)));
            CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

            multiply_bf16_cublas(handle, dA, dB, dC, m, n, k, /*transa*/false, /*transb*/false);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

            // Compare against float reference
            std::vector<float> ref(m * n, 0.0f);
            for (int row = 0; row < m; ++row) {
                for (int col = 0; col < n; ++col) {
                    float sum = 0.0f;
                    for (int kk = 0; kk < k; ++kk) sum += fA[row*k + kk] * fB[kk*n + col];
                    ref[row*n + col] = sum;
                }
            }
            int bad = 0;
            for (int i = 0; i < m*n; ++i) {
                float c = __bfloat162float(hC[i]);
                if (!nearly_equal(c, ref[i], 2e-2f)) {
                    ++bad;
                    if (bad < 10) std::cerr << "bf16 GEMM mismatch at " << i << ": got " << c << ", exp " << ref[i] << std::endl;
                }
            }
            if (bad) return 2;

            CUDA_CHECK(cudaFree(dA));
            CUDA_CHECK(cudaFree(dB));
            CUDA_CHECK(cudaFree(dC));
            CUBLAS_CHECK(cublasDestroy(handle));
        }

        std::cout << "PASS: test_gemm" << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return 3;
    }
}
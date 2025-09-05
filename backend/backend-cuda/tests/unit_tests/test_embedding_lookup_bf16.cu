#include <cassert>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../../src/common.cuh"

// Test bf16 embedding lookup correctness by constructing deterministic table
// and comparing copied rows element-wise (after float conversion) with tolerance.
int main() {
    using T = __nv_bfloat16;
    using I = int32_t;
    const int embed_width = 16; // make divisible by 16 bytes (16 * 2 = 32B) OK
    const size_t num_rows = 7;
    const size_t num_indices = 4;

    std::vector<float> h_embedding_f(num_rows * embed_width);
    for (size_t r = 0; r < num_rows; ++r) {
        for (int c = 0; c < embed_width; ++c) {
            h_embedding_f[r * embed_width + c] = static_cast<float>(r * 0.5f + c * 0.25f);
        }
    }
    std::vector<I> h_indices = {6, 0, 3, 5};

    // Convert to bf16 host buffer
    std::vector<T> h_embedding(num_rows * embed_width);
    for (size_t i = 0; i < h_embedding_f.size(); ++i) {
        h_embedding[i] = __float2bfloat16(h_embedding_f[i]);
    }

    T *d_embedding = nullptr, *d_result = nullptr; I *d_indices = nullptr;
    cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(&d_embedding, h_embedding.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_indices, h_indices.size() * sizeof(I)));
    CUDA_CHECK(cudaMalloc(&d_result, num_indices * embed_width * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_embedding, h_embedding.data(), h_embedding.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(), h_indices.size() * sizeof(I), cudaMemcpyHostToDevice));

    embed<T,I>(d_embedding, num_rows, d_indices, num_indices, d_result, embed_width, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<T> h_out(num_indices * embed_width);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_result, h_out.size() * sizeof(T), cudaMemcpyDeviceToHost));

    int errors = 0; const float rel_tol = 2e-2f; const float abs_tol = 3e-2f;
    for (size_t i = 0; i < num_indices; ++i) {
        I row = h_indices[i];
        for (int c = 0; c < embed_width; ++c) {
            float expected = h_embedding_f[row * embed_width + c];
            float got = __bfloat162float(h_out[i * embed_width + c]);
            float denom = std::max(1.0f, std::fabs(expected));
            if (std::fabs(got - expected) > std::max(abs_tol, rel_tol * denom)) {
                if (errors < 8) {
                    std::cerr << "Mismatch at idx " << i << ", col " << c << ": got "
                              << got << ", exp " << expected << std::endl;
                }
                ++errors;
            }
        }
    }

    CUDA_CHECK(cudaFree(d_embedding));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaStreamDestroy(stream));

    if (errors) {
        std::cerr << "FAIL: test_embedding_lookup_bf16 (errors=" << errors << ")" << std::endl;
        return 1;
    }
    std::cout << "PASS: test_embedding_lookup_bf16" << std::endl;
    return 0;
}

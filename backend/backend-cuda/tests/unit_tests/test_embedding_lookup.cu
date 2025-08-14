#include <cassert>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "../../src/common.cuh"

int main() {
    using T = float;
    using I = int32_t;

    const int embed_width = 8; // multiple of 4 floats => 32 bytes
    const size_t num_rows = 5;
    const size_t num_indices = 3;

    std::vector<T> h_embedding(num_rows * embed_width);
    for (size_t r = 0; r < num_rows; ++r) {
        for (int c = 0; c < embed_width; ++c) {
            h_embedding[r * embed_width + c] = static_cast<T>(r * 100 + c);
        }
    }
    std::vector<I> h_indices = {3, 1, 4};

    T *d_embedding = nullptr, *d_result = nullptr;
    I *d_indices = nullptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMalloc(&d_embedding, h_embedding.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_indices, h_indices.size() * sizeof(I)));
    CUDA_CHECK(cudaMalloc(&d_result, num_indices * embed_width * sizeof(T)));

    CUDA_CHECK(cudaMemcpy(d_embedding, h_embedding.data(), h_embedding.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(), h_indices.size() * sizeof(I), cudaMemcpyHostToDevice));

    embed<T,I>(d_embedding, num_rows, d_indices, num_indices, d_result, embed_width, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<T> h_out(num_indices * embed_width);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_result, h_out.size() * sizeof(T), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < num_indices; ++i) {
        I row = h_indices[i];
        for (int c = 0; c < embed_width; ++c) {
            T expected = static_cast<T>(row * 100 + c);
            if (std::fabs(h_out[i * embed_width + c] - expected) > 1e-6f) {
                std::cerr << "Mismatch at idx " << i << ", col " << c << ": got "
                          << h_out[i * embed_width + c] << ", expected " << expected << std::endl;
                return 1;
            }
        }
    }

    CUDA_CHECK(cudaFree(d_embedding));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::cout << "PASS: test_embedding_lookup" << std::endl;
    return 0;
}
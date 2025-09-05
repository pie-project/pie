#include <cassert>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "../../src/common.cuh"
#include "../../src/tensor.hpp"

int main() {
    try {
        // Prepare host data
        std::vector<float> h = {0.5f, -1.25f, 3.0f, 7.75f};
        const size_t bytes = h.size() * sizeof(float);

        // Allocate ByteTensor and copy bytes
        ByteTensor bytes_dev(bytes);
        CUDA_CHECK(cudaMemcpy(bytes_dev.data(), h.data(), bytes, cudaMemcpyHostToDevice));

        // Create a float Tensor view over the byte buffer
        Tensor<float> view(bytes_dev, /*byte_offset*/0, /*count*/h.size());
        std::vector<float> got = view.to_vector();

        for (size_t i = 0; i < h.size(); ++i) {
            if (std::fabs(got[i] - h[i]) > 1e-6f) {
                std::cerr << "Mismatch at " << i << ": got " << got[i] << ", exp " << h[i] << std::endl;
                return 1;
            }
        }

        std::cout << "PASS: test_tensor_view" << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return 2;
    }
}
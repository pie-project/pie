#include <cstdint>
#include <iostream>
#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>

#include "store/elastic.hpp"
#include "tensor.hpp"

int main() {
    try {
        if (cuInit(0) != CUDA_SUCCESS || cudaSetDevice(0) != cudaSuccess) {
            std::cerr << "CUDA initialization failed\n";
            return 1;
        }

        auto pool = std::make_shared<pie_cuda_driver::CudaPhysicalPool>(
            0,
            128ull << 20);
        pie_cuda_driver::CudaArena arena(pool, 64ull << 20, "smoke");
        const std::uint64_t base = arena.base();

        arena.ensure_committed(1);
        if (cudaMemset(
                reinterpret_cast<void*>(base),
                0x5a,
                1ull << 20) != cudaSuccess) {
            return 2;
        }
        arena.ensure_committed(48ull << 20);
        if (arena.base() != base ||
            cudaMemset(
                reinterpret_cast<void*>(base + (40ull << 20)),
                0xa5,
                1ull << 20) != cudaSuccess) {
            return 3;
        }

        arena.trim_committed(16ull << 20);
        arena.ensure_committed(48ull << 20);
        if (arena.base() != base ||
            cudaMemset(
                reinterpret_cast<void*>(base + (40ull << 20)),
                0x3c,
                1ull << 20) != cudaSuccess) {
            return 4;
        }

        auto allocator =
            std::make_shared<pie_cuda_driver::CudaArenaAllocator>(
                pool,
                "tensor-smoke",
                false);
        pie_cuda_driver::DeviceTensor tensor;
        {
            pie_cuda_driver::ScopedCudaArenaAllocator scope(*allocator);
            tensor = pie_cuda_driver::DeviceTensor::allocate(
                pie_cuda_driver::DType::UINT8,
                {8ll << 20});
        }
        allocator->ensure_all();
        if (cudaMemset(tensor.data(), 0, tensor.nbytes()) != cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess) {
            return 5;
        }
        if (pool->committed_pages() == 0 ||
            pool->committed_pages() > pool->budget_pages()) {
            return 6;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 7;
    }
}

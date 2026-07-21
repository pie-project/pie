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

        const std::uint64_t generation_before_trim = pool->generation();
        arena.trim_committed(16ull << 20);
        if (pool->generation() <= generation_before_trim) {
            return 4;
        }
        arena.ensure_committed(48ull << 20);
        if (arena.base() != base ||
            cudaMemset(
                reinterpret_cast<void*>(base + (40ull << 20)),
                0x3c,
                1ull << 20) != cudaSuccess) {
            return 5;
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
            return 6;
        }
        if (pool->committed_pages() == 0 ||
            pool->charged_pages() !=
                pool->committed_pages() + pool->held_pages() ||
            pool->charged_pages() > pool->budget_pages()) {
            return 7;
        }

        auto floor_pool =
            std::make_shared<pie_cuda_driver::CudaPhysicalPool>(
                0,
                pie::elastic::kLogicalPageBytes +
                    pie::elastic::kLogicalPageBytes / 2);
        if (floor_pool->budget_pages() != 1) {
            return 8;
        }

        auto rollback_pool =
            std::make_shared<pie_cuda_driver::CudaPhysicalPool>(
                0,
                128ull << 20);
        pie_cuda_driver::CudaArenaAllocator rollback_allocator(
            rollback_pool,
            "rollback",
            false);
        {
            pie_cuda_driver::ScopedCudaArenaAllocator scope(
                rollback_allocator);
            auto first = pie_cuda_driver::DeviceTensor::allocate(
                pie_cuda_driver::DType::UINT8,
                {8ll << 20});
            auto second = pie_cuda_driver::DeviceTensor::allocate(
                pie_cuda_driver::DType::UINT8,
                {8ll << 20});
            static_cast<void>(first);
            static_cast<void>(second);
        }
        rollback_pool->fail_mapping_after_for_test(1);
        bool failed = false;
        try {
            static_cast<void>(
                pie_cuda_driver::commit_cuda_arena_targets_atomically(
                    rollback_pool,
                    {{&rollback_allocator, 1, 1}}));
        } catch (const std::exception&) {
            failed = true;
        }
        if (!failed ||
            rollback_allocator.committed_bytes() != 0 ||
            rollback_pool->committed_pages() != 0 ||
            rollback_pool->held_pages() != 0) {
            return 9;
        }
        const auto committed =
            pie_cuda_driver::commit_cuda_arena_targets_atomically(
                rollback_pool,
                {{&rollback_allocator, 1, 1}});
        if (committed.outcome !=
                pie_cuda_driver::CudaCommitOutcome::Committed ||
            rollback_pool->charged_pages() >
                rollback_pool->budget_pages()) {
            return 10;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 11;
    }
}

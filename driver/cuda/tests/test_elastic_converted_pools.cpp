#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>

#include <cuda_runtime.h>

#include "ops/gemm.hpp"
#include "store/elastic.hpp"
#include "store/mla_cache.hpp"

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

std::shared_ptr<pie_cuda_driver::CudaPhysicalPool> make_pool() {
    int device = 0;
    cudaGetDevice(&device);
    return std::make_shared<pie_cuda_driver::CudaPhysicalPool>(
        device, 512ull * 1024 * 1024);
}

}  // namespace

int main() {
    {
        auto pool = make_pool();
        pie_cuda_driver::CudaArenaAllocator allocator(
            pool, "test-mla", false);
        pie_cuda_driver::MlaCache cache;
        {
            pie_cuda_driver::ScopedCudaArenaAllocator scoped(allocator);
            cache = pie_cuda_driver::MlaCache::allocate(
                2, 4, 16, 8, 4, pie_cuda_driver::DType::BF16);
        }
        constexpr std::size_t expected =
            2 * 4 * 16 * (8 + 4) * sizeof(std::uint16_t);
        check(
            allocator.allocated_bytes() == expected,
            "MLA cache must reserve its virtual storage in the KV arena");
        check(
            allocator.committed_bytes() == 0,
            "MLA allocation must not commit physical memory");
        allocator.ensure_fraction(1, 4);
        check(
            cudaMemset(cache.ckv(0), 0, 16 * 8 * sizeof(std::uint16_t)) ==
                cudaSuccess,
            "committed MLA page must be writable");
        check(
            cudaDeviceSynchronize() == cudaSuccess,
            "MLA page write must complete before trim");
        allocator.trim_fraction(0, 4);
        check(
            allocator.committed_bytes() == 0,
            "MLA trim must release all physical mappings");
    }

    {
        auto pool = make_pool();
        pie_cuda_driver::CudaArenaAllocator allocator(
            pool, "test-runtime-quant", false);
        const pie_cuda_driver::ops::RuntimeQuantScratchSpec spec{
            .max_tokens = 4,
            .max_weight_rows = 128,
            .max_weight_cols = 128,
            .has_fp8 = true,
            .has_int8 = true,
        };
        pie_cuda_driver::ops::RuntimeQuantContext quant_context;
        pie_cuda_driver::ops::ScopedRuntimeQuantContext quant_scope(
            quant_context);
        {
            pie_cuda_driver::ScopedCudaArenaAllocator scoped(allocator);
            pie_cuda_driver::ops::reserve_runtime_quant_scratch(
                spec, true);
        }
        check(
            allocator.allocated_bytes() ==
                pie_cuda_driver::ops::runtime_quant_scratch_bytes(spec),
            "runtime-quant scratch must reserve in the workspace arena");
        check(
            allocator.committed_bytes() == 0,
            "runtime-quant reservation must remain physically lazy");
        allocator.ensure_all();
        check(
            allocator.committed_bytes() > 0,
            "runtime-quant scratch must commit with the workspace arena");
        quant_context.reset();
    }

    std::puts("elastic converted pools: OK");
    return 0;
}

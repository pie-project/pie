#include "kernels/dequant_fp8.hpp"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// NOTE on the design choice: at runtime, the right move is to skip
// dequant entirely and use cuBLAS's FP8 GEMM (cublasGemmEx with
// CUDA_R_8F_E4M3 + CUBLAS_COMPUTE_32F) — that path also accepts a
// scalar scale per operand and avoids the 2× memory blow-up. We keep
// this load-time dequant kernel as the simple option that lets
// existing bf16 GEMMs run unchanged. Larger Mistral checkpoints
// (24B+) will need the fused-GEMM path; the kernel here is a
// reference + small-model fallback.

namespace pie_cuda_driver::kernels {

namespace {

constexpr int BLOCK = 256;

__global__ void dequant_fp8_e4m3_kernel(
    const __nv_fp8_storage_t* __restrict__ src,
    __nv_bfloat16*            __restrict__ dst,
    float                                  scale,
    std::size_t                            n)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    // Use the half-conversion intrinsic to keep this dependency on the
    // CUDA FP8 ABI minimal; older CUDA toolkits don't expose a direct
    // fp8 → fp32 free function, but the storage → __half path is stable
    // back to 11.8.
    const __half h = __nv_cvt_fp8_to_halfraw(src[i], __NV_E4M3);
    const float f = __half2float(h) * scale;
    dst[i] = __float2bfloat16(f);
}

}  // namespace

void launch_dequant_fp8_e4m3_to_bf16(
    const std::uint8_t* fp8_in, void* bf16_out,
    float scale, std::size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    dequant_fp8_e4m3_kernel<<<blocks, BLOCK, 0, stream>>>(
        reinterpret_cast<const __nv_fp8_storage_t*>(fp8_in),
        static_cast<__nv_bfloat16*>(bf16_out),
        scale, n);
}

}  // namespace pie_cuda_driver::kernels

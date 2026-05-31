#include "dequant_fp8.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Lifted from driver/cuda/src/kernels/dequant_fp8.cu (base scalar-scale
// variant). Verbatim apart from the namespace and the dropped `launch_` prefix;
// the `_per_channel` and `_per_group` kernels/launchers are omitted.

namespace pie_cuda_device::kernels {

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
    const __half h = __nv_cvt_fp8_to_halfraw(src[i], __NV_E4M3);
    const float f = __half2float(h) * scale;
    dst[i] = __float2bfloat16(f);
}

}  // namespace

void dequant_fp8_e4m3_to_bf16(
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

}  // namespace pie_cuda_device::kernels

// The host launcher, and nothing else. The `__global__` and the block
// reduction it folds through live in `attn/attn_res.cuh` -- ONE definition,
// read by nvcc here and by NVRTC from the same text at run time.
//
// `T` is still an argument to this function because the grid is `T` blocks
// and because `block_rows > 0 ? block_rows : T` defaults through it. It is no
// longer an argument to the KERNEL: the row states `LaunchRule::Rms`, whose
// grid is exactly `rows`, so the bound check it guarded is the rule's.
#include "attn/attn_res.cuh"
#include "attn/attn_res.hpp"

namespace pie_cuda_driver::kernels::attn {

void attn_res_blend_bf16(
    const void* prefix, const void* blocks, const void* norm_weight,
    const void* proj_weight, void* out, int T, int B, int H, int block_rows,
    float eps, cudaStream_t stream)
{
    using bf16 = ::pie_cuda_driver::kernels::device::bf16;
    if (T <= 0 || H <= 0) return;
    device::attn_res_blend<bf16>
        <<<static_cast<unsigned>(T), device::kThreads, 0, stream>>>(
            static_cast<const bf16*>(prefix),
            static_cast<const bf16*>(blocks),
            static_cast<const bf16*>(norm_weight),
            static_cast<const bf16*>(proj_weight),
            static_cast<bf16*>(out), B, H,
            block_rows > 0 ? block_rows : T, eps);
}

}  // namespace pie_cuda_driver::kernels::attn

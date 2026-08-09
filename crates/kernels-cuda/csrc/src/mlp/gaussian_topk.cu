// The launcher, and nothing else.
//
// The `__global__` moved to `mlp/gaussian_topk.cuh`, which the JIT compiles
// at run time and which this file includes so the ahead-of-time archive keeps
// exactly ONE definition of it. `<cooperative_groups.h>` left with it: the
// prelude's `block_sum` is the same fold, and NVRTC has no include path to
// resolve the header on.
//
// This launcher IS `LaunchRule::Rms` -- one block per row, 256 threads,
// `(256 / 32) * sizeof(float)` bytes of dynamic shared memory -- so the row
// in `kernels_cuda_new::families::mlp` states everything below the include.
#include "pie_device.cuh"
#include "mlp/gaussian_topk.cuh"
#include "mlp/gaussian_topk.hpp"

namespace pie_cuda_driver::kernels::mlp {

void gaussian_topk_bf16(
    void* x, int N, int dim,
    float std_multiplier, cudaStream_t stream)
{
    if (N <= 0 || dim <= 0) return;
    constexpr int BLOCK = 256;
    const int smem_bytes = (BLOCK / 32) * sizeof(float);
    device::gaussian_topk<device::bf16><<<N, BLOCK, smem_bytes, stream>>>(
        static_cast<device::bf16*>(x), dim, std_multiplier);
}

}  // namespace pie_cuda_driver::kernels::mlp

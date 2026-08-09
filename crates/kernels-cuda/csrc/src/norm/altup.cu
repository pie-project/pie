// The kernels live in the header; this file is the two entry points that
// still launch them ahead of time -- and, for altup, the only way they are
// launched at all: their three-axis grid matches no stated `LaunchRule`, so
// no JIT row names them. `altup.cuh` says why.
#include "pie_device.cuh"
#include "norm/altup.cuh"
#include "norm/altup.hpp"


namespace pie_cuda_driver::kernels::norm {

void altup_predict_bf16(
    const void* streams, const float* coefs, void* predictions,
    int K, int T, int H, cudaStream_t stream)
{
    if (T <= 0 || K <= 0 || H <= 0) return;
    constexpr int BLOCK = 128;
    const dim3 grid(T, K, (H + BLOCK - 1) / BLOCK);
    device::altup_predict<device::bf16><<<grid, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(streams), coefs,
        static_cast<device::bf16*>(predictions),
        K, T, H);
}

void altup_correct_bf16(
    const void* predictions, const void* activated,
    const float* correction_coefs_plus_one, void* corrected,
    int K, int T, int H, int active_idx, cudaStream_t stream)
{
    if (T <= 0 || K <= 0 || H <= 0) return;
    constexpr int BLOCK = 128;
    const dim3 grid(T, K, (H + BLOCK - 1) / BLOCK);
    device::altup_correct<device::bf16><<<grid, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(predictions),
        static_cast<const device::bf16*>(activated),
        correction_coefs_plus_one,
        static_cast<device::bf16*>(corrected),
        K, T, H, active_idx);
}

}  // namespace pie_cuda_driver::kernels::norm

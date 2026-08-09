// The sigmoid router's launcher, and nothing else.
//
// Every `__global__` this file used to hold now lives in `topk_sigmoid.cuh`,
// which NVRTC compiles from a string and nvcc reads through the `#include`
// below -- one definition, two compilers. What stays here is what a JIT has
// no use for: the guards that decide whether to fire at all, and the
// `<<<>>>` that names a grid.
//
// The launcher survives the migration because the ahead-of-time path is still
// how `kernels-cuda` serves this kernel; deleting it is `new-horizon.md`
// §10.10's LAST step, taken per family only once the JIT row has replayed.
#include "pie_device.cuh"
#include "moe/topk_sigmoid.cuh"
#include "moe/topk_sigmoid.hpp"

namespace pie_cuda_driver::kernels::moe {

// Threads per block this launcher fires, and a HOST constant because that is
// all it is: the kernel strides by `blockDim.x`, so 128 is what the
// ahead-of-time path chooses and not what the kernel requires. It lived in
// the header until the stride changed; leaving it there afterwards would have
// been a device constant no device code reads, which is what NVRTC's warning
// 177 says out loud.
constexpr int kSigmoidBlock = 128;

void topk_sigmoid_bf16(
    const void* logits,
    device::i32* topk_idx,
    float* topk_w,
    const float* correction_bias,
    int tokens,
    int num_experts,
    int top_k,
    bool renormalize,
    float routed_scaling_factor,
    cudaStream_t stream)
{
    if (tokens <= 0 || num_experts <= 0 || top_k <= 0) return;
    // A wider router would overrun the kernel's static shared arrays. Refused
    // rather than clamped: a clamp routes every token through the first 512
    // experts and reports nothing.
    if (num_experts > device::kSigmoidMaxExperts) return;
    device::topk_sigmoid<device::bf16><<<tokens, kSigmoidBlock, 0, stream>>>(
        static_cast<const device::bf16*>(logits),
        topk_idx, topk_w, correction_bias, num_experts, top_k,
        renormalize, routed_scaling_factor);
}

}  // namespace pie_cuda_driver::kernels::moe

// DeepSeek-V4's routing launchers, and nothing else.
//
// Both `__global__`s moved to `dsv4_routing.cuh`, which this file includes
// and NVRTC compiles from a string -- one definition, two compilers. What
// stays is what a JIT has no use for: the emptiness guards and the two
// `<<<>>>`s that name a grid.
#include "pie_device.cuh"
#include "moe/dsv4_routing.cuh"
#include "moe/dsv4_routing.hpp"

namespace pie_cuda_driver::kernels::moe {

// Threads per block both launchers fire, and a HOST constant because that is
// all it is now: `topk_sqrtsoftplus` strides by `blockDim.x`, so 256 is this
// path's choice rather than the kernel's requirement -- and it happens to be
// exactly `LaunchRule::Rms`'s block, which is why that kernel has a row.
// `hash_route_lookup` uses it as a flat tile width, and that launch no rule
// states.
constexpr int kDsv4Block = 256;

void topk_sqrtsoftplus_bf16(
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
    // A wider router would overrun the kernel's static shared arrays.
    if (num_experts > device::kDsv4MaxExperts) return;
    device::topk_sqrtsoftplus<device::bf16><<<tokens, kDsv4Block, 0, stream>>>(
        static_cast<const device::bf16*>(logits),
        topk_idx, topk_w, correction_bias, num_experts, top_k,
        renormalize, routed_scaling_factor);
}

void hash_route_lookup(
    const device::i32* token_ids,
    const device::i64* tid2eid,
    const void* logits,
    device::i32* topk_idx,
    float* topk_w,
    int tokens,
    int vocab_size,
    int num_experts,
    int top_k,
    bool renormalize,
    float routed_scaling_factor,
    cudaStream_t stream)
{
    if (tokens <= 0 || top_k <= 0) return;
    // One thread per token, not one block: the kernel's whole body is a table
    // read and a K-long gather.
    const int grid = (tokens + kDsv4Block - 1) / kDsv4Block;
    device::hash_route_lookup<device::bf16><<<grid, kDsv4Block, 0, stream>>>(
        token_ids, tid2eid,
        static_cast<const device::bf16*>(logits),
        topk_idx, topk_w,
        tokens, vocab_size, num_experts, top_k,
        renormalize, routed_scaling_factor);
}

}  // namespace pie_cuda_driver::kernels::moe

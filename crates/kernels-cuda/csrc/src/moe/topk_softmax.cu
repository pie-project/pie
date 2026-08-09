// The routers' launchers, and nothing else.
//
// Every `__global__` this file used to hold now lives in `topk_softmax.cuh`,
// which NVRTC compiles from a string and nvcc reads through the `#include`
// below -- one definition, two compilers. What stays here is what a JIT has
// no use for and cannot have: `std::getenv`, `throw`, and the run-time ladder
// that picks a `PER_LANE` from the expert count. Those are host decisions,
// and they are why this file is not empty.
#include "pie_device.cuh"
#include "moe/topk_softmax.cuh"
#include "moe/topk_softmax.hpp"

#include <cstdlib>
#include <stdexcept>

namespace pie_cuda_driver::kernels::moe {

// Threads per block for `apply_per_expert_scale`, and a HOST constant because
// the kernel never names it: that kernel is flat pointwise, one thread per
// routed slot, and 256 is `LaunchRule::Elementwise`'s block -- which is why
// it is the one kernel in this file with a row.
constexpr int kScaleBlock = 256;

void topk_softmax_bf16(
    const void* logits,
    device::i32* topk_idx, float* topk_w,
    int N, int num_experts, int K,
    cudaStream_t stream)
{
    if (N <= 0 || num_experts <= 0 || K <= 0) return;
    if (num_experts > device::kSoftmaxMaxExperts) {
        throw std::runtime_error("topk_softmax_bf16: num_experts exceeds MAX_EXPERTS");
    }
    // PIE_TOPK_WARP=0 forces the block form, for A/B measurement.
    static const bool warp_ok = [] {
        const char* v = std::getenv("PIE_TOPK_WARP");
        return v == nullptr || v[0] != '0';
    }();
    topk_softmax_bf16_form(logits, topk_idx, topk_w, N, num_experts, K,
                                  warp_ok, stream);
}

void topk_softmax_bf16_form(
    const void* logits,
    device::i32* topk_idx, float* topk_w,
    int N, int num_experts, int K,
    bool use_warp,
    cudaStream_t stream)
{
    if (N <= 0 || num_experts <= 0 || K <= 0) return;
    if (num_experts > device::kSoftmaxMaxExperts) {
        throw std::runtime_error("topk_softmax_bf16: num_experts exceeds MAX_EXPERTS");
    }
    // The warp form keeps the experts in registers, so it applies while they
    // fit (<= 512, which is kSoftmaxMaxExperts) and while the K winners fit the
    // small result array (<= 8). Qwen3.6-35B-A3B routes through more than 128
    // and was falling back to the block form at 7.56 us/call, 4.9% of its step.
    //
    // The ladder is a host decision -- one rung is one instantiation, and a
    // JIT row can state one of them but not the choice between them.
    if (use_warp && K <= 8 && num_experts <= device::kSoftmaxMaxExperts) {
        const auto* in = static_cast<const device::bf16*>(logits);
        if (num_experts <= 32) {
            device::topk_softmax_warp_x1<device::bf16><<<N, 32, 0, stream>>>(
                in, topk_idx, topk_w, num_experts, K);
        } else if (num_experts <= 64) {
            device::topk_softmax_warp_x2<device::bf16><<<N, 32, 0, stream>>>(
                in, topk_idx, topk_w, num_experts, K);
        } else if (num_experts <= 128) {
            device::topk_softmax_warp_x4<device::bf16><<<N, 32, 0, stream>>>(
                in, topk_idx, topk_w, num_experts, K);
        } else if (num_experts <= 256) {
            device::topk_softmax_warp_x8<device::bf16><<<N, 32, 0, stream>>>(
                in, topk_idx, topk_w, num_experts, K);
        } else {
            device::topk_softmax_warp_x16<device::bf16><<<N, 32, 0, stream>>>(
                in, topk_idx, topk_w, num_experts, K);
        }
        return;
    }
    device::topk_softmax<device::bf16><<<N, device::kSoftmaxBlock, 0, stream>>>(
        static_cast<const device::bf16*>(logits),
        nullptr, nullptr, topk_idx, topk_w,
        num_experts, K, 0);
}

void router_topk_softmax_bf16(
    const void* act,
    const void* router_weight,
    const void* router_bias,
    device::i32* topk_idx,
    float* topk_w,
    int N, int num_experts, int K, int hidden,
    cudaStream_t stream)
{
    if (N <= 0 || num_experts <= 0 || K <= 0 || hidden <= 0) return;
    if (num_experts > device::kSoftmaxMaxExperts) {
        throw std::runtime_error(
            "router_topk_softmax_bf16: num_experts exceeds MAX_EXPERTS");
    }
    device::router_topk_softmax<device::bf16><<<N, device::kSoftmaxBlock, 0, stream>>>(
        static_cast<const device::bf16*>(router_weight),
        static_cast<const device::bf16*>(act),
        static_cast<const device::bf16*>(router_bias),
        topk_idx, topk_w, num_experts, K, hidden);
}

void apply_per_expert_scale_bf16(
    const device::i32* topk_idx,
    float* topk_w,
    const void* per_expert_scale_bf16,
    int N, int K,
    cudaStream_t stream)
{
    const int total = N * K;
    if (total <= 0) return;
    const int grid = (total + kScaleBlock - 1) / kScaleBlock;
    device::apply_per_expert_scale<device::bf16><<<grid, kScaleBlock, 0, stream>>>(
        topk_idx, topk_w,
        static_cast<const device::bf16*>(per_expert_scale_bf16),
        total);
}

void topk_sigmoid_bias_bf16(
    const void* logits,
    const float* correction_bias,
    device::i32* topk_idx,
    float* topk_w,
    int N,
    int num_experts,
    int K,
    bool normalize,
    float routed_scaling_factor,
    cudaStream_t stream)
{
    if (N <= 0 || num_experts <= 0 || K <= 0) return;
    device::topk_sigmoid_bias<device::bf16><<<N, device::kSoftmaxBlock, 0, stream>>>(
        static_cast<const device::bf16*>(logits),
        correction_bias,
        topk_idx,
        topk_w,
        num_experts,
        K,
        normalize ? 1 : 0,
        routed_scaling_factor);
}

void topk_sigmoid_bias_fp32(
    const float* logits,
    const float* correction_bias,
    device::i32* topk_idx,
    float* topk_w,
    int N,
    int num_experts,
    int K,
    bool normalize,
    float routed_scaling_factor,
    cudaStream_t stream)
{
    if (N <= 0 || num_experts <= 0 || K <= 0) return;
    // The same kernel as above at a different element type -- the fp32 router
    // was a second copy of it until this family became templates.
    device::topk_sigmoid_bias<device::f32><<<N, device::kSoftmaxBlock, 0, stream>>>(
        logits,
        correction_bias,
        topk_idx,
        topk_w,
        num_experts,
        K,
        normalize ? 1 : 0,
        routed_scaling_factor);
}

}  // namespace pie_cuda_driver::kernels::moe

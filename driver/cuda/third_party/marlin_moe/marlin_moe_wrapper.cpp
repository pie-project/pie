#include "marlin_moe_wrapper.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "../marlin/scalar_type.hpp"

namespace marlin_moe_wna16 {

// Defined by the detoxified `ops.cu`: the torch-free launcher that selects a
// thread config, sizes shared memory and launches the expert-indexed kernel.
void marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* b_bias,
               void* a_s, void* b_s, void* g_s, void* zp, void* g_idx,
               void* perm, void* a_tmp, void* sorted_token_ids,
               void* expert_ids, void* num_tokens_past_padded,
               void* topk_weights, int moe_block_size, int num_experts,
               int top_k, bool mul_topk_weights, int prob_m, int prob_n,
               int prob_k, void* workspace, vllm::ScalarType const& a_type,
               vllm::ScalarType const& b_type, vllm::ScalarType const& c_type,
               vllm::ScalarType const& s_type, bool has_bias,
               bool has_act_order, bool is_k_full, bool has_zp, int num_groups,
               int group_size, int dev, cudaStream_t stream, int thread_k,
               int thread_n, int sms, int blocks_per_sm, bool use_atomic_add,
               bool use_fp32_reduce, bool is_zp_float);

}  // namespace marlin_moe_wna16

namespace pie_cuda_driver::marlin_moe {

namespace {

int sm_count(int dev) {
    int sms = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) !=
        cudaSuccess) {
        throw std::runtime_error("marlin_moe: failed to query SM count");
    }
    return sms;
}

}  // namespace

std::size_t marlin_moe_workspace_bytes(int prob_n, int block_size) {
    // `locks`: one int per output column tile, which the kernel uses for the
    // cross-threadblock reduction barrier. Sized from the widest tile the
    // selector can pick (64 columns) so the buffer is valid whatever thread
    // config the heuristic lands on, and rounded up generously -- it is a few
    // kilobytes and a too-small buffer is a silent race, not an error.
    const int n_tiles = (prob_n + 63) / 64;
    const int blocks = n_tiles > 0 ? n_tiles : 1;
    (void)block_size;
    return static_cast<std::size_t>(blocks + 1024) * sizeof(int);
}

void launch_mxfp4_moe_gemm_w4a16_bf16(
    const void* act_bf16,
    const void* w_mxfp4_packed,
    const void* scales_e8m0,
    const void* bias_bf16,
    void* out_bf16,
    void* reduce_scratch,
    void* workspace,
    const std::int32_t* sorted_token_ids,
    const std::int32_t* expert_ids,
    const std::int32_t* num_tokens_past_padded,
    const float* topk_weights,
    int block_size,
    int num_experts,
    int top_k,
    bool mul_topk_weights,
    int prob_m,
    int prob_n,
    int prob_k,
    cudaStream_t stream)
{
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        throw std::runtime_error("marlin_moe: cudaGetDevice failed");
    }

    constexpr int group_size = 32;  // MXFP4 groups are 32 elements
    if (prob_k % group_size != 0) {
        throw std::runtime_error("marlin_moe: MXFP4 K must be divisible by 32");
    }
    const int num_groups = prob_k / group_size;

    marlin_moe_wna16::marlin_mm(
        /*A=*/          act_bf16,
        /*B=*/          w_mxfp4_packed,
        /*C=*/          out_bf16,
        /*C_tmp=*/      reduce_scratch,
        /*b_bias=*/     const_cast<void*>(bias_bf16),
        /*a_s=*/        nullptr,
        /*b_s=*/        const_cast<void*>(scales_e8m0),
        /*g_s=*/        nullptr,
        /*zp=*/         nullptr,
        /*g_idx=*/      nullptr,
        /*perm=*/       nullptr,
        /*a_tmp=*/      nullptr,
        /*sorted_token_ids=*/ const_cast<std::int32_t*>(sorted_token_ids),
        /*expert_ids=*/ const_cast<std::int32_t*>(expert_ids),
        /*num_tokens_past_padded=*/
            const_cast<std::int32_t*>(num_tokens_past_padded),
        /*topk_weights=*/ const_cast<float*>(topk_weights),
        /*moe_block_size=*/ block_size,
        num_experts,
        top_k,
        mul_topk_weights,
        /*prob_m=*/     prob_m,
        /*prob_n=*/     prob_n,
        /*prob_k=*/     prob_k,
        /*workspace=*/  workspace,
        vllm::kBFloat16, vllm::kFE2M1f, vllm::kBFloat16, vllm::kFE8M0fnu,
        /*has_bias=*/       bias_bf16 != nullptr,
        /*has_act_order=*/  false,
        /*is_k_full=*/      true,
        /*has_zp=*/         false,
        num_groups,
        group_size,
        dev,
        stream,
        /*thread_k=*/       -1,
        /*thread_n=*/       -1,
        /*sms=*/            sm_count(dev),
        /*blocks_per_sm=*/  -1,
        // Routes for one expert land in disjoint output rows, so no two blocks
        // accumulate into the same element and the atomic path buys nothing.
        /*use_atomic_add=*/  false,
        /*use_fp32_reduce=*/ reduce_scratch != nullptr,
        /*is_zp_float=*/     false);
}

}  // namespace pie_cuda_driver::marlin_moe

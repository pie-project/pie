#include "marlin_wrapper.hpp"

#include <stdexcept>
#include <string>

#include "scalar_type.hpp"

// Forward-declare the torch-free internal launchers that live in
// marlin.cu / gptq_marlin_repack.cu. Kept out of `marlin.cuh` (which
// still has some torch-typed stubs) so this TU doesn't pull in torch.
namespace marlin {
void marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* b_bias,
               void* a_s, void* b_s, void* g_s, void* zp, void* g_idx,
               void* perm, void* a_tmp, int prob_m, int prob_n, int prob_k,
               int lda, void* workspace, vllm::ScalarType const& a_type,
               vllm::ScalarType const& b_type, vllm::ScalarType const& c_type,
               vllm::ScalarType const& s_type, bool has_bias,
               bool has_act_order, bool is_k_full, bool has_zp, int num_groups,
               int group_size, int dev, cudaStream_t stream, int thread_k_init,
               int thread_n_init, int sms, bool use_atomic_add,
               bool use_fp32_reduce, bool is_zp_float);

void pie_gptq_marlin_repack_w4_no_perm(
    const std::uint32_t* b_q_weight, std::uint32_t* out,
    int size_k, int size_n, cudaStream_t stream);

void pie_awq_marlin_repack_w4(
    const std::uint32_t* b_q_weight, std::uint32_t* out,
    int size_k, int size_n, cudaStream_t stream);
}  // namespace marlin

namespace pie_cuda_driver::marlin {

std::size_t marlin_gptq_workspace_bytes(int M, int N, int K, int group_size) {
    // Marlin's `min_workspace_size` upstream is `sms` int32s. We allocate
    // a generous 128 KiB scratch — covers up to ~32K SMs (well past any
    // current GPU) and is small enough to one-shot allocate at engine
    // init. The caller passes the full buffer in to launch_*; marlin
    // checks `workspace.numel() >= sms` against this.
    (void)M; (void)N; (void)K; (void)group_size;
    return 128 * 1024;
}

void launch_gptq_gemm_w4a16_bf16(
    const void* act_bf16,
    const void* w_q4_packed,
    const void* scales_bf16,
    const void* qzeros_marlin,
    void*       out_bf16,
    void*       workspace,
    int         M,
    int         N,
    int         K,
    int         group_size,
    bool        use_fp32_reduce,
    cudaStream_t stream)
{
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        throw std::runtime_error("marlin: cudaGetDevice failed");
    }
    int sms = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev)
            != cudaSuccess) {
        throw std::runtime_error("marlin: failed to query SM count");
    }

    // Symmetric (GPTQ): kU4B8 with implicit bias-8 — no zero-point.
    // Asymmetric (AWQ):  kU4 with explicit zero-points.
    const bool has_zp = (qzeros_marlin != nullptr);
    const auto a_type = vllm::kBFloat16;
    const auto b_type = has_zp ? vllm::kU4 : vllm::kU4B8;
    const auto c_type = vllm::kBFloat16;
    const auto s_type = vllm::kBFloat16;

    const int num_groups = (group_size > 0)
        ? (K / group_size)
        : 1;  // -1 / per-channel collapses to a single group of length K

    // c_tmp is only consumed when `use_fp32_reduce=true` — it's the fp32
    // accumulator buffer for marlin's parallel split-K reduce. We wire
    // a NULL c_tmp + use_fp32_reduce=false in v1; the GEMM dispatcher
    // can opt into fp32 accumulation later by allocating a sized scratch.
    void* c_tmp = nullptr;

    ::marlin::marlin_mm(
        /*A=*/         act_bf16,
        /*B=*/         w_q4_packed,
        /*C=*/         out_bf16,
        /*C_tmp=*/     c_tmp,
        /*b_bias=*/    nullptr,
        /*a_s=*/       nullptr,
        /*b_s=*/       const_cast<void*>(scales_bf16),
        /*g_s=*/       nullptr,
        /*zp=*/        const_cast<void*>(qzeros_marlin),
        /*g_idx=*/     nullptr,
        /*perm=*/      nullptr,
        /*a_tmp=*/     nullptr,
        /*prob_m=*/    M,
        /*prob_n=*/    N,
        /*prob_k=*/    K,
        /*lda=*/       K,
        /*workspace=*/ workspace,
        a_type, b_type, c_type, s_type,
        /*has_bias=*/      false,
        /*has_act_order=*/ false,
        /*is_k_full=*/     true,
        has_zp,
        num_groups,
        group_size,
        dev,
        stream,
        /*thread_k_init=*/ -1,
        /*thread_n_init=*/ -1,
        sms,
        /*use_atomic_add=*/ false,
        /*use_fp32_reduce=*/ use_fp32_reduce && (c_tmp != nullptr),
        /*is_zp_float=*/    false);
}

void launch_gptq_repack_w4_no_perm(
    const void*  qweight_in,
    void*        repacked_out,
    int          size_k,
    int          size_n,
    cudaStream_t stream)
{
    ::marlin::pie_gptq_marlin_repack_w4_no_perm(
        static_cast<const std::uint32_t*>(qweight_in),
        static_cast<std::uint32_t*>(repacked_out),
        size_k, size_n, stream);
}

void launch_awq_repack_w4(
    const void*  qweight_in,
    void*        repacked_out,
    int          size_k,
    int          size_n,
    cudaStream_t stream)
{
    ::marlin::pie_awq_marlin_repack_w4(
        static_cast<const std::uint32_t*>(qweight_in),
        static_cast<std::uint32_t*>(repacked_out),
        size_k, size_n, stream);
}

}  // namespace pie_cuda_driver::marlin

#include "marlin_wrapper.hpp"

// Forward-declare the torch-free launcher in gptq_marlin_repack.cu.  Keeping
// this adapter out of marlin_wrapper.cpp lets the repacker link without the
// dense Marlin GEMM translation unit.
namespace marlin {
void pie_gptq_marlin_repack_w4_no_perm(
    const std::uint32_t* b_q_weight, std::uint32_t* out,
    int size_k, int size_n, cudaStream_t stream);
}  // namespace marlin

namespace pie_cuda_driver::marlin {

void launch_gptq_repack_w4_no_perm(
    const void* qweight_in,
    void* repacked_out,
    int size_k,
    int size_n,
    cudaStream_t stream)
{
    ::marlin::pie_gptq_marlin_repack_w4_no_perm(
        static_cast<const std::uint32_t*>(qweight_in),
        static_cast<std::uint32_t*>(repacked_out),
        size_k, size_n, stream);
}

}  // namespace pie_cuda_driver::marlin

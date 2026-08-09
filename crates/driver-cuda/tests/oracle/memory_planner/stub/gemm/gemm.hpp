#pragma once
#include <cstddef>
namespace pie_cuda_driver::kernels::gemm {
struct RuntimeQuantScratchSpec {
    std::size_t max_tokens = 0;
    int hidden = 0;
    int group = 0;
};
std::size_t runtime_quant_scratch_bytes(const RuntimeQuantScratchSpec& spec);
}  // namespace pie_cuda_driver::kernels::gemm

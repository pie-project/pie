// The host launcher, and nothing else. The `__global__` lives in
// `attn/softcap.cuh` -- ONE definition, compiled by nvcc into the archive
// here and by NVRTC from the same text at run time. A copy in this file would
// be a second kernel that agrees until the day it does not.
#include "attn/softcap.cuh"
#include "attn/softcap.hpp"


namespace pie_cuda_driver::kernels::attn {

namespace {

constexpr int BLOCK = 256;

}  // namespace

void logit_softcap_bf16(
    void* x, float cap, std::size_t n, cudaStream_t stream)
{
    if (n == 0 || !(cap > 0.f)) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    device::logit_softcap<::pie_cuda_driver::kernels::device::bf16>
        <<<blocks, BLOCK, 0, stream>>>(
            static_cast<::pie_cuda_driver::kernels::device::bf16*>(x), cap, n);
}

}  // namespace pie_cuda_driver::kernels::attn

//===-- quant_bf16_to_mxfp4.cu - the ahead-of-time entry point -----------===//
//
// One launcher and no device text. The packer lives in
// `quant_bf16_to_mxfp4.cuh`, which the JIT compiles from the same bytes --
// see the header for why the split exists and what the `<<<>>>` became.
//
// This entry point is NOT going away: `model-loader` calls
// `quantize_bf16_to_mxfp4_e2m1_per_block` directly while the checkpoint path
// is still ahead-of-time.
//
//===----------------------------------------------------------------------===//
#include "quant/quant_bf16_to_mxfp4.hpp"

#include "quant/quant_bf16_to_mxfp4.cuh"

namespace pie_cuda_driver::kernels::quant {

namespace {

constexpr int BLOCK = 256;

}  // namespace

void quantize_bf16_to_mxfp4_e2m1_per_block(
    const void* W_bf16, device::u8* W_packed, device::u8* W_scale_e8m0,
    int rows, int cols, cudaStream_t stream)
{
    if (rows == 0 || cols == 0) return;
    device::quant_bf16_to_mxfp4_row<device::bf16><<<rows, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(W_bf16),
        W_packed, W_scale_e8m0, cols);
}

}  // namespace pie_cuda_driver::kernels::quant

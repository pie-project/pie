//===-- dtype_cast.cu - the ahead-of-time entry points -------------------===//
//
// Two launchers and no device text. Every `__global__` this file fires
// lives in `dtype_cast.cuh`, which the JIT compiles from the same bytes --
// see the header for why the split exists, what each `<<<>>>` became, and
// which four have no launch rule.
//
// `model-loader` calls `pie_k_quant_cast_fp32_to_bf16` and `scale_rows_bf16`
// by name from Rust, so none of these entry points is going away.
//
//===----------------------------------------------------------------------===//
#include "quant/dtype_cast.hpp"

#include "quant/dtype_cast.cuh"

namespace pie_cuda_driver::kernels::quant {

namespace {

constexpr int BLOCK = 256;


}  // namespace

// # Nine of eleven launchers went, and only two entry points were ever used
//
// §43 deleted `cast_fp16_to_bf16`, `cast_bf16_to_fp32`, `cast_e8m0_to_fp32`,
// `scale_bf16`, `scale_fp32`, `scale_fp16`, `marlin_permute_scales_bf16`,
// `awq_dequant_to_bf16` and `gptq_dequant_to_bf16`. `MARLIN_GROUP_PERM_LEN`
// went with the permute launcher, its only reader.
//
// The header above used to say "none of these entry points is going away"
// because `model-loader` names two of them. It names exactly two:
// `pie_k_quant_cast_fp32_to_bf16` and `scale_rows_bf16`, and those two stay.
// The sentence was true about the file and false about the other nine -- the
// §28 error at file granularity.
//
// Four of the nine are still jobs: `cast_bf16_to_fp32`, `cast_e8m0_to_fp32`,
// `scale_bf16` and `scale_fp32` have rows in `families::quant`, which fires
// the templates in `quant/dtype_cast.cuh` under NVRTC. The other five --
// `cast_fp16_to_bf16`, `scale_fp16`, the Marlin permute and the AWQ/GPTQ
// dequantisers -- have no row and no caller in any language; the AWQ and
// GPTQ templates stay in the `.cuh`, where `families::quant` documents their
// two-dimensional blocks as the reason they carry no rule.

void cast_fp32_to_bf16(
    const void* src_fp32, void* dst_bf16,
    std::size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    device::cast_f32_to<device::bf16><<<blocks, BLOCK, 0, stream>>>(
        static_cast<const float*>(src_fp32),
        static_cast<device::bf16*>(dst_bf16), n);
}


void scale_rows_bf16(
    void*         buf_bf16,
    const void*   l_bf16,
    int           rows,
    int           width,
    cudaStream_t  stream)
{
    if (rows == 0 || width == 0) return;
    // One block per row, columns strided -- `LaunchRule::RouteRows`' shape.
    // The block width is the launcher's to pick because the kernel reads
    // `blockDim.x`; 256 here, `ceil_warp(width)` under the rule, same answer.
    device::scale_rows<device::bf16><<<rows, BLOCK, 0, stream>>>(
        static_cast<device::bf16*>(buf_bf16),
        static_cast<const device::bf16*>(l_bf16),
        width);
}

}  // namespace pie_cuda_driver::kernels::quant

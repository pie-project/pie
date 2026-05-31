#pragma once

// Element-wise dtype casts used by the loader to bring non-bf16
// checkpoints into our standard bf16 format. Lifted verbatim from
// driver/cuda/src/kernels/dtype_cast.{hpp,cu} (the three element-wise
// casts only; the marlin/awq/gptq dequant + permute entries are lifted
// separately as the quantized-loader paths land). The only changes are
// the namespace (pie_cuda_driver -> pie_cuda_device::kernels) and the
// dropped `launch_` prefix on the entry points.

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_device::kernels {

/// `dst[i] = (bf16)src[i]` for `n` elements.
void cast_fp16_to_bf16(
    const void*   src_fp16,
    void*         dst_bf16,
    std::size_t   n,
    cudaStream_t  stream);

/// `dst[i] = (bf16)src[i]` for `n` fp32 elements. Used when ckpts ship
/// projection scales as fp32 but our GEMM dispatcher (or the cuBLASLt
/// scale path) expects bf16.
void cast_fp32_to_bf16(
    const void*   src_fp32,
    void*         dst_bf16,
    std::size_t   n,
    cudaStream_t  stream);

/// `dst[i] = (fp32)src[i]` for `n` bf16 elements. Used by the
/// compressed-tensors FP8 loader when scales ship as bf16 but the
/// dispatcher (cuBLASLt scale-pointer / dequant fallback) requires
/// fp32. Equivalent to `(int32(bf16_bits) << 16) reinterpreted as fp32`.
void cast_bf16_to_fp32(
    const void*   src_bf16,
    void*         dst_fp32,
    std::size_t   n,
    cudaStream_t  stream);

}  // namespace pie_cuda_device::kernels

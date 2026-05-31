#pragma once
//
// Compile-time scalar-id -> CUDA vector types + conversions for the fused
// quant GEMM. (De-branded port of `marlin_dtypes.cuh`.)
//
// Only half and bfloat16 compute/output dtypes are mapped here, since this
// slice instantiates only the bf16 path. fp8 / int8 activation dtypes can be
// added as additional QTypeMap specializations when those weight paths land.

#include "qgemm_common.cuh"
#include "qtype.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace pie_cuda_device::qgemm {

template <long scalar_type_id>
class QTypeMap {};

template <>
class QTypeMap<kFloat16.id()> {
 public:
  using scalar_t = half;
  using scalar_t2 = half2;
  using scalar_t4 = half2;
  using scalar_32bit_t = half2;

  // Matrix fragments for the m16n8k16 fp16/bf16 MMA. See:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
  using FragA = Vec<half2, 4>;
  using FragB = Vec<half2, 2>;
  using FragC = Vec<float, 4>;
  using FragS = Vec<half2, 1>;
  using FragZP = Vec<half2, 4>;

  static __device__ float inline num2float(const half x) {
    return __half2float(x);
  }
  static __device__ half2 inline num2num2(const half x) {
    return __half2half2(x);
  }
  static __device__ half2 inline nums2num2(const half x1, const half x2) {
    return __halves2half2(x1, x2);
  }
  static __host__ __device__ half inline float2num(const float x) {
    return __float2half(x);
  }
  static __host__ __device__ float2 inline num22float2(const half2 x) {
    return __half22float2(x);
  }
};

template <>
class QTypeMap<kBFloat16.id()> {
 public:
  using scalar_t = nv_bfloat16;
  using scalar_t2 = nv_bfloat162;
  using scalar_t4 = nv_bfloat162;
  using scalar_32bit_t = nv_bfloat162;

  using FragA = Vec<nv_bfloat162, 4>;
  using FragB = Vec<nv_bfloat162, 2>;
  using FragC = Vec<float, 4>;
  using FragS = Vec<nv_bfloat162, 1>;
  using FragZP = Vec<nv_bfloat162, 4>;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  static __device__ float inline num2float(const nv_bfloat16 x) {
    return __bfloat162float(x);
  }
  static __device__ nv_bfloat162 inline num2num2(const nv_bfloat16 x) {
    return __bfloat162bfloat162(x);
  }
  static __device__ nv_bfloat162 inline nums2num2(const nv_bfloat16 x1,
                                                  const nv_bfloat16 x2) {
    return __halves2bfloat162(x1, x2);
  }
  static __host__ __device__ nv_bfloat16 inline float2num(const float x) {
    return __float2bfloat16(x);
  }
  static __host__ __device__ float2 inline num22float2(const nv_bfloat162 x) {
    return __bfloat1622float2(x);
  }
#endif
};

}  // namespace pie_cuda_device::qgemm

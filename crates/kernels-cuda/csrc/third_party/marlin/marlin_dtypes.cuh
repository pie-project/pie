
#ifndef _data_types_cuh
#define _data_types_cuh
#include "marlin.cuh"
#include "scalar_type.hpp"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// PIE: the carried fp16/bf16 shims are a census of the names the ALREADY
// migrated tree spelled; Marlin is the first source to reach for the packed
// broadcasts and `__hsub2`. See `pie_marlin_intrinsics.h` for the four names
// and for the note that they belong in the shims, not here.
#ifdef __CUDACC_RTC__
  #include "pie_marlin_intrinsics.h"
#endif

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

namespace MARLIN_NAMESPACE_NAME {

template <long scalar_type_id>
class MarlinScalarType {};

template <>
class MarlinScalarType<vllm::kFloat16.id()> {
 public:
  using scalar_t = half;
  using scalar_t2 = half2;
  using scalar_t4 = half2;
  using scalar_32bit_t = half2;

  // Matrix fragments for tensor core instructions; their precise layout is
  // documented here:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
  using FragA = Vec<half2, 4>;
  using FragB = Vec<half2, 2>;
  using FragC = Vec<float, 4>;
  using FragS = Vec<half2, 1>;
  using FragS0 = Vec<__nv_fp8x2_e4m3, 1>;
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
class MarlinScalarType<vllm::kBFloat16.id()> {
 public:
  using scalar_t = nv_bfloat16;
  using scalar_t2 = nv_bfloat162;
  using scalar_t4 = nv_bfloat162;
  using scalar_32bit_t = nv_bfloat162;

  using FragA = Vec<nv_bfloat162, 4>;
  using FragB = Vec<nv_bfloat162, 2>;
  using FragC = Vec<float, 4>;
  using FragS = Vec<nv_bfloat162, 1>;
  using FragS0 = Vec<__nv_fp8x2_e4m3, 1>;
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

template <>
class MarlinScalarType<vllm::kFE4M3fn.id()> {
 public:
  using scalar_t = __nv_fp8_e4m3;
  using scalar_t2 = __nv_fp8x2_e4m3;
  using scalar_t4 = __nv_fp8x4_e4m3;
  using scalar_32bit_t = __nv_fp8x4_e4m3;

  using FragA = Vec<__nv_fp8x4_e4m3, 4>;
  using FragB = Vec<__nv_fp8x4_e4m3, 2>;
  using FragC = Vec<float, 4>;
  using FragZP = Vec<__nv_fp8x2_e4m3, 4>;

  static __host__ __device__
      float2 inline num22float2(const __nv_fp8x2_e4m3 x) {
#ifdef __CUDACC_RTC__
    // PIE: the carried `cuda_fp8.h` declares `__nv_fp8x2_e4m3` as STORAGE and
    // says so -- "no constructor is spelled anywhere, so none is written
    // here" -- which leaves the vendor's `explicit operator float2()` absent
    // and this cast without a conversion. Spelled out through the two names
    // the shim does carry, which is the vendor's own body: an fp8 byte widens
    // to an fp16 bit pattern exactly, and fp16 widens to fp32 exactly, so
    // both hops are lossless and the pair is bit-identical to `(float2)x`.
    // No shape in `kernels.def` outputs fp8; this specialisation is compiled
    // regardless because a full class specialisation's members are.
    float2 out;
    out.x = __half2float(static_cast<__half>(__nv_cvt_fp8_to_halfraw(
        static_cast<__nv_fp8_storage_t>(x.__x & 0xFFu), __NV_E4M3)));
    out.y = __half2float(static_cast<__half>(__nv_cvt_fp8_to_halfraw(
        static_cast<__nv_fp8_storage_t>(x.__x >> 8), __NV_E4M3)));
    return out;
#else
    return (float2)x;
#endif
  }
};

template <>
class MarlinScalarType<vllm::kS8.id()> {
 public:
  using scalar_t = int8_t;
  using scalar_t2 = int16_t;
  using scalar_t4 = int32_t;
  using scalar_32bit_t = int32_t;

  using FragA = Vec<int32_t, 4>;
  using FragB = Vec<int32_t, 2>;
  using FragC = Vec<float, 4>;
  using FragZP = Vec<int16_t, 4>;
};

template <typename scalar_t>
class MarlinScalarType2 {};

template <>
class MarlinScalarType2<half> : public MarlinScalarType<vllm::kFloat16.id()> {};

template <>
class MarlinScalarType2<nv_bfloat16>
    : public MarlinScalarType<vllm::kBFloat16.id()> {};

template <>
class MarlinScalarType2<__nv_fp8_e4m3>
    : public MarlinScalarType<vllm::kFE4M3fn.id()> {};

template <>
class MarlinScalarType2<int8_t> : public MarlinScalarType<vllm::kS8.id()> {};

}  // namespace MARLIN_NAMESPACE_NAME

#endif

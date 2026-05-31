#pragma once
//
// Fast in-register dequantization for the fused quant GEMM. (De-branded port
// of the int4 routines in upstream `dequant.h`.)
//
// Strategy: weight nibble -> (bitwise ops) -> raw fp16/bf16 bits -> (one
// floating-point op) -> dequantized value. The symmetric zero point (-8 for
// u4b8) is fused into the float step.
//
// THIS SLICE defines only the int4 (u4b8 / u4) -> half2 / bf162 specializations
// needed by the bf16 x u4b8 path. The primary `dequant` /
// `dequant_fp8_scales` / `sub_zp_and_dequant` templates are declared (but not
// defined) so the kernel's compile-time-discarded branches for other weight
// dtypes still type-check; adding fp8/fp4/int8 weights later means defining the
// corresponding specializations here.

#include "qgemm_dtypes.cuh"

namespace pie_cuda_device::qgemm {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750

// LUT-based 3-input logical op; explicitly emitted because the compiler does
// not always fold the dequant bit-ops into a LOP3.
template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(res)
               : "r"(a), "r"(b), "r"(c), "n"(lut));
  return res;
}

// Byte permute helper.
template <int start_byte, int mask>
__device__ inline uint32_t prmt(uint32_t a) {
  uint32_t res;
  asm volatile("prmt.b32 %0, %1, %2, %3;\n"
               : "=r"(res)
               : "r"(a), "n"(start_byte), "n"(mask));
  return res;
}

// Primary templates (declared; specializations follow / are added later).
template <typename scalar_t2, QTypeId w_type_id, bool skip_flop = false>
__device__ inline void dequant(int q, scalar_t2* frag_b);

template <typename scalar_t2, QTypeId s_type_id>
__device__ inline void dequant_fp8_scales(int q, scalar_t2* frag_b);

template <typename scalar_t2, QTypeId w_type_id, bool skip_flop = false>
__device__ inline void sub_zp_and_dequant(int q, scalar_t2* frag_b, int zp);

// ---- int4 -> fp16 ---------------------------------------------------------
// Efficient 4bit -> 4x fp16. Follows FasterTransformer's interleaved numeric
// conversion (BF16/FP16 variants).

template <>
__device__ inline void dequant<half2, kU4B8.id(), true>(int q, half2* frag_b) {
  const int MASK = 0x000f000f;
  const int EX = 0x64006400;
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  frag_b[0] = *reinterpret_cast<half2*>(&lo);
  frag_b[1] = *reinterpret_cast<half2*>(&hi);
}

template <>
__device__ inline void dequant<half2, kU4B8.id(), false>(int q, half2* frag_b) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // Signed int4 outputs: fuse the -8 symmetric zero point into SUB / ADD.
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&MUL),
                      *reinterpret_cast<const half2*>(&ADD));
}

template <>
__device__ inline void dequant<half2, kU4.id(), true>(int q, half2* frag_b) {
  dequant<half2, kU4B8.id(), true>(q, frag_b);
}

template <>
__device__ inline void dequant<half2, kU4.id(), false>(int q, half2* frag_b) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  const int SUB = 0x64006400;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd400d400;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo),
                      *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi),
                      *reinterpret_cast<const half2*>(&MUL),
                      *reinterpret_cast<const half2*>(&ADD));
}

// ---- int4 -> bf16 ---------------------------------------------------------

template <>
__device__ inline void dequant<nv_bfloat162, kU4B8.id(), true>(
    int q, nv_bfloat162* frag_b) {
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t EX = 0x43004300;
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  frag_b[0] = *reinterpret_cast<nv_bfloat162*>(&lo);
  frag_b[1] = *reinterpret_cast<nv_bfloat162*>(&hi);
}

template <>
__device__ inline void dequant<nv_bfloat162, kU4B8.id(), false>(
    int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, kU4B8.id(), true>(q, frag_b);
  static constexpr uint32_t SUB = 0x43084308;
  frag_b[0] = __hsub2(frag_b[0], *reinterpret_cast<const nv_bfloat162*>(&SUB));
  frag_b[1] = __hsub2(frag_b[1], *reinterpret_cast<const nv_bfloat162*>(&SUB));
}

template <>
__device__ inline void dequant<nv_bfloat162, kU4.id(), true>(
    int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, kU4B8.id(), true>(q, frag_b);
}

template <>
__device__ inline void dequant<nv_bfloat162, kU4.id(), false>(
    int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, kU4.id(), true>(q, frag_b);
  static constexpr uint32_t SUB = 0x43004300;
  frag_b[0] = __hsub2(frag_b[0], *reinterpret_cast<const nv_bfloat162*>(&SUB));
  frag_b[1] = __hsub2(frag_b[1], *reinterpret_cast<const nv_bfloat162*>(&SUB));
}

// ---- fp8 (fe4m3fn) -> bf16 ------------------------------------------------
// Efficiently dequantize 4x packed fe4m3fn bytes into 4x bf16. Follows the
// upstream FP8->BF16 numeric conversion (see third_party/marlin/dequant.h):
// the FP8 sign + 7-bit (exp4|mant3) field is splatted into the BF16
// [sign|exp8|mant7] layout, shifting the exponent right by (8-4)=4 bits but
// *without* re-biasing. The result is therefore the true fp8 value scaled by
// 2^-(127-7) == 2^-120; the matching 2^120 correction is folded into the
// per-group/per-channel scale on the host (see qgemm.cu scale_permute), which
// is exactly what `skip_flop == true` means for this format.
//
// `frag_b[1]`/`frag_b[0]` are written in reverse on purpose: the prepacked
// weight layout permutes the two bytes, so the kernel expects the high byte
// first.

template <>
__device__ inline void dequant<nv_bfloat162, kFE4M3fn.id(), true>(
    int q, nv_bfloat162* frag_b) {
  constexpr int FP8_EXPONENT = 4, BF16_EXPONENT = 8;
  constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP8_EXPONENT;
  constexpr int MASK = 0x7F007F00;

  int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 8;
  int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);

  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
}

// `skip_flop == false` variant: applies the 2^120 bias correction in-register
// (used when the scale does NOT carry the bias fold). Provided for
// completeness / type-checking of the kernel's discarded branches; the wired
// bf16-scale fp8 path uses the `true` variant above with a host-folded scale.
template <>
__device__ inline void dequant<nv_bfloat162, kFE4M3fn.id(), false>(
    int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, kFE4M3fn.id(), true>(q, frag_b);

  constexpr int FP8_EXPONENT = 4, BF16_EXPONENT = 8;
  constexpr int BIAS_OFFSET =
      (1 << (BF16_EXPONENT - 1)) - (1 << (FP8_EXPONENT - 1));
  constexpr uint32_t BIAS = (BIAS_OFFSET + 127) << 23;
  const nv_bfloat162 bias_reg =
      __float2bfloat162_rn(*reinterpret_cast<const float*>(&BIAS));

  frag_b[1] = __hmul2(frag_b[1], bias_reg);
  frag_b[0] = __hmul2(frag_b[0], bias_reg);
}

// ---- mxfp4 (fe2m1f) -> bf16 -----------------------------------------------
// e2m1 sign + 3-bit (exp2|mant1) field splatted into BF16, exponent shifted
// right by (8-2)=6 without re-bias; the 2^(127-1)=2^126 correction is folded
// into the (e8m0) block scale. See qgemm.cu (mxfp4 path) / kernel matmul_a8.
// Provided so the kFE2M1f branch of the kernel type-checks; the mxfp4 host
// entry is NOT wired in this slice (see qgemm.h notes).
template <>
__device__ inline void dequant<nv_bfloat162, kFE2M1f.id(), true>(
    int q, nv_bfloat162* frag_b) {
  constexpr int FP4_EXPONENT = 2, BF16_EXPONENT = 8;
  constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP4_EXPONENT;
  constexpr int MASK = 0x70007000;

  int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 4;
  int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);

  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
}

template <>
__device__ inline void dequant<nv_bfloat162, kFE2M1f.id(), false>(
    int q, nv_bfloat162* frag_b) {
  dequant<nv_bfloat162, kFE2M1f.id(), true>(q, frag_b);

  constexpr int FP4_EXPONENT = 2, BF16_EXPONENT = 8;
  constexpr int BIAS_OFFSET =
      (1 << (BF16_EXPONENT - 1)) - (1 << (FP4_EXPONENT - 1));
  constexpr uint32_t BIAS = (BIAS_OFFSET + 127) << 23;
  const nv_bfloat162 bias_reg =
      __float2bfloat162_rn(*reinterpret_cast<const float*>(&BIAS));

  frag_b[1] = __hmul2(frag_b[1], bias_reg);
  frag_b[0] = __hmul2(frag_b[0], bias_reg);
}

// ---- fp8 (fe4m3fn) scale dequant ------------------------------------------
// Used only by the e8m0/fp8-scale microscaling branches in the kernel; defined
// here so `dequant_fp8_scales<...>` resolves when the kernel body references
// it under `if constexpr (s_type == kFE4M3fn || s_type == kFE8M0fnu)`. The
// wired fp8-weight path uses bf16 scales and never hits this.
template <>
__device__ inline void dequant_fp8_scales<nv_bfloat162, kFE4M3fn.id()>(
    int q, nv_bfloat162* frag_b) {
  constexpr int FP8_EXPONENT = 4, BF16_EXPONENT = 8;
  constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP8_EXPONENT;
  constexpr int MASK = 0x7F007F00;

  int Out1 = ((q & 0x80008000) >> 1) | ((q & MASK) >> RIGHT_SHIFT);
  q <<= 8;
  int Out2 = ((q & 0x80008000) >> 1) | ((q & MASK) >> RIGHT_SHIFT);

  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
}

template <>
__device__ inline void dequant_fp8_scales<nv_bfloat162, kFE8M0fnu.id()>(
    int q, nv_bfloat162* frag_b) {
  int Out1 = (q & 0xFF00FF00) >> 1;
  q <<= 7;
  int Out2 = q & 0x7F807F80;

  frag_b[1] = *reinterpret_cast<const nv_bfloat162*>(&Out1);
  frag_b[0] = *reinterpret_cast<const nv_bfloat162*>(&Out2);
}

#endif  // __CUDA_ARCH__ >= 750

}  // namespace pie_cuda_device::qgemm

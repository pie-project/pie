#pragma once
//
// Tensor-core MMA wrappers for the fused quant GEMM. (De-branded port of
// `marlin_mma.h`.) Only the m16n8k16 fp16/bf16 instructions are kept, which
// is all the bf16 x u4b8 -> bf16 path needs. fp8/int8 MMA variants can be
// re-added alongside their dtype-map entries when those paths land.

#include "qgemm_dtypes.cuh"
#include <type_traits>

namespace pie_cuda_device::qgemm {

// m16n8k16 tensor core mma with fp16/bf16 inputs and fp32 accumulation.
template <QTypeId type_id, bool use_fp16_accum, int k_size = 16>
__device__ inline void mma(const typename QTypeMap<type_id>::FragA& a_frag,
                           const typename QTypeMap<type_id>::FragB& frag_b,
                           typename QTypeMap<type_id>::FragC& frag_c,
                           int idx = 0) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  using scalar_t = typename QTypeMap<type_id>::scalar_t;
  // This slice only emits the m16n8k16 fp16/bf16 instruction. The k_size==32
  // (fp8/int8 activation) path is never reached for the instantiated bf16
  // configs, but its lambda body is still type-checked, so compile it to an
  // empty body rather than failing a static_assert.
  if constexpr (k_size != 16) {
    (void)a;
    (void)b;
    (void)idx;
    return;
  } else if constexpr (std::is_same<scalar_t, half>::value && !use_fp16_accum) {
    float* c = reinterpret_cast<float*>(&frag_c);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  } else if constexpr (std::is_same<scalar_t, half>::value && use_fp16_accum) {
    uint32_t* c = reinterpret_cast<uint32_t*>(&frag_c);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
        : "=r"(c[0]), "=r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]));
  } else if constexpr (std::is_same<scalar_t, nv_bfloat16>::value) {
    float* c = reinterpret_cast<float*>(&frag_c);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  }
}

// Transposed variant used for the m_block_size_8 path (single 8-row tile).
template <QTypeId type_id, bool use_fp16_accum, int k_size = 16>
__device__ inline void mma_trans(
    const typename QTypeMap<type_id>::FragA& a_frag,
    const typename QTypeMap<type_id>::FragB& frag_b,
    const typename QTypeMap<type_id>::FragB& frag_b2,
    typename QTypeMap<type_id>::FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  const uint32_t* b2 = reinterpret_cast<const uint32_t*>(&frag_b2);
  using scalar_t = typename QTypeMap<type_id>::scalar_t;
  if constexpr (k_size != 16) {
    (void)a;
    (void)b;
    (void)b2;
    return;
  } else if constexpr (std::is_same<scalar_t, half>::value && !use_fp16_accum) {
    float* c = reinterpret_cast<float*>(&frag_c);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(b[0]), "r"(b2[0]), "r"(b[1]), "r"(b2[1]), "r"(a[0]), "r"(a[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  } else if constexpr (std::is_same<scalar_t, half>::value && use_fp16_accum) {
    uint32_t* c = reinterpret_cast<uint32_t*>(&frag_c);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
        : "=r"(c[0]), "=r"(c[1])
        : "r"(b[0]), "r"(b2[0]), "r"(b[1]), "r"(b2[1]), "r"(a[0]), "r"(a[1]),
          "r"(c[0]), "r"(c[1]));
  } else if constexpr (std::is_same<scalar_t, nv_bfloat16>::value) {
    float* c = reinterpret_cast<float*>(&frag_c);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(b[0]), "r"(b2[0]), "r"(b[1]), "r"(b2[1]), "r"(a[0]), "r"(a[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  }
}

}  // namespace pie_cuda_device::qgemm

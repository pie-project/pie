#pragma once
//
// Kernel parameter-list macro + forward declaration for the fused quant GEMM
// kernel template. (De-branded equivalent of upstream `kernel.h`.) Kept as a
// macro so the long parameter pack is written once and reused by the template
// definition, the forward declaration, and explicit instantiations.

#include "qgemm_common.cuh"
#include "qgemm_dtypes.cuh"
#include "qtype.h"

#define QGEMM_KERNEL_PARAMS                                                    \
  const int4 *__restrict__ A, const int4 *__restrict__ B,                      \
      int4 *__restrict__ C, int4 *__restrict__ C_tmp,                          \
      const int4 *__restrict__ b_bias_ptr,                                     \
      const float *__restrict__ a_scales_ptr,                                  \
      const int4 *__restrict__ scales_ptr,                                     \
      const float *__restrict__ global_scale_ptr,                              \
      const int4 *__restrict__ zp_ptr, const int *__restrict__ g_idx,          \
      int num_groups, int prob_m, int prob_n, int prob_k, int lda, int *locks, \
      bool has_bias, bool use_atomic_add, bool use_fp32_reduce,                \
      int max_shared_mem

namespace pie_cuda_device::qgemm {

template <const QTypeId a_type_id,       // A scalar id
          const QTypeId b_type_id,       // B (weight) scalar id
          const QTypeId c_type_id,       // C (output) scalar id
          const QTypeId s_type_id,       // B-scale scalar id
          const int threads,             // threads per block
          const int thread_m_blocks,     // # of 16x16 blocks in M
          const int thread_n_blocks,     // # of 16x16 blocks in N
          const int thread_k_blocks,     // # of 16x16 blocks in K
          const bool m_block_size_8,     // m_block_size == 8 (only tm_blocks==1)
          const int stages,              // async global->shared pipeline depth
          const int group_blocks,        // 16x16 blocks per quant scale group
          const bool is_zp_float>        // float zero point?
__global__ void QGemm(QGEMM_KERNEL_PARAMS);

}  // namespace pie_cuda_device::qgemm

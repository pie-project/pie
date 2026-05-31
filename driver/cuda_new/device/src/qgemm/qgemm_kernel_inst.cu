// Explicit instantiations of the fused quant GEMM kernel for the only path
// this slice ships: bf16 activations x u4b8 weights -> bf16 output, fp32
// accumulation, no float zero point, 4 pipeline stages.
//
// The (threads, thread_m/n/k_blocks, m_block_size_8) tuples below are the
// compact set the host dispatch (qgemm.cu) selects among. They mirror the
// thread_config priority used by the launcher:
//   small batch (thread_m_blocks == 1): {128,128,256} {64,128,128} {128,64,128}
//   large batch (thread_m_blocks  > 1): {64,256,256}  {64,128,128} {128,64,128}
// expressed as (threads, thread_n_blocks, thread_k_blocks):
//   {128,128,256} -> (256, 8, 8)
//   {64,128,128}  -> (128, 8, 4)
//   {128,64,128}  -> (128, 4, 8)
//   {64,256,256}  -> (256,16, 4)
// instantiated for thread_m_blocks in {1,2,3,4} and group_blocks in
// {-1 (per-channel), 8 (group_size 128)}.
//
// Adding fp8/mxfp4/int8 later = add new (a_type,b_type,s_type) instantiations
// here plus the matching dequant specializations; no template changes needed.

#include "qgemm_kernel_params.h"
#include "qgemm_kernel.cuh"

namespace pie_cuda_device::qgemm {

#define QGEMM_INST(THREADS, TM, TN, TK, M8, GB)                              \
  template __global__ void                                                   \
  QGemm<kBFloat16.id(), kU4B8.id(), kBFloat16.id(), kBFloat16.id(), THREADS, \
        TM, TN, TK, M8, 4, GB, false>(QGEMM_KERNEL_PARAMS);

// Instantiate both group_blocks values via this expansion macro.
#define QGEMM_INST_ALL_GB(GB)             \
  /* thread_m_blocks == 1 */              \
  QGEMM_INST(256, 1, 8, 8, true, GB)      \
  QGEMM_INST(128, 1, 8, 4, true, GB)      \
  QGEMM_INST(128, 1, 4, 8, true, GB)      \
  QGEMM_INST(256, 1, 8, 8, false, GB)     \
  QGEMM_INST(128, 1, 8, 4, false, GB)     \
  QGEMM_INST(128, 1, 4, 8, false, GB)     \
  /* thread_m_blocks == 2 */              \
  QGEMM_INST(256, 2, 16, 4, false, GB)    \
  QGEMM_INST(128, 2, 8, 4, false, GB)     \
  QGEMM_INST(128, 2, 4, 8, false, GB)     \
  /* thread_m_blocks == 3 */              \
  QGEMM_INST(256, 3, 16, 4, false, GB)    \
  QGEMM_INST(128, 3, 8, 4, false, GB)     \
  QGEMM_INST(128, 3, 4, 8, false, GB)     \
  /* thread_m_blocks == 4 */              \
  QGEMM_INST(256, 4, 16, 4, false, GB)    \
  QGEMM_INST(128, 4, 8, 4, false, GB)     \
  QGEMM_INST(128, 4, 4, 8, false, GB)

QGEMM_INST_ALL_GB(-1)  // per-channel
QGEMM_INST_ALL_GB(8)   // group_size == 128

#undef QGEMM_INST_ALL_GB
#undef QGEMM_INST

// fp8 (fe4m3fn) weight path: bf16 activations x fe4m3fn weights -> bf16 output,
// bf16 group scales, fp32 accumulation. Same (threads, m/n/k_blocks,
// m_block_size_8, group_blocks) tuples as the u4b8 path above (the upstream
// thread_config tables are identical for 4- and 8-bit weights); only the B
// scalar type changes from kU4B8 to kFE4M3fn.
#define QGEMM_INST_FP8(THREADS, TM, TN, TK, M8, GB)                            \
  template __global__ void                                                     \
  QGemm<kBFloat16.id(), kFE4M3fn.id(), kBFloat16.id(), kBFloat16.id(),         \
        THREADS, TM, TN, TK, M8, 4, GB, false>(QGEMM_KERNEL_PARAMS);

#define QGEMM_INST_FP8_ALL_GB(GB)             \
  QGEMM_INST_FP8(256, 1, 8, 8, true, GB)      \
  QGEMM_INST_FP8(128, 1, 8, 4, true, GB)      \
  QGEMM_INST_FP8(128, 1, 4, 8, true, GB)      \
  QGEMM_INST_FP8(256, 1, 8, 8, false, GB)     \
  QGEMM_INST_FP8(128, 1, 8, 4, false, GB)     \
  QGEMM_INST_FP8(128, 1, 4, 8, false, GB)     \
  QGEMM_INST_FP8(256, 2, 16, 4, false, GB)    \
  QGEMM_INST_FP8(128, 2, 8, 4, false, GB)     \
  QGEMM_INST_FP8(128, 2, 4, 8, false, GB)     \
  QGEMM_INST_FP8(256, 3, 16, 4, false, GB)    \
  QGEMM_INST_FP8(128, 3, 8, 4, false, GB)     \
  QGEMM_INST_FP8(128, 3, 4, 8, false, GB)     \
  QGEMM_INST_FP8(256, 4, 16, 4, false, GB)    \
  QGEMM_INST_FP8(128, 4, 8, 4, false, GB)     \
  QGEMM_INST_FP8(128, 4, 4, 8, false, GB)

QGEMM_INST_FP8_ALL_GB(-1)  // per-channel
QGEMM_INST_FP8_ALL_GB(8)   // group_size == 128

#undef QGEMM_INST_FP8_ALL_GB
#undef QGEMM_INST_FP8

}  // namespace pie_cuda_device::qgemm

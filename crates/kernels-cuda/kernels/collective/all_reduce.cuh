#include <cuda_bf16.h>
#include <cstdint>

#include "flashinfer/comm/vllm_custom_all_reduce.cuh"
#include "flashinfer/comm/trtllm_allreduce_fusion.cuh"

namespace pie::collective {

using DType = __nv_bfloat16;

}

#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace pie_cuda_driver::ops {

bool flashinfer_mamba_ssu_enabled();

bool flashinfer_mamba_ssu_bf16(
    const std::uint16_t* conv_out,
    const std::uint16_t* dt,
    const float* A,
    const std::uint16_t* D,
    const std::uint16_t* dt_bias,
    std::uint16_t* state_base,
    const std::int32_t* slot_ids,
    std::uint16_t* y,
    int batch,
    int num_heads,
    int head_dim,
    int state_size,
    int num_groups,
    int conv_dim,
    int intermediate,
    int state_cache_size,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::ops

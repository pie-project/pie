#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace pie_cuda_driver::ops {

// Subset of TensorRT-LLM's ActivationType that pie routes through the CUTLASS
// MoE runner; kept as a separate enum so callers don't need nv_internal/ on
// their include path. Every value used here must also appear as a
// PIE_MOE_ACTIVATION entry in kernels.def or the runner throws at run time.
//
// Note on Swiglu: the runner reads the gate half from the *second* half of the
// fc1 output and the linear half from the first, i.e. silu(w[I:]) * w[:I] --
// the opposite of pie's chunked_swiglu. fc1 weights must be stacked as
// [up; gate], not pie's usual [gate; up].
enum class MoeActivation {
    Relu2,
    Swiglu,
};

bool flashinfer_cutlass_moe_enabled();

std::size_t flashinfer_cutlass_moe_workspace_bytes(
    MoeActivation activation,
    int num_rows,
    int hidden_size,
    int inter_size,
    int num_experts,
    int experts_per_token,
    int tp_size,
    int tp_rank);

bool flashinfer_cutlass_moe_bf16(
    MoeActivation activation,
    const std::uint16_t* input,
    const std::int32_t* token_selected_experts,
    const float* token_final_scales,
    const std::uint16_t* fc1_expert_weights,
    const std::uint16_t* fc2_expert_weights,
    std::uint16_t* output,
    std::uint8_t* workspace,
    std::size_t workspace_bytes,
    std::int32_t* unpermuted_row_to_permuted_row,
    int num_rows,
    int hidden_size,
    int inter_size,
    int num_experts,
    int experts_per_token,
    int tp_size,
    int tp_rank,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::ops

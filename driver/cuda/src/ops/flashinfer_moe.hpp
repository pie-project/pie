#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace pie_cuda_driver::ops {

// Which gated/ungated epilogue the fused MoE runs. Each value costs one
// more CUTLASS grouped-GEMM instantiation, so the set is declared in
// kernels.def and kept to what a shipped arch actually reaches.
//
// Note on Swiglu: the runner reads the gate half from the *second* half of the
// fc1 output and the linear half from the first, i.e. silu(w[I:]) * w[:I] --
// the opposite of pie's chunked_swiglu. fc1 weights must be stacked as
// [up; gate], not pie's usual [gate; up]. Geglu has the same convention; only
// the scalar function on the gate half differs.
enum class MoeActivation {
    Relu2,    // nemotron_h
    Swiglu,   // qwen3.5 / qwen3.6 MoE, glm5 / kimi / deepseek_v4
    Geglu,    // gemma-4 26B-A4B routed experts (GELU-tanh gate)
};

bool flashinfer_cutlass_moe_enabled();

/// Row budget the runner's workspace is sized for. The workspace holds the
/// permuted activations, so it scales with `rows * experts_per_token *
/// hidden_size` -- at the full prefill token budget that is gigabytes of
/// VMM-backed arena, and mapping it makes every later `cuMemCreate` /
/// `cuMemSetAccess` on the shared physical pool an order of magnitude more
/// expensive. The fused path pays off at decode-sized batches, so cap the
/// budget there and let anything larger fall back. Override with
/// `PIE_MOE_FUSED_MAX_ROWS`.
int flashinfer_cutlass_moe_max_rows();

/// Row count below which the fused path is declined, so a model can keep its
/// small-batch GEMM for the shapes where the grouped GEMM's permute/finalize
/// overhead is not amortised. Off (`0`) unless `PIE_MOE_FUSED_MIN_ROWS` says
/// otherwise.
int flashinfer_cutlass_moe_min_rows();

/// Token count at or below which a model keeps its W4A16 per-route MoE GEMV
/// instead of the BF16 batched / fused grouped GEMM. The GEMV dequantises int4
/// with scalar FP32 ALU while the batched paths run on tensor cores, so the
/// crossover is far lower than its weight-traffic model suggests. Each model
/// passes its own compiled-in default; `PIE_MOE_GEMV_MAX_TOKENS` overrides it,
/// and `0` disables the GEMV path entirely.
int moe_gemv_max_tokens(int fallback);

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

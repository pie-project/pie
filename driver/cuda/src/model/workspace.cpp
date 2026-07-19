#include "model/workspace.hpp"

#include <algorithm>

namespace pie_cuda_driver::model {

Workspace Workspace::allocate_full(
    const HfConfig& cfg, int max_tokens,
    int max_intermediate, int max_Hq, int max_Hk,
    int max_output_rows,
    int max_mtp_draft_rows)
{
    const int H  = cfg.hidden_size;
    const int Hq = max_Hq;
    const int Hk = max_Hk;
    const int I  = max_intermediate;
    const int V  = cfg.vocab_size;
    const int N  = max_tokens;
    const int O  = max_output_rows > 0 ? max_output_rows : max_tokens;
    const int D  = std::max(0, max_mtp_draft_rows);

    Workspace ws;
    ws.y             = DeviceTensor::allocate(DType::BF16, {N, H});
    ws.norm_x        = DeviceTensor::allocate(DType::BF16, {N, H});
    ws.spec_hidden   = DeviceTensor::allocate(DType::BF16, {N, H});
    // Fused QKV / gate-up matmul outputs. Always allocated — costs ~12 MiB
    // at N=10240 for Qwen3 dims and lets the forward dispatch decide per
    // layer whether to use the fused or unfused projection.
    ws.qkv_fused     = DeviceTensor::allocate(DType::BF16, {N, Hq + 2 * Hk});
    ws.gate_up_fused = DeviceTensor::allocate(DType::BF16, {N, 2 * I});
    ws.mtp_concat    = DeviceTensor::allocate(DType::BF16, {N, 2 * H});
    ws.mtp_row0_save = DeviceTensor::allocate(DType::BF16, {1, V});
    ws.rope_table    = DeviceTensor::allocate(DType::FP32, {N, cfg.head_dim});
    ws.q             = DeviceTensor::allocate(DType::BF16, {N, Hq});
    ws.k             = DeviceTensor::allocate(DType::BF16, {N, Hk});
    ws.v             = DeviceTensor::allocate(DType::BF16, {N, Hk});
    ws.attn_out      = DeviceTensor::allocate(DType::BF16, {N, Hq});
    ws.norm_y        = DeviceTensor::allocate(DType::BF16, {N, H});
    ws.gate          = DeviceTensor::allocate(DType::BF16, {N, I});
    ws.up            = DeviceTensor::allocate(DType::BF16, {N, I});
    ws.logits        = DeviceTensor::allocate(
        DType::BF16, {workspace_logits_rows(N, D), V});
    ws.mtp_draft_row_base = workspace_mtp_draft_row_base(N);
    ws.mtp_draft_row_capacity = D;
    ws.probs         = DeviceTensor::allocate(DType::FP32, {O, V});
    // Padded q/k/v/attn_out only when head_dim != head_dim_kernel
    // (currently only Phi-3 at 96 → 128). Empty allocations otherwise
    // — the forward path detects the empty-state and aliases the
    // packed buffers.
    if (cfg.head_dim != cfg.head_dim_kernel) {
        const int q_heads = Hq / std::max(1, cfg.head_dim);
        const int kv_heads = Hk / std::max(1, cfg.head_dim);
        const int Hq_pad = q_heads * cfg.head_dim_kernel;
        const int Hk_pad = kv_heads * cfg.head_dim_kernel;
        ws.q_padded        = DeviceTensor::allocate(DType::BF16, {N, Hq_pad});
        ws.k_padded        = DeviceTensor::allocate(DType::BF16, {N, Hk_pad});
        ws.v_padded        = DeviceTensor::allocate(DType::BF16, {N, Hk_pad});
        ws.attn_out_padded = DeviceTensor::allocate(DType::BF16, {N, Hq_pad});
    }
    return ws;
}

Workspace Workspace::allocate_with_max_intermediate(
    const HfConfig& cfg, int max_tokens, int max_intermediate,
    int max_output_rows)
{
    const int Hq = cfg.num_attention_heads * cfg.head_dim;
    const int Hk = cfg.num_key_value_heads * cfg.head_dim;
    return allocate_full(
        cfg, max_tokens, max_intermediate, Hq, Hk, max_output_rows);
}

Workspace Workspace::allocate(const HfConfig& cfg, int max_tokens) {
    return allocate_with_max_intermediate(cfg, max_tokens, cfg.intermediate_size);
}

std::size_t workspace_bytes(const HfConfig& cfg,
                                  int N,
                                  int output_rows,
                                  int max_intermediate,
                                  int max_Hq,
                                  int max_Hk,
                                  int max_mtp_draft_rows) {
    const auto bf16 = [](std::size_t elems) { return elems * 2; };
    const auto fp32 = [](std::size_t elems) { return elems * 4; };
    const std::size_t n = static_cast<std::size_t>(N);
    const std::size_t o = static_cast<std::size_t>(std::max(1, output_rows));
    std::size_t bytes = 0;
    bytes += bf16(n * cfg.hidden_size);
    bytes += bf16(n * cfg.hidden_size);
    bytes += bf16(n * cfg.hidden_size);
    bytes += bf16(n * (max_Hq + 2 * max_Hk));
    bytes += bf16(n * (2 * max_intermediate));
    bytes += bf16(n * (2 * cfg.hidden_size));
    bytes += fp32(n * cfg.head_dim);
    bytes += bf16(n * max_Hq);
    bytes += bf16(n * max_Hk);
    bytes += bf16(n * max_Hk);
    bytes += bf16(n * max_Hq);
    bytes += bf16(n * cfg.hidden_size);
    bytes += bf16(n * max_intermediate);
    bytes += bf16(n * max_intermediate);
    bytes += bf16(
        (n + static_cast<std::size_t>(
                 std::max(0, max_mtp_draft_rows))) *
        cfg.vocab_size);
    bytes += fp32(o * cfg.vocab_size);
    if (cfg.head_dim != cfg.head_dim_kernel) {
        const int q_heads = max_Hq / std::max(1, cfg.head_dim);
        const int kv_heads = max_Hk / std::max(1, cfg.head_dim);
        const int Hq_pad = q_heads * cfg.head_dim_kernel;
        const int Hk_pad = kv_heads * cfg.head_dim_kernel;
        bytes += bf16(n * Hq_pad);
        bytes += bf16(n * Hk_pad);
        bytes += bf16(n * Hk_pad);
        bytes += bf16(n * Hq_pad);
    }
    return bytes;
}

}  // namespace pie_cuda_driver::model

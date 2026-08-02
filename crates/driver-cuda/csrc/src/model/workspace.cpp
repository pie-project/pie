#include <cstdio>
#include <cstdlib>
#include "model/workspace.hpp"

#include "kernels/argmax.hpp"

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
    // One [N, H] value's worth, which is what the converted island asks
    // for; every further island widens this and nothing else.
    ws.declared_values = DeviceTensor::allocate(DType::BF16, {N, H + I});
    ws.gate          = DeviceTensor::allocate(DType::BF16, {N, I});
    ws.up            = DeviceTensor::allocate(DType::BF16, {N, I});
    // NOTE (measured, do not "optimise" this without reading it):
    // this slab is sized by TOKENS, not by the sampled-row bound `O`, and that
    // is load-bearing. On Qwen3.6-27B it is N=8192 D=192 V=248320 -> 3970.9 MiB,
    // against `probs` at [O=64, V] fp32 = 60.6 MiB for the same sampling step,
    // and the workspace arena is charged IN FULL at every frame commit
    // (context.cpp passes {used=1, capacity=1}), so it costs ~3.85 GiB of lane
    // budget on every fire. Sizing it [O + D, V] was TRIED and REVERTED: output
    // stayed byte-identical at C=1 with a short prompt, but a 1024-token prompt
    // failed EVERY request in 0.2 s, at C=8 and C=64 alike -- prefill needs the
    // per-token rows. Recovering that memory needs the prefill path to honour
    // the compact row list, not a smaller allocation.
    const int logits_rows = workspace_logits_rows(N, D);
    ws.logits        = DeviceTensor::allocate(DType::BF16, {logits_rows, V});
    ws.mtp_draft_row_base = workspace_mtp_draft_row_base(N);
    ws.mtp_draft_row_capacity = D;
    ws.probs         = DeviceTensor::allocate(DType::FP32, {O, V});
    {
        static const bool dbg = [] {
            const char* v = std::getenv("PIE_WS_DEBUG");
            return v != nullptr && v[0] == '1';
        }();
        if (dbg) {
            const double mb = 1048576.0;
            std::fprintf(stderr,
                "[ws] N=%lld O=%lld D=%d V=%lld logits_rows=%d "
                "logits=%.1fMiB probs=%.1fMiB\n",
                (long long)N, (long long)O, D, (long long)V, logits_rows,
                logits_rows * (double)V * 2 / mb,
                (double)O * V * 4 / mb);
        }
    }
    ws.sampled_tokens = DeviceTensor::allocate(
        DType::INT32, {logits_rows, 1});
    ws.argmax_acc_val = DeviceTensor::allocate(
        DType::FP32,
        {logits_rows, kernels::kArgmaxAccumSlots});
    ws.argmax_acc_idx = DeviceTensor::allocate(
        DType::INT32,
        {logits_rows, kernels::kArgmaxAccumSlots});
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
    const auto i32  = [](std::size_t elems) { return elems * 4; };
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
    // `sampled_tokens` + `argmax_acc_val` + `argmax_acc_idx`. `allocate_full`
    // creates these for every family, not only the two that can fuse, so they
    // belong in the estimate unconditionally -- this function is the planner's
    // arena figure (`memory_planner.cpp`) and every byte missing from it is a
    // byte handed to the KV pool instead.
    {
        const std::size_t logits_rows = static_cast<std::size_t>(
            workspace_logits_rows(N, max_mtp_draft_rows));
        bytes += i32(logits_rows);
        bytes += fp32(logits_rows * kernels::kArgmaxAccumSlots);
        bytes += i32(logits_rows * kernels::kArgmaxAccumSlots);
    }
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

#include <cstdio>
#include <cstdlib>
#include "model/workspace.hpp"

#include "sample/argmax.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace pie_cuda_driver::model {

namespace {

// Every buffer `Workspace` owns, in allocation order.
//
// This list exists because the layout used to be written down TWICE — once in
// `allocate_full`, which creates the tensors, and once in `workspace_bytes`,
// which adds up what they cost so the memory planner can subtract the arena
// from the KV pool. Nothing compared them, and they had drifted:
// `declared_values` and `mtp_row0_save` were allocated and never budgeted, so
// the planner under-charged the arena by 17 MB on Qwen3-0.6B and 503 MB on
// Qwen3-32B, on every boot. The comment in `workspace_bytes` had stated the
// stakes correctly and been wrong about the fact.
//
// So neither function states the layout any more; both walk this. Adding a
// buffer here budgets it, and there is no second list to forget.
enum class Slot {
    kY, kNormX, kSpecHidden, kQkvFused, kGateUpFused, kMtpConcat,
    kMtpRow0Save, kRopeTable, kQ, kK, kV, kAttnOut, kNormY,
    kDeclaredValues, kGate, kUp, kLogits, kProbs, kSampledTokens,
    kArgmaxAccVal, kArgmaxAccIdx,
    kQPadded, kKPadded, kVPadded, kAttnOutPadded,
};

struct SlotSpec {
    Slot slot;
    DType dtype;
    std::int64_t rows;
    std::int64_t cols;
};

std::vector<SlotSpec> workspace_slots(
    const HfConfig& cfg, int max_tokens,
    int max_intermediate, int max_Hq, int max_Hk,
    int max_output_rows,
    int max_mtp_draft_rows)
{
    const std::int64_t H  = cfg.hidden_size;
    const std::int64_t Hq = max_Hq;
    const std::int64_t Hk = max_Hk;
    const std::int64_t I  = max_intermediate;
    const std::int64_t V  = cfg.vocab_size;
    const std::int64_t N  = max_tokens;
    // One definition of the sampled-row count, where there used to be two:
    // `allocate_full` fell back to `max_tokens` and `workspace_bytes` clamped
    // to 1, so at `output_rows == 0` the allocator made N rows and the budget
    // charged for one. The planner has only ever passed a positive count, so
    // the divergence was unreachable rather than harmless; the allocator's
    // reading is the one kept, because it is the one that reserves memory.
    const std::int64_t O  = max_output_rows > 0 ? max_output_rows : N;
    const std::int64_t logits_rows =
        workspace_logits_rows(max_tokens, max_mtp_draft_rows);
    const std::int64_t slots = kernels::sample::kArgmaxAccumSlots;

    std::vector<SlotSpec> out = {
        {Slot::kY,             DType::BF16,  N, H},
        {Slot::kNormX,         DType::BF16,  N, H},
        {Slot::kSpecHidden,    DType::BF16,  N, H},
        // Fused QKV / gate-up matmul outputs. Always allocated — costs ~12 MiB
        // at N=10240 for Qwen3 dims and lets the forward dispatch decide per
        // layer whether to use the fused or unfused projection.
        {Slot::kQkvFused,      DType::BF16,  N, Hq + 2 * Hk},
        {Slot::kGateUpFused,   DType::BF16,  N, 2 * I},
        {Slot::kMtpConcat,     DType::BF16,  N, 2 * H},
        {Slot::kMtpRow0Save,   DType::BF16,  1, V},
        {Slot::kRopeTable,     DType::FP32,  N, cfg.head_dim},
        {Slot::kQ,             DType::BF16,  N, Hq},
        {Slot::kK,             DType::BF16,  N, Hk},
        {Slot::kV,             DType::BF16,  N, Hk},
        {Slot::kAttnOut,       DType::BF16,  N, Hq},
        {Slot::kNormY,         DType::BF16,  N, H},
        // One [N, H] value's worth, which is what the converted island asks
        // for; every further island widens this and nothing else.
        {Slot::kDeclaredValues, DType::BF16, N, H + I},
        {Slot::kGate,          DType::BF16,  N, I},
        {Slot::kUp,            DType::BF16,  N, I},
        // NOTE (measured, do not "optimise" this without reading it):
        // this slab is sized by TOKENS, not by the sampled-row bound `O`, and
        // that is load-bearing. On Qwen3.6-27B it is N=8192 D=192 V=248320 ->
        // 3970.9 MiB, against `probs` at [O=64, V] fp32 = 60.6 MiB for the
        // same sampling step, and the workspace arena is charged IN FULL at
        // every frame commit (context.cpp passes {used=1, capacity=1}), so it
        // costs ~3.85 GiB of lane budget on every fire. Sizing it [O + D, V]
        // was TRIED and REVERTED: output stayed byte-identical at C=1 with a
        // short prompt, but a 1024-token prompt failed EVERY request in 0.2 s,
        // at C=8 and C=64 alike -- prefill needs the per-token rows.
        // Recovering that memory needs the prefill path to honour the compact
        // row list, not a smaller allocation.
        {Slot::kLogits,        DType::BF16,  logits_rows, V},
        {Slot::kProbs,         DType::FP32,  O, V},
        // `sampled_tokens` + `argmax_acc_val` + `argmax_acc_idx` are created
        // for every family, not only the two that can fuse.
        {Slot::kSampledTokens, DType::INT32, logits_rows, 1},
        {Slot::kArgmaxAccVal,  DType::FP32,  logits_rows, slots},
        {Slot::kArgmaxAccIdx,  DType::INT32, logits_rows, slots},
    };

    // Padded q/k/v/attn_out only when head_dim != head_dim_kernel (currently
    // only Phi-3 at 96 → 128). Absent otherwise — the forward path detects
    // the empty-state and aliases the packed buffers.
    if (cfg.head_dim != cfg.head_dim_kernel) {
        const std::int64_t q_heads = Hq / std::max(1, cfg.head_dim);
        const std::int64_t kv_heads = Hk / std::max(1, cfg.head_dim);
        const std::int64_t Hq_pad = q_heads * cfg.head_dim_kernel;
        const std::int64_t Hk_pad = kv_heads * cfg.head_dim_kernel;
        out.push_back({Slot::kQPadded,       DType::BF16, N, Hq_pad});
        out.push_back({Slot::kKPadded,       DType::BF16, N, Hk_pad});
        out.push_back({Slot::kVPadded,       DType::BF16, N, Hk_pad});
        out.push_back({Slot::kAttnOutPadded, DType::BF16, N, Hq_pad});
    }
    return out;
}

}  // namespace

Workspace Workspace::allocate_full(
    const HfConfig& cfg, int max_tokens,
    int max_intermediate, int max_Hq, int max_Hk,
    int max_output_rows,
    int max_mtp_draft_rows)
{
    Workspace ws;
    for (const SlotSpec& s : workspace_slots(
             cfg, max_tokens, max_intermediate, max_Hq, max_Hk,
             max_output_rows, max_mtp_draft_rows)) {
        DeviceTensor t = DeviceTensor::allocate(s.dtype, {s.rows, s.cols});
        switch (s.slot) {
            case Slot::kY:              ws.y = std::move(t); break;
            case Slot::kNormX:          ws.norm_x = std::move(t); break;
            case Slot::kSpecHidden:     ws.spec_hidden = std::move(t); break;
            case Slot::kQkvFused:       ws.qkv_fused = std::move(t); break;
            case Slot::kGateUpFused:    ws.gate_up_fused = std::move(t); break;
            case Slot::kMtpConcat:      ws.mtp_concat = std::move(t); break;
            case Slot::kMtpRow0Save:    ws.mtp_row0_save = std::move(t); break;
            case Slot::kRopeTable:      ws.rope_table = std::move(t); break;
            case Slot::kQ:              ws.q = std::move(t); break;
            case Slot::kK:              ws.k = std::move(t); break;
            case Slot::kV:              ws.v = std::move(t); break;
            case Slot::kAttnOut:        ws.attn_out = std::move(t); break;
            case Slot::kNormY:          ws.norm_y = std::move(t); break;
            case Slot::kDeclaredValues: ws.declared_values = std::move(t); break;
            case Slot::kGate:           ws.gate = std::move(t); break;
            case Slot::kUp:             ws.up = std::move(t); break;
            case Slot::kLogits:         ws.logits = std::move(t); break;
            case Slot::kProbs:          ws.probs = std::move(t); break;
            case Slot::kSampledTokens:  ws.sampled_tokens = std::move(t); break;
            case Slot::kArgmaxAccVal:   ws.argmax_acc_val = std::move(t); break;
            case Slot::kArgmaxAccIdx:   ws.argmax_acc_idx = std::move(t); break;
            case Slot::kQPadded:        ws.q_padded = std::move(t); break;
            case Slot::kKPadded:        ws.k_padded = std::move(t); break;
            case Slot::kVPadded:        ws.v_padded = std::move(t); break;
            case Slot::kAttnOutPadded:  ws.attn_out_padded = std::move(t); break;
        }
    }
    ws.mtp_draft_row_base = workspace_mtp_draft_row_base(max_tokens);
    ws.mtp_draft_row_capacity = std::max(0, max_mtp_draft_rows);

    {
        static const bool dbg = [] {
            const char* v = std::getenv("PIE_WS_DEBUG");
            return v != nullptr && v[0] == '1';
        }();
        if (dbg) {
            const double mb = 1048576.0;
            const int logits_rows =
                workspace_logits_rows(max_tokens, max_mtp_draft_rows);
            const long long O =
                max_output_rows > 0 ? max_output_rows : max_tokens;
            std::fprintf(stderr,
                "[ws] N=%lld O=%lld D=%d V=%lld logits_rows=%d "
                "logits=%.1fMiB probs=%.1fMiB\n",
                (long long)max_tokens, O, std::max(0, max_mtp_draft_rows),
                (long long)cfg.vocab_size, logits_rows,
                logits_rows * (double)cfg.vocab_size * 2 / mb,
                (double)O * cfg.vocab_size * 4 / mb);
        }
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
    // The planner's arena figure. It is the sum of the SAME list
    // `allocate_full` walks, so a buffer cannot be allocated and not
    // budgeted; note the parameter order differs from `allocate_full`'s,
    // which is why this forwards by name rather than by position.
    std::size_t bytes = 0;
    for (const SlotSpec& s : workspace_slots(
             cfg, N, max_intermediate, max_Hq, max_Hk, output_rows,
             max_mtp_draft_rows)) {
        bytes += static_cast<std::size_t>(std::max<std::int64_t>(0, s.rows)) *
                 static_cast<std::size_t>(std::max<std::int64_t>(0, s.cols)) *
                 dtype_bytes(s.dtype);
    }
    return bytes;
}

}  // namespace pie_cuda_driver::model

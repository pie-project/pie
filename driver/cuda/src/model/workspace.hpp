#pragma once

// De-facto universal per-fire forward workspace (`Workspace`) and its
// byte-budget helper (`workspace_bytes`), shared by every llama-like
// forward path. Originally scoped to Qwen3; the standalone prefill/paged
// Qwen3 forward functions that once lived here were superseded by the
// wire-driven `llama_like` forward and removed.

#include <cstdint>
#include <vector>

#include "device_buffer.hpp"
#include "model/loaded_model.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver::model {

constexpr int workspace_mtp_draft_row_base(int max_tokens) {
    return max_tokens;
}

constexpr int workspace_logits_rows(
    int max_tokens, int max_mtp_draft_rows) {
    return max_tokens +
        (max_mtp_draft_rows > 0 ? max_mtp_draft_rows : 0);
}

// Reusable scratch buffers, sized once for `max_tokens`. The forward pass
// only writes prefixes of these, so reusing across calls is safe as long as
// you don't exceed `max_tokens`.
// The lora staging arena (lora-graph campaign step 1, north-star-dsl.md):
// per-fire BUMP allocations for the adapter cast buffers and the grouped
// pointer slab, replacing per-fire cudaMallocAsync — a captured lora fire
// must allocate nothing at body time. Stream-safety of the per-fire
// reset/reuse: every write into the arena is stream-ordered (cast
// kernels, async uploads), so reuse is ordered behind the previous
// fire's reads. Growth keeps the old block alive in `retired` — an
// in-flight fire may still read it — and frees at destruction.
struct LoraStageArena {
    DeviceBuffer<std::uint8_t> buf;
    std::size_t used = 0;
    std::vector<DeviceBuffer<std::uint8_t>> retired;

    void reset() { used = 0; }

    void* alloc(std::size_t bytes) {
        constexpr std::size_t kAlign = 256;
        const std::size_t at = (used + kAlign - 1) / kAlign * kAlign;
        if (at + bytes > buf.size()) {
            std::size_t want = (at + bytes) * 2;
            if (want < 1 << 20) want = 1 << 20;
            if (buf.size() > 0) retired.push_back(std::move(buf));
            buf = DeviceBuffer<std::uint8_t>::alloc(want);
        }
        used = at + bytes;
        return buf.data() + at;
    }
};

struct Workspace {
    // Stage-2 MTP: extra rows reserved at the TAIL of `logits` (beyond the
    // `max_tokens` target rows) to hold the K native MTP draft-logit rows
    // an `Intrinsic::MtpLogits` [K,vocab] binding reads. `mtp_draft_row_base` is
    // the first reserved row; drafts live at [base, base+K) and never collide
    // with the target rows [0, max_tokens). A program may request at most
    // 32 drafts, while the aggregate batch reserve is one row per possible
    // output row so several MTP programs can coexist.
    static constexpr int kMtpDraftRowsPerProgram = 32;
    int mtp_draft_row_base = 0;
    int mtp_draft_row_capacity = 0;

    DeviceTensor y;          // [max_tokens, hidden]
    DeviceTensor norm_x;     // [max_tokens, hidden]
    DeviceTensor spec_hidden; // [max_tokens, hidden] saved verifier hidden rows
    DeviceTensor qkv_fused;  // [max_tokens, Hq + 2*Hk]   — only allocated when fused
                             // QKV path is in use; empty otherwise.
    DeviceTensor rope_table; // [max_tokens, head_dim] FP32; first half of
                             // each row is standard-RoPE cos, second is sin.
    DeviceTensor q;          // [max_tokens, h_q  * head_dim]   — packed
    DeviceTensor k;          // [max_tokens, h_kv * head_dim]   — packed
    DeviceTensor v;          // [max_tokens, h_kv * head_dim]   — packed
    DeviceTensor attn_out;   // [max_tokens, h_q  * head_dim]   — packed
    DeviceTensor norm_y;     // [max_tokens, hidden]
    DeviceTensor gate_up_fused; // [max_tokens, 2*I] — fused gate+up output, empty
                                // when unfused
    DeviceTensor mtp_concat;    // [max_tokens, 2*hidden] — Qwen3.6 MTP fc input
    DeviceTensor mtp_row0_save; // [1, vocab] preserves target row 0 while MTP drafts run
    DeviceTensor gate;       // [max_tokens, intermediate]
    DeviceTensor up;         // [max_tokens, intermediate]
    DeviceTensor logits;     // [max_tokens, vocab]
    DeviceTensor probs;      // [max_tokens, vocab] FP32 — softmax scratch for sampling
    // [max_tokens] i32. Written instead of `logits` when the forward folds a
    // greedy argmax into the LM head GEMM (§20.37), so the vocabulary is
    // reduced as it is produced and never materialised. The chunked path's slab
    // scratch is carved out of `logits`, which by construction is the buffer
    // that path is not filling.
    DeviceTensor sampled_tokens;
    // [max_tokens, kArgmaxAccumSlots] running (value, index) pair per row for
    // that same path. These do NOT share `logits`: at `chunk >= vocab` with
    // every row sampling, the slab alone consumes all of it, and a scratch
    // block that only sometimes fits would turn into a silent fallback the
    // epilogue cannot see. Sized off the same row count, so it always fits.
    //
    // All three are allocated unconditionally. They cost ~2 MiB against a
    // multi-GiB arena, and making them conditional would reintroduce exactly
    // the "is this buffer real for this fire?" question that the fused path is
    // built to never have to ask.
    DeviceTensor argmax_acc_val;
    DeviceTensor argmax_acc_idx;

    // Padded variants for the attention kernel when `head_dim_kernel >
    // head_dim` (Phi-3 ships head_dim=96; flashinfer's TC kernel only
    // works at {64, 128, 256, 512}, so we round up to 128). Empty
    // (numel()==0) for every other model — the forward graph aliases
    // the packed buffers directly.
    // Lora staging (per fire, bump-allocated; see LoraStageArena above).
    LoraStageArena lora_arena;

    DeviceTensor q_padded;        // [max_tokens, h_q  * head_dim_kernel]
    DeviceTensor k_padded;        // [max_tokens, h_kv * head_dim_kernel]
    DeviceTensor v_padded;        // [max_tokens, h_kv * head_dim_kernel]
    DeviceTensor attn_out_padded; // [max_tokens, h_q  * head_dim_kernel]

    static Workspace allocate(const HfConfig& cfg, int max_tokens);

    // Variant for architectures whose per-layer MLP `intermediate_size`
    // exceeds the base `cfg.intermediate_size` (Gemma-4's
    // `use_double_wide_mlp` doubles the width on shared layers).
    // Caller passes the worst-case value; ws.gate / ws.up / logits are
    // sized accordingly. Other shapes match the standard `allocate`.
    static Workspace allocate_with_max_intermediate(
        const HfConfig& cfg, int max_tokens, int max_intermediate,
        int max_output_rows = -1);

    // Variant for architectures whose per-layer attention dimensions
    // (Hq = num_q_heads * head_dim, Hk = num_kv_heads * head_dim) vary
    // across layers — Gemma-4's full-attention layers run at
    // head_dim_global=512 while sliding layers run at head_dim=256, so
    // a single ws.q sized at the sliding width overflows on full
    // layers. Caller passes the worst-case `Hq` and `Hk`.
    static Workspace allocate_full(
        const HfConfig& cfg, int max_tokens,
        int max_intermediate, int max_Hq, int max_Hk,
        int max_output_rows = -1,
        int max_mtp_draft_rows = 0);
};

// Byte budget for the per-fire Workspace tensors, parameterized by
// the HF config and the per-fire token/output shape. Used by the memory
// planner to size the persistent workspace arena.
std::size_t workspace_bytes(const HfConfig& cfg,
                                  int N,
                                  int output_rows,
                                  int max_intermediate,
                                  int max_Hq,
                                  int max_Hk,
                                  int max_mtp_draft_rows = 0);

}  // namespace pie_cuda_driver::model

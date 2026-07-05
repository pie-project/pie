// RawMetalDecoder — the reusable host wrapper around the MLX-free raw_metal decode
// pipeline. It packages the one-time lifecycle that decode_run.cpp::main() performs
// (open checkpoint → plan/build heap+DAG → stage weights/state/KV → bind → load PSOs →
// make resident) behind setup(), and the per-token inner loop (write IO scalars →
// ping-pong GDN conv-state → encode_decode_step → logits) behind step().
//
// This is the reusable direct-launch decode body: the entry path can hold ONE
// decoder and, per direct launch view (batch=1 single-stream), thread token_ids
// / position_ids through step() and read logits()/argmax() back out.
// State (GDN conv/recurrent + the contiguous KV ring) lives in the decoder's resident
// heap and accumulates IN-PLACE across step() calls AND across run_forward calls, so
// prefill→decode is seamless. reset_state() zeroes it for a fresh sequence (rs_slot NEW).
//
// Shipped config is fixed here (the 3.755ms qwen3.6 path): GdnPrep ON, GDN-input
// concurrency ON (compiled into encode_decode_step), force_barriers OFF, residual-fusion
// OFF. No env A/B knobs on the e2e path — decode_run keeps those for benching.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "decode_abi.hpp"
#include "decode_psos.hpp"
#include "decode_step.hpp"
#include "decode_timing.hpp"
#include "heap_bind_metal.hpp"
#include "heap_layout.hpp"
#include "mtl4_context.hpp"
#include "scratch_schedule.hpp"

namespace pie_metal_driver::raw_metal {

class RawMetalDecoder {
public:
    RawMetalDecoder() = default;
    ~RawMetalDecoder() = default;
    RawMetalDecoder(const RawMetalDecoder&)            = delete;
    RawMetalDecoder& operator=(const RawMetalDecoder&) = delete;

    // One-time lifecycle: open the checkpoint (zero-copy mmap, weights memcpy'd into the
    // resident heap then released), size+build the heap/DAG, stage+bind every weight/state/
    // KV/IO/scratch slot, bind const-params, compile the PSOs, make the heap resident.
    // `geom` defaults to Qwen3.5-0.8B (qwen3.6). Returns false + *err on failure.
    bool setup(const std::string& ckpt_dir, const std::string& kernels_dir,
               const DecodeGeometry& geom = DecodeGeometry{}, std::string* err = nullptr);

    bool ready() const { return ctx_ != nullptr; }

    // Zero the persistent sequence state (GDN conv/recurrent per layer + the KV ring) for a
    // fresh sequence — call when the runtime marks the rs_slot NEW (or position 0 prefill).
    void reset_state();

    // S>1: zero only `slot`'s GDN conv/recurrent slab region (no-op-equivalent to the GDN half
    // of reset_state() at slot=0). Call per NEW request (RsSlotFlags) under multi-batch; KV is
    // reset via the runtime's paged page-table, not here.
    void reset_state(uint32_t slot);

    // Process ONE M=1 token at absolute `position`: writes IO {TokenId, Position, SeqLen=
    // position+1}, advances the GDN conv-state ping-pong by (position % 2), then encodes the
    // full decode DAG (one run_step). Logits for this token land in the IO logits buffer;
    // read them via logits()/argmax(). State accumulates in-place. Returns the step timing.
    StepTiming step(uint32_t token_id, uint32_t position);

    // Borrowed pointer to the current IO logits produced by the last step(). The lm_head
    // (affine_qmv_*_bfloat16) writes BF16 (raw uint16_t bit patterns) — NOT f32, despite the
    // IoSlot::Logits doc tag. Use copy_logits_f32() to materialize f32 for sample_tokens.
    const uint16_t* logits_bf16() const;

    // Convert the current bf16 logits → f32 into `out` (must hold vocab() floats). For the
    // runtime's sample_tokens path; greedy callers can use argmax() directly.
    void copy_logits_f32(float* out) const;

    // Greedy/argmax over the current bf16 logits (the deterministic bench sampler). qwen3.6
    // golden pos-7 cross-check = 264.
    uint32_t argmax() const;

    const DecodeGeometry& geometry() const { return g_; }
    int vocab() const { return g_.vocab; }

private:
    DecodeGeometry                       g_{};
    HeapPlan                             plan_{};
    std::vector<Dispatch>                dag_{};
    ScratchSchedule                      sched_{};
    std::unique_ptr<RawMetalContext>     ctx_{};
    BoundDecode                          b_{};
    std::vector<SlotHandle>              pool_{};
    DecodeStepPsos                       psos_{};

    // GdnCore (+ GdnPrep when split) dispatches whose conv-state binds ping-pong per step.
    struct GdnDisp { int ord; int layer; Kernel kind; };
    std::vector<GdnDisp>                 gdn_disp_{};

    // Shipped 3.755ms config (fixed, no env on the e2e path).
    static constexpr bool gdn_prep_      = true;
    static constexpr bool fuse_residual_ = false;
    static constexpr bool force_barriers_= false;
    static constexpr int  max_ctx_       = 4096;

    int step_count_ = 0;  // monotonic per-sequence step index (resets with reset_state()).
};

}  // namespace pie_metal_driver::raw_metal

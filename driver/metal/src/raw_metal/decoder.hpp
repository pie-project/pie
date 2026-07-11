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
#include "batch_schedule.hpp"
#include "decode_psos.hpp"
#include "decode_step.hpp"
#include "decode_step_mb.hpp"
#include "decode_timing.hpp"
#include "heap_bind_metal.hpp"
#include "heap_layout.hpp"
#include "mtl4_context.hpp"
#include "scratch_schedule.hpp"

namespace pie_metal_driver::raw_metal {

// Phase 1b/3 paged-KV bridge: a SEPARATE, independently-resizable NHD
// (page-major: [num_pages, page_size, n_kv_heads, head_dim]) K/V pool, one
// pair of STANDALONE (non-heap-placed) buffers per full-attention layer —
// deliberately NOT part of the single resident decode heap (I2), so growing
// it (`RawMetalDecoder::resize_kv_pool`) never disturbs the core weights/
// state/scratch/IO layout at all. Distinct from the sealed M=1 HND
// contiguous ring (`BoundDecode::kv`), which this bridge does not
// reinterpret or touch.
struct KvPagePool {
    struct LayerPages { SlotHandle k_pages, v_pages; };
    std::vector<LayerPages> layers;  // size n_layers; only full-attn entries valid
    uint32_t total_pages = 0;
    uint32_t page_size = 0;
    bool enabled = false;
};

// Fully-owned host representation of one paged fire.  The executor validates
// this before calling run_batch_step; the decoder repeats capacity-critical
// checks because it is the final owner of the GPU writes.
struct BatchStepInputs {
    std::vector<uint32_t> token_ids;
    std::vector<uint32_t> position_ids;
    std::vector<uint32_t> qo_indptr;
    std::vector<uint32_t> kv_page_indptr;
    std::vector<uint32_t> kv_page_indices;
    std::vector<uint32_t> kv_last_page_lens;
    std::vector<uint32_t> rs_slot_ids;
    std::vector<uint8_t> rs_slot_flags;
    std::vector<uint32_t> w_page;
    std::vector<uint32_t> w_off;
};

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

    // Phase 1b/3: allocate the paged-KV pool (one standalone K/V buffer pair
    // per full-attention layer, `total_pages*page_size` NHD rows each) —
    // separate from the core resident heap (see KvPagePool doc). Idempotent
    // to call again with a DIFFERENT size (delegates to resize semantics
    // internally the first time there's nothing to preserve). Requires
    // `ready()`. Returns false + *err on failure (e.g. no full-attention
    // layers, zero-sized request, or allocation failure).
    bool setup_kv_pool(uint32_t total_pages, uint32_t page_size, std::string* err = nullptr);

    const KvPagePool& kv_pool() const { return kv_pool_; }
    uint64_t paged_bind_generation() const { return paged_bind_generation_; }

    // Phase 3 (review item 4): host-visible standalone-buffer allocation probe
    // (paged-KV pool buffers only) — for a lifecycle test proving grow/shrink
    // returns to a bounded baseline (no per-resize leak). 0 before a context
    // exists.
    size_t standalone_buffer_count() const {
        return ctx_ ? ctx_->standalone_buffer_count() : 0;
    }
    size_t standalone_bytes() const { return ctx_ ? ctx_->standalone_bytes() : 0; }

    // Whole-PAGE copy: for every full-attention layer, copy K and V rows
    // `src_pages[i] -> dst_pages[i]` (both must be < kv_pool().total_pages;
    // `src_pages.size() == dst_pages.size()`). A real memcpy over the
    // Shared-storage (unified-memory) standalone pool buffers — safe
    // without a blit encoder because `step()`/forward is synchronous today
    // (no concurrent GPU access at the time control ops run).
    bool copy_kv_pages(const std::vector<uint32_t>& src_pages,
                       const std::vector<uint32_t>& dst_pages, std::string* err = nullptr);

    // Per-TOKEN cell copy (mirrors PieKvMoveCell): for every full-attention
    // layer, copy the ONE [n_kv_heads, head_dim] row at
    // (src_page_id*page_size + src_token_offset) to
    // (dst_page_id*page_size + dst_token_offset), for each cell in `cells`.
    struct KvMoveCell {
        uint32_t dst_page_id, dst_token_offset, src_page_id, src_token_offset;
    };
    bool copy_kv_cells(const std::vector<KvMoveCell>& cells, std::string* err = nullptr);

    // Grow or logically shrink the pool to `new_total_pages`. Grow: allocate
    // NEW, bigger standalone buffers and copy every existing page [0,
    // total_pages) into the corresponding low range of the new buffers (page
    // ids are stable — old page N is still page N after a grow). Shrink:
    // truthfully rejected UNLESS the caller attests (`unmapped_tail_pages`)
    // that every page in [new_total_pages, total_pages) is unmapped/free —
    // this driver has no independent way to know which physical pages the
    // runtime still considers live, so it never silently truncates without
    // that attestation.  A successful resize rebuilds the paged argument-table
    // bindings before returning; paged_bind_generation() advances so tests can
    // prove no stale standalone GPU address survives into the next fire.
    bool resize_kv_pool(uint32_t new_total_pages, bool unmapped_tail_pages, std::string* err = nullptr);

    // Zero the persistent sequence state (GDN conv/recurrent per layer + the KV ring) for a
    // fresh sequence — call when the runtime marks the rs_slot NEW (or position 0 prefill).
    void reset_state();


    // S>1: zero only `slot`'s GDN conv/recurrent slab region (no-op-equivalent to the GDN half
    // of reset_state() at slot=0). Call per NEW request (RsSlotFlags) under multi-batch; KV is
    // reset via the runtime's paged page-table, not here. ALSO resets `slot`'s own GDN
    // ping-pong step parity (Phase 1b state-slot fix) so a fresh sequence on this slot starts
    // from parity 0 regardless of any other slot's step history.
    void reset_state(uint32_t slot);

    // Phase 1b: copy `src_slot`'s resident GDN conv+recurrent state (every
    // GDN layer, both conv-state ping-pong halves) to `dst_slot` — a real,
    // bounds-checked, whole-slot memcpy over the Shared-storage (unified
    // memory) heap region `plan_heap`/`build_bound_decode` already size for
    // `g_.max_slots` slots (independent of whether any decode DAG currently
    // reads a slot other than 0). False + `*err` if `src_slot`/`dst_slot`
    // is >= `g_.max_slots`, the decoder is not ready, or the checkpoint has
    // no GDN layers at all (nothing to copy). A same-slot copy is a no-op
    // success. ALSO copies `src_slot`'s ping-pong step-count PARITY to
    // `dst_slot` (state-slot fix) — both ConvState/ConvStateOut halves are
    // copied verbatim (never swapped), so `dst_slot` must know which half
    // currently holds the latest data via the SAME step-count parity
    // `src_slot` had, or a later `step(..., dst_slot)` could read the stale
    // half.
    bool copy_state_slot(uint32_t src_slot, uint32_t dst_slot, std::string* err = nullptr);

    // Total resident bytes ONE slot's state occupies, summed over every GDN
    // layer (both conv-state ping-pong buffers + the recurrent-state
    // buffer) — the real, current per-slot cost `copy_state_slot` moves,
    // for caps ("rs_cache_slot_bytes") to report truthfully. 0 before
    // `setup()` or for a checkpoint with no GDN layers.
    uint64_t rs_slot_bytes() const;

    // Process ONE M=1 token at absolute `position`, operating on GDN state `slot` (S>1,
    // Phase 1b state-slot fix): writes IO {TokenId, Position, SeqLen=position+1}, REBINDS
    // every GDN dispatch's ConvState/ConvStateOut/RecurrentState argument-table entries to
    // `slot`'s byte range within the shared per-layer slab (arg_bind_ordinal's byte-offset
    // form — the M=1 kernel itself is unchanged; it always operates at "offset 0 of whatever
    // it's bound to", so sliding the bound GPU address by `slot*stride` transparently
    // retargets it), advances slot's OWN GDN conv-state ping-pong parity (tracked
    // independently per slot, NOT the decoder-wide step count — a slot that has taken 5
    // steps then yields to a different slot for 3 steps must resume its own ping-pong
    // parity correctly when selected again), then encodes the full decode DAG (one
    // run_step). Logits for this token land in the IO logits buffer; read them via
    // logits()/argmax(). State accumulates in-place at `slot`. `slot` defaults to 0
    // (byte-identical to the pre-Phase-1b sealed single-slot behavior). Returns the step
    // timing. `slot` MUST be < geometry().max_slots (undefined slot selection otherwise —
    // callers validate this; see MetalExecutor::rs_slots()).
    StepTiming step(uint32_t token_id, uint32_t position, uint32_t slot = 0);

    // One command buffer for the entire paged fire.  This is intentionally a
    // separate path from step(): it uses the NHD standalone pool and never
    // reinterprets or mutates the sealed HND M=1 ring.
    bool run_batch_step(const BatchSchedule& schedule, const BatchStepInputs& in,
                        std::string* err = nullptr);

    // Borrowed pointer to the current IO logits produced by the last step(). The lm_head
    // (affine_qmv_*_bfloat16) writes BF16 (raw uint16_t bit patterns) — NOT f32, despite the
    // IoSlot::Logits doc tag. Use copy_logits_f32() to materialize f32 for sample_tokens.
    const uint16_t* logits_bf16() const;

    // Convert the current bf16 logits → f32 into `out` (must hold vocab() floats). For the
    // runtime's sample_tokens path; greedy callers can use argmax() directly.
    void copy_logits_f32(float* out) const;
    void copy_batch_logits_f32(uint32_t token_row, float* out) const;

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
    MultiBatchPsos                       mb_psos_{};
    KvPagePool                           kv_pool_{};
    std::vector<Dispatch>                mb_dag_{};
    ScratchSchedule                      mb_sched_{};
    std::vector<std::vector<Dispatch>>   prefill_dags_{};
    ScratchSchedule                      prefill_sched_{};
    bool                                 mb_bound_ = false;
    uint64_t                             paged_bind_generation_ = 0;

    // GdnCore (+ GdnPrep when split) dispatches whose conv-state binds ping-pong per step.
    struct GdnDisp { int ord; int layer; Kernel kind; };
    std::vector<GdnDisp>                 gdn_disp_{};

    // Shipped 3.755ms config (fixed, no env on the e2e path).
    static constexpr bool gdn_prep_      = true;
    static constexpr bool fuse_residual_ = false;
    static constexpr bool force_barriers_= false;
    static constexpr int  max_ctx_       = 4096;

    // Phase 1b state-slot fix: EACH slot's own monotonic ping-pong step index (was a single
    // decoder-wide `step_count_`, which silently corrupted ping-pong parity whenever a fire
    // targeted a slot other than whichever one had most recently stepped). Sized to
    // `max(1, g_.max_slots)` in setup(); index i tracks slot i. `step_count(slot)` clamps
    // defensively to slot 0 if `slot >= size()` (should never happen — callers validate
    // bounds against geometry().max_slots first).
    std::vector<int> step_count_by_slot_{0};
    int& step_count_for(uint32_t slot) {
        return step_count_by_slot_[slot < step_count_by_slot_.size() ? slot : 0];
    }
    bool bind_paged_dag(std::string* err);
    bool run_prefill_step(const BatchSchedule& schedule, std::string* err);
};

}  // namespace pie_metal_driver::raw_metal

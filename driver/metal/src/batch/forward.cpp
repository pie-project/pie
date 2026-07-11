#include "forward.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <type_traits>

#if defined(__APPLE__)
#include "batch_schedule.hpp"
#include "decode_consts.hpp"
#include "decode_psos.hpp"
#include "decode_step.hpp"
#include "decode_step_mb.hpp"
#include "decode_timing.hpp"
#include "heap_bind.hpp"
#include "heap_bind_metal.hpp"
#include "heap_layout.hpp"
#include "mtl4_context.hpp"
#include "safetensors_view.hpp"
#include "scratch.hpp"
#include "store/kv_pool.hpp"
#include "store/linear_state_slots.hpp"
#endif

namespace pie::metal::batch {

bool validate_linear_sequence_geometry(const LinearSequenceState& state,
                                       bool other_slot_ring_backed_different_sequence,
                                       const MemberForwardDesc& desc,
                                       std::string* reject_reason) {
    auto reject = [&](const std::string& why) {
        if (reject_reason != nullptr) *reject_reason = why;
        return false;
    };
    if (desc.token_ids.empty()) return reject("forward fire carries no tokens");
    if (desc.position_ids.size() != desc.token_ids.size()) {
        return reject("forward fire token/position count mismatch");
    }
    for (std::size_t i = 0; i + 1 < desc.position_ids.size(); ++i) {
        if (desc.position_ids[i + 1] != desc.position_ids[i] + 1) {
            return reject(
                "Metal Phase 1a requires in-order positions within a fire "
                "(non-monotone or gapped position run)");
        }
    }
    // Every page id in the fire's full list must be unique. Physical page
    // NUMBERING is reused across sequences by the runtime's free list and is
    // not required to be arithmetically adjacent (e.g. {5, 9} is a valid
    // two-page allocation) — a duplicate within ONE sequence's own list is
    // the actual fork/share/corruption signal.
    {
        std::vector<std::uint32_t> sorted_pages = desc.kv_pages;
        std::sort(sorted_pages.begin(), sorted_pages.end());
        if (std::adjacent_find(sorted_pages.begin(), sorted_pages.end()) != sorted_pages.end()) {
            return reject(
                "Metal Phase 1a supports only a single contiguous KV run per "
                "sequence (a duplicated physical page id indicates a fork, a "
                "shared prefix, or scattered/aliased pages, which are "
                "unsupported)");
        }
    }

    const bool is_fresh =
        desc.has_rs_slot ? desc.rs_reset : desc.position_ids.front() == 0;
    if (is_fresh) {
        if (other_slot_ring_backed_different_sequence) {
            return reject(
                "Metal Phase 1a supports exactly one resident linear KV "
                "sequence at a time; a different sequence is still resident "
                "(close it first — MetalExecutor::close_sequence — before a "
                "new sequence may start fresh)");
        }
        return true;
    }
    if (!state.has_resident) {
        return reject(
            "Metal Phase 1a: no resident sequence to continue (fire is not "
            "marked fresh but nothing has been reset yet)");
    }
    if (state.resident_sequence_id != desc.sequence_id) {
        return reject(
            "Metal Phase 1a supports exactly one resident linear KV sequence "
            "at a time; this member belongs to a different sequence than the "
            "one currently resident (interleaved concurrent sequences are "
            "unsupported)");
    }
    if (!state.ring_backed) {
        return reject(
            "Metal Phase 1b: this slot's recurrent state was copied "
            "(MetalExecutor::copy_state) but its KV history is not resident "
            "in the shared M=1 ring — continuing it requires the paged-KV "
            "CSR path; the sealed M=1 fast path can continue only the slot "
            "the ring is currently backing");
    }
    if (desc.position_ids.front() != state.resident_next_position) {
        return reject(
            "Metal Phase 1a: this member's positions do not extend the "
            "currently resident sequence (forks/shared-prefix/interleaved "
            "sequences are unsupported)");
    }
    // The resident page list must survive as a literal PREFIX of this fire's
    // full list — the prior pages must still be exactly where they were;
    // only the tail may grow with newly appended (unique) pages.
    if (desc.kv_pages.size() < state.resident_pages.size()) {
        return reject(
            "Metal Phase 1a: this member's KV page list is shorter than the "
            "currently resident sequence's page list (a truncation/rewrite "
            "of prior pages is unsupported)");
    }
    for (std::size_t i = 0; i < state.resident_pages.size(); ++i) {
        if (desc.kv_pages[i] != state.resident_pages[i]) {
            return reject(
                "Metal Phase 1a: this member's KV page list does not preserve "
                "the currently resident sequence's page-list prefix (a "
                "rewrite of already-committed pages is unsupported)");
        }
    }
    return true;
}

void close_linear_sequence(LinearSequenceState& state, std::uint64_t sequence_id) {
    if (state.has_resident && (state.ring_backed || state.paged_backed) &&
        state.resident_sequence_id == sequence_id) {
        state = LinearSequenceState{};
    }
}

BatchExecPlan plan_batch_execution(
    const std::unordered_map<std::uint32_t, LinearSequenceState>& slot_states,
    const std::vector<MemberForwardDesc>& descs) {
    BatchExecPlan plan;
    plan.member_ok.assign(descs.size(), 1);
    plan.member_reason.assign(descs.size(), std::string{});

    // The (at most one) sequence whose KV is currently resident in the shared
    // M=1 ring — the only sequence a CONTINUATION member can be served against.
    bool have_ring = false;
    std::uint64_t ring_seq = 0;
    for (const auto& [slot, state] : slot_states) {
        static_cast<void>(slot);
        if (state.ring_backed) {
            have_ring = true;
            ring_seq = state.resident_sequence_id;
            break;
        }
    }

    auto is_fresh = [](const MemberForwardDesc& d) {
        if (d.has_rs_slot) return d.rs_reset;
        return !d.position_ids.empty() && d.position_ids.front() == 0;
    };

    // Is the currently ring-backed sequence referenced by SOME member of this
    // batch (continued, or explicitly re-reset by its own instance)? If it is,
    // its residency is being deliberately taken over / continued in-batch, so a
    // sibling fresh member clobbering the ring afterwards is expected. If it is
    // NOT — a different, still-live sequence the engine has not closed — then a
    // fresh member silently clobbering it is the "steal the ring out from under
    // a resident sequence" hazard the sealed single-member path rejects; keep
    // that protection so single-member semantics are byte-identical.
    bool ring_handled_in_batch = false;
    if (have_ring) {
        for (const MemberForwardDesc& d : descs) {
            if (d.sequence_id == ring_seq) {
                ring_handled_in_batch = true;
                break;
            }
        }
    }

    // First pass: gate the members the single ring cannot serve, and elect the
    // leading continuation (the member that continues the currently ring-backed
    // sequence — it must run before any fresh member clobbers the ring).
    bool leading_taken = false;
    std::size_t leading_index = 0;
    for (std::size_t i = 0; i < descs.size(); ++i) {
        const MemberForwardDesc& d = descs[i];
        if (is_fresh(d)) {
            // Fresh members clobber the ring with their own reset+replay. Allow
            // that only when it does not silently discard a DIFFERENT resident
            // sequence the engine still considers live (sealed-path protection).
            if (have_ring && !ring_handled_in_batch && d.sequence_id != ring_seq) {
                plan.member_ok[i] = 0;
                plan.member_reason[i] =
                    "Metal Phase 1a supports exactly one resident linear KV sequence at a "
                    "time; a different sequence is still resident (close it first before a "
                    "new sequence may start fresh)";
            }
            continue;
        }
        // Continuation: serviceable only if it continues the CURRENTLY
        // ring-backed sequence, and only one such member per batch.
        if (have_ring && d.sequence_id == ring_seq && !leading_taken) {
            leading_taken = true;
            leading_index = i;
            continue;
        }
        plan.member_ok[i] = 0;
        plan.member_reason[i] =
            have_ring
                ? "Metal serves at most one continuation per batch against the single "
                  "shared M=1 KV ring; this member continues a sequence whose KV is not "
                  "the currently ring-backed one (this legacy M=1 planner cannot select "
                  "the paged path)"
                : "Metal has no resident KV to continue this sequence in the shared M=1 "
                  "ring (a fresh sequence must be reset/prefilled first; concurrent "
                  "multi-sequence decode needs the per-request paged-KV path)";
    }

    // Emit the execution order: leading continuation first, then every other
    // serviceable member in input order.
    if (leading_taken) plan.order.push_back(leading_index);
    for (std::size_t i = 0; i < descs.size(); ++i) {
        if (plan.member_ok[i] == 0) continue;
        if (leading_taken && i == leading_index) continue;
        plan.order.push_back(i);
    }
    return plan;
}

std::vector<std::uint32_t> global_readout_rows(
    std::uint32_t qo_begin, const std::vector<std::uint32_t>& local_indices) {
    std::vector<std::uint32_t> rows;
    rows.reserve(local_indices.size());
    for (uint32_t local : local_indices) rows.push_back(qo_begin + local);
    return rows;
}

// Platform-agnostic: mutates only `slot_states_`, no decoder dependency.
// Only the ring-backed entry (if any) matching `sequence_id` is released —
// per-slot entries holding copy_state'd metadata for the SAME sequence_id
// (but `ring_backed == false`) are untouched (close must not erase copied
// destination metadata).
void MetalExecutor::close_sequence(std::uint64_t sequence_id) {
    for (auto& [slot, state] : slot_states_) {
        static_cast<void>(slot);
        close_linear_sequence(state, sequence_id);
    }
}

#if defined(__APPLE__)

struct MetalExecutor::Impl {
    DecodeGeometry g_{};
    HeapPlan plan_{};
    std::vector<Dispatch> dag_{};
    ScratchSchedule sched_{};
    std::unique_ptr<RawMetalContext> ctx_{};
    BoundDecode b_{};
    std::vector<SlotHandle> pool_{};
    DecodeStepPsos psos_{};
    MultiBatchPsos mb_psos_{};
    KvPagePool kv_pool_{};
    std::vector<Dispatch> mb_dag_{};
    ScratchSchedule mb_sched_{};
    std::vector<std::vector<Dispatch>> prefill_dags_{};
    ScratchSchedule prefill_sched_{};
    bool mb_bound_ = false;
    std::uint64_t paged_bind_generation_ = 0;

    struct GdnDisp {
        int ord;
        int layer;
        Kernel kind;
    };
    std::vector<GdnDisp> gdn_disp_{};
    LinearStateSlots linear_state_slots_{};

    static constexpr bool gdn_prep_ = true;
    static constexpr bool fuse_residual_ = false;
    static constexpr bool force_barriers_ = false;
    static constexpr int max_ctx_ = 4096;

    bool setup(
        const std::string& checkpoint_dir,
        const std::string& kernels_dir,
        const DecodeGeometry& geometry,
        std::string* error);
    bool ready() const { return ctx_ != nullptr; }
    int vocab() const { return g_.vocab; }
    const DecodeGeometry& geometry() const { return g_; }
    bool setup_kv_pool(
        std::uint32_t total_pages,
        std::uint32_t page_size,
        std::string* error);
    const KvPagePool& kv_pool() const { return kv_pool_; }
    std::size_t standalone_buffer_count() const {
        return ctx_ ? ctx_->standalone_buffer_count() : 0;
    }
    std::size_t standalone_bytes() const {
        return ctx_ ? ctx_->standalone_bytes() : 0;
    }
    bool copy_kv_pages(
        const std::vector<std::uint32_t>& src_pages,
        const std::vector<std::uint32_t>& dst_pages,
        std::string* error);
    bool copy_kv_cells(
        const std::vector<KvMoveCell>& cells,
        std::string* error);
    bool resize_kv_pool(
        std::uint32_t total_pages,
        bool unmapped_tail_pages,
        std::string* error);
    void reset_state();
    void reset_state(std::uint32_t slot);
    bool copy_state_slot(
        std::uint32_t src_slot,
        std::uint32_t dst_slot,
        std::string* error);
    std::uint64_t rs_slot_bytes() const;
    StepTiming step(
        std::uint32_t token_id,
        std::uint32_t position,
        std::uint32_t slot = 0);
    bool run_batch_step(
        const BatchSchedule& schedule,
        const BatchStepInputs& inputs,
        std::string* error);
    const std::uint16_t* logits_bf16() const;
    void copy_logits_f32(float* output) const;
    void copy_batch_logits_f32(
        std::uint32_t token_row,
        float* output) const;
    std::uint32_t argmax() const;

  private:
    int& step_count_for(std::uint32_t slot) {
        return linear_state_slots_.at(slot);
    }
    bool bind_paged_dag(std::string* error);
    bool run_prefill_step(
        const BatchSchedule& schedule,
        std::string* error);
};

namespace {

void write_u32(const SlotHandle& s, uint32_t v) {
    std::memcpy(s.contents(), &v, sizeof(v));
}

inline float bf16_to_f32(uint16_t h) {
    uint32_t bits = uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

void zero_slot(const SlotHandle& s) {
    if (s.contents() && s.size) std::memset(s.contents(), 0, s.size);
}

// Zero one [off, off+len) byte window of a slot (a single slot's GDN-state slab region).
void zero_slot_region(const SlotHandle& s, size_t off, size_t len) {
    if (s.contents() && off + len <= s.size)
        std::memset(static_cast<char*>(s.contents()) + off, 0, len);
}

// Copy one [off, off+len) byte window from `src`'s contents to the SAME window in a
// DIFFERENT SlotHandle `dst` (used when growing the KV pool into a new, bigger standalone
// buffer — old and new pages share the SAME byte offset, just a different backing buffer).
bool copy_between_slots(const SlotHandle& dst, const SlotHandle& src, size_t off, size_t len) {
    if (!dst.contents() || !src.contents()) return false;
    if (off + len > dst.size || off + len > src.size) return false;
    std::memcpy(static_cast<char*>(dst.contents()) + off,
               static_cast<const char*>(src.contents()) + off, len);
    return true;
}

// Copy one [src_off, src_off+len) byte window from `s` to [dst_off, dst_off+len) of the
// SAME SlotHandle `s` (a single slot's GDN-state slab region — different slot OFFSETS
// within one shared per-layer buffer). Returns false if either window is out of range (a
// no-op, never a partial/garbage copy).
bool copy_slot_region(const SlotHandle& s, size_t src_off, size_t dst_off, size_t len) {
    if (!s.contents() || src_off + len > s.size || dst_off + len > s.size) return false;
    if (src_off == dst_off || len == 0) return true;  // no-op
    std::memcpy(static_cast<char*>(s.contents()) + dst_off,
               static_cast<const char*>(s.contents()) + src_off, len);
    return true;
}

}  // namespace

bool MetalExecutor::Impl::setup(const std::string& ckpt_dir, const std::string& kernels_dir,
                            const DecodeGeometry& geom, std::string* err) {
    g_ = geom;

    // ── Open the checkpoint (zero-copy mmap) + size the heap from the manifest. The view is
    //    transient: stage_decode_weights memcpy's every weight into the resident heap, so the
    //    mmap is released at the end of setup(). ──
    SafetensorsView view(ckpt_dir);
    size_t weights_bytes = 0;
    for (const auto& name : decode_weight_tensors(g_))
        weights_bytes += view.get(name).nbytes;
    plan_ = plan_heap(g_, weights_bytes, max_ctx_);

    // ── Build the decode DAG (shipped config: GdnPrep ON, no argmax dispatch — host samples). ──
    dag_ = build_decode_dag(g_, /*with_argmax=*/false, fuse_residual_, gdn_prep_);
    if (g_.paged_kv_enabled) {
        mb_dag_ = build_decode_dag_mb(g_, std::max(1, g_.max_tokens),
                                      kMultiBatchOrdinalBase, fuse_residual_, gdn_prep_);
        mb_sched_ = build_scratch_schedule(mb_dag_, g_, /*no_recycle=*/false);
        prefill_dags_ = build_decode_prefill_dags(g_, std::max(1, g_.max_tokens),
                                                   fuse_residual_, gdn_prep_);
        prefill_sched_ = build_scratch_schedule(prefill_dags_.front(), g_, /*no_recycle=*/false);
    }

    // ── beta's scratch schedule (WAR/WAW coloring). e2e path always recycles. ──
    sched_ = build_scratch_schedule(dag_, g_, /*no_recycle=*/false);

    size_t prefill_consts_budget = 0;
    for (const auto& dag : prefill_dags_) prefill_consts_budget += decode_consts_budget(dag);
    const size_t consts_budget = decode_consts_budget(dag_) +
                                 (mb_dag_.empty() ? 0 : decode_consts_budget(mb_dag_)) +
                                 prefill_consts_budget;
    const size_t heap_bytes = plan_.total + consts_budget
                            + size_t(std::max({sched_.colors_used, mb_sched_.colors_used,
                                               prefill_sched_.colors_used})) *
                                  plan_.scratch_slot_bytes + (32u << 20);

    ctx_ = RawMetalContext::create(heap_bytes);
    if (!ctx_) {
        if (err) *err = "RawMetalContext::create failed";
        return false;
    }

    // ── Stage weights/state/KV/IO; bind weight/state/KV/IO slots by ordinal. ──
    b_ = stage_decode_weights(*ctx_, view, g_, plan_);
    bind_decode_dag(*ctx_, b_, dag_, g_, gdn_prep_);

    // ── Scratch pool (colors_used slots) → beta's bind pass. ──
    pool_.resize(std::max({sched_.colors_used, mb_sched_.colors_used,
                           prefill_sched_.colors_used}));
    for (size_t i = 0; i < pool_.size(); ++i)
        pool_[i] = ctx_->heap_alloc(plan_.scratch_slot_bytes);
    bind_scratch(*ctx_, dag_, sched_, pool_.data(), int(pool_.size()));

    // ── Geometry const-params. ──
    bind_decode_consts(*ctx_, dag_, g_, max_ctx_, gdn_prep_);

    // ── Compile the kernel PSOs. ──
    std::string load_err;
    if (!load_decode_psos(*ctx_, kernels_dir, psos_, /*with_argmax=*/false, &load_err,
                          fuse_residual_, gdn_prep_)) {
        if (err) *err = "PSO load failed: " + load_err;
        ctx_.reset();
        return false;
    }
    if (g_.paged_kv_enabled &&
        !load_multibatch_psos(*ctx_, kernels_dir, mb_psos_, /*with_d512=*/false, &load_err)) {
        if (err) *err = "multi-batch PSO load failed: " + load_err;
        ctx_.reset();
        return false;
    }

    // ── Residency (I2): one set, after all binds. ──
    ctx_->make_resident();

    // ── Precompute the GDN dispatches whose conv-state binds ping-pong per step. ──
    gdn_disp_.clear();
    for (const auto& d : dag_)
        if (d.kind == Kernel::GdnCore || d.kind == Kernel::GdnPrep)
            gdn_disp_.push_back({d.ordinal, d.layer, d.kind});

    // Phase 1b: one independent ping-pong step counter per resident-state slot.
    linear_state_slots_.resize(size_t(g_.max_slots));
    return true;
}

void MetalExecutor::Impl::reset_state() {
    for (auto& gs : b_.gdn) {
        zero_slot(gs.conv_state);
        zero_slot(gs.conv_state_out);
        zero_slot(gs.recurrent_state);
    }
    for (auto& ks : b_.kv) {
        zero_slot(ks.k_pages);
        zero_slot(ks.v_pages);
    }
    linear_state_slots_.reset_all();
}

// Zero only `slot`'s GDN conv/recurrent region within each layer's slab (the per-slot stride
// laid out by build_bound_decode: conv = gdn_conv_dim*gdn_conv_k, recurrent =
// gdn_v_heads*gdn_v_dim*gdn_k_dim, f32). KV is paged per-request → reset via the runtime's
// page table (kv_last_page_lens=0 for a NEW request), not by zeroing the shared pool here.
// At max_slots=1, slot=0, off=0 → equivalent to the GDN half of reset_state(). ALSO resets
// this slot's own ping-pong step parity (Phase 1b state-slot fix) so a fresh sequence on
// `slot` always starts at parity 0, independent of any other slot's step history.
void MetalExecutor::Impl::reset_state(uint32_t slot) {
    const size_t conv_stride  = g_.gdn_conv_stride_bytes();
    const size_t recur_stride = g_.gdn_recurrent_stride_bytes();
    const size_t conv_off  = size_t(slot) * conv_stride;
    const size_t recur_off = size_t(slot) * recur_stride;
    for (auto& gs : b_.gdn) {
        zero_slot_region(gs.conv_state, conv_off, conv_stride);
        zero_slot_region(gs.conv_state_out, conv_off, conv_stride);
        zero_slot_region(gs.recurrent_state, recur_off, recur_stride);
    }
    linear_state_slots_.reset(slot);
}

// Phase 1b: real, bounds-checked whole-slot copy of every GDN layer's
// resident conv+recurrent state — the SAME per-slot stride formula
// reset_state(slot) already uses (build_bound_decode/plan_heap: conv =
// gdn_conv_dim*gdn_conv_k, recurrent = gdn_v_heads*gdn_v_dim*gdn_k_dim, f32,
// `g_.max_slots` slots packed per layer). Only GDN layers have a real
// (non-zero-sized) `b_.gdn[L]` slab — full-attn layer entries are default-
// constructed (size 0) and `copy_slot_region` safely no-ops on them, so this
// loop does not need an `is_full_attn` filter (mirrors reset_state(slot)'s
// own style). `src_slot`/`dst_slot` are bounds-checked against `g_.max_slots`
// up front — never a partial/silent-garbage copy.
bool MetalExecutor::Impl::copy_state_slot(uint32_t src_slot, uint32_t dst_slot, std::string* err) {
    if (!ready()) {
        if (err) *err = "MetalExecutor::Impl::copy_state_slot: decoder not initialized";
        return false;
    }
    if (src_slot >= uint32_t(g_.max_slots) || dst_slot >= uint32_t(g_.max_slots)) {
        if (err) {
            *err = "MetalExecutor::Impl::copy_state_slot: slot id out of range [0, " +
                   std::to_string(g_.max_slots) + ")";
        }
        return false;
    }
    const size_t conv_stride  = g_.gdn_conv_stride_bytes();
    const size_t recur_stride = g_.gdn_recurrent_stride_bytes();
    const size_t src_conv_off  = size_t(src_slot) * conv_stride;
    const size_t dst_conv_off  = size_t(dst_slot) * conv_stride;
    const size_t src_recur_off = size_t(src_slot) * recur_stride;
    const size_t dst_recur_off = size_t(dst_slot) * recur_stride;
    int gdn_layers_copied = 0;
    for (auto& gs : b_.gdn) {
        if (!gs.conv_state.valid()) continue;  // a full-attn layer's unused slot
        const bool ok = copy_slot_region(gs.conv_state, src_conv_off, dst_conv_off, conv_stride) &&
                        copy_slot_region(gs.conv_state_out, src_conv_off, dst_conv_off, conv_stride) &&
                        copy_slot_region(gs.recurrent_state, src_recur_off, dst_recur_off, recur_stride);
        if (!ok) {
            if (err) *err = "MetalExecutor::Impl::copy_state_slot: internal bounds check failed";
            return false;
        }
        ++gdn_layers_copied;
    }
    if (gdn_layers_copied == 0) {
        if (err) *err = "MetalExecutor::Impl::copy_state_slot: this checkpoint has no GDN layers (nothing to copy)";
        return false;
    }
    // The ping-pong PARITY (which of ConvState/ConvStateOut currently holds
    // the LATEST data) is a function of how many steps a slot has taken
    // (`step_count_by_slot_[slot] % 2` — see step()). Since both ping-pong
    // buffers were just copied VERBATIM (A stays A, C stays C, never
    // swapped), `dst_slot` must inherit `src_slot`'s exact step count too —
    // otherwise a later step() on `dst_slot` could read the STALE half
    // instead of the one that actually holds the copied-in latest data
    // (silently correct only when src/dst happened to share the same
    // parity by coincidence).
    linear_state_slots_.copy(src_slot, dst_slot);
    return true;
}

uint64_t MetalExecutor::Impl::rs_slot_bytes() const {
    if (!ready()) return 0;
    const size_t conv_stride  = g_.gdn_conv_stride_bytes();
    const size_t recur_stride = g_.gdn_recurrent_stride_bytes();
    uint64_t total = 0;
    for (const auto& gs : b_.gdn) {
        if (!gs.conv_state.valid()) continue;
        total += 2 * conv_stride + recur_stride;  // ConvState + ConvStateOut + RecurrentState
    }
    return total;
}

namespace {
// One NHD paged-pool row's byte size: [n_kv_heads, head_dim], bf16 (matches the M=1
// ring's activation dtype — kv_append.metal/kv_append_paged.metal both instantiate bf16).
size_t kv_pool_row_bytes(const DecodeGeometry& g) {
    return size_t(g.n_kv_heads) * size_t(g.head_dim) * 2u;
}
}  // namespace

bool MetalExecutor::Impl::setup_kv_pool(uint32_t total_pages, uint32_t page_size, std::string* err) {
    if (!ready()) {
        if (err) *err = "MetalExecutor::Impl::setup_kv_pool: decoder not initialized";
        return false;
    }
    if (total_pages == 0 || page_size == 0) {
        if (err) *err = "MetalExecutor::Impl::setup_kv_pool: total_pages and page_size must be > 0";
        return false;
    }
    int n_full = 0;
    for (int L = 0; L < g_.n_layers; ++L) {
        if (DecodeGeometry::is_full_attn(L)) ++n_full;
    }
    if (n_full == 0) {
        if (err) {
            *err = "MetalExecutor::Impl::setup_kv_pool: this checkpoint has no full-attention "
                   "layers (nothing to allocate a KV page pool for)";
        }
        return false;
    }
    const size_t layer_bytes =
        size_t(total_pages) * size_t(page_size) * kv_pool_row_bytes(g_);
    KvPagePool pool;
    pool.layers.resize(size_t(g_.n_layers));
    for (int L = 0; L < g_.n_layers; ++L) {
        if (!DecodeGeometry::is_full_attn(L)) continue;
        pool.layers[size_t(L)].k_pages = ctx_->create_standalone_buffer(layer_bytes);
        pool.layers[size_t(L)].v_pages = ctx_->create_standalone_buffer(layer_bytes);
        if (!pool.layers[size_t(L)].k_pages.valid() || !pool.layers[size_t(L)].v_pages.valid()) {
            if (err) {
                *err = "MetalExecutor::Impl::setup_kv_pool: standalone buffer allocation failed "
                       "(layer " + std::to_string(L) + ", " + std::to_string(layer_bytes) +
                       " bytes/buffer)";
            }
            return false;
        }
    }
    pool.total_pages = total_pages;
    pool.page_size = page_size;
    pool.enabled = true;
    // Phase 3 (review item 4): if a pool was already allocated (setup_kv_pool
    // re-called with a different size), release the OLD standalone buffers
    // before replacing it, so re-setup does not leak the previous allocation.
    KvPagePool old_pool = std::move(kv_pool_);
    kv_pool_ = std::move(pool);
    for (auto& lp : old_pool.layers) {
        if (lp.k_pages.valid()) ctx_->release_standalone_buffer(lp.k_pages);
        if (lp.v_pages.valid()) ctx_->release_standalone_buffer(lp.v_pages);
    }
    return bind_paged_dag(err);
}

bool MetalExecutor::Impl::bind_paged_dag(std::string* err) {
    if (!ready() || !g_.paged_kv_enabled || !kv_pool_.enabled || mb_dag_.empty()) {
        if (err) *err = "MetalExecutor::Impl::bind_paged_dag: paged decode is not initialized";
        return false;
    }
    try {
        std::vector<SlotHandle> k_pages(size_t(g_.n_layers));
        std::vector<SlotHandle> v_pages(size_t(g_.n_layers));
        for (int L = 0; L < g_.n_layers; ++L) {
            if (!DecodeGeometry::is_full_attn(L)) continue;
            k_pages[size_t(L)] = kv_pool_.layers[size_t(L)].k_pages;
            v_pages[size_t(L)] = kv_pool_.layers[size_t(L)].v_pages;
        }
        bind_decode_dag_mb(*ctx_, b_, mb_dag_, g_, k_pages, v_pages, gdn_prep_);
        const size_t prefill_scratch_row = size_t(scratch_widest_elems(g_)) * 2u;
        const size_t prefill_logits_row = size_t(g_.vocab) * 2u;
        for (size_t t = 0; t < prefill_dags_.size(); ++t) {
            const MbBindOffsets offsets{
                .token_row = t,
                .logits_bytes = t * prefill_logits_row,
            };
            bind_decode_dag_mb(*ctx_, b_, prefill_dags_[t], g_, k_pages, v_pages, gdn_prep_,
                               offsets);
        }
        if (!mb_bound_) {
            bind_scratch(*ctx_, mb_dag_, mb_sched_, pool_.data(), int(pool_.size()));
            bind_decode_consts(*ctx_, mb_dag_, g_, max_ctx_, gdn_prep_);
            for (size_t t = 0; t < prefill_dags_.size(); ++t) {
                bind_scratch(*ctx_, prefill_dags_[t], prefill_sched_, pool_.data(),
                             int(pool_.size()), t * prefill_scratch_row);
                bind_decode_consts(*ctx_, prefill_dags_[t], g_, max_ctx_, gdn_prep_);
            }
            mb_bound_ = true;
        }
        ++paged_bind_generation_;
        return true;
    } catch (const std::exception& e) {
        if (err) *err = std::string("MetalExecutor::Impl::bind_paged_dag: ") + e.what();
        return false;
    }
}

bool MetalExecutor::Impl::copy_kv_pages(const std::vector<uint32_t>& src_pages,
                                    const std::vector<uint32_t>& dst_pages, std::string* err) {
    if (!ready() || !kv_pool_.enabled) {
        if (err) *err = "MetalExecutor::Impl::copy_kv_pages: KV page pool not allocated";
        return false;
    }
    if (src_pages.size() != dst_pages.size()) {
        if (err) *err = "MetalExecutor::Impl::copy_kv_pages: src/dst page count mismatch";
        return false;
    }
    // Bounds-check EVERY page first — never a partial copy on a late failure.
    for (size_t i = 0; i < src_pages.size(); ++i) {
        if (src_pages[i] >= kv_pool_.total_pages || dst_pages[i] >= kv_pool_.total_pages) {
            if (err) {
                *err = "MetalExecutor::Impl::copy_kv_pages: page id out of range [0, " +
                       std::to_string(kv_pool_.total_pages) + ")";
            }
            return false;
        }
    }
    const size_t page_bytes = size_t(kv_pool_.page_size) * kv_pool_row_bytes(g_);
    // NOTE: copies within one call are applied in the given order; a chain like
    // {1->0, 2->1} reads page 1 for the second copy AFTER the first already
    // overwrote it. Matches the typical device-copy convention (each pair is
    // independent; the caller sequences non-conflicting moves, or issues them as
    // separate calls when a true swap/rotate is needed).
    for (int L = 0; L < g_.n_layers; ++L) {
        if (!DecodeGeometry::is_full_attn(L)) continue;
        const auto& lp = kv_pool_.layers[size_t(L)];
        for (size_t i = 0; i < src_pages.size(); ++i) {
            const size_t src_off = size_t(src_pages[i]) * page_bytes;
            const size_t dst_off = size_t(dst_pages[i]) * page_bytes;
            if (!copy_slot_region(lp.k_pages, src_off, dst_off, page_bytes) ||
                !copy_slot_region(lp.v_pages, src_off, dst_off, page_bytes)) {
                if (err) *err = "MetalExecutor::Impl::copy_kv_pages: internal bounds check failed";
                return false;
            }
        }
    }
    return true;
}

bool MetalExecutor::Impl::copy_kv_cells(const std::vector<KvMoveCell>& cells, std::string* err) {
    if (!ready() || !kv_pool_.enabled) {
        if (err) *err = "MetalExecutor::Impl::copy_kv_cells: KV page pool not allocated";
        return false;
    }
    for (const auto& c : cells) {
        if (c.src_page_id >= kv_pool_.total_pages || c.dst_page_id >= kv_pool_.total_pages ||
            c.src_token_offset >= kv_pool_.page_size || c.dst_token_offset >= kv_pool_.page_size) {
            if (err) {
                *err = "MetalExecutor::Impl::copy_kv_cells: cell page id/token offset out of "
                       "range (total_pages=" + std::to_string(kv_pool_.total_pages) +
                       ", page_size=" + std::to_string(kv_pool_.page_size) + ")";
            }
            return false;
        }
    }
    const size_t row_bytes = kv_pool_row_bytes(g_);
    const size_t page_bytes = size_t(kv_pool_.page_size) * row_bytes;
    for (int L = 0; L < g_.n_layers; ++L) {
        if (!DecodeGeometry::is_full_attn(L)) continue;
        const auto& lp = kv_pool_.layers[size_t(L)];
        for (const auto& c : cells) {
            const size_t src_off = size_t(c.src_page_id) * page_bytes +
                                   size_t(c.src_token_offset) * row_bytes;
            const size_t dst_off = size_t(c.dst_page_id) * page_bytes +
                                   size_t(c.dst_token_offset) * row_bytes;
            if (!copy_slot_region(lp.k_pages, src_off, dst_off, row_bytes) ||
                !copy_slot_region(lp.v_pages, src_off, dst_off, row_bytes)) {
                if (err) *err = "MetalExecutor::Impl::copy_kv_cells: internal bounds check failed";
                return false;
            }
        }
    }
    return true;
}

bool MetalExecutor::Impl::resize_kv_pool(uint32_t new_total_pages, bool unmapped_tail_pages,
                                     std::string* err) {
    if (!ready()) {
        if (err) *err = "MetalExecutor::Impl::resize_kv_pool: decoder not initialized";
        return false;
    }
    if (!kv_pool_.enabled) {
        if (err) {
            *err = "MetalExecutor::Impl::resize_kv_pool: KV page pool not allocated "
                   "(call setup_kv_pool first)";
        }
        return false;
    }
    if (new_total_pages == kv_pool_.total_pages) return true;  // no-op success
    if (new_total_pages == 0) {
        if (err) *err = "MetalExecutor::Impl::resize_kv_pool: resize to 0 pages is not supported";
        return false;
    }
    if (new_total_pages < kv_pool_.total_pages && !unmapped_tail_pages) {
        if (err) {
            *err = "MetalExecutor::Impl::resize_kv_pool: shrink would truncate pages [" +
                   std::to_string(new_total_pages) + ", " + std::to_string(kv_pool_.total_pages) +
                   ") that the caller has not attested are unmapped/free — refusing to "
                   "silently discard potentially-live pages";
        }
        return false;
    }
    const size_t row_bytes = kv_pool_row_bytes(g_);
    const size_t new_layer_bytes =
        size_t(new_total_pages) * size_t(kv_pool_.page_size) * row_bytes;
    const size_t copy_pages = std::min<uint32_t>(new_total_pages, kv_pool_.total_pages);
    const size_t copy_bytes = size_t(copy_pages) * size_t(kv_pool_.page_size) * row_bytes;
    KvPagePool new_pool;
    new_pool.layers.resize(size_t(g_.n_layers));
    for (int L = 0; L < g_.n_layers; ++L) {
        if (!DecodeGeometry::is_full_attn(L)) continue;
        SlotHandle new_k = ctx_->create_standalone_buffer(new_layer_bytes);
        SlotHandle new_v = ctx_->create_standalone_buffer(new_layer_bytes);
        if (!new_k.valid() || !new_v.valid()) {
            if (err) *err = "MetalExecutor::Impl::resize_kv_pool: new buffer allocation failed";
            return false;
        }
        if (copy_bytes > 0) {
            const auto& old_lp = kv_pool_.layers[size_t(L)];
            if (!copy_between_slots(new_k, old_lp.k_pages, 0, copy_bytes) ||
                !copy_between_slots(new_v, old_lp.v_pages, 0, copy_bytes)) {
                if (err) *err = "MetalExecutor::Impl::resize_kv_pool: page-preserving copy failed";
                return false;
            }
        }
        new_pool.layers[size_t(L)].k_pages = new_k;
        new_pool.layers[size_t(L)].v_pages = new_v;
    }
    new_pool.total_pages = new_total_pages;
    new_pool.page_size = kv_pool_.page_size;
    new_pool.enabled = true;
    // Phase 3 (review item 4): install the new pool, then RELEASE the old
    // standalone buffers (drop from residency + retained-alive so ARC frees
    // them). The synchronous copy_between_slots above has already read every
    // preserved page out of the old buffers, so nothing still references them.
    // Without this, repeated grow/shrink would leak the old K/V allocations
    // unbounded (they'd stay retained + resident forever).
    KvPagePool old_pool = std::move(kv_pool_);
    kv_pool_ = std::move(new_pool);
    for (auto& lp : old_pool.layers) {
        if (lp.k_pages.valid()) ctx_->release_standalone_buffer(lp.k_pages);
        if (lp.v_pages.valid()) ctx_->release_standalone_buffer(lp.v_pages);
    }
    return bind_paged_dag(err);
}

StepTiming MetalExecutor::Impl::step(uint32_t token_id, uint32_t position, uint32_t slot) {
    write_u32(b_.io[int(IoSlot::TokenId)],  token_id);
    write_u32(b_.io[int(IoSlot::Position)], position);
    write_u32(b_.io[int(IoSlot::SeqLen)],   position + 1u);

    int& sc = step_count_for(slot);

    // GDN conv-state cross-step ping-pong: ConvState (RO) and ConvStateOut are DISTINCT
    // buffers, advanced token-to-token by swapping their bind each step (step i reads what
    // i-1 wrote). Parity follows `slot`'s OWN monotonic step index (Phase 1b state-slot fix:
    // each slot tracks its own parity independently, so switching between slots between
    // forward calls resumes each slot's ping-pong correctly) — NOT the absolute position
    // (which can start non-zero) and NOT a single decoder-wide counter (which would
    // silently corrupt a slot's parity whenever a DIFFERENT slot had stepped in between).
    //
    // Slot selection (Phase 1b state-slot fix): the M=1 kernels (gdn_prep_bfloat16 /
    // gdn_core_recurrent_bfloat16, the shipped config) have NO slot_ids input — they always
    // operate at byte offset 0 of whatever buffer they're bound to. `arg_bind_ordinal`'s
    // offset form (`setAddress:(slot.gpu_address + offset)`) lets us slide the GPU address
    // the kernel sees by `slot*stride`, so binding "the same conv/recurrent slab, offset by
    // slot*stride" transparently retargets the UNCHANGED kernel at slot's own byte range —
    // no shader change, no new PSO. RecurrentState (unlike ConvState/ConvStateOut) is bound
    // ONCE at setup() and never touched again by the OLD code — it must ALSO be rebound here
    // every step, or every slot would silently alias slot 0's recurrent state forever.
    const bool even = (sc % 2 == 0);
    const size_t conv_stride  = g_.gdn_conv_stride_bytes();
    const size_t recur_stride = g_.gdn_recurrent_stride_bytes();
    const size_t conv_off  = size_t(slot) * conv_stride;
    const size_t recur_off = size_t(slot) * recur_stride;
    for (const auto& gd : gdn_disp_) {
        const SlotHandle& A = b_.gdn[gd.layer].conv_state;
        const SlotHandle& C = b_.gdn[gd.layer].conv_state_out;
        const SlotHandle& R = b_.gdn[gd.layer].recurrent_state;
        uint8_t cs_bind, cso_bind;
        int rs_bind = -1;  // -1: this dispatch kind has no RecurrentState bind (GdnPrep)
        if (gd.kind == Kernel::GdnPrep) {                // prep writes q/k conv_state channels
            cs_bind  = (uint8_t)bind::GdnPrep::ConvState;
            cso_bind = (uint8_t)bind::GdnPrep::ConvStateOut;
        } else if (gdn_prep_) {                           // recurrent writes v conv_state channels
            cs_bind  = (uint8_t)bind::GdnCoreRecurrent::ConvState;
            cso_bind = (uint8_t)bind::GdnCoreRecurrent::ConvStateOut;
            rs_bind  = (uint8_t)bind::GdnCoreRecurrent::RecurrentState;
        } else {                                          // in-kernel-share GdnCore
            cs_bind  = (uint8_t)bind::GdnCore::ConvState;
            cso_bind = (uint8_t)bind::GdnCore::ConvStateOut;
            rs_bind  = (uint8_t)bind::GdnCore::RecurrentState;
        }
        ctx_->arg_bind_ordinal(gd.ord, cs_bind,  even ? A : C, conv_off);
        ctx_->arg_bind_ordinal(gd.ord, cso_bind, even ? C : A, conv_off);
        if (rs_bind >= 0) ctx_->arg_bind_ordinal(gd.ord, uint8_t(rs_bind), R, recur_off);
    }

    StepTiming t = ctx_->run_step(
        [&](StepEncoder& se) { encode_decode_step(se, dag_, psos_, force_barriers_); },
        sc & 1);
    ++sc;
    return t;
}

bool MetalExecutor::Impl::run_batch_step(const BatchSchedule& schedule, const BatchStepInputs& in,
                                     std::string* err) {
    auto fail = [&](const std::string& why) {
        if (err) *err = "MetalExecutor::Impl::run_batch_step: " + why;
        return false;
    };
    if (!ready() || !g_.paged_kv_enabled || !kv_pool_.enabled || !mb_bound_)
        return fail("paged decode DAG/pool is not initialized");
    if (schedule.N <= 0 || schedule.R <= 0)
        return fail("paged batch has no tokens or requests");
    std::string capacity_err;
    if (!validate_paged_batch_capacity(schedule, uint32_t(g_.max_tokens),
                                       uint32_t(g_.max_requests), &capacity_err))
        return fail(capacity_err);
    if (in.token_ids.size() != size_t(schedule.N) || in.position_ids.size() != size_t(schedule.N) ||
        in.qo_indptr.size() != size_t(schedule.R + 1) ||
        in.kv_page_indptr.size() != size_t(schedule.R + 1) ||
        in.kv_last_page_lens.size() != size_t(schedule.R) ||
        in.rs_slot_ids.size() != size_t(schedule.R) ||
        in.rs_slot_flags.size() != size_t(schedule.R) ||
        in.w_page.size() != size_t(schedule.N) || in.w_off.size() != size_t(schedule.N))
        return fail("inconsistent fixed IO vector sizes");
    if (in.kv_page_indices.size() > size_t(g_.max_requests) * size_t(g_.total_pages))
        return fail("flattened KV CSR exceeds configured reference capacity");
    std::string geometry_err;
    if (!validate_paged_batch(schedule, in.position_ids, in.kv_page_indices, in.w_page, in.w_off,
                              kv_pool_.total_pages, uint32_t(g_.max_slots), &geometry_err))
        return fail(geometry_err);
    for (int r = 0; r < schedule.R; ++r) {
        const RequestSpan& sp = schedule.spans[size_t(r)];
        if (sp.rs_slot >= uint32_t(g_.max_slots))
            return fail("recurrent-state slot is out of range");
        if (sp.num_pages == 0 || sp.pages_first + sp.num_pages > in.kv_page_indices.size())
            return fail("request has an invalid KV page span");
        if (sp.seqlen == 0 || sp.qo_lo >= uint32_t(schedule.N) ||
            in.position_ids[sp.qo_lo] >= sp.seqlen)
            return fail("position is outside its request KV extent");
    }
    for (int t = 0; t < schedule.N; ++t) {
        const uint32_t r = schedule.req_of_token[size_t(t)];
        if (r >= uint32_t(schedule.R) || in.w_page[size_t(t)] >= kv_pool_.total_pages ||
            in.w_off[size_t(t)] >= kv_pool_.page_size)
            return fail("write page/offset is out of range");
        const RequestSpan& sp = schedule.spans[r];
        const uint32_t pos = in.position_ids[size_t(t)];
        const uint32_t page_at_pos =
            in.kv_page_indices[sp.pages_first + pos / kv_pool_.page_size];
        if (in.w_page[size_t(t)] != page_at_pos || in.w_off[size_t(t)] != pos % kv_pool_.page_size)
            return fail("write descriptor does not match the request CSR position");
    }

    auto copy_to = [&](IoSlot slot, const auto& values) {
        std::memcpy(b_.io[static_cast<int>(slot)].contents(), values.data(),
                    values.size() * sizeof(typename std::decay_t<decltype(values)>::value_type));
    };
    copy_to(IoSlot::TokenId, in.token_ids);
    copy_to(IoSlot::Position, in.position_ids);
    copy_to(IoSlot::QoIndptr, in.qo_indptr);
    copy_to(IoSlot::KvPageIndptr, in.kv_page_indptr);
    copy_to(IoSlot::KvPageIndices, in.kv_page_indices);
    copy_to(IoSlot::KvLastPageLens, in.kv_last_page_lens);
    copy_to(IoSlot::RsSlotIds, in.rs_slot_ids);
    copy_to(IoSlot::RsSlotFlags, in.rs_slot_flags);
    copy_to(IoSlot::ReqOfToken, schedule.req_of_token);
    copy_to(IoSlot::SlotOfToken, schedule.slot_of_token);
    copy_to(IoSlot::WPage, in.w_page);
    copy_to(IoSlot::WOff, in.w_off);
    std::vector<uint32_t> seq_len(size_t(schedule.N));
    for (int t = 0; t < schedule.N; ++t)
        seq_len[size_t(t)] = schedule.spans[schedule.req_of_token[size_t(t)]].seqlen;
    copy_to(IoSlot::SeqLen, seq_len);

    if (!schedule.is_pure_decode) return run_prefill_step(schedule, err);

    std::vector<uint32_t> active_slots;
    active_slots.reserve(size_t(schedule.R));
    for (const RequestSpan& sp : schedule.spans) {
        if (sp.rs_is_new) reset_state(sp.rs_slot);
        if (std::find(active_slots.begin(), active_slots.end(), sp.rs_slot) == active_slots.end())
            active_slots.push_back(sp.rs_slot);
    }
    // The paged kernels always read ConvState and write ConvStateOut.  Normalize
    // slots last touched by the M=1 ping-pong path before dispatch, then fold the
    // completed result back into ConvState for the next paged fire.
    for (uint32_t slot : active_slots) {
        if ((step_count_for(slot) & 1) == 0) continue;
        const size_t off = size_t(slot) * g_.gdn_conv_stride_bytes();
        // copy C -> A (different handles, same offset).
        for (auto& gs : b_.gdn) {
            if (!gs.conv_state.valid() ||
                !copy_between_slots(gs.conv_state, gs.conv_state_out, off,
                                    g_.gdn_conv_stride_bytes()))
                return fail("failed to normalize GDN ping-pong state");
        }
    }

    const std::vector<Dispatch> fire_dag =
        build_decode_dag_mb(g_, schedule.N, kMultiBatchOrdinalBase, fuse_residual_, gdn_prep_);
    ctx_->run_step([&](StepEncoder& se) {
        encode_decode_step_mb(se, fire_dag, psos_, mb_psos_, force_barriers_);
    });

    for (uint32_t slot : active_slots) {
        const size_t off = size_t(slot) * g_.gdn_conv_stride_bytes();
        for (auto& gs : b_.gdn) {
            if (!gs.conv_state.valid() ||
                !copy_between_slots(gs.conv_state, gs.conv_state_out, off,
                                    g_.gdn_conv_stride_bytes()))
                return fail("failed to commit GDN ping-pong state");
        }
    }
    for (uint32_t slot : schedule.slot_of_token) ++step_count_for(slot);
    return true;
}

bool MetalExecutor::Impl::run_prefill_step(const BatchSchedule& schedule, std::string* err) {
    auto fail = [&](const std::string& why) {
        if (err) *err = "MetalExecutor::Impl::run_prefill_step: " + why;
        return false;
    };
    if (size_t(schedule.N) > prefill_dags_.size())
        return fail("batch exceeds prebuilt sequential prefill command-stream capacity");

    // Reset once per request, before its first encoded token.  Do not reset in
    // the token loop: later prompt rows must consume the preceding GDN/KV state.
    for (const RequestSpan& sp : schedule.spans)
        if (sp.rs_is_new) reset_state(sp.rs_slot);

    std::vector<int> next_step(size_t(g_.max_slots), 0);
    for (int s = 0; s < g_.max_slots; ++s) next_step[size_t(s)] = step_count_for(uint32_t(s));
    for (int t = 0; t < schedule.N; ++t) {
        const uint32_t slot = schedule.slot_of_token[size_t(t)];
        bind_prefill_gdn_state(*ctx_, b_, prefill_dags_[size_t(t)], slot,
                               (next_step[slot] & 1) == 0);
        ++next_step[slot];
    }

    // One command buffer, request-major token order.  Every complete layer DAG
    // ends in a barrier, so token t+1 observes token t's GDN and paged KV writes.
    ctx_->run_step([&](StepEncoder& se) {
        for (int t = 0; t < schedule.N; ++t)
            encode_decode_step_mb(se, prefill_dags_[size_t(t)], psos_, mb_psos_,
                                  force_barriers_);
    });
    for (uint32_t slot : schedule.slot_of_token) ++step_count_for(slot);
    return true;
}

const uint16_t* MetalExecutor::Impl::logits_bf16() const {
    return static_cast<const uint16_t*>(b_.io[int(IoSlot::Logits)].contents());
}

void MetalExecutor::Impl::copy_logits_f32(float* out) const {
    const uint16_t* lb = logits_bf16();
    for (int i = 0; i < g_.vocab; ++i) out[i] = bf16_to_f32(lb[i]);
}

void MetalExecutor::Impl::copy_batch_logits_f32(uint32_t token_row, float* out) const {
    const uint16_t* lb = logits_bf16() + size_t(token_row) * size_t(g_.vocab);
    for (int i = 0; i < g_.vocab; ++i) out[i] = bf16_to_f32(lb[i]);
}

uint32_t MetalExecutor::Impl::argmax() const {
    const uint16_t* lb = logits_bf16();   // lm_head writes bf16, not f32
    uint32_t best = 0;
    float bv = bf16_to_f32(lb[0]);
    for (int i = 1; i < g_.vocab; ++i) {
        float v = bf16_to_f32(lb[i]);
        if (v > bv) { bv = v; best = uint32_t(i); }
    }
    return best;
}

MetalExecutor::MetalExecutor() = default;
MetalExecutor::~MetalExecutor() = default;

bool MetalExecutor::setup_native(
    const std::string& checkpoint_dir,
    const std::string& kernels_dir,
    const DecodeGeometry& geometry,
    std::string* error) {
    auto impl = std::make_unique<Impl>();
    if (!impl->setup(checkpoint_dir, kernels_dir, geometry, error)) {
        return false;
    }
    impl_ = std::move(impl);
    vocab_ = static_cast<std::uint32_t>(impl_->vocab());
    slot_states_.clear();
    return true;
}

bool MetalExecutor::setup_kv_pool_native(
    std::uint32_t total_pages,
    std::uint32_t page_size,
    std::string* error) {
    return ready() && impl_->setup_kv_pool(total_pages, page_size, error);
}

void MetalExecutor::reset_state_native() {
    if (ready()) impl_->reset_state();
}

void MetalExecutor::reset_state_native(std::uint32_t slot) {
    if (ready()) impl_->reset_state(slot);
}

bool MetalExecutor::copy_state_slot_native(
    std::uint32_t src_slot,
    std::uint32_t dst_slot,
    std::string* error) {
    return ready() && impl_->copy_state_slot(src_slot, dst_slot, error);
}

StepTiming MetalExecutor::step_native(
    std::uint32_t token_id,
    std::uint32_t position,
    std::uint32_t slot) {
    return ready() ? impl_->step(token_id, position, slot) : StepTiming{};
}

bool MetalExecutor::run_batch_step_native(
    const BatchSchedule& schedule,
    const BatchStepInputs& inputs,
    std::string* error) {
    return ready() && impl_->run_batch_step(schedule, inputs, error);
}

std::uint64_t MetalExecutor::paged_bind_generation_native() const {
    return ready() ? impl_->paged_bind_generation_ : 0;
}

const KvPagePool& MetalExecutor::kv_pool_native() const {
    static const KvPagePool empty;
    return ready() ? impl_->kv_pool() : empty;
}

int MetalExecutor::vocab_native() const {
    return ready() ? impl_->vocab() : 0;
}

void MetalExecutor::copy_logits_f32_native(float* output) const {
    if (ready()) impl_->copy_logits_f32(output);
}

void MetalExecutor::copy_batch_logits_f32_native(
    std::uint32_t token_row,
    float* output) const {
    if (ready()) impl_->copy_batch_logits_f32(token_row, output);
}

std::uint32_t MetalExecutor::argmax_native() const {
    return ready() ? impl_->argmax() : 0;
}

bool MetalExecutor::setup(const SetupConfig& cfg, std::string* err) {
    // Phase 1a targets exactly the shipped qwen3.6 (GDN-hybrid) geometry —
    // refuse truthfully rather than let caps advertise a forward that does
    // not exist (metal_ptir_plan.md §5.5, §12 "Caps honesty").
    if (!cfg.has_linear_attn) {
        if (err != nullptr) {
            *err = "Metal PTIR forward requires the qwen3.6 (GDN-hybrid) checkpoint "
                   "geometry in this increment (config '" +
                   cfg.arch_name + "' has no linear-attention layers)";
        }
        return false;
    }
    auto impl = std::make_unique<Impl>();
    DecodeGeometry geom{};  // shipped qwen3.6 defaults
    // Phase 1b (review fix B): really allocate `kPhase1bRsSlots` resident
    // GDN conv+recurrent state slots — heap_layout.hpp's `plan_heap` sizes
    // the State region as `slots * per_slot_bytes` and heap_bind.cpp binds
    // the M=1 kernels at slot 0's (unchanged) base offset regardless of
    // slot count, so this only grows reserved-but-idle memory; it does not
    // change the sealed M=1 decode path's behavior. `copy_state` operates
    // truthfully over these slots (real memory, not aspirational).
    geom.max_slots = kPhase1bRsSlots;
    // Bounded, actually allocated/bound multi-batch capacity.  The paged path
    // has no hidden ring fallback: every advertised row/request has an IO,
    // scratch, logits, slot-state, and CSR binding.
    geom.max_requests = static_cast<int>(std::min(cfg.max_forward_requests,
                                                  kPagedMaxForwardRequests));
    geom.max_tokens = static_cast<int>(std::min(cfg.max_forward_tokens,
                                                kPagedMaxForwardTokens));
    geom.max_slots = std::max(geom.max_slots, geom.max_requests);
    geom.kv_page_size = static_cast<int>(cfg.kv_page_size);
    geom.total_pages = static_cast<int>(cfg.total_pages);
    geom.paged_kv_enabled = cfg.total_pages > 0 && cfg.kv_page_size > 0 &&
                            geom.max_tokens > 0 && geom.max_requests > 0;
    if (cfg.vocab_size != 0 &&
        cfg.vocab_size != static_cast<std::uint32_t>(geom.vocab)) {
        if (err != nullptr) {
            *err = "checkpoint vocab_size (" + std::to_string(cfg.vocab_size) +
                   ") does not match the shipped qwen3.6 geometry (" +
                   std::to_string(geom.vocab) +
                   "); only the qwen3.6 checkpoint is supported in this increment";
        }
        return false;
    }
    std::string derr;
    if (!impl->setup(cfg.checkpoint_dir, cfg.kernels_dir, geom, &derr)) {
        if (err != nullptr) *err = "Metal forward setup failed: " + derr;
        return false;
    }
    // Phase 1b/3 paged-KV bridge: allocate a REAL paged KV pool sized from
    // the runtime's configured capacity, so copy_kv/resize_pool operate on
    // genuine storage matching caps (rather than being aspirational stubs).
    // Failure here does NOT fail executor setup — the forward path (and
    // copy_state) do not depend on the pool at all; only copy_kv/resize_pool
    // would report UNSUPPORTED if this didn't succeed (e.g. total_pages==0
    // in config, the default, deliberately leaves the pool disabled).
    if (cfg.total_pages > 0 && cfg.kv_page_size > 0) {
        std::string pool_err;
        if (!impl->setup_kv_pool(cfg.total_pages, cfg.kv_page_size, &pool_err)) {
            std::cerr << "[pie-driver-metal] MetalExecutor::setup: KV page pool allocation "
                         "failed, copy_kv/resize_pool will be UNSUPPORTED: "
                      << pool_err << "\n";
        }
    }
    impl_ = std::move(impl);
    vocab_ = static_cast<std::uint32_t>(impl_->vocab());
    slot_states_.clear();
    return true;
}

bool MetalExecutor::ready() const { return impl_ != nullptr && impl_->ready(); }

std::uint32_t MetalExecutor::vocab() const { return vocab_; }

std::uint32_t MetalExecutor::rs_slots() const {
    return ready() ? static_cast<std::uint32_t>(impl_->geometry().max_slots) : 0u;
}

std::uint64_t MetalExecutor::rs_slot_bytes() const {
    return ready() ? impl_->rs_slot_bytes() : 0u;
}

std::uint32_t MetalExecutor::kv_pool_total_pages() const {
    return ready() && impl_->kv_pool().enabled ? impl_->kv_pool().total_pages : 0u;
}

std::uint32_t MetalExecutor::kv_pool_page_size() const {
    return ready() && impl_->kv_pool().enabled ? impl_->kv_pool().page_size : 0u;
}

bool MetalExecutor::copy_kv_pages(const std::vector<std::uint32_t>& src_pages,
                                  const std::vector<std::uint32_t>& dst_pages, std::string* err) {
    if (!ready()) {
        if (err != nullptr) *err = "Metal executor not initialized";
        return false;
    }
    return impl_->copy_kv_pages(src_pages, dst_pages, err);
}

bool MetalExecutor::copy_kv_cells(const std::vector<KvMoveCell>& cells, std::string* err) {
    if (!ready()) {
        if (err != nullptr) *err = "Metal executor not initialized";
        return false;
    }
    std::vector<KvMoveCell> mapped;
    mapped.reserve(cells.size());
    for (const auto& c : cells) {
        mapped.push_back({c.dst_page_id, c.dst_token_offset, c.src_page_id, c.src_token_offset});
    }
    return impl_->copy_kv_cells(mapped, err);
}

bool MetalExecutor::resize_kv_pool(std::uint32_t new_total_pages, bool unmapped_tail_pages,
                                   std::string* err) {
    if (!ready()) {
        if (err != nullptr) *err = "Metal executor not initialized";
        return false;
    }
    return impl_->resize_kv_pool(new_total_pages, unmapped_tail_pages, err);
}

bool MetalExecutor::copy_state(std::uint32_t src_slot, std::uint32_t dst_slot, std::string* err) {
    if (!ready()) {
        if (err != nullptr) *err = "Metal executor not initialized";
        return false;
    }
    if (!impl_->copy_state_slot(src_slot, dst_slot, err)) return false;
    // Phase 1b state-slot fix: propagate `src_slot`'s tracked sequence
    // metadata to `dst_slot` too — a real memcpy without carrying the
    // matching bookkeeping would leave `dst_slot` either stale (if it had
    // its own prior metadata) or silently untracked (has_resident=false,
    // even though it now holds real, meaningful bytes). The destination is
    // explicitly NOT marked ring-backed (see LinearSequenceState doc) —
    // only an actual forward()/reset through dst_slot can promote it.
    const auto it = slot_states_.find(src_slot);
    if (it == slot_states_.end()) {
        // src_slot was never forwarded/reset — nothing meaningful to carry
        // forward; any STALE metadata already at dst_slot no longer
        // corresponds to the bytes just copied in, so drop it.
        slot_states_.erase(dst_slot);
    } else {
        LinearSequenceState copied = it->second;
        copied.resident_slot = dst_slot;
        copied.ring_backed = false;
        slot_states_[dst_slot] = std::move(copied);
    }
    return true;
}

bool MetalExecutor::forward(const MemberForwardDesc& desc, LogitsOut& out, std::string* err) {
    const std::uint32_t slot = desc.has_rs_slot ? desc.rs_slot_id : 0u;
    const auto state = slot_states_.find(slot);
    if (desc.requires_paged || desc.has_write_desc ||
        (state != slot_states_.end() && state->second.paged_backed)) {
        std::vector<LogitsOut> outs;
        std::vector<std::uint8_t> success;
        std::vector<std::string> errors;
        run_paged_batch_forward({desc}, outs, success, errors);
        if (!success.empty() && success[0] != 0) {
            out = std::move(outs[0]);
            return true;
        }
        if (err != nullptr) *err = errors.empty() ? "paged forward failed" : errors[0];
        return false;
    }
    return run_member_forward(desc, out, /*batch_serialized=*/false, err);
}

void MetalExecutor::forward_batch(const std::vector<MemberForwardDesc>& descs,
                                  std::vector<LogitsOut>& outs,
                                  std::vector<std::uint8_t>& success,
                                  std::vector<std::string>& errors) {
    outs.assign(descs.size(), LogitsOut{});
    success.assign(descs.size(), 0);
    errors.assign(descs.size(), std::string{});
    if (!ready()) {
        for (auto& e : errors) e = "Metal executor not initialized";
        return;
    }
    if (descs.size() == 1 && !descs[0].requires_paged && !descs[0].has_write_desc) {
        std::string member_err;
        if (forward(descs[0], outs[0], &member_err)) success[0] = 1;
        else errors[0] = std::move(member_err);
        return;
    }
    run_paged_batch_forward(descs, outs, success, errors);
}

bool MetalExecutor::run_paged_batch_forward(const std::vector<MemberForwardDesc>& descs,
                                            std::vector<LogitsOut>& outs,
                                            std::vector<std::uint8_t>& success,
                                            std::vector<std::string>& errors) {
    outs.assign(descs.size(), LogitsOut{});
    success.assign(descs.size(), 0);
    errors.assign(descs.size(), std::string{});
    if (!ready()) {
        for (auto& e : errors) e = "Metal executor not initialized";
        return false;
    }
    const auto& pool = impl_->kv_pool();
    if (!pool.enabled) {
        for (auto& e : errors) e = "paged KV pool is not allocated";
        return false;
    }

    BatchStepInputs in;
    std::vector<std::size_t> member_of_request;
    std::vector<std::uint32_t> token_base_of_request;
    std::unordered_map<std::uint32_t, std::size_t> slot_owner;
    auto reject = [&](std::size_t i, const std::string& reason) { errors[i] = reason; };
    for (std::size_t i = 0; i < descs.size(); ++i) {
        const MemberForwardDesc& d = descs[i];
        const std::uint32_t slot = d.has_rs_slot ? d.rs_slot_id : 0u;
        if (d.token_ids.empty() || d.token_ids.size() != d.position_ids.size()) {
            reject(i, "paged forward token/position count mismatch or empty span");
            continue;
        }
        if (slot >= rs_slots()) {
            reject(i, "recurrent-state slot is out of range");
            continue;
        }
        const bool fresh = d.has_rs_slot ? d.rs_reset : d.position_ids[0] == 0;
        const auto prior = slot_states_.find(slot);
        if (!fresh &&
            (prior == slot_states_.end() || !prior->second.has_resident ||
             !prior->second.paged_backed || prior->second.resident_sequence_id != d.sequence_id ||
             prior->second.resident_next_position != d.position_ids[0])) {
            reject(i, "paged continuation has no matching resident paged GDN state");
            continue;
        }
        if (!slot_owner.emplace(slot, i).second) {
            reject(i, "two paged requests target the same recurrent-state slot in one fire");
            continue;
        }
        if (d.kv_pages.empty()) {
            reject(i, "paged request has no KV page CSR");
            continue;
        }
        bool positions_ok = true;
        for (size_t t = 1; t < d.position_ids.size(); ++t) {
            if (d.position_ids[t] != d.position_ids[t - 1] + 1u) {
                positions_ok = false;
                break;
            }
        }
        if (!positions_ok) {
            reject(i, "paged prefill positions must be contiguous within a request");
            continue;
        }
        const uint32_t pos = d.position_ids.back();
        const uint32_t last = d.kv_last_page_len != 0
                                  ? d.kv_last_page_len
                                  : (pos % pool.page_size) + 1u;
        const uint64_t extent = uint64_t(d.kv_pages.size() - 1) * pool.page_size + last;
        if (last == 0 || last > pool.page_size ||
            std::any_of(d.position_ids.begin(), d.position_ids.end(),
                        [&](uint32_t p) { return p >= extent; })) {
            reject(i, "position is outside the request's paged KV extent");
            continue;
        }
        bool pages_ok = true;
        for (uint32_t p : d.kv_pages) {
            if (p >= pool.total_pages) { pages_ok = false; break; }
        }
        if (!pages_ok) {
            reject(i, "KV page id is outside the paged pool");
            continue;
        }
        if (d.has_write_desc &&
            (d.w_page.size() != d.token_ids.size() || d.w_off.size() != d.token_ids.size())) {
            reject(i, "explicit w_page/w_off must have one entry per prompt token");
            continue;
        }
        for (uint32_t local : d.readout_local_indices) {
            if (local >= d.token_ids.size()) {
                reject(i, "readout index exceeds this prefill member's token span");
                break;
            }
        }
        if (!errors[i].empty()) continue;

        in.qo_indptr.push_back(static_cast<uint32_t>(in.token_ids.size()));
        in.kv_page_indptr.push_back(static_cast<uint32_t>(in.kv_page_indices.size()));
        token_base_of_request.push_back(static_cast<uint32_t>(in.token_ids.size()));
        in.token_ids.insert(in.token_ids.end(), d.token_ids.begin(), d.token_ids.end());
        in.position_ids.insert(in.position_ids.end(), d.position_ids.begin(), d.position_ids.end());
        in.kv_page_indices.insert(in.kv_page_indices.end(), d.kv_pages.begin(), d.kv_pages.end());
        in.kv_last_page_lens.push_back(last);
        in.rs_slot_ids.push_back(slot);
        in.rs_slot_flags.push_back(d.has_rs_slot && d.rs_reset ? 1u : 0u);
        for (size_t t = 0; t < d.token_ids.size(); ++t) {
            const uint32_t token_pos = d.position_ids[t];
            const uint32_t csr_page = d.kv_pages[token_pos / pool.page_size];
            in.w_page.push_back(d.has_write_desc ? d.w_page[t] : csr_page);
            in.w_off.push_back(d.has_write_desc ? d.w_off[t] : token_pos % pool.page_size);
        }
        member_of_request.push_back(i);
    }
    if (in.token_ids.empty()) return false;
    in.qo_indptr.push_back(static_cast<uint32_t>(in.token_ids.size()));
    in.kv_page_indptr.push_back(static_cast<uint32_t>(in.kv_page_indices.size()));

    const BatchSchedule schedule = build_batch_schedule(
        in.token_ids.data(), int(in.token_ids.size()), in.qo_indptr.data(),
        in.kv_page_indptr.data(), in.kv_last_page_lens.data(), in.rs_slot_ids.data(),
        in.rs_slot_flags.data(), int(in.qo_indptr.size()), int(pool.page_size));
    std::string batch_err;
    if (!impl_->run_batch_step(schedule, in, &batch_err)) {
        for (std::size_t i : member_of_request) errors[i] = batch_err;
        return false;
    }
    for (std::size_t r = 0; r < member_of_request.size(); ++r) {
        const std::size_t i = member_of_request[r];
        const MemberForwardDesc& d = descs[i];
        LogitsOut& out = outs[i];
        out.vocab = vocab_;
        out.rows = static_cast<uint32_t>(d.readout_local_indices.size());
        out.data.resize(size_t(out.rows) * out.vocab);
        const std::vector<uint32_t> rows =
            global_readout_rows(token_base_of_request[r], d.readout_local_indices);
        for (uint32_t row = 0; row < out.rows; ++row)
            impl_->copy_batch_logits_f32(
                rows[row],
                                                 out.data.data() + size_t(row) * out.vocab);
        success[i] = 1;
        const uint32_t slot = d.has_rs_slot ? d.rs_slot_id : 0u;
        LinearSequenceState& state = slot_states_[slot];
        state.has_resident = true;
        state.resident_sequence_id = d.sequence_id;
        state.resident_slot = slot;
        state.resident_next_position = d.position_ids.back() + 1;
        state.resident_pages = d.kv_pages;
        state.ring_backed = false;
        state.paged_backed = true;
    }
    return true;
}

bool MetalExecutor::run_member_forward(const MemberForwardDesc& desc, LogitsOut& out,
                                       bool batch_serialized, std::string* err) {
    if (!ready()) {
        if (err != nullptr) *err = "Metal executor not initialized";
        return false;
    }
    const std::uint32_t slot = desc.has_rs_slot ? desc.rs_slot_id : 0u;
    if (desc.has_rs_slot && slot >= rs_slots()) {
        if (err != nullptr) {
            *err = "this member's recurrent-state slot (" + std::to_string(slot) +
                   ") is out of range [0, " + std::to_string(rs_slots()) + ")";
        }
        return false;
    }
    // Only one slot may be ring-backed system-wide (the shared M=1 KV ring
    // holds exactly one sequence's history) — precompute whether some
    // OTHER slot is ring-backed for a DIFFERENT sequence, for the pure
    // gate's fresh-acceptance check. Within a serialized BATCH pass this
    // arbitration already happened in `plan_batch_execution`; the ring is
    // deliberately clobbered member-to-member, so the per-member gate must
    // NOT re-reject a fresh member merely because a sibling is ring-backed.
    bool other_ring_backed_different_sequence = false;
    if (!batch_serialized) {
        for (const auto& [other_slot, other_state] : slot_states_) {
            if (other_slot == slot) continue;
            if (other_state.ring_backed && other_state.resident_sequence_id != desc.sequence_id) {
                other_ring_backed_different_sequence = true;
                break;
            }
        }
    }
    LinearSequenceState& state = slot_states_[slot];
    std::string reject;
    if (!validate_linear_sequence_geometry(state, other_ring_backed_different_sequence, desc,
                                           &reject)) {
        if (err != nullptr) *err = reject;
        return false;
    }
    for (const std::uint32_t local : desc.readout_local_indices) {
        if (local >= desc.token_ids.size()) {
            if (err != nullptr) *err = "readout index exceeds this fire's token span";
            return false;
        }
    }

    const bool is_fresh =
        desc.has_rs_slot ? desc.rs_reset : desc.position_ids.front() == 0;
    if (is_fresh) impl_->reset_state(slot);

    out.vocab = vocab_;
    out.rows = static_cast<std::uint32_t>(desc.readout_local_indices.size());
    out.data.assign(static_cast<std::size_t>(out.rows) * out.vocab, 0.0f);

    for (std::size_t i = 0; i < desc.token_ids.size(); ++i) {
        impl_->step(desc.token_ids[i], desc.position_ids[i], slot);
        for (std::uint32_t r = 0; r < desc.readout_local_indices.size(); ++r) {
            if (desc.readout_local_indices[r] != static_cast<std::uint32_t>(i)) continue;
            impl_->copy_logits_f32(out.data.data() +
                                           static_cast<std::size_t>(r) * out.vocab);
        }
    }

    // This slot is now the (only) ring-backed one — clear any other slot's
    // stale ring_backed flag (should be at most one anyway, given the
    // fresh-acceptance gate above, but defensive) without disturbing their
    // tracked metadata (copy_state'd destinations stay intact).
    for (auto& [other_slot, other_state] : slot_states_) {
        if (other_slot != slot) other_state.ring_backed = false;
    }
    state.has_resident = true;
    state.resident_sequence_id = desc.sequence_id;
    state.resident_slot = slot;
    state.resident_next_position = desc.position_ids.back() + 1;
    state.resident_pages = desc.kv_pages;
    state.ring_backed = true;
    state.paged_backed = false;
    return true;
}

#else  // !defined(__APPLE__)

// Linux/CI stub build: the direct-ABI surface still validates (abi.cpp,
// metal_direct_stub_test) but there is no Metal to run a forward on. Every
// call reports a clear, truthful error instead of silently no-op'ing.
struct MetalExecutor::Impl {};

MetalExecutor::MetalExecutor() = default;
MetalExecutor::~MetalExecutor() = default;

bool MetalExecutor::setup(const SetupConfig&, std::string* err) {
    if (err != nullptr) *err = "Metal executor requires an Apple build";
    return false;
}

bool MetalExecutor::ready() const { return false; }

std::uint32_t MetalExecutor::vocab() const { return 0; }

std::uint32_t MetalExecutor::rs_slots() const { return 0; }

std::uint64_t MetalExecutor::rs_slot_bytes() const { return 0; }

bool MetalExecutor::copy_state(std::uint32_t, std::uint32_t, std::string* err) {
    if (err != nullptr) *err = "Metal executor requires an Apple build";
    return false;
}

std::uint32_t MetalExecutor::kv_pool_total_pages() const { return 0; }
std::uint32_t MetalExecutor::kv_pool_page_size() const { return 0; }

bool MetalExecutor::copy_kv_pages(const std::vector<std::uint32_t>&,
                                  const std::vector<std::uint32_t>&, std::string* err) {
    if (err != nullptr) *err = "Metal executor requires an Apple build";
    return false;
}

bool MetalExecutor::copy_kv_cells(const std::vector<KvMoveCell>&, std::string* err) {
    if (err != nullptr) *err = "Metal executor requires an Apple build";
    return false;
}

bool MetalExecutor::resize_kv_pool(std::uint32_t, bool, std::string* err) {
    if (err != nullptr) *err = "Metal executor requires an Apple build";
    return false;
}

bool MetalExecutor::forward(const MemberForwardDesc&, LogitsOut&, std::string* err) {
    if (err != nullptr) *err = "Metal executor requires an Apple build";
    return false;
}

bool MetalExecutor::run_member_forward(const MemberForwardDesc&, LogitsOut&, bool,
                                       std::string* err) {
    if (err != nullptr) *err = "Metal executor requires an Apple build";
    return false;
}

void MetalExecutor::forward_batch(const std::vector<MemberForwardDesc>& descs,
                                  std::vector<LogitsOut>& outs,
                                  std::vector<std::uint8_t>& success,
                                  std::vector<std::string>& errors) {
    outs.assign(descs.size(), LogitsOut{});
    success.assign(descs.size(), 0);
    errors.assign(descs.size(), std::string("Metal executor requires an Apple build"));
}

#endif

}  // namespace pie::metal::batch

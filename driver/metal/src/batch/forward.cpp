#include "forward.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <type_traits>
#include <unistd.h>

#include "mtl4_context.hpp"

#if defined(__APPLE__)
#include "batch_schedule.hpp"
#include "model/facts.hpp"
#include "model/gemma4/decode_step.hpp"
#include "model/gemma4/geometry.hpp"
#include "model/qwen3_5/geometry_facts.hpp"
#include "simple_family.hpp"
#include "decode_consts.hpp"
#include "decode_dispatch_mb.hpp"
#include "decode_psos.hpp"
#include "decode_step.hpp"
#include "decode_step_mb.hpp"
#include "decode_timing.hpp"
#include "golden_tap.hpp"
#include "heap_bind.hpp"
#include "heap_bind_metal.hpp"
#include "heap_layout.hpp"
#include "logits_convert.hpp"
#include "pie_loader/checkpoint_source.hpp"
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
            "currently resident sequence (starts at " +
            std::to_string(desc.position_ids.front()) + ", resident sequence " +
            "next expects " + std::to_string(state.resident_next_position) +
            "; forks/shared-prefix/interleaved sequences are unsupported)");
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

bool validate_request_local_positions(
    const MemberForwardDesc& desc,
    std::string* error) {
    if (desc.qo_indptr.size() < 2 ||
        desc.qo_indptr.front() != 0 ||
        desc.qo_indptr.back() != desc.position_ids.size()) {
        if (error != nullptr) {
            *error = "paged query CSR does not cover positions";
        }
        return false;
    }
    for (std::size_t request = 0;
         request + 1 < desc.qo_indptr.size();
         ++request) {
        const std::uint32_t begin = desc.qo_indptr[request];
        const std::uint32_t end = desc.qo_indptr[request + 1];
        if (end <= begin || end > desc.position_ids.size()) {
            if (error != nullptr) {
                *error = "paged request has an empty or invalid position span";
            }
            return false;
        }
        for (std::uint32_t token = begin + 1; token < end; ++token) {
            if (desc.position_ids[token] !=
                desc.position_ids[token - 1] + 1u) {
                if (error != nullptr) {
                    *error =
                        "paged prefill positions must be contiguous within each request";
                }
                return false;
            }
        }
    }
    return true;
}

namespace {

bool request_rs_binding(
    const MemberForwardDesc& desc,
    std::size_t request,
    std::uint32_t& slot,
    bool& reset,
    bool& read,
    bool& write) {
    if (!desc.has_rs_slot) return false;
    if (desc.request_rs_slot_ids.empty()) {
        if (request != 0) return false;
        slot = desc.rs_slot_id;
        reset = desc.rs_reset;
        read = !reset;
        write = true;
        return true;
    }
    if (request >= desc.request_rs_slot_ids.size() ||
        request >= desc.request_rs_reset.size() ||
        request >= desc.request_rs_read.size() ||
        request >= desc.request_rs_write.size()) {
        return false;
    }
    slot = desc.request_rs_slot_ids[request];
    reset = desc.request_rs_reset[request] != 0;
    read = desc.request_rs_read[request] != 0;
    write = desc.request_rs_write[request] != 0;
    return true;
}

}  // namespace

bool validate_paged_request_state(
    const std::unordered_map<std::uint32_t, LinearSequenceState>& states,
    const MemberForwardDesc& desc,
    std::size_t request,
    std::string* error) {
    auto fail = [&](const std::string& message) {
        if (error != nullptr) *error = message;
        return false;
    };
    std::uint32_t slot = 0;
    bool reset = false, read = false, write = false;
    if (!request_rs_binding(
            desc, request, slot, reset, read, write)) {
        return fail("missing per-request recurrent-state binding");
    }
    if ((reset && read) || (!reset && !read) || !write) {
        return fail("invalid recurrent-state reset/read/write flags");
    }
    if (request + 1 >= desc.qo_indptr.size() ||
        request + 1 >= desc.kv_page_indptr.size() ||
        desc.qo_indptr[request + 1] <=
            desc.qo_indptr[request] ||
        desc.qo_indptr[request + 1] >
            desc.position_ids.size() ||
        desc.kv_page_indptr[request + 1] <
            desc.kv_page_indptr[request] ||
        desc.kv_page_indptr[request + 1] >
            desc.kv_pages.size()) {
        return fail("invalid recurrent-state request CSR");
    }
    if (reset) return true;
    const auto found = states.find(slot);
    // Five ways to not match; say which, or the caller cannot tell a slot that
    // was never resident from one that is a token behind.
    const std::uint32_t want_position =
        desc.position_ids[desc.qo_indptr[request]];
    if (found == states.end()) {
        return fail("paged continuation: recurrent slot " +
                    std::to_string(slot) + " has no state at all");
    }
    if (!found->second.has_resident) {
        return fail("paged continuation: recurrent slot " +
                    std::to_string(slot) + " holds no resident sequence");
    }
    if (!found->second.paged_backed) {
        return fail("paged continuation: recurrent slot " +
                    std::to_string(slot) +
                    " is ring-backed, not paged — the prior fire took the "
                    "sealed M=1 path and this one is paged");
    }
    if (found->second.resident_sequence_id != desc.sequence_id) {
        return fail("paged continuation: recurrent slot " +
                    std::to_string(slot) + " holds sequence " +
                    std::to_string(found->second.resident_sequence_id) +
                    ", this fire is sequence " +
                    std::to_string(desc.sequence_id));
    }
    if (found->second.resident_next_position != want_position) {
        return fail("paged continuation: recurrent slot " +
                    std::to_string(slot) + " is at position " +
                    std::to_string(found->second.resident_next_position) +
                    ", this fire starts at " + std::to_string(want_position));
    }
    const std::size_t page_begin =
        desc.kv_page_indptr[request];
    const std::size_t page_end =
        desc.kv_page_indptr[request + 1];
    if (found->second.resident_pages.size() >
            page_end - page_begin ||
        !std::equal(
            found->second.resident_pages.begin(),
            found->second.resident_pages.end(),
            desc.kv_pages.begin() + page_begin)) {
        const auto list = [](const std::vector<std::uint32_t>& v,
                            std::size_t from, std::size_t to) {
            std::string out = "[";
            for (std::size_t i = from; i < to && i < v.size(); ++i) {
                if (i != from) out += ",";
                out += std::to_string(v[i]);
            }
            return out + "]";
        };
        return fail(
            "paged continuation does not preserve recurrent-state KV lineage: "
            "resident pages " +
            list(found->second.resident_pages, 0,
                 found->second.resident_pages.size()) +
            " is not a prefix of this fire's " +
            list(desc.kv_pages, page_begin, page_end));
    }
    return true;
}

void commit_paged_request_state(
    std::unordered_map<std::uint32_t, LinearSequenceState>& states,
    const MemberForwardDesc& desc,
    std::size_t request) {
    std::uint32_t slot = 0;
    bool reset = false, read = false, write = false;
    if (!request_rs_binding(
            desc, request, slot, reset, read, write) ||
        !write ||
        request + 1 >= desc.qo_indptr.size() ||
        request + 1 >= desc.kv_page_indptr.size()) {
        return;
    }
    LinearSequenceState& state = states[slot];
    state.has_resident = true;
    state.resident_sequence_id = desc.sequence_id;
    state.resident_slot = slot;
    state.resident_next_position =
        desc.position_ids[desc.qo_indptr[request + 1] - 1] + 1;
    state.resident_pages.assign(
        desc.kv_pages.begin() + desc.kv_page_indptr[request],
        desc.kv_pages.begin() + desc.kv_page_indptr[request + 1]);
    state.ring_backed = false;
    state.paged_backed = true;
}

std::vector<PtirCommandCallbacks> compact_ptir_callbacks(
    const std::vector<MemberForwardDesc>& descs,
    const std::vector<std::size_t>& accepted_members,
    const std::vector<std::uint32_t>& accepted_token_bases,
    const std::vector<PtirCommandCallbacks>& callbacks) {
    std::vector<PtirCommandCallbacks> compacted;
    compacted.reserve(accepted_members.size());
    for (std::size_t request = 0;
         request < accepted_members.size();
         ++request) {
        const std::size_t member = accepted_members[request];
        if (member >= descs.size() || member >= callbacks.size() ||
            request >= accepted_token_bases.size()) {
            continue;
        }
        PtirCommandCallbacks callback = callbacks[member];
        std::vector<std::uint32_t> rows;
        rows.reserve(
            descs[member].readout_local_indices.size());
        for (const std::uint32_t local :
             descs[member].readout_local_indices) {
            rows.push_back(
                accepted_token_bases[request] + local);
        }
        if (callback.set_logits_rows) {
            callback.set_logits_rows(rows);
        }
        if (callback.set_logits_row &&
            rows.size() == 1) {
            callback.set_logits_row(rows[0]);
        }
        compacted.push_back(std::move(callback));
    }
    for (auto& callback : compacted)
        if (callback.finalize_group) callback.finalize_group();
    return compacted;
}

std::vector<PtirCommandCallbacks> compact_ptir_member_callbacks(
    const std::vector<std::size_t>& accepted_members,
    const std::vector<std::vector<std::uint32_t>>& readout_rows,
    const std::vector<PtirCommandCallbacks>& callbacks) {
    std::vector<PtirCommandCallbacks> compacted;
    compacted.reserve(accepted_members.size());
    for (const std::size_t member : accepted_members) {
        if (member >= callbacks.size() ||
            member >= readout_rows.size()) {
            continue;
        }
        PtirCommandCallbacks callback = callbacks[member];
        const auto& rows = readout_rows[member];
        if (callback.set_logits_rows) {
            callback.set_logits_rows(rows);
        }
        if (callback.set_logits_row && rows.size() == 1) {
            callback.set_logits_row(rows[0]);
        }
        compacted.push_back(std::move(callback));
    }
    for (auto& callback : compacted) {
        if (callback.finalize_group) callback.finalize_group();
    }
    return compacted;
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
    // The token count `mb_dag_`'s width-dependent constants are currently bound
    // at. Seeded from the setup bind so a first fire at `max_tokens` rebinds
    // nothing; every other width rebinds once. See the fire path.
    int mb_bound_tokens_ = 0;
    std::uint64_t paged_bind_generation_ = 0;
    SlotHandle ptir_logits_{};
    SlotHandle ptir_logits_copy_params_{};
    // The prefill's uniform scratch row pitch, in elements, for the batched
    // projection (`affine_qmm_t_strided` reads it at buffer 8).
    SlotHandle prefill_row_stride_{};
    SlotHandle prefill_rows_{};
    SlotHandle prefill_fp16_input_{};
    // One scan-length buffer per prompt row.  A grouped prefill carries several
    // requests, each with its own scan length, and the row's argument table is
    // what selects the segment -- so the length has to be per row too.
    std::vector<SlotHandle> prefill_scan_rows_{};
    // A slot whose conv history was last written by the prefill's ping-pong may
    // hold it in ConvStateOut; the paged decode writes in place and always
    // leaves it in ConvState, so only the handover needs a copy.
    std::vector<std::uint8_t> conv_in_out_{};
    Pso ptir_logits_copy_pso_{};
    std::uint32_t ptir_logits_capacity_rows_ = 0;
    std::uint32_t ptir_logits_next_row_ = 0;

    struct GdnDisp {
        int ord;
        int layer;
        Kernel kind;
    };
    std::vector<GdnDisp> gdn_disp_{};
    LinearStateSlots linear_state_slots_{};

    // ── Families with no recurrent state ──
    // gemma4 and gpt-oss share none of the machinery above -- no GDN slots, no
    // conv ping-pong, no split-K -- so their family-shaped half lives in
    // `SimpleFamilyEngine` and everything that is NOT family-shaped (this
    // context, the logits staging, the sequence bookkeeping) stays here.
    model::ModelFamily family_ = model::ModelFamily::Qwen35;
    std::unique_ptr<SimpleFamilyEngine> simple_{};
    bool is_simple() const { return simple_ != nullptr; }
    SimpleFamilyEngine* simple_engine() const { return simple_.get(); }
    StepTiming fire_simple(const SimpleFamilyEngine::FireCsr& csr,
                           const SimpleFamilyEngine::EncodeHook& pre = {},
                           const SimpleFamilyEngine::EncodeHook& post = {}) {
        const StepTiming t = simple_->fire(*ctx_, csr, pre, post);
        if (golden_taps_enabled()) simple_->dump_taps(int(csr.token_ids.size()));
        return t;
    }

    /// Build one of those, and the context it lives in.
    bool setup_simple(model::ModelFamily family, const std::string& kernels_dir,
                      const SetupConfig& cfg, const pie_loader::LoadPlan& load_plan,
                      std::string* err);

    static constexpr bool gdn_prep_ = true;
    // Folds the block/MLP residual add into the projection that feeds it,
    // dropping 48 of the DAG's 411 dispatches. The decode DAG is launch-bound
    // -- ~10us a dispatch against qmv kernels that are themselves only ~10us of
    // weight streaming -- so removing dispatches is the lever, and this one is
    // free: the fused epilogue is already implemented (affine_qmv_fast_residual)
    // and the scratch schedule already models it. Measured 45.7 -> 52.5 tok/s
    // with byte-identical output.
    static constexpr bool fuse_residual_ = true;
    static constexpr bool force_barriers_ = false;
    /// The KV ring, in tokens. `kMetalMaxCtxTokens` unless a `SetupConfig`
    /// asked for less; see `SetupConfig::max_ctx_tokens`.
    int max_ctx_ = int(kMetalMaxCtxTokens);

    // No checkpoint directory: since §6 the plan declares the files it reads,
    // so staging weights needs the plan and nothing else.
    bool setup(
        const std::string& kernels_dir,
        const DecodeGeometry& geometry,
        const pie_loader::LoadPlan& load_plan,
        std::size_t storage_page_size,
        bool stream_routed_experts,
        std::uint32_t max_ctx_tokens,
        std::string* error);
    bool ready() const { return ctx_ != nullptr; }
    int vocab() const { return simple_ ? simple_->vocab() : g_.vocab; }
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
    bool resize_elastic_pool(
        std::uint64_t pool_id,
        std::uint64_t target_pages,
        std::string* error);
    bool ensure_elastic_storage(
        std::uint32_t kv_pages,
        std::uint32_t state_slots,
        std::uint32_t token_rows,
        std::uint32_t ring_tokens,
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
        std::uint32_t slot = 0,
        const PtirCommandCallbacks* ptir = nullptr);
    bool run_batch_step(
        const BatchSchedule& schedule,
        const BatchStepInputs& inputs,
        std::string* error,
        const std::vector<PtirCommandCallbacks>* ptir = nullptr);
    const std::uint16_t* logits_bf16() const;
    void copy_logits_f32(float* output) const;
    void copy_batch_logits_f32(
        std::uint32_t token_row,
        float* output) const;
    std::uint32_t argmax() const;
    bool ensure_ptir_logits_rows(std::uint32_t rows, std::string* error);
    std::uint32_t reserve_ptir_logits_rows(std::uint32_t rows);
    // Rows the forward should copy into the PTIR staging buffer as part of its
    // OWN command buffer. Staging used to be a second `run_step` -- a whole
    // command-buffer submit and completion wait per token, for a copy that is
    // ~1 us of bandwidth. Set before `run_batch_step`, consumed by its encoder.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> pending_logits_stage_{};
    /// Set when the staging copy could not be encoded into a step's own
    /// command buffer. Recorded rather than returned because the failure is
    /// discovered inside the encode callback, which has nowhere to fail to.
    std::string step_stage_error_{};
    bool encode_logits_stage(StepEncoder& encoder, std::string* error);
    void attach_ptir_logits_view(LogitsOut& output) const;

  private:
    int& step_count_for(std::uint32_t slot) {
        return linear_state_slots_.at(slot);
    }
    bool bind_paged_dag(std::string* error);
    bool run_prefill_step(
        const BatchSchedule& schedule,
        const BatchStepInputs& in,
        std::string* error,
        const std::vector<PtirCommandCallbacks>* ptir);
};

namespace {

struct PtirLogitsCopyParams {
    std::uint32_t source_row = 0;
    std::uint32_t destination_row = 0;
    std::uint32_t vocab = 0;
    std::uint32_t reserved = 0;
};


// Rows one staging dispatch can carry. Bounded by the paged forward's row
// capacity, which is what `LogitsOut::rows` is drawn from.
inline constexpr std::size_t kPtirLogitsCopyMaxRows = kPagedMaxForwardTokensCeiling;
// The prefill's ordinal blocks and PTIR's allocations share one argument-table
// namespace, and the ceiling above is what decides where the first ends.
static_assert(int(kPagedMaxForwardTokensCeiling) <= kPrefillOrdinalMaxRows,
              "prefill rows would claim ordinals PTIR hands out");

void write_u32(const SlotHandle& s, uint32_t v) {
    std::memcpy(s.contents(), &v, sizeof(v));
}

// Both setup paths compute a heap size and an elastic budget, and the line
// after that is where arithmetic turns into spending. A model that does not
// fit does NOT fail at that line -- the heap is created, nineteen gigabytes
// are copied into it over forty-six seconds, every bind succeeds, and the
// first command buffer comes back with "The operation couldn't be completed",
// whose real error is three `NSUnderlyingError` levels down:
// `kIOGPUCommandBufferCallbackErrorOutOfMemory`. Asking the device what it
// will hold is one call, and it turns three quarters of a minute and an
// unreadable failure into a sentence with numbers in it.
//
// This lives here, rather than at either call site, because it was written at
// one of them: qwen3.5 got the check when a 35B mixture would not load, and
// `setup_simple` -- llama, qwen3, gemma and gpt-oss, which is most of what
// runs -- had none at all. A refusal that only one caller performs is not a
// refusal the driver makes.
//
// On the accuracy of the arithmetic being checked: measured on Qwen3.5-35B-A3B
// the plan wanted 22.59 GiB and the device reported 22.61 GiB allocated at the
// first dispatch. The sizing is right to twenty megabytes. What is optimistic
// is `recommendedMaxWorkingSetSize` itself -- on this M1 Max it is 24.96 GiB,
// a flat 78% of the 32 GiB the machine has, taking no account of the 6 GiB the
// kernel had wired down by the time prefill ran. That is why the check below
// asks the kernel a second question -- what is reclaimable right now -- and
// refuses on whichever bound is tighter. The device ceiling catches models
// that are plainly too big for the GPU; the host bound catches models that
// would fit a quiet machine and not this one.
// The elastic budget's four addends, kept apart so a refusal can name the one
// that is large. Assembled by the caller because only it knows the scratch
// pool, which the heap plan does not carry.
struct ElasticBreakdown {
    std::size_t kv_ring = 0;
    std::size_t kv_pool = 0;
    std::size_t state = 0;
    std::size_t scratch = 0;
};

// Say once, before anything is allocated, that this machine is carrying GPU
// memory nobody owns.
//
// A model small enough to fit alongside the leak is admitted by the check
// below and runs fine, so nothing on the path to a successful load would
// otherwise mention it -- and the danger is not to the run. Wired pages
// cannot be paged out and are charged to no live process, so `ps`, Activity
// Monitor and free-memory readings all look ordinary while the window server
// is one composite away from being unable to allocate. When it cannot, it
// blocks in the kernel inside its own Metal submit, misses the 120-second
// userspace watchdog, and the desktop dies. Observed here twice: once during
// a run, and once **ten hours after** the run that leaked the memory, with no
// pie process alive to connect it to.
//
// So this warns rather than refuses. The load is not what is unsafe; leaving
// the machine up is. The threshold is half of RAM because a healthy idle Mac
// sits near 3%, and the leaks seen here were 59% -- there is no ambiguous
// middle to tune against.
void warn_once_if_the_gpu_leaked_memory_before_this_run() {
    [[maybe_unused]] static const bool said = [] {
        const auto [wired, installed] =
            RawMetalContext::host_wired_and_installed_bytes();
        if (installed == 0 || wired * 2 <= installed) return true;
        const auto gib = [](std::size_t b) { return double(b) / (1024.0 * 1024.0 * 1024.0); };
        std::fprintf(stderr,
                     "[pie-metal] warning: %.2f GiB of this machine's %.2f GiB is wired "
                     "before this model is loaded. No process has to be holding it: a GPU "
                     "context whose command buffer never signalled is abandoned rather "
                     "than released, and its pages stay wired until reboot. They cannot "
                     "be paged out, so the window server can be starved of memory hours "
                     "later and take the desktop down with it. Reboot before leaving this "
                     "machine unattended.\n",
                     gib(wired), gib(installed));
        return true;
    }();
}

bool fits_on_this_gpu(std::size_t heap_bytes,
                      std::size_t elastic_bytes,
                      std::size_t resident_weights,
                      std::string* err,
                      const ElasticBreakdown* parts = nullptr) {
    warn_once_if_the_gpu_leaked_memory_before_this_run();
    const std::size_t limit = RawMetalContext::device_working_set_bytes();
    if (limit == 0) return true;  // the device would not say; do not invent one
    const std::size_t want = heap_bytes + elastic_bytes;

    // The device ceiling is what this GPU would hold on an idle machine. What
    // the machine will actually give us right now is a second, independent
    // bound, and on unified memory it is usually the smaller one. Checking
    // only the first is how a 14 GiB model was admitted onto a box with 18 GiB
    // left, allocated its pools, and then hung: the command buffer never
    // signalled, the context was abandoned as unsafe to release, and the
    // process became unkillable. Every retry left another one, so free memory
    // fell with each attempt while the ceiling being checked never moved.
    //
    // Refusing here is the only cheap moment. Afterwards there is no failure
    // path -- the allocation does not fail, the dispatch does not return, and
    // nothing short of a reboot recovers the memory.
    //
    // The margin is headroom for what the load itself adds beyond the plan:
    // the mmap'd weights file leaves a file-backed copy roughly the size of
    // the model, and the kernel needs room to keep running. It is deliberately
    // a flat floor rather than a fraction, so that the refusal stays legible.
    const std::size_t reclaimable =
        RawMetalContext::device_working_set_is_forced()
            ? 0  // a forced ceiling describes a device, not this machine
            : RawMetalContext::host_reclaimable_bytes();
    constexpr std::size_t kHostMargin = 2ull * 1024 * 1024 * 1024;
    const bool host_bound =
        reclaimable != 0 && want + kHostMargin > reclaimable && want <= limit;

    if (want <= limit && !host_bound) return true;
    if (err) {
        const auto gib = [](std::size_t b) {
            return std::to_string(double(b) / (1024.0 * 1024.0 * 1024.0)).substr(0, 5);
        };
        if (host_bound) {
            *err = "this model does not fit the memory this machine has left: "
                   "it needs " + gib(want) + " GiB resident (" +
                   gib(resident_weights) + " GiB of weights, " +
                   gib(elastic_bytes) + " GiB of KV, state and scratch) and only " +
                   gib(reclaimable) + " GiB is reclaimable. The GPU itself would "
                   "hold " + gib(limit) + " GiB, so this is the machine, not the "
                   "device: something else already has the memory. On macOS a "
                   "previously wedged run is the usual cause -- it survives "
                   "kill -9, holds its pages, and is only cleared by a reboot.";
            return false;
        }
        *err = "this model does not fit this GPU: it needs " + gib(want) +
               " GiB resident (" + gib(resident_weights) + " GiB of weights, " +
               gib(elastic_bytes) + " GiB of KV, state and scratch) and the device "
               "will hold " + gib(limit) + " GiB.";
        // Which region to shrink, and with which knob. "A shorter context
        // shrinks the KV" was the whole of the old advice, and on a paged
        // family it is wrong twice: the operator reaches for `total_pages`,
        // the number does not move, and nothing says why. It does not move
        // because a paged model allocates BOTH the paged pool that
        // `total_pages` sizes AND the M=1 contiguous ring that `max_ctx`
        // sizes -- two KV regions, one knob each. Naming them separately is
        // the difference between a refusal an operator can act on and one
        // they can only read.
        if (parts != nullptr) {
            *err += " Of that: " + gib(parts->kv_ring) +
                    " GiB M=1 KV ring (from max_model_len), " +
                    gib(parts->kv_pool) +
                    " GiB paged KV pool (total_pages x kv_page_size), " +
                    gib(parts->state) +
                    " GiB recurrent state (max_forward_requests), " +
                    gib(parts->scratch) + " GiB scratch.";
        }
        *err += " The weights do not shrink.";
    }
    return false;
}

}  // namespace

bool MetalExecutor::Impl::setup_simple(model::ModelFamily family,
                                       const std::string& kernels_dir,
                                       const SetupConfig& cfg,
                                       const pie_loader::LoadPlan& load_plan,
                                       std::string* err) {
    family_ = family;
    if (cfg.max_ctx_tokens > 0) {
        max_ctx_ = int(std::min<std::uint32_t>(cfg.max_ctx_tokens, kMetalMaxCtxTokens));
    }
    // The heap: the weights the plan already sized, plus what the family needs
    // on top of them. `plan_heap` is not consulted -- it is `DecodeGeometry`'s.
    const std::size_t weights = load_plan.view().memory.persistent_bytes;
    // Streamed tensors are bound over a pack, so they must not be counted here:
    // a heap sized for weights that are then ALSO mapped is the footprint
    // doubled rather than halved, and on a machine where the model only just
    // fits that is the difference between running and reading zeros.
    // A budget for the routed experts changes BOTH numbers below, in opposite
    // directions, and that is the whole shape of the feature. The bank leaves
    // the heap as it does when streamed -- but a slab of the budget's size
    // joins it, and unlike a mapping the bank never becomes resident at all.
    // So the ask falls from the model to the dense weights plus the budget,
    // which is what lets a model bigger than the GPU be admitted.
    const bool slab = cfg.expert_slab_bytes > 0;
    // Asked BEFORE the sizing, because the sizing below already believes the
    // budget: it takes the paging rule for what must be resident on the
    // strength of this field alone. A family that cannot page then reached the
    // fit check having had its bank subtracted from the ask and the budget
    // added to the heap, and refused with "this model does not fit" -- true,
    // but about the wrong thing, and it never reached the engine that knew the
    // real reason. One site, before either number is computed, so the message
    // is the reason.
    if (slab) {
        const char* why = nullptr;
        switch (family) {
        case model::ModelFamily::Llama:
            break;
        case model::ModelFamily::GptOss:
            // Was refused here for its per-expert bias, which is indexed by
            // the very buffer paging renumbers. The fix was to page the bias
            // too, as one more band of the same slot, rather than to keep a
            // family out -- see `stream_predicate`.
            break;
        case model::ModelFamily::Gemma4:
            // The family has both shapes. E2B/E4B are dense and have no bank to
            // page; the 26B mixture has 128 experts per layer and pages exactly
            // as the others do. Asked of the CONFIG rather than of the loaded
            // tensors because this runs before the engine exists.
            if (cfg.gemma4.n_experts <= 1) {
                why = "gemma4: expert_slab_bytes has nothing to page -- this checkpoint is "
                      "dense and has no routed expert bank";
            }
            break;
        default:
            why = "expert_slab_bytes is only supported for the llama family";
            break;
        }
        if (why) {
            if (err) *err = why;
            return false;
        }
    }
    const auto streams =
        SimpleFamilyEngine::stream_predicate(cfg.stream_routed_experts || slab, slab);
    // Not gated on `streams`: a checkpoint that places its tensors where a
    // device pointer may point has EVERY weight bound over its own mapping,
    // whether or not anything asked for expert streaming.
    //
    // With a slab the mapping is off, so only the routed banks leave the heap.
    const std::size_t streamed =
        std::size_t(streamable_plan_bytes(load_plan, streams, slab));
    // Saturating, not `>` -- when EVERY weight is bound where it lies the two
    // are equal, and a guard that fell back to the full `weights` on equality
    // is the one case that matters most: it re-reserved the entire model
    // alongside its own mapping and the first command buffer came back out of
    // memory.
    const std::size_t extra = SimpleFamilyEngine::extra_heap_bytes(family, cfg, max_ctx_);
    const std::size_t heap_bytes = (weights >= streamed ? weights - streamed : 0) +
                                   (slab ? std::size_t(cfg.expert_slab_bytes) : 0) + extra;
    // These families carry their KV inside `extra_heap_bytes` rather than in
    // an elastic budget, so the breakdown the other path traces is one number
    // here -- but it is the number `max_model_len` moves, and a knob whose
    // effect cannot be observed is a knob nobody can tune.
    if (std::getenv("PIE_METAL_LOAD_TRACE") != nullptr) {
        std::fprintf(stderr,
                     "[pie-metal] load: ctx %d tokens; kv, state and scratch %.1f MB\n",
                     max_ctx_, double(extra) / (1024.0 * 1024.0));
    }
    // What to ALLOCATE and what must be RESIDENT are two numbers, and this
    // refusal is about the second. Weights bound where they lie leave the heap
    // but not the working set -- `wrap_host_memory` puts them in the residency
    // set and asks for them, and paging in Qwen3-30B's mapping costs 9.3 s of
    // real I/O, which is not what an evictable byte costs. Subtracting them
    // from the ask made a 17.17 GB model report that it needed 0.326 GiB, so
    // the one guard meant to say "this will not fit" said nothing at all.
    //
    // `create` here takes no elastic budget, so the whole ask is the heap plus
    // whatever was bound outside it.
    //
    // A slab makes `streamed` mean something else: those bytes are read from an
    // mmap the GPU never sees, so they are neither allocated nor resident, and
    // adding them here would refuse exactly the models this exists to run.
    if (!fits_on_this_gpu(slab ? heap_bytes : heap_bytes + streamed, 0,
                          slab ? heap_bytes : weights, err)) {
        return false;
    }
    // Two marks, because between them lies the answer to "why is loading slow"
    // and the two halves have completely different causes. Everything up to
    // `staged` is the driver's own work -- copying, dequantizing, allocating --
    // and `PIE_METAL_LOAD_TRACE`'s `load:` line breaks it down further. What
    // follows is `make_resident`, which is the kernel paging the weights in,
    // and no amount of driver work makes it cheaper.
    //
    // Binding a checkpoint where it lies moves the whole cost across that line:
    // Qwen3-30B off safetensors is 31.3 s of copying and 2.2 s of residency,
    // and off the same weights in a `.zt` it is 1 ms and 9.3 s. The second
    // number rose because the pages are now faulted from the file instead of
    // from a heap that the copy had already warmed -- and the total still fell
    // from 33.8 s to 9.6 s, because the file is read once instead of twice.
    const auto _t0 = std::chrono::steady_clock::now();
    const auto _mark = [&](const char* what) {
        if (std::getenv("PIE_METAL_LOAD_TRACE") == nullptr) return;
        std::fprintf(stderr, "[pie-metal] setup: %s %.0f ms\n", what,
                     std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - _t0).count());
    };
    ctx_ = RawMetalContext::create(heap_bytes);
    if (!ctx_) {
        if (err) *err = "RawMetalContext::create failed";
        return false;
    }
    simple_ = SimpleFamilyEngine::create(family, *ctx_, kernels_dir, cfg, load_plan, max_ctx_,
                                         err);
    _mark("staged");
    if (!simple_) return false;

    // The sampler reads through the same staging path qwen3.5 uses, so the
    // engine's logits slot takes the place of `b_.io[Logits]` at its ordinal --
    // and the copy that fills it needs its own pipeline, which qwen's `setup`
    // builds and this one bypasses.
    g_.vocab = simple_->vocab();
    std::string pso_err;
    ptir_logits_copy_pso_ = ctx_->compile_ptir_pso_from_file(
        (std::filesystem::path(kernels_dir) / "ptir_logits_copy.metal").string(),
        "ptir_copy_logits_bf16", &pso_err);
    if (!ptir_logits_copy_pso_.valid()) {
        if (err) *err = "compiling the logits staging copy: " + pso_err;
        return false;
    }
    ptir_logits_copy_params_ =
        ctx_->create_standalone_buffer(sizeof(PtirLogitsCopyParams) * kPtirLogitsCopyMaxRows);
    if (!ptir_logits_copy_params_.valid()) {
        if (err) *err = "allocating the logits staging params";
        return false;
    }
    ctx_->arg_bind_ordinal(kPtirLogitsCopyOrdinal, 2, ptir_logits_copy_params_);
    if (!ensure_ptir_logits_rows(1, err)) return false;
    ctx_->arg_bind_ordinal(kPtirLogitsCopyOrdinal, 0, simple_->logits_slot());

    // The KV pool the engine plans against.
    //
    // A paged family reports the pages it really allocated, which is what makes
    // the runtime's physical page ids mean something: they index the engine's
    // per-layer pools directly. A ring-backed one reports the capacity the RING
    // has -- not a lie about storage, since `ensure_kv_pages` bounds demand
    // against the same number, but a page is not a thing it stores into. What
    // is genuinely absent for both -- moving pages around, which is what prefix
    // sharing and forking need -- is refused as absent.
    const std::uint32_t page =
        simple_->paged() ? std::uint32_t(simple_->page_size())
                         : (cfg.kv_page_size > 0 ? cfg.kv_page_size : 1u);
    kv_pool_.enabled = true;
    kv_pool_.page_size = page;
    kv_pool_.total_pages = simple_->paged() ? std::uint32_t(simple_->total_pages())
                                            : std::uint32_t(max_ctx_) / page;
    kv_pool_.capacity_pages = kv_pool_.total_pages;
    kv_pool_.committed_pages = kv_pool_.total_pages;
    kv_pool_.layers.clear();

    ctx_->make_resident();
    _mark("resident");
    return true;
}

bool MetalExecutor::Impl::setup(const std::string& kernels_dir,
                            const DecodeGeometry& geom,
                            const pie_loader::LoadPlan& load_plan,
                            std::size_t storage_page_size,
                            bool stream_routed_experts,
                            std::uint32_t max_ctx_tokens,
                            std::string* err) {
    g_ = geom;
    // The same line `setup_simple` has, for the same reason: `max_ctx_` sizes
    // the M=1 ring `plan_heap` reserves, and this path used to leave it at the
    // ceiling. So `max_model_len` moved nothing here while the refusal that
    // printed the ring named `max_model_len` as its knob -- an operator could
    // set it to 512, watch 8 GiB not move, and have nothing to read.
    if (max_ctx_tokens > 0) {
        max_ctx_ = int(std::min<std::uint32_t>(max_ctx_tokens, kMetalMaxCtxTokens));
    }

    // The mmap is transient: the LoadPlan copies each finalized tensor
    // once into the resident weights region.
    const auto storage = load_plan.view();
    auto view = std::make_shared<pie_loader::CheckpointSource>(storage);
    plan_ = plan_heap(
        g_,
        storage.memory.persistent_bytes,
        max_ctx_,
        4,
        2,
        std::max<std::size_t>(1, storage_page_size),
        std::max<std::size_t>(1, load_plan.preferred_alignment()));

    // ── Build the decode DAG (shipped config: GdnPrep ON, no argmax dispatch — host samples). ──
    // Under the accuracy gate every activation value gets its own pool buffer, so a
    // tap's producer is still readable once the command buffer retires.
    const bool taps = golden_taps_enabled() && !golden_taps_recycle();
    dag_ = build_decode_dag(g_, /*with_argmax=*/false, fuse_residual_, gdn_prep_);
    // Each DAG owns one region of the argument-table namespace. A DAG longer
    // than its region reaches into the next one's, and the two then share
    // tables silently -- see `ordinals_fit`.
    if (!ordinals_fit(dag_.size())) {
        if (err) {
            *err = "this decoder emits " + std::to_string(dag_.size()) +
                   " dispatches and one argument-table region holds " +
                   std::to_string(kPrefillOrdinalStride);
        }
        return false;
    }
    if (g_.paged_kv_enabled) {
        mb_dag_ = build_decode_dag_mb(g_, std::max(1, g_.max_tokens),
                                      kMultiBatchOrdinalBase, fuse_residual_, gdn_prep_);
        if (!ordinals_fit(mb_dag_.size())) {
            if (err) {
                *err = "this decoder's batched DAG emits " + std::to_string(mb_dag_.size()) +
                       " dispatches and one argument-table region holds " +
                       std::to_string(kPrefillOrdinalStride);
            }
            return false;
        }
        mb_sched_ = build_scratch_schedule(mb_dag_, g_, /*no_recycle=*/taps);
        prefill_dags_ = build_decode_prefill_dags(g_, std::max(1, g_.max_tokens),
                                                   fuse_residual_, gdn_prep_);
        prefill_sched_ = build_scratch_schedule(prefill_dags_.front(), g_, /*no_recycle=*/taps);
    }

    // ── beta's scratch schedule (WAR/WAW coloring). e2e path always recycles. ──
    sched_ = build_scratch_schedule(dag_, g_, /*no_recycle=*/taps);

    size_t prefill_consts_budget = 0;
    for (const auto& dag : prefill_dags_) prefill_consts_budget += decode_consts_budget(dag);
    const size_t consts_budget = decode_consts_budget(dag_) +
                                 (mb_dag_.empty() ? 0 : decode_consts_budget(mb_dag_)) +
                                 prefill_consts_budget;
    const size_t scratch_pool_bytes =
        size_t(std::max({sched_.colors_used, mb_sched_.colors_used,
                         prefill_sched_.colors_used})) *
        plan_.scratch_slot_bytes;
    // Streamed weights are bound over a pack, so the heap must be created
    // WITHOUT them: a heap sized for weights that are then also mapped is the
    // footprint doubled rather than halved, which on a machine where the model
    // only just fits is the difference between running and reading zeros.
    const auto streams = SimpleFamilyEngine::stream_predicate(stream_routed_experts);
    // Not gated on `streams`; see `setup_simple`.
    const size_t streamed = size_t(streamable_plan_bytes(load_plan, streams));
    // Saturating; see `setup_simple` on why equality is the case that matters.
    const size_t resident_weights =
        plan_.weights_bytes >= streamed ? plan_.weights_bytes - streamed : 0;
    const size_t heap_bytes =
        resident_weights + plan_.io_bytes + plan_.mb_io_bytes +
        consts_budget + (32u << 20);
    const size_t elastic_budget =
        plan_.kv_bytes + plan_.state_bytes + plan_.scratch_bytes +
        plan_.kv_pool_bytes + (taps ? 0u : scratch_pool_bytes);

    // Heap plus mapping: see `setup_simple`. What leaves the heap does not
    // leave the working set.
    const ElasticBreakdown elastic_parts{
        plan_.kv_bytes, plan_.kv_pool_bytes, plan_.state_bytes,
        plan_.scratch_bytes + (taps ? 0u : scratch_pool_bytes)};
    // The same four numbers the refusal prints, printed on the path that
    // SUCCEEDS too. A knob whose effect is only visible when the model fails
    // to load is a knob nobody can tune.
    if (std::getenv("PIE_METAL_LOAD_TRACE") != nullptr) {
        const auto mb = [](std::size_t b) { return double(b) / (1024.0 * 1024.0); };
        std::fprintf(stderr,
                     "[pie-metal] load: ctx %d tokens; kv ring %.1f MB, kv pool %.1f MB, "
                     "state %.1f MB, scratch %.1f MB\n",
                     max_ctx_, mb(elastic_parts.kv_ring), mb(elastic_parts.kv_pool),
                     mb(elastic_parts.state), mb(elastic_parts.scratch));
    }
    if (!fits_on_this_gpu(heap_bytes + streamed, elastic_budget, plan_.weights_bytes, err,
                          &elastic_parts))
        return false;

    ctx_ = RawMetalContext::create(heap_bytes, elastic_budget);
    if (!ctx_) {
        if (err) *err = "RawMetalContext::create failed";
        return false;
    }

    // ── Stage weights/state/KV/IO; bind weight/state/KV/IO slots by ordinal. ──
    b_ = stage_decode_storage(*ctx_, std::move(view), load_plan, g_, plan_, streams);
    bind_decode_dag(*ctx_, b_, dag_, g_, gdn_prep_);

    // ── Scratch pool (colors_used slots) → beta's bind pass. ──
    pool_.resize(std::max({sched_.colors_used, mb_sched_.colors_used,
                           prefill_sched_.colors_used}));
    // Commit the scratch slots. `create_elastic_buffer`'s second argument
    // defaults to zero, which leaves a placement-sparse VA with no memory
    // behind it -- and every activation in the graph passes through this pool,
    // so an uncommitted slot is read as whatever the sparse mapping returns.
    // The GDN and KV allocations next door go through `alloc_zeroed`, which
    // always passes a commit size; this one was the exception.
    for (size_t i = 0; i < pool_.size(); ++i) {
        pool_[i] = taps ? ctx_->create_standalone_buffer(plan_.scratch_slot_bytes)
                        : ctx_->create_elastic_buffer(
                              plan_.scratch_slot_bytes, plan_.scratch_slot_bytes);
        if (!pool_[i].valid()) {
            if (err) {
                *err = "scratch slot " + std::to_string(i) + " (" +
                       std::to_string(plan_.scratch_slot_bytes) +
                       " bytes) failed to commit";
            }
            ctx_.reset();
            return false;
        }
    }
    bind_scratch(*ctx_, dag_, sched_, pool_.data(), int(pool_.size()));

    // ── Geometry const-params. ──
    bind_decode_consts(*ctx_, dag_, g_, max_ctx_, gdn_prep_);

    // ── Compile the kernel PSOs. ──
    std::string load_err;
    if (!load_decode_psos(
            *ctx_, kernels_dir, psos_, g_.quant, &load_err,
            DecodePsoFeatures{
                .residual_qmv = fuse_residual_,
                .gdn = gdn_prep_,
                .gated_attention = true,
                .sdpa_d256 = true,
                .routed = g_.is_moe(),
                .untied = !g_.tied_embeddings})) {
        if (err) *err = "PSO load failed: " + load_err;
        ctx_.reset();
        return false;
    }
    // The second quantized set. A checkpoint that spares its two routing
    // projections at another width gets one more table, and only the two kinds
    // that read it are taken from it -- everything else stays on the pipelines
    // the model-wide format named. `mb_geometry` and the strided branch both
    // keep those kinds on the matvec, so there is no batched shape to build.
    if (g_.has_alt_quant()) {
        DecodeStepPsos alt{};
        if (!load_decode_psos(*ctx_, kernels_dir, alt, g_.alt_quant, &load_err,
                              DecodePsoFeatures{.routing_only = true})) {
            if (err) *err = "PSO load failed (second quantized set): " + load_err;
            ctx_.reset();
            return false;
        }
        psos_[Kernel::LlRouter] = alt[Kernel::LlRouter];
        psos_[Kernel::LlSharedGateProj] = alt[Kernel::LlSharedGateProj];
    }
    if (g_.paged_kv_enabled &&
        !load_multibatch_psos(
            *ctx_, kernels_dir, mb_psos_, g_.quant, &load_err,
            MultiBatchPsoFeatures{
                .sdpa_d256 = true,
                .gdn = true,
                .residual = fuse_residual_,
                .routed = g_.is_moe(),
                .strided = true,
                .fp16_strided = !g_.is_moe() &&
                    g_.quant.bits == 4 && g_.quant.group == 64})) {
        if (err) *err = "multi-batch PSO load failed: " + load_err;
        ctx_.reset();
        return false;
    }
    ptir_logits_copy_pso_ = ctx_->compile_ptir_pso_from_file(
        (std::filesystem::path(kernels_dir) / "ptir_logits_copy.metal").string(),
        "ptir_copy_logits_bf16",
        &load_err);
    ptir_logits_copy_params_ =
        ctx_->create_standalone_buffer(
            sizeof(PtirLogitsCopyParams) * kPtirLogitsCopyMaxRows);
    prefill_row_stride_ = ctx_->create_standalone_buffer(sizeof(std::int32_t));
    prefill_rows_ = ctx_->create_standalone_buffer(sizeof(std::int32_t));
    if (prefill_row_stride_.valid()) {
        *static_cast<std::int32_t*>(prefill_row_stride_.contents()) =
            static_cast<std::int32_t>(scratch_widest_elems(g_));
    }
    if (!g_.is_moe() && g_.quant.bits == 4 && g_.quant.group == 64) {
        prefill_fp16_input_ = ctx_->heap_alloc(
            std::size_t(std::max(1, g_.max_tokens)) *
            std::size_t(scratch_widest_elems(g_)) * sizeof(std::uint16_t));
        if (!prefill_fp16_input_.valid()) {
            if (err) *err = "allocating Qwen prefill FP16 input staging";
            ctx_.reset();
            return false;
        }
    }
    prefill_scan_rows_.clear();
    if (!ptir_logits_copy_pso_.valid() ||
        !ptir_logits_copy_params_.valid() ||
        !ensure_ptir_logits_rows(1, &load_err)) {
        if (err) *err = "PTIR logits staging setup failed: " + load_err;
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
        ctx_->ensure_elastic_buffer(gs.conv_state, gs.conv_state.size);
        ctx_->ensure_elastic_buffer(gs.conv_state_out, gs.conv_state_out.size);
        ctx_->ensure_elastic_buffer(
            gs.recurrent_state, gs.recurrent_state.size);
        ctx_->zero_buffer_range(gs.conv_state, 0, gs.conv_state.size);
        ctx_->zero_buffer_range(
            gs.conv_state_out, 0, gs.conv_state_out.size);
        ctx_->zero_buffer_range(
            gs.recurrent_state, 0, gs.recurrent_state.size);
    }
    for (auto& ks : b_.kv) {
        ctx_->ensure_elastic_buffer(ks.k_pages, ks.k_pages.size);
        ctx_->ensure_elastic_buffer(ks.v_pages, ks.v_pages.size);
        ctx_->zero_buffer_range(ks.k_pages, 0, ks.k_pages.size);
        ctx_->zero_buffer_range(ks.v_pages, 0, ks.v_pages.size);
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
    if (is_simple()) {
        // A PAGED family resets through the runtime's page table -- a fresh
        // sequence gets fresh pages -- exactly as qwen3.5 does. Zeroing here
        // would zero EVERY page, which with several sequences resident is not a
        // reset but the destruction of its neighbours.
        //
        // A ring-backed one has nowhere else to put the boundary: the KV is the
        // only thing it carries between tokens, and there is no conv history to
        // ping-pong and no recurrent state to clear.
        if (!simple_->paged()) simple_->reset();
        return;
    }
    const size_t conv_stride  = g_.gdn_conv_stride_bytes();
    const size_t recur_stride = g_.gdn_recurrent_stride_bytes();
    const size_t conv_off  = size_t(slot) * conv_stride;
    const size_t recur_off = size_t(slot) * recur_stride;
    for (auto& gs : b_.gdn) {
        ctx_->ensure_elastic_buffer(
            gs.conv_state, conv_off + conv_stride);
        ctx_->ensure_elastic_buffer(
            gs.conv_state_out, conv_off + conv_stride);
        ctx_->ensure_elastic_buffer(
            gs.recurrent_state, recur_off + recur_stride);
        ctx_->zero_buffer_range(
            gs.conv_state, conv_off, conv_stride);
        ctx_->zero_buffer_range(
            gs.conv_state_out, conv_off, conv_stride);
        ctx_->zero_buffer_range(
            gs.recurrent_state, recur_off, recur_stride);
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
        const size_t conv_extent =
            std::max(src_conv_off, dst_conv_off) + conv_stride;
        const size_t recur_extent =
            std::max(src_recur_off, dst_recur_off) + recur_stride;
        const bool ok =
            ctx_->ensure_elastic_buffer(gs.conv_state, conv_extent) &&
            ctx_->ensure_elastic_buffer(gs.conv_state_out, conv_extent) &&
            ctx_->ensure_elastic_buffer(gs.recurrent_state, recur_extent) &&
            ctx_->copy_buffer_range(
                gs.conv_state, dst_conv_off,
                gs.conv_state, src_conv_off, conv_stride) &&
            ctx_->copy_buffer_range(
                gs.conv_state_out, dst_conv_off,
                gs.conv_state_out, src_conv_off, conv_stride) &&
            ctx_->copy_buffer_range(
                gs.recurrent_state, dst_recur_off,
                gs.recurrent_state, src_recur_off, recur_stride);
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

/// Why a step did not succeed, in the GPU's own words when it has any.
///
/// Every caller used to say "timed out before its completion fence", which is
/// true of exactly one of the two ways a step fails. The other -- the command
/// buffer ran and came back with an error -- was reported as a timeout, and
/// the error itself was printed to stderr and dropped. A nineteen-gigabyte
/// model whose every fire failed out of memory produced a clean run, token
/// zero, and four passing checks.
std::string step_failure_reason(const StepTiming& timing) {
    if (timing.gpu_error) {
        return "the GPU rejected this command buffer: " +
               (timing.gpu_error_text.empty() ? std::string("no reason given")
                                              : timing.gpu_error_text);
    }
    return "Metal command timed out before its completion fence";
}

std::uint64_t rs_slot_bytes_for(const DecodeGeometry& g) {
    const std::uint64_t conv = g.gdn_conv_stride_bytes();
    const std::uint64_t recur = g.gdn_recurrent_stride_bytes();
    std::uint64_t gdn_layers = 0;
    for (int l = 0; l < g.n_layers; ++l) {
        if (!g.is_full_attn(l)) ++gdn_layers;
    }
    return gdn_layers * (2 * conv + recur);
}

std::uint64_t rs_slot_budget_bytes() {
    const std::uint64_t working_set =
        static_cast<std::uint64_t>(RawMetalContext::device_working_set_bytes());
    return std::max<std::uint64_t>(kRsSlotBudgetBytes, working_set / 10);
}

std::uint32_t rs_slots_for_budget(const DecodeGeometry& g, std::uint64_t budget_bytes,
                                  std::uint32_t requested_slots) {
    const std::uint64_t per_slot = rs_slot_bytes_for(g);
    // A geometry with no linear-attention layers has no state to slot; the
    // count is then whatever the caller needs, at no cost.
    if (per_slot == 0) return std::max<std::uint32_t>(requested_slots, 1);
    const std::uint64_t affordable = budget_bytes / per_slot;
    // `requested_slots` is a CEILING, not a floor. It used to be applied as
    // `max(slots, requested)`, which made `budget_bytes` decorative: the
    // request count is `kPagedMaxForwardRequests`, a constant, so every
    // checkpoint reserved 64 slots at whatever they cost. Qwen3.6-27B's slots
    // are 170 MiB, so that is 10.6 GiB of recurrent state -- which put the
    // model 5.2 GiB over this device and, at the batch sizes that did load,
    // returned `kIOGPUCommandBufferCallbackErrorOutOfMemory` from the command
    // queue. That arrived as a command buffer which never ran, so every PTIR
    // lane's status still held its zero fill, and the runtime read the zeros
    // back as every lane faulting -- deterministic above concurrency 4, and
    // naming neither memory nor the batch it came from.
    //
    // Reserving fewer slots than the driver ADVERTISES would hang rather than
    // queue, so the two are kept equal: `context.cpp` derives the advertised
    // `max_forward_requests` from this same call. Fewer admitted requests
    // queue; a device that is overrun does not.
    std::uint64_t slots = std::min<std::uint64_t>(affordable, kPhase1bRsSlots);
    slots = std::min<std::uint64_t>(
        slots, std::max<std::uint32_t>(requested_slots, 1));
    return std::uint32_t(std::max<std::uint64_t>(slots, 1));
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

bool MetalExecutor::Impl::ensure_elastic_storage(
    std::uint32_t kv_pages,
    std::uint32_t state_slots,
    std::uint32_t token_rows,
    std::uint32_t ring_tokens,
    std::string* err) {
    std::vector<std::pair<SlotHandle, size_t>> targets;
    auto add_target = [&](const SlotHandle& slot, size_t bytes) {
        if (!slot.valid() || !slot.elastic) return;
        targets.emplace_back(slot, std::min(bytes, slot.size));
    };

    if (ring_tokens != 0) {
        for (const auto& layer : b_.kv) {
            if (!layer.k_pages.valid()) continue;
            const size_t bytes =
                layer.k_pages.size *
                std::min<std::uint32_t>(ring_tokens, max_ctx_) /
                size_t(max_ctx_);
            add_target(layer.k_pages, bytes);
            add_target(layer.v_pages, bytes);
        }
    }
    if (kv_pool_.enabled && kv_pages != 0) {
        const size_t page_bytes =
            size_t(kv_pool_.page_size) *
            size_t(g_.n_kv_heads) * size_t(g_.head_dim) * 2u;
        const size_t bytes = size_t(kv_pages) * page_bytes;
        for (const auto& layer : kv_pool_.layers) {
            if (!layer.k_pages.valid()) continue;
            add_target(layer.k_pages, bytes);
            add_target(layer.v_pages, bytes);
        }
    }
    if (state_slots != 0) {
        const size_t conv_bytes =
            size_t(state_slots) * g_.gdn_conv_stride_bytes();
        const size_t recurrent_bytes =
            size_t(state_slots) * g_.gdn_recurrent_stride_bytes();
        for (const auto& layer : b_.gdn) {
            if (!layer.conv_state.valid()) continue;
            add_target(layer.conv_state, conv_bytes);
            add_target(layer.conv_state_out, conv_bytes);
            add_target(layer.recurrent_state, recurrent_bytes);
        }
    }
    const std::uint32_t rows = std::max<std::uint32_t>(1, token_rows);
    const std::uint32_t capacity =
        std::max<std::uint32_t>(1, g_.max_tokens);
    for (const auto& slot : pool_) {
        const size_t bytes =
            (slot.size * std::min(rows, capacity) + capacity - 1) /
            capacity;
        add_target(slot, bytes);
    }
    for (const auto& slot : b_.scratch) {
        if (!slot.valid()) continue;
        const size_t bytes =
            (slot.size * std::min(rows, capacity) + capacity - 1) /
            capacity;
        add_target(slot, bytes);
    }
    if (ctx_->ensure_elastic_buffers_atomically(targets)) return true;
    if (err != nullptr) {
        *err = "Metal elastic physical budget exhausted";
    }
    return false;
}

namespace {
// One NHD paged-pool row's byte size: [n_kv_heads, head_dim], bf16 (matches the M=1
// ring's activation dtype — kv_append.metal/kv_append_paged.metal both instantiate bf16).
size_t kv_pool_row_bytes(const DecodeGeometry& g) {
    return size_t(g.n_kv_heads) * size_t(g.head_dim) * 2u;
}
}  // namespace

bool MetalExecutor::Impl::setup_kv_pool(uint32_t total_pages, uint32_t page_size, std::string* err) {
    auto release_pool = [&](KvPagePool& candidate) {
        for (auto& layer : candidate.layers) {
            if (layer.k_pages.valid()) ctx_->release_elastic_buffer(layer.k_pages);
            if (layer.v_pages.valid()) ctx_->release_elastic_buffer(layer.v_pages);
        }
        candidate = {};
    };
    if (!ready()) {
        if (err) *err = "MetalExecutor::Impl::setup_kv_pool: decoder not initialized";
        return false;
    }
    if (total_pages == 0 || page_size == 0) {
        if (err) *err = "MetalExecutor::Impl::setup_kv_pool: total_pages and page_size must be > 0";
        return false;
    }
    if (!paged_pool_size_supported(g_, total_pages) ||
        page_size != static_cast<std::uint32_t>(g_.kv_page_size)) {
        if (err) {
            *err =
                "MetalExecutor::Impl::setup_kv_pool: pool geometry exceeds "
                "the fixed paged-IO allocation";
        }
        return false;
    }
    int n_full = 0;
    for (int L = 0; L < g_.n_layers; ++L) {
        if (g_.is_full_attn(L)) ++n_full;
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
        if (!g_.is_full_attn(L)) continue;
        const size_t initial = std::min(layer_bytes, size_t{2} << 20);
        pool.layers[size_t(L)].k_pages =
            ctx_->create_elastic_buffer(layer_bytes, initial);
        pool.layers[size_t(L)].v_pages =
            ctx_->create_elastic_buffer(layer_bytes, initial);
        if (!pool.layers[size_t(L)].k_pages.valid() || !pool.layers[size_t(L)].v_pages.valid()) {
            if (err) {
                *err = "MetalExecutor::Impl::setup_kv_pool: sparse arena allocation failed "
                       "(layer " + std::to_string(L) + ", " + std::to_string(layer_bytes) +
                       " bytes/buffer)";
            }
            release_pool(pool);
            return false;
        }
    }
    pool.total_pages = total_pages;
    pool.capacity_pages = total_pages;
    const size_t page_bytes = size_t(page_size) * kv_pool_row_bytes(g_);
    pool.committed_pages = static_cast<std::uint32_t>(
        std::min<size_t>(
            total_pages,
            (std::min(layer_bytes, size_t{2} << 20) + page_bytes - 1) /
                page_bytes));
    pool.page_size = page_size;
    pool.enabled = true;
    const std::uint64_t generation = paged_bind_generation_;
    const bool was_bound = mb_bound_;
    KvPagePool old_pool = std::move(kv_pool_);
    kv_pool_ = std::move(pool);
    if (!bind_paged_dag(err)) {
        KvPagePool failed = std::move(kv_pool_);
        kv_pool_ = std::move(old_pool);
        mb_bound_ = was_bound;
        if (kv_pool_.enabled) {
            std::string restore_error;
            if (!bind_paged_dag(&restore_error) && err != nullptr) {
                *err += "; restore failed: " + restore_error;
            }
            paged_bind_generation_ = generation;
        }
        release_pool(failed);
        return false;
    }
    release_pool(old_pool);
    return true;
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
            if (!g_.is_full_attn(L)) continue;
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
            bind_decode_consts(*ctx_, mb_dag_, g_, max_ctx_, gdn_prep_,
                               std::max(1, g_.max_tokens));
            mb_bound_tokens_ = std::max(1, g_.max_tokens);
            // Decode writes the shifted conv history straight back over the one
            // it read.  Safe because each channel is read and written by the
            // same thread, in that order, and prep and recurrent touch disjoint
            // channels -- which saves copying every slot's whole conv slab back
            // on the host after every single token.
            alias_decode_conv_state_out(*ctx_, b_, mb_dag_);
            prefill_scan_rows_.assign(prefill_dags_.size(), SlotHandle{});
            for (size_t t = 0; t < prefill_dags_.size(); ++t) {
                bind_scratch(*ctx_, prefill_dags_[t], prefill_sched_, pool_.data(),
                             int(pool_.size()), t * prefill_scratch_row);
                bind_decode_consts(*ctx_, prefill_dags_[t], g_, max_ctx_, gdn_prep_);
                // A scan launched off row t's argument table reads its own
                // length, so every row carries one.
                prefill_scan_rows_[t] = ctx_->create_standalone_buffer(sizeof(std::int32_t));
                if (!prefill_scan_rows_[t].valid() || !prefill_row_stride_.valid()) continue;
                for (const Dispatch& d : prefill_dags_[t]) {
                    if (d.kind == Kernel::GdnPrepSlotted) {
                        ctx_->arg_bind_ordinal(d.ordinal, 14, prefill_row_stride_);
                        ctx_->arg_bind_ordinal(d.ordinal, 15, prefill_scan_rows_[t]);
                    } else if (d.kind == Kernel::GdnCoreSlotted) {
                        ctx_->arg_bind_ordinal(d.ordinal, 12, prefill_row_stride_);
                        ctx_->arg_bind_ordinal(d.ordinal, 13, prefill_scan_rows_[t]);
                    }
                }
            }
            // The batched projection runs off token 0's argument tables -- they
            // already point at row 0 of every scratch tensor -- and walks the
            // rest of the prompt with this pitch.
            if (prefill_row_stride_.valid() && !prefill_dags_.empty()) {
                for (const Dispatch& d : prefill_dags_[0]) {
                    if (qmv_out_size(d.kind, g_) != 0) {
                        ctx_->arg_bind_ordinal(d.ordinal, 8, prefill_row_stride_);
                        if (prefill_rows_.valid())
                            ctx_->arg_bind_ordinal(d.ordinal, 9, prefill_rows_);
                        if (prefill_fp16_input_.valid())
                            ctx_->arg_bind_ordinal(d.ordinal, 12, prefill_fp16_input_);
                        continue;
                    }
                    // The row-blocked elementwise/norm kernels take the same pitch,
                    // at whichever index follows their own signature.
                    switch (d.kind) {
                        case Kernel::Rms:
                        case Kernel::FfnRms:
                        case Kernel::FinalRms:
                        case Kernel::SiluMul:
                            ctx_->arg_bind_ordinal(d.ordinal, 4, prefill_row_stride_);
                            break;
                        case Kernel::GatedRms:
                            ctx_->arg_bind_ordinal(d.ordinal, 5, prefill_row_stride_);
                            break;
                        case Kernel::GdnInA:
                        case Kernel::GdnInB:
                            ctx_->arg_bind_ordinal(d.ordinal, 5, prefill_row_stride_);
                            break;

                        default:
                            break;
                    }
                }
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
    if (!src_pages.empty()) {
        const std::uint32_t highest = std::max(
            *std::max_element(src_pages.begin(), src_pages.end()),
            *std::max_element(dst_pages.begin(), dst_pages.end()));
        if (!ensure_elastic_storage(
                highest + 1, 0, 0, 0, err)) {
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
        if (!g_.is_full_attn(L)) continue;
        const auto& lp = kv_pool_.layers[size_t(L)];
        for (size_t i = 0; i < src_pages.size(); ++i) {
            const size_t src_off = size_t(src_pages[i]) * page_bytes;
            const size_t dst_off = size_t(dst_pages[i]) * page_bytes;
            if (!ctx_->copy_buffer_range(
                    lp.k_pages, dst_off,
                    lp.k_pages, src_off, page_bytes) ||
                !ctx_->copy_buffer_range(
                    lp.v_pages, dst_off,
                    lp.v_pages, src_off, page_bytes)) {
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
    if (!cells.empty()) {
        std::uint32_t highest = 0;
        for (const auto& cell : cells) {
            highest = std::max(
                highest,
                std::max(cell.src_page_id, cell.dst_page_id));
        }
        if (!ensure_elastic_storage(
                highest + 1, 0, 0, 0, err)) {
            return false;
        }
    }
    const size_t row_bytes = kv_pool_row_bytes(g_);
    const size_t page_bytes = size_t(kv_pool_.page_size) * row_bytes;
    for (int L = 0; L < g_.n_layers; ++L) {
        if (!g_.is_full_attn(L)) continue;
        const auto& lp = kv_pool_.layers[size_t(L)];
        for (const auto& c : cells) {
            const size_t src_off = size_t(c.src_page_id) * page_bytes +
                                   size_t(c.src_token_offset) * row_bytes;
            const size_t dst_off = size_t(c.dst_page_id) * page_bytes +
                                   size_t(c.dst_token_offset) * row_bytes;
            if (!ctx_->copy_buffer_range(
                    lp.k_pages, dst_off,
                    lp.k_pages, src_off, row_bytes) ||
                !ctx_->copy_buffer_range(
                    lp.v_pages, dst_off,
                    lp.v_pages, src_off, row_bytes)) {
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
    if (new_total_pages == kv_pool_.committed_pages) return true;
    if (new_total_pages == 0) {
        if (err) *err = "MetalExecutor::Impl::resize_kv_pool: resize to 0 pages is not supported";
        return false;
    }
    if (!paged_pool_size_supported(g_, new_total_pages) ||
        new_total_pages > kv_pool_.capacity_pages) {
        if (err) {
            *err =
                "MetalExecutor::Impl::resize_kv_pool: growth exceeds the "
                "fixed paged-IO allocation";
        }
        return false;
    }
    if (new_total_pages < kv_pool_.committed_pages && !unmapped_tail_pages) {
        if (err) {
            *err = "MetalExecutor::Impl::resize_kv_pool: shrink would truncate pages [" +
                   std::to_string(new_total_pages) + ", " +
                   std::to_string(kv_pool_.committed_pages) +
                   ") that the caller has not attested are unmapped/free — refusing to "
                   "silently discard potentially-live pages";
        }
        return false;
    }
    const size_t row_bytes = kv_pool_row_bytes(g_);
    const size_t committed_bytes =
        size_t(new_total_pages) * size_t(kv_pool_.page_size) * row_bytes;
    for (int L = 0; L < g_.n_layers; ++L) {
        if (!g_.is_full_attn(L)) continue;
        auto& layer = kv_pool_.layers[size_t(L)];
        if (!ctx_->ensure_elastic_buffer(layer.k_pages, committed_bytes) ||
            !ctx_->ensure_elastic_buffer(layer.v_pages, committed_bytes)) {
            if (err) {
                *err =
                    "MetalExecutor::Impl::resize_kv_pool: shared elastic "
                    "budget exhausted";
            }
            return false;
        }
        ctx_->trim_elastic_buffer(layer.k_pages, committed_bytes);
        ctx_->trim_elastic_buffer(layer.v_pages, committed_bytes);
    }
    kv_pool_.committed_pages = new_total_pages;
    return true;
}

bool MetalExecutor::Impl::resize_elastic_pool(
    std::uint64_t pool_id,
    std::uint64_t target_pages,
    std::string* err) {
    std::vector<SlotHandle> slots;
    if (pool_id == 1) {
        for (const auto& layer : b_.gdn) {
            if (!layer.conv_state.valid()) continue;
            slots.push_back(layer.conv_state);
            slots.push_back(layer.conv_state_out);
            slots.push_back(layer.recurrent_state);
        }
    } else if (pool_id == 2) {
        slots.insert(slots.end(), pool_.begin(), pool_.end());
        for (const auto& slot : b_.scratch) {
            if (slot.valid()) slots.push_back(slot);
        }
    } else {
        if (err != nullptr) *err = "unknown elastic pool id";
        return false;
    }
    size_t capacity = 0;
    for (const auto& slot : slots) capacity += slot.size;
    const std::uint64_t page_bytes = ctx_->elastic_page_bytes();
    const std::uint64_t capacity_pages =
        page_bytes == 0
            ? 0
            : (capacity + page_bytes - 1) / page_bytes;
    const std::uint64_t requested =
        std::min(target_pages, capacity_pages) * page_bytes;
    const size_t target = static_cast<size_t>(
        std::min<std::uint64_t>(requested, capacity));
    for (const auto& slot : slots) {
        const size_t bytes = capacity == 0
            ? 0
            : (slot.size * target + capacity - 1) / capacity;
        if (!ctx_->ensure_elastic_buffer(slot, bytes) ||
            !ctx_->trim_elastic_buffer(slot, bytes)) {
            if (err != nullptr) {
                *err = "elastic pool resize exceeded the shared budget";
            }
            return false;
        }
    }
    return true;
}

StepTiming MetalExecutor::Impl::step(
    uint32_t token_id,
    uint32_t position,
    uint32_t slot,
    const PtirCommandCallbacks* ptir) {
    if (is_simple()) {
        // No elastic commit, no state slot, no ping-pong: the engine owns one
        // DAG and a contiguous KV, and `slot` has nothing to select.
        (void)slot;
        (void)ptir;
        const StepTiming t = simple_->step(*ctx_, token_id, position);
        // The walk, pointed at the ENGINE rather than at the raw path. Off
        // unless PIE_METAL_GOLDEN_DIR is set; when it is, the LAST step's
        // activations are what land there, so the caller chooses which step is
        // examined by choosing the prompt.
        if (golden_taps_enabled()) simple_->dump_taps(1);
        return t;
    }
    std::string commit_error;
    if (!ensure_elastic_storage(
            0,
            slot + 1,
            1,
            position + 1,
            &commit_error)) {
        throw std::runtime_error(commit_error);
    }
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
        [&](StepEncoder& se) {
            if (ptir != nullptr && ptir->pre_forward) {
                ptir->pre_forward(se);
            }
            encode_decode_step(se, dag_, psos_, force_barriers_);
            if (ptir != nullptr && ptir->post_forward) {
                ptir->post_forward(se);
            }
            // The staging copy rides this buffer too, for the same reason it
            // rides the batch paths': a second submission and a second fence
            // per token is most of a decode's host cost, and the destination
            // row is a bump allocation the caller made before the fire.
            std::string stage_err;
            if (!encode_logits_stage(se, &stage_err)) step_stage_error_ = stage_err;
        },
        sc & 1);
    ++sc;
    return t;
}

namespace {
// The step meter's `gap` covers everything between one forward's encode and the
// next's, and `run_batch_step` does a lane-wide preamble before it ever reaches
// that point.  These two marks split the gap into the caller's share
// (`forward_batch` composing the batch) and this function's own, which is what
// separated "the engine is late" from "the driver is busy" -- it is the driver.
std::chrono::steady_clock::time_point g_forward_batch_t0{};
}  // namespace

bool MetalExecutor::Impl::run_batch_step(const BatchSchedule& schedule, const BatchStepInputs& in,
                                     std::string* err,
                                     const std::vector<PtirCommandCallbacks>* ptir) {
    const auto fn_t0 = std::chrono::steady_clock::now();
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
    std::uint32_t highest_page = 0;
    if (!in.kv_page_indices.empty()) {
        highest_page = std::max(
            highest_page,
            *std::max_element(
                in.kv_page_indices.begin(),
                in.kv_page_indices.end()));
    }
    if (!in.w_page.empty()) {
        highest_page = std::max(
            highest_page,
            *std::max_element(in.w_page.begin(), in.w_page.end()));
    }
    const std::uint32_t highest_slot =
        in.rs_slot_ids.empty()
            ? 0
            : *std::max_element(
                  in.rs_slot_ids.begin(),
                  in.rs_slot_ids.end());
    std::string commit_error;
    if (!ensure_elastic_storage(
            highest_page + 1,
            highest_slot + 1,
            static_cast<std::uint32_t>(schedule.N),
            0,
            &commit_error)) {
        return fail(commit_error);
    }
    if (in.token_ids.size() != size_t(schedule.N) || in.position_ids.size() != size_t(schedule.N) ||
        in.qo_indptr.size() != size_t(schedule.R + 1) ||
        in.kv_page_indptr.size() != size_t(schedule.R + 1) ||
        in.kv_last_page_lens.size() != size_t(schedule.R) ||
        in.rs_slot_ids.size() != size_t(schedule.R) ||
        in.rs_slot_flags.size() != size_t(schedule.R) ||
        in.w_page.size() != size_t(schedule.N) || in.w_off.size() != size_t(schedule.N))
        return fail("inconsistent fixed IO vector sizes");
    const bool has_attention_mask =
        !in.attention_mask.empty() ||
        !in.attention_mask_enabled.empty();
    if (has_attention_mask &&
        (in.attention_mask_stride == 0 ||
         in.attention_mask_enabled.size() !=
             size_t(schedule.N) ||
         in.attention_mask.size() !=
             size_t(schedule.N) *
                 in.attention_mask_stride ||
         in.attention_mask.size() >
             b_.io[int(IoSlot::AttnMask)].size)) {
        return fail("inconsistent dense attention mask sizes");
    }
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
    if (has_attention_mask) {
        copy_to(IoSlot::AttnMask, in.attention_mask);
        copy_to(
            IoSlot::AttnMaskEnabled,
            in.attention_mask_enabled);
        write_u32(
            b_.io[int(IoSlot::AttnMaskStride)],
            in.attention_mask_stride);
    } else {
        std::memset(
            b_.io[int(IoSlot::AttnMaskEnabled)].contents(),
            0,
            size_t(schedule.N));
        write_u32(
            b_.io[int(IoSlot::AttnMaskStride)],
            static_cast<std::uint32_t>(
                b_.io[int(IoSlot::AttnMask)].size /
                std::max(schedule.N, 1)));
    }
    std::vector<uint32_t> seq_len(size_t(schedule.N));
    for (int t = 0; t < schedule.N; ++t)
        seq_len[size_t(t)] = schedule.spans[schedule.req_of_token[size_t(t)]].seqlen;
    copy_to(IoSlot::SeqLen, seq_len);

    const auto step_t0 = std::chrono::steady_clock::now();
    if (!schedule.is_pure_decode) return run_prefill_step(schedule, in, err, ptir);

    std::vector<uint32_t> active_slots;
    active_slots.reserve(size_t(schedule.R));
    for (const RequestSpan& sp : schedule.spans) {
        if (sp.rs_is_new) reset_state(sp.rs_slot);
        if (std::find(active_slots.begin(), active_slots.end(), sp.rs_slot) == active_slots.end())
            active_slots.push_back(sp.rs_slot);
    }
    // The paged decode shifts the conv history in place -- each channel of each
    // slot is read by exactly one thread before that same thread writes it, and
    // a pure-decode fire gives every request its own slot -- so there is nothing
    // to fold back afterwards.  The prefill still ping-pongs per prompt token,
    // so a slot handed over mid-parity is copied once, here.
    for (uint32_t slot : active_slots) {
        if (slot >= conv_in_out_.size() || conv_in_out_[slot] == 0) continue;
        conv_in_out_[slot] = 0;
        const size_t off = size_t(slot) * g_.gdn_conv_stride_bytes();
        // copy C -> A (different handles, same offset). A full-attention layer
        // has no GDN slab; skip it exactly as the commit below does.
        for (auto& gs : b_.gdn) {
            if (!gs.conv_state.valid() && !gs.conv_state_out.valid()) continue;
            if (!gs.conv_state.valid() ||
                !ctx_->copy_buffer_range(
                    gs.conv_state, off,
                    gs.conv_state_out, off,
                    g_.gdn_conv_stride_bytes()))
                return fail("failed to normalize GDN ping-pong state");
        }
    }

    // Alternate the two arms fire by fire so both see the same machine.
    static bool ab_flip = false;
    if (ab_enabled()) ab_set_arm(ab_flip = !ab_flip);
    std::string stage_err;
    bool stage_failed = false;
    const std::vector<Dispatch> fire_dag =
        build_decode_dag_mb(g_, schedule.N, kMultiBatchOrdinalBase, fuse_residual_, gdn_prep_);
    // The mixture's routing is the one constant this DAG's shape does not
    // carry: the sort is told how many (token, slot) pairs to place and how
    // many rows their tile-padded runs occupy, and a value bound at one width
    // and fired at another sorts pairs the router never wrote. Everything else
    // here is per-row and was bound once at setup.
    //
    // Rebound only when the width actually changes -- a serving loop that
    // settles at one batch size pays this once. It allocates nothing (const
    // slots are cached by (ordinal, index)) and moves no encoded byte, since
    // the argument table already holds the address whose contents change.
    if (g_.is_moe() && mb_bound_tokens_ != schedule.N) {
        bind_token_consts(*ctx_, fire_dag, g_, schedule.N);
        mb_bound_tokens_ = schedule.N;
    }
    const StepTiming timing = ctx_->run_step([&](StepEncoder& se) {
        if (ptir != nullptr) {
            for (const auto& callbacks : *ptir)
                if (callbacks.pre_forward) callbacks.pre_forward(se);
        }
        encode_decode_step_mb(se, fire_dag, psos_, mb_psos_, force_barriers_);
        if (ptir != nullptr) {
            for (const auto& callbacks : *ptir)
                if (callbacks.post_forward) callbacks.post_forward(se);
        }
        if (!encode_logits_stage(se, &stage_err)) stage_failed = true;
    });
    if (!timing.succeeded())
        return fail(step_failure_reason(timing));
    if (stage_failed) return fail(stage_err);
    // Step meter.  This machine is permanently contended (the agent process
    // alone runs at ~250% CPU), so wall-clock A/B swings 3x and cannot decide
    // anything; the command buffer's own execution time can.  Bucketed by lane
    // count, since the cost is strongly batch-dependent.
    //
    // `wall` is this function's own span, so `host` = wall - gpu - encode is the
    // driver's per-step CPU with nothing else folded in.  It reads 0.013ms at 32
    // lanes, which is how the search for a throughput gap was steered away from
    // the driver and into the engine.
    if constexpr (false) {
        static double sum[2][33] = {};
        static double wall[33] = {};
        static double enc[33] = {};
        // Gap between one step returning and the next arriving: the GPU is idle
        // for all of it, and nothing inside this driver can see it otherwise.
        static double gap[33] = {};
        // Bucketed, because the mean cannot tell "every step round-trips" from
        // "only frame boundaries do".
        static int gap_lt1[33] = {};
        static int gap_ge1[33] = {};
        static double gap_max[33] = {};
        static double pre_fb[33] = {};
        static double pre_fn[33] = {};
        static std::chrono::steady_clock::time_point last[33] = {};
        static int n[2][33] = {};
        const int lanes = schedule.N < 33 ? schedule.N : 32;
        const int arm = ab_enabled() && ab_arm() ? 1 : 0;
        sum[arm][lanes] += timing.gpu_exec_ms;
        const auto now_tp = std::chrono::steady_clock::now();
        wall[lanes] += std::chrono::duration<double, std::milli>(now_tp - step_t0).count();
        enc[lanes] += timing.encode_ms;
        pre_fb[lanes] +=
            std::chrono::duration<double, std::milli>(fn_t0 - g_forward_batch_t0).count();
        pre_fn[lanes] += std::chrono::duration<double, std::milli>(step_t0 - fn_t0).count();
        if (last[lanes].time_since_epoch().count() != 0) {
            const double g =
                std::chrono::duration<double, std::milli>(step_t0 - last[lanes]).count();
            gap[lanes] += g;
            if (g < 1.0) ++gap_lt1[lanes]; else ++gap_ge1[lanes];
            if (g > gap_max[lanes]) gap_max[lanes] = g;
        }
        last[lanes] = now_tp;
        if (++n[arm][lanes] % 128 == 0) {
            std::fprintf(stderr,
                         "[gpu] lanes=%d A n=%d %.4f | B n=%d %.4f | wall %.4f enc %.4f "
                         "host %.4f ms\n",
                         lanes, n[0][lanes],
                         n[0][lanes] ? sum[0][lanes] / n[0][lanes] : 0.0, n[1][lanes],
                         n[1][lanes] ? sum[1][lanes] / n[1][lanes] : 0.0,
                         wall[lanes] / (n[0][lanes] + n[1][lanes]),
                         enc[lanes] / (n[0][lanes] + n[1][lanes]),
                         (wall[lanes] - sum[0][lanes] - sum[1][lanes] - enc[lanes]) /
                             (n[0][lanes] + n[1][lanes]));
            std::fprintf(stderr,
                         "[gap] lanes=%d mean %.4f ms  <1ms=%d  >=1ms=%d  max %.2f ms"
                         "  | fb_pre %.4f  step_pre %.4f ms\n",
                         lanes, gap[lanes] / (n[0][lanes] + n[1][lanes]), gap_lt1[lanes],
                         gap_ge1[lanes], gap_max[lanes],
                         pre_fb[lanes] / (n[0][lanes] + n[1][lanes]),
                         pre_fn[lanes] / (n[0][lanes] + n[1][lanes]));
        }
    }

    for (uint32_t slot : schedule.slot_of_token) ++step_count_for(slot);
    return true;
}

bool MetalExecutor::Impl::run_prefill_step(
    const BatchSchedule& schedule,
    const BatchStepInputs& in,
    std::string* err,
    const std::vector<PtirCommandCallbacks>* ptir) {
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

    // The GDN scan replaces the per-token chain only when the prompt is one
    // request's (a shared recurrent slot) and its length is odd -- the conv
    // ping-pong alternates per token, so an odd count leaves the history in the
    // buffer token `scan_rows` reads, and an even prompt keeps one trailing
    // token on the per-token path.
    // One scan per request: the rows of a request are contiguous and share its
    // recurrent slot, so each is its own independent recurrence.  The conv
    // ping-pong alternates per row, so an odd scan length leaves the history
    // where the segment's trailing per-token dispatch expects it -- an even
    // segment keeps its last row on the per-token path.
    std::vector<GdnScanSegment> gdn_scans;
    if (prefill_row_stride_.valid() &&
        prefill_scan_rows_.size() >= size_t(schedule.N)) {
        int t = 0;
        while (t < schedule.N) {
            const uint32_t slot = schedule.slot_of_token[size_t(t)];
            int end = t;
            while (end < schedule.N && schedule.slot_of_token[size_t(end)] == slot) ++end;
            const int len = end - t;
            int rows = len % 2 == 1 ? len : len - 1;
            if (rows >= 3) {
                *static_cast<std::int32_t*>(prefill_scan_rows_[size_t(t)].contents()) =
                    static_cast<std::int32_t>(rows);
                gdn_scans.push_back(GdnScanSegment{t, rows});
            }
            t = end;
        }
    }
    // One command buffer, request-major token order.  Every complete layer DAG
    // ends in a barrier, so token t+1 observes token t's GDN and paged KV writes.
    // Alternate the arms fire by fire so both see the same machine, exactly as
    // the decode step does -- prefill fires are few, so without interleaving a
    // single contended window decides the answer.
    if (prefill_rows_.valid())
        *static_cast<std::int32_t*>(prefill_rows_.contents()) = schedule.N;
    static bool prefill_ab_flip = false;
    if (ab_enabled()) ab_set_arm(prefill_ab_flip = !prefill_ab_flip);
    // The staging copy rides this command buffer, exactly as it rides the
    // decode step's. It used to ride only the decode step's: a prefill fire
    // computed its logits, wrote them, and never copied the sampled row out, so
    // the caller read whatever the staging buffer held from the fire before --
    // zeros on the first, which reads back as a confident argmax of token 0.
    // Nothing failed and nothing was slow; the first token of every prompt was
    // simply the wrong one.
    std::string stage_err;
    bool stage_failed = false;
    const StepTiming timing = ctx_->run_step([&](StepEncoder& se) {
        if (ptir != nullptr) {
            for (const auto& callbacks : *ptir)
                if (callbacks.pre_forward) callbacks.pre_forward(se);
        }
        encode_prefill_dags_mb(se, prefill_dags_, schedule.N, psos_, mb_psos_,
                               force_barriers_, in.row_needs_logits, &g_,
                               int(prefill_dags_.size()), gdn_scans);
        if (ptir != nullptr) {
            for (const auto& callbacks : *ptir)
                if (callbacks.post_forward) callbacks.post_forward(se);
        }
        if (!encode_logits_stage(se, &stage_err)) stage_failed = true;
    });
    if (!timing.succeeded())
        return fail(step_failure_reason(timing));
    if (stage_failed) return fail(stage_err);
    // Prefill meter.  Prefill fires are few and long, so the per-row cost is what
    // compares across arms -- a raw total confuses "faster" with "shorter prompt".
    if constexpr (false) {
        static double ms[2] = {};
        static double rows[2] = {};
        static double enc[2] = {};
        static int n[2] = {};
        const int arm = ab_enabled() && ab_arm() ? 1 : 0;
        ms[arm] += timing.gpu_ms > 0.0 ? timing.gpu_ms : timing.gpu_exec_ms;
        enc[arm] += timing.encode_ms;
        rows[arm] += double(schedule.N);
        ++n[arm];
        std::fprintf(stderr,
                     "[prefill] A n=%d %.2f ms %.4f ms/row | B n=%d %.2f ms %.4f ms/row"
                     " | enc A %.2f B %.2f ms\n",
                     n[0], n[0] ? ms[0] / n[0] : 0.0, rows[0] > 0 ? ms[0] / rows[0] : 0.0,
                     n[1], n[1] ? ms[1] / n[1] : 0.0, rows[1] > 0 ? ms[1] / rows[1] : 0.0,
                     n[0] ? enc[0] / n[0] : 0.0, n[1] ? enc[1] / n[1] : 0.0);
    }
    if (golden_taps_enabled()) {
        // Every prefill token walks its own copy of the DAG, and bind_scratch lays
        // token t's row at t * (widest * sizeof(bf16)) inside each pool slot — so one
        // walk of dag[0]'s schedule reads every row of every tap.
        dump_golden_taps(prefill_dags_.front(), prefill_sched_, pool_.data(),
                         int(pool_.size()), g_, schedule.N,
                         size_t(scratch_widest_elems(g_)) * 2u);
        dump_golden_bf16("logits", logits_bf16(), schedule.N, g_.vocab,
                         size_t(g_.vocab));
        dump_golden_tokens(
            static_cast<const std::uint32_t*>(b_.io[int(IoSlot::TokenId)].contents()),
            schedule.N);
    }
    for (uint32_t slot : schedule.slot_of_token) ++step_count_for(slot);
    // The prompt's last token decides which half of the ping-pong holds the
    // history; the paged decode reads ConvState, so record an odd handover.
    if (conv_in_out_.size() < size_t(g_.max_slots))
        conv_in_out_.assign(size_t(g_.max_slots), 0);
    for (uint32_t slot : schedule.slot_of_token)
        conv_in_out_[slot] = std::uint8_t(step_count_for(slot) & 1);
    return true;
}

const uint16_t* MetalExecutor::Impl::logits_bf16() const {
    return static_cast<const uint16_t*>(b_.io[int(IoSlot::Logits)].contents());
}

bool MetalExecutor::Impl::ensure_ptir_logits_rows(
    std::uint32_t rows,
    std::string* error) {
    rows = std::max<std::uint32_t>(rows, 1);
    if (ptir_logits_.valid() && rows <= ptir_logits_capacity_rows_) return true;
    SlotHandle replacement = ctx_->create_standalone_buffer(
        static_cast<std::size_t>(rows) * g_.vocab * sizeof(std::uint16_t));
    if (!replacement.valid()) {
        if (error != nullptr) *error = "shared bf16 staging allocation failed";
        return false;
    }
    SlotHandle old = ptir_logits_;
    ptir_logits_ = replacement;
    ptir_logits_capacity_rows_ = rows;
    // The copy's SOURCE. `b_.io[Logits]` is qwen3.5's; a simple family's tail
    // writes a slot of its own, and rebinding qwen's here would leave the copy
    // reading an unallocated buffer -- which is silent, and reads downstream as
    // logits of exactly zero. Only reachable on GROWTH, so it survived every
    // single-row fire.
    ctx_->arg_bind_ordinal(kPtirLogitsCopyOrdinal, 0,
                           is_simple() ? simple_->logits_slot() : b_.io[int(IoSlot::Logits)]);
    ctx_->arg_bind_ordinal(kPtirLogitsCopyOrdinal, 1, ptir_logits_);
    ctx_->arg_bind_ordinal(
        kPtirLogitsCopyOrdinal, 2, ptir_logits_copy_params_);
    if (old.valid()) ctx_->release_standalone_buffer(old);
    return true;
}

std::uint32_t MetalExecutor::Impl::reserve_ptir_logits_rows(
    std::uint32_t rows) {
    const std::uint32_t base = ptir_logits_next_row_;
    ptir_logits_next_row_ += rows;
    return base;
}

bool MetalExecutor::Impl::encode_logits_stage(StepEncoder& encoder, std::string* error) {
    const auto& rows = pending_logits_stage_;
    if (rows.empty()) return true;
    if (!ptir_logits_copy_pso_.valid() || rows.size() > kPtirLogitsCopyMaxRows) {
        if (error != nullptr) *error = "PTIR logits staging is not ready";
        return false;
    }
    auto* params =
        static_cast<PtirLogitsCopyParams*>(ptir_logits_copy_params_.contents());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].second >= ptir_logits_capacity_rows_) {
            if (error != nullptr) *error = "PTIR logits staging is not ready";
            return false;
        }
        params[i] = {
            .source_row = rows[i].first,
            .destination_row = rows[i].second,
            .vocab = static_cast<std::uint32_t>(g_.vocab),
            .reserved = 0,
        };
    }
    // The copy reads the logits the forward just wrote.
    encoder.barrier();
    encoder.set_pso(ptir_logits_copy_pso_);
    encoder.set_argtable_ordinal(kPtirLogitsCopyOrdinal);
    encoder.dispatch(
        Grid{static_cast<std::uint32_t>(g_.vocab),
             static_cast<std::uint32_t>(rows.size()), 1},
        Threadgroup{256, 1, 1});
    return true;
}

void MetalExecutor::Impl::attach_ptir_logits_view(LogitsOut& output) const {
    output.device_buffer = ptir_logits_.buffer;
    output.device_contents = ptir_logits_.contents();
    output.device_gpu_address = ptir_logits_.gpu_address;
    output.device_bytes = ptir_logits_.size;
}

void MetalExecutor::Impl::copy_logits_f32(float* out) const {
    const uint16_t* lb = logits_bf16();
    copy_bf16_to_f32(lb, out, static_cast<std::size_t>(g_.vocab));
}

void MetalExecutor::Impl::copy_batch_logits_f32(uint32_t token_row, float* out) const {
    const uint16_t* lb = logits_bf16() + size_t(token_row) * size_t(g_.vocab);
    copy_bf16_to_f32(lb, out, static_cast<std::size_t>(g_.vocab));
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

bool MetalExecutor::setup(const SetupConfig& cfg, std::string* err) {
    // Which family is this? Answered from the config's own `model_type`, so a
    // checkpoint gets a diagnosis about ITSELF rather than about the family that
    // happens to be wired up.
    switch (model::model_family_of(cfg.model_type)) {
    case model::ModelFamily::Qwen35:
        break;
    // gemma4 and gpt-oss run through `SimpleFamilyEngine`: one DAG, contiguous
    // KV, no recurrent state. `MetalExecutor::Impl`'s own machinery is
    // qwen3.5's and stays untouched.
    case model::ModelFamily::Gemma4:
    case model::ModelFamily::GptOss:
    case model::ModelFamily::Llama:
        break;
    case model::ModelFamily::Unknown:
        if (err != nullptr) {
            *err = "Metal has no model family for config '" + cfg.arch_name +
                   "' (model_type '" + cfg.model_type + "')";
        }
        return false;
    }
    auto impl = std::make_unique<Impl>();
    // Compile for this device. The loader is told what Metal can do and answers
    // with a plan that stays inside it, so there is nothing to re-check here
    // (`loader/architecture.md` §9).
    pie_loader::LoadPlan load_plan;
    try {
        // `tie_word_embeddings` is the one config fact a contract needs and
        // cannot see: it decides whether the head is its own tensor or the
        // embedding table read a second time, and a contract only sees tensors.
        model::ContractFacts contract_facts;
        // Whichever family this is, its own reading of the flag. They are
        // separate fields because they are separate configs, and a single
        // shared one would have made the qwen3.5 half silently inherit
        // llama's default the moment a checkpoint of either kind loaded.
        contract_facts.tied_embeddings =
            model::model_family_of(cfg.model_type) == model::ModelFamily::Qwen35
                ? cfg.qwen35.tied_embeddings
                : cfg.llama.tied_embeddings;
        contract_facts.quant_bits = cfg.quant_bits;
        contract_facts.quant_group_size = cfg.quant_group_size;
        // Gemma4's KV sharing, which the author asks for and this path was
        // sending as zeros. A layer past `n_layers - num_kv_shared_layers`
        // attends the KV an earlier layer wrote, so the author declines to
        // declare its k/v/k_norm.
        //
        // Measured, so the change is not mistaken for a saving: the MLX
        // conversions we gate already ship only the KV-OWNING layers' k/v
        // (e2b has `self_attn.k_proj` for layers 0-14 and nothing for 15-34),
        // so the authored plan is byte-identical either way -- 1096 tensors
        // with the truthful 35/20 and with 0/0 -- and resident weights stay
        // 2.43 GiB. What this fixes is the request, not the plan: a conversion
        // that does ship the dead tensors would have them staged and never
        // bound, because the skip branch it is asking about was unreachable.
        if (model::model_family_of(cfg.model_type) == model::ModelFamily::Gemma4) {
            contract_facts.num_hidden_layers = std::uint32_t(cfg.gemma4.n_layers);
            contract_facts.num_kv_shared_layers =
                std::uint32_t(cfg.gemma4.num_kv_shared_layers);
        }
        // A serving boot forwards the document it was handed; a test harness
        // that set up `SetupConfig` by hand states the facts its family needs
        // and gets a synthesized one. The branch is on which of the two this
        // is, and nothing else.
        load_plan = compile_load_plan(
            cfg.snapshot_dir, metal_device_target(),
            cfg.descriptor_json.empty()
                ? descriptor_for_testing(cfg.model_type, contract_facts)
                : cfg.descriptor_json);
    } catch (const std::exception& error) {
        if (err != nullptr) {
            *err = std::string("LoadPlan compile failed: ") + error.what();
        }
        return false;
    }
    // The shape this checkpoint actually has, from its own config.
    //
    // This was `DecodeGeometry geom{}` -- default-constructed, with one preview
    // checkpoint's dimensions compiled in as the defaults. Every other member
    // of the family therefore ran at the wrong hidden size, layer count and
    // linear-attention geometry, and said nothing about it: the loader binds by
    // NAME, and a name carries no dimension to disagree with.
    DecodeGeometry geom{};
    const model::ModelFamily family = model::model_family_of(cfg.model_type);
    // Only this family's. `geometry_from_facts` reads `cfg.qwen35`, which a
    // llama or gemma4 checkpoint never fills, and it refuses an empty one --
    // correctly, but it was being asked unconditionally, so every simple
    // family's setup died on "qwen3.5 geometry: config carried no decoder
    // shape" before it ever reached `setup_simple`, which builds its own
    // geometry from its own facts.
    if (family == model::ModelFamily::Qwen35) {
        std::string gerr;
        if (cfg.quant_bits != 0) geom.quant.bits = cfg.quant_bits;
        if (cfg.quant_group_size != 0) geom.quant.group = cfg.quant_group_size;
        if (!geometry_from_facts(cfg.qwen35, geom, &gerr)) {
            if (err != nullptr) *err = gerr;
            return false;
        }
        // The routed FFN is built -- DAG, kernels, launch shapes, constants,
        // names, pool, and the per-step rebinding of the one constant the
        // batch width can change (`bind_token_consts`, below). There is no
        // refusal here any more: what a mixture still cannot have is a SHARED
        // expert, and `geometry_from_facts` above already refuses that from
        // the config, before any of this is built.

        // Whether the head is its own tensor is decided ONCE, by the contract,
        // and read back here rather than decided a second time -- see
        // `plan_ties_embeddings`.
        geom.tied_embeddings = plan_ties_embeddings(load_plan);
        // The routing projections' format, read off the checkpoint rather than
        // the config: mlx_lm's quantization predicate singles out tensors by
        // NAME, and `config.json` records only the model-wide choice beside a
        // list of exceptions this driver does not parse. Both routing weights
        // are read, and a disagreement between them is refused rather than
        // resolved -- there is one alternate pipeline table.
        {
            const auto view = load_plan.view();
            AffineFormat found{0, 0};
            bool conflict = false;
            for (std::size_t i = 0; i < view.tensors.len; ++i) {
                const auto& t = view.tensors.ptr[i];
                const std::string name(reinterpret_cast<const char*>(t.name.ptr), t.name.len);
                const bool routing = name.find("mlp.gate.weight") != std::string::npos ||
                                     name.find("mlp.shared_expert_gate.weight") != std::string::npos;
                if (!routing) continue;
                const AffineFormat f{int(t.quant_bits_per_element), int(t.quant_group_size)};
                if (f.bits == 0 || f.group == 0) continue;
                if (f.bits == geom.quant.bits && f.group == geom.quant.group) continue;
                if (found.bits != 0 && (found.bits != f.bits || found.group != f.group)) {
                    conflict = true;
                    break;
                }
                found = f;
            }
            if (conflict) {
                if (err != nullptr) {
                    *err = "qwen3.5: the router and the shared expert's gate are quantized "
                           "differently from each other, and this driver builds one "
                           "alternate pipeline table";
                }
                return false;
            }
            geom.alt_quant = found;
        }
    }
    // Phase 1b (review fix B): really allocate `kPhase1bRsSlots` resident
    // GDN conv+recurrent state slots — heap_layout.hpp's `plan_heap` sizes
    // the State region as `slots * per_slot_bytes` and heap_bind.cpp binds
    // the M=1 kernels at slot 0's (unchanged) base offset regardless of
    // slot count, so this only grows reserved-but-idle memory; it does not
    // change the sealed M=1 decode path's behavior. `copy_state` operates
    // truthfully over these slots (real memory, not aspirational).
    // As many slots as the budget buys for THIS checkpoint's state, not a
    // count chosen against a model whose slot was three times smaller. The
    // floor is the concurrency the caller asked for: a driver that reserves
    // fewer slots than it accepts requests hangs rather than queues.
    geom.max_slots = int(rs_slots_for_budget(
        geom, rs_slot_budget_bytes(),
        std::min(cfg.max_forward_requests, kPagedMaxForwardRequests)));
    // Bounded, actually allocated/bound multi-batch capacity.  The paged path
    // has no hidden ring fallback: every advertised row/request has an IO,
    // scratch, logits, slot-state, and CSR binding.
    geom.max_requests = static_cast<int>(
        std::min<std::uint32_t>(cfg.max_forward_requests, std::uint32_t(geom.max_slots)));
    geom.max_tokens = static_cast<int>(std::min(cfg.max_forward_tokens,
                                                kPagedMaxForwardTokensCeiling));
    geom.max_slots = std::max(geom.max_slots, geom.max_requests);
    geom.rope_theta = cfg.rope_theta;
    // rope_dims = 2*floor(0.5 * partial_rotary_factor * head_dim), matching
    // tests/mlx/model/qwen3_5.cpp. A factor >= 1 rotates the whole head.
    geom.rotary_dims =
        cfg.partial_rotary_factor < 1.0f
            ? std::max(2, 2 * int(std::floor(0.5f * cfg.partial_rotary_factor *
                                             float(geom.head_dim))))
            : geom.head_dim;
    geom.kv_page_size = static_cast<int>(cfg.kv_page_size);
    geom.total_pages = static_cast<int>(cfg.total_pages);
    // `total_pages` and `max_ctx_tokens` are two spellings of one capacity, and
    // only the simple families read the second: `setup_simple` derives
    // `total_pages = kv_max_ctx / page_size` for itself. The native path read
    // the first alone, so a caller that sized its ring in TOKENS -- which is
    // what the field is for, and what `llama_bench` does -- got zero pages, a
    // DAG built unpaged, and a pool left disabled. Nothing failed: setup
    // succeeded and `kv_pool_page_size()` returned zero to a caller with no
    // reason to expect it. One capacity, one derivation, in both halves.
    //
    // Scoped to this family on purpose. The simple engines compute their own
    // pool from `max_ctx` a few lines into `setup_simple`, and handing them a
    // second answer here would be two derivations of one fact again.
    if (family == model::ModelFamily::Qwen35 && geom.total_pages == 0 &&
        geom.kv_page_size > 0) {
        const std::uint32_t ctx =
            cfg.max_ctx_tokens > 0 ? cfg.max_ctx_tokens : kMetalMaxCtxTokens;
        const std::uint32_t ps = std::uint32_t(geom.kv_page_size);
        geom.total_pages = static_cast<int>((ctx + ps - 1) / ps);
    }
    geom.paged_kv_enabled = geom.total_pages > 0 && geom.kv_page_size > 0 &&
                            geom.max_tokens > 0 && geom.max_requests > 0;
    // The vocabulary is a property of the checkpoint, so take it from the
    // checkpoint. It used to be cross-checked against the hard-coded 248320 and
    // REFUSED on mismatch, which meant even another size of the same family
    // could not load. Every consumer reads `geom.vocab`; nothing else pinned it.
    if (cfg.vocab_size != 0) {
        geom.vocab = static_cast<int>(cfg.vocab_size);
    }
    std::string derr;
    if (family != model::ModelFamily::Qwen35) {
        if (!impl->setup_simple(family, cfg.kernels_dir, cfg, load_plan, &derr)) {
            if (err != nullptr) *err = "Metal forward setup failed: " + derr;
            return false;
        }
        impl_ = std::move(impl);
        vocab_ = static_cast<std::uint32_t>(impl_->vocab());
        slot_states_.clear();
        return true;
    }
    if (!impl->setup(
            cfg.kernels_dir,
            geom,
            load_plan,
            cfg.storage_page_size,
            cfg.stream_routed_experts,
            cfg.max_ctx_tokens,
            &derr)) {
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
    if (geom.total_pages > 0 && geom.kv_page_size > 0) {
        std::string pool_err;
        if (!impl->setup_kv_pool(std::uint32_t(geom.total_pages),
                                 std::uint32_t(geom.kv_page_size), &pool_err)) {
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

WeightBytes MetalExecutor::weight_bytes() const {
    if (!ready()) return {};
    // Two decoders live here -- the llama/qwen3 path with its own `BoundDecode`
    // and everything that runs through a `SimpleFamilyEngine` -- and both stage
    // the same kind of map. Asking each for its map rather than teaching each
    // to do the arithmetic keeps the counting rule in one place.
    if (const SimpleFamilyEngine* eng = impl_->simple_engine(); eng != nullptr) {
        return eng->weight_bytes();
    }
    return pie::metal::weight_bytes(impl_->b_.weights, impl_->g_.n_experts,
                                    impl_->g_.experts_per_token);
}

RawMetalContext* MetalExecutor::command_context() {
    return ready() ? impl_->ctx_.get() : nullptr;
}

SlotHandle MetalExecutor::logits_device_slot() const {
    if (!ready()) return SlotHandle{};
    // `b_.io[Logits]` is qwen3.5's. A simple family's tail writes a slot of its
    // own, and PTIR's device program samples from whatever this names -- so
    // naming an unallocated buffer is silent, and reads as logits of zero.
    return impl_->is_simple() ? impl_->simple_engine()->logits_slot()
                              : impl_->b_.io[int(IoSlot::Logits)];
}

std::uint32_t MetalExecutor::rs_slots() const {
    return ready() ? static_cast<std::uint32_t>(impl_->geometry().max_slots) : 0u;
}

std::uint64_t MetalExecutor::rs_slot_bytes() const {
    return ready() ? impl_->rs_slot_bytes() : 0u;
}

std::uint64_t MetalExecutor::elastic_page_bytes() const {
    return ready() ? impl_->ctx_->elastic_page_bytes() : 0u;
}

std::uint64_t MetalExecutor::elastic_budget_pages() const {
    return ready() ? impl_->ctx_->elastic_budget_pages() : 0u;
}

std::uint64_t MetalExecutor::elastic_committed_pages() const {
    return ready() ? impl_->ctx_->elastic_committed_pages() : 0u;
}

bool MetalExecutor::ensure_launch_storage(
    std::uint32_t kv_pages,
    std::uint32_t state_slots,
    std::uint32_t token_rows,
    std::string* error) {
    return ready() && impl_->ensure_elastic_storage(
        kv_pages, state_slots, token_rows, 0, error);
}

std::uint32_t MetalExecutor::kv_pool_total_pages() const {
    return ready() && impl_->kv_pool().enabled ? impl_->kv_pool().total_pages : 0u;
}

std::uint32_t MetalExecutor::kv_pool_committed_pages() const {
    return ready() && impl_->kv_pool().enabled
        ? impl_->kv_pool().committed_pages
        : 0u;
}

std::uint32_t MetalExecutor::kv_pool_page_size() const {
    return ready() && impl_->kv_pool().enabled ? impl_->kv_pool().page_size : 0u;
}

bool MetalExecutor::ensure_kv_pages(
    std::uint32_t pages,
    std::string* error) {
    if (ready() && impl_->is_simple()) {
        // Nothing to commit: the ring is allocated whole at setup. The demand is
        // still checked, against the capacity that ring actually has.
        if (pages > impl_->kv_pool().total_pages) {
            if (error != nullptr) {
                *error = "this family's KV ring holds " +
                         std::to_string(impl_->kv_pool().total_pages) +
                         " pages' worth of tokens, and this fire asked for " +
                         std::to_string(pages);
            }
            return false;
        }
        return true;
    }
    if (!ready() || !impl_->kv_pool().enabled) {
        if (error != nullptr) *error = "Metal KV pool is unavailable";
        return false;
    }
    if (pages > impl_->kv_pool().total_pages) {
        if (error != nullptr) *error = "Metal KV commit demand exceeds capacity";
        return false;
    }
    if (pages <= impl_->kv_pool().committed_pages) return true;
    return impl_->resize_kv_pool(pages, true, error);
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

bool MetalExecutor::resize_elastic_pool(
    std::uint64_t pool_id,
    std::uint64_t target_pages,
    std::string* err) {
    return ready() &&
        impl_->resize_elastic_pool(pool_id, target_pages, err);
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
    if (!ready()) {
        if (err != nullptr) *err = "Metal executor not initialized";
        return false;
    }
    impl_->ptir_logits_next_row_ = 0;
    const SimpleFamilyEngine* simple = impl_->is_simple() ? impl_->simple_engine() : nullptr;
    const bool device_greedy =
        simple != nullptr && simple->paged() && simple->greedy_tokens_slot().valid();
    if (!(desc.greedy_token_only && device_greedy) &&
        !impl_->ensure_ptir_logits_rows(
            static_cast<std::uint32_t>(desc.readout_local_indices.size()), err)) {
        return false;
    }
    const std::uint32_t slot = desc.has_rs_slot ? desc.rs_slot_id : 0u;
    const auto state = slot_states_.find(slot);
    // gemma4 and gpt-oss hold their KV as a contiguous per-layer ring indexed by
    // position, which is qwen3.5's M=1 fast path and not its paged one. The
    // engine still marks a fire `requires_paged` because the POOL is configured;
    // for these families that says nothing about how the KV is stored, and
    // `validate_linear_sequence_geometry` below is what actually enforces the
    // one-resident-sequence invariant the ring needs.
    if (impl_->is_simple()) {
        SimpleFamilyEngine* eng = impl_->simple_engine();
        if (eng != nullptr && eng->paged()) {
            // One member is a batch of one: the same fire, the same CSR.
            std::vector<LogitsOut> outs(1);
            std::vector<std::uint8_t> ok(1, 0);
            std::vector<std::string> errs(1);
            run_simple_batch_forward({desc}, outs, ok, errs);
            if (ok[0] != 0) {
                out = std::move(outs[0]);
                return true;
            }
            if (err != nullptr) *err = errs[0].empty() ? "Metal forward failed" : errs[0];
            return false;
        }
        if (desc.has_write_desc) {
            if (err != nullptr) {
                *err = "this family stores KV as a position-indexed ring, so an explicit "
                       "page-write descriptor has nothing to address";
            }
            return false;
        }
        std::string member_err;
        if (run_member_forward(desc, out, /*batch_serialized=*/false, &member_err, nullptr)) {
            return true;
        }
        // The ring holds one sequence. Say what would lift that, since the
        // generic message is qwen3.5's and names a phase rather than a gap.
        if (err != nullptr) {
            *err = member_err +
                   " [this family serves one sequence at a time: its KV is a "
                   "position-indexed ring, so a second resident sequence would clobber the "
                   "first]";
        }
        return false;
    }
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
    return run_member_forward(
        desc, out, /*batch_serialized=*/false, err, nullptr);
}

void MetalExecutor::forward_batch(const std::vector<MemberForwardDesc>& descs,
                                  std::vector<LogitsOut>& outs,
                                  std::vector<std::uint8_t>& success,
                                  std::vector<std::string>& errors,
                                  const std::vector<PtirCommandCallbacks>* ptir) {
    g_forward_batch_t0 = std::chrono::steady_clock::now();
    outs.assign(descs.size(), LogitsOut{});
    success.assign(descs.size(), 0);
    errors.assign(descs.size(), std::string{});
    if (!ready()) {
        for (auto& e : errors) e = "Metal executor not initialized";
        return;
    }
    if (ptir != nullptr && ptir->size() != descs.size()) {
        for (auto& e : errors) e = "PTIR callback/member count mismatch";
        return;
    }
    const SimpleFamilyEngine* simple = impl_->is_simple() ? impl_->simple_engine() : nullptr;
    const bool device_greedy =
        simple != nullptr && simple->paged() && simple->greedy_tokens_slot().valid();
    std::uint32_t total_readout_rows = 0;
    for (const auto& desc : descs) {
        if (!(desc.greedy_token_only && device_greedy && ptir == nullptr)) {
            total_readout_rows +=
                static_cast<std::uint32_t>(desc.readout_local_indices.size());
        }
    }
    std::string staging_error;
    if (!impl_->ensure_ptir_logits_rows(total_readout_rows, &staging_error)) {
        for (auto& e : errors) e = staging_error;
        return;
    }
    impl_->ptir_logits_next_row_ = 0;
    // One member at a time for the families on the ring: it holds exactly one
    // sequence's history, so a second member in the same fire would clobber the
    // first. Refused per member rather than per batch, which is what the caller
    // poisons on.
    if (impl_->is_simple()) {
        SimpleFamilyEngine* eng = impl_->simple_engine();
        if (eng != nullptr && eng->paged()) {
            run_simple_batch_forward(descs, outs, success, errors, ptir);
            for (std::size_t i = 0; i < descs.size(); ++i) {
                if (success[i] == 0 && !errors[i].empty()) {
                    std::cerr << "[pie-driver-metal] member forward rejected: " << errors[i]
                              << "\n";
                }
            }
            return;
        }
        if (descs.size() != 1) {
            for (auto& e : errors) {
                e = "this family's KV is a single-sequence ring, so a batch of " +
                    std::to_string(descs.size()) + " members cannot be forwarded together";
            }
            return;
        }
        std::string member_err;
        if (run_member_forward(descs[0], outs[0], /*batch_serialized=*/false, &member_err,
                               ptir != nullptr ? &(*ptir)[0] : nullptr)) {
            success[0] = 1;
        } else {
            // A rejected member becomes a poison epoch by the time the client
            // sees it, which says nothing about why. This is the only place the
            // reason exists.
            std::cerr << "[pie-driver-metal] member forward rejected: " << member_err << "\n";
            errors[0] = member_err;
        }
        return;
    }
    if (descs.size() == 1 && !descs[0].requires_paged && !descs[0].has_write_desc) {
        if (ptir != nullptr) {
            if ((*ptir)[0].set_logits_row)
                (*ptir)[0].set_logits_row(0);
            if ((*ptir)[0].finalize_group)
                (*ptir)[0].finalize_group();
        }
        std::string member_err;
        if (run_member_forward(
                descs[0],
                outs[0],
                /*batch_serialized=*/false,
                &member_err,
                ptir != nullptr ? &(*ptir)[0] : nullptr)) {
            success[0] = 1;
        }
        else errors[0] = std::move(member_err);
        return;
    }
    run_paged_batch_forward(descs, outs, success, errors, ptir);
}

bool MetalExecutor::run_paged_batch_forward(const std::vector<MemberForwardDesc>& descs,
                                            std::vector<LogitsOut>& outs,
                                            std::vector<std::uint8_t>& success,
                                            std::vector<std::string>& errors,
                                            const std::vector<PtirCommandCallbacks>* ptir) {
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
    const std::uint64_t mask_stride64 =
        paged_attention_mask_pitch_bytes(
            impl_->geometry());
    if (mask_stride64 == 0 ||
        mask_stride64 > std::numeric_limits<std::uint32_t>::max()) {
        for (auto& e : errors) {
            e = "paged attention mask stride exceeds u32";
        }
        return false;
    }
    in.attention_mask_stride =
        static_cast<std::uint32_t>(mask_stride64);
    // The dense mask is one byte per addressable KV token per row -- 131072 bytes
    // here -- so materializing it costs `rows * stride` of zero-fill plus the same
    // again copying it to the IO slot. `run_batch_step` treats a non-empty
    // `attention_mask_enabled` as "this batch is masked", so pushing a zero per
    // token made every batch pay both, for a buffer no kernel reads: 8.4 MB of
    // memory traffic per step at 32 lanes, growing linearly with lane count.
    // Nothing downstream needs either vector when no member carries a mask.
    bool any_attention_mask = false;
    for (const auto& d : descs) any_attention_mask = any_attention_mask || d.has_attention_mask;
    struct AcceptedRequest {
        std::size_t member = 0;
        std::uint32_t local_request = 0;
        bool write = false;
    };
    std::vector<AcceptedRequest> accepted_requests;
    std::vector<std::vector<std::uint32_t>> member_readout_rows(
        descs.size());
    std::vector<std::size_t> accepted_members;
    std::unordered_map<std::uint32_t, std::size_t> slot_owner;
    auto reject = [&](std::size_t i, const std::string& reason) { errors[i] = reason; };
    for (std::size_t i = 0; i < descs.size(); ++i) {
        const MemberForwardDesc& d = descs[i];
        if (d.token_ids.empty() || d.token_ids.size() != d.position_ids.size()) {
            reject(i, "paged forward token/position count mismatch or empty span");
            continue;
        }
        if (d.qo_indptr.size() < 2 ||
            d.qo_indptr.front() != 0 ||
            d.qo_indptr.back() != d.token_ids.size() ||
            d.kv_page_indptr.size() != d.qo_indptr.size() ||
            d.kv_page_indptr.front() != 0 ||
            d.kv_page_indptr.back() != d.kv_pages.size() ||
            d.kv_last_page_lens.size() + 1 !=
                d.qo_indptr.size() ||
            d.sampling_indptr.size() != d.qo_indptr.size() ||
            d.sampling_indptr.front() != 0 ||
            d.sampling_indptr.back() !=
                d.readout_local_indices.size()) {
            reject(i, "paged member CSR is malformed");
            continue;
        }
        std::string position_error;
        if (!validate_request_local_positions(
                d, &position_error)) {
            reject(i, position_error);
            continue;
        }
        const std::size_t request_count =
            d.qo_indptr.size() - 1;
        // A hybrid family's linear attention state is not in the KV pages. A
        // member that names no slot is asking a recurrent decoder to run with
        // no history and to write its own nowhere -- which is exactly what a
        // caller that never learned about slots sends, and exactly what used
        // to be accepted. It ran: every member fell to slot zero, the fire
        // computed each sequence on top of the last one's state, and the
        // answers came back confident and wrong. The two-member gate caught it
        // at a relative error of 0.5 only because someone went looking.
        if (rs_slots() > 0 && !d.has_rs_slot) {
            reject(i,
                   "this decoder carries recurrent state, so a paged member "
                   "must name the slot its state lives in");
            continue;
        }
        const bool legacy_single_rs =
            d.has_rs_slot && request_count == 1 &&
            d.request_rs_slot_ids.empty();
        if (d.has_rs_slot &&
            !legacy_single_rs &&
            (d.request_rs_slot_ids.size() != request_count ||
             d.request_rs_reset.size() != request_count ||
             d.request_rs_read.size() != request_count ||
             d.request_rs_write.size() != request_count)) {
            reject(
                i,
                "paged member recurrent-state bindings do not match its "
                "request count");
            continue;
        }
        if (d.has_write_desc &&
            (d.w_page.size() != d.token_ids.size() ||
             d.w_off.size() != d.token_ids.size())) {
            reject(
                i,
                "explicit w_page/w_off must have one entry per prompt token");
            continue;
        }
        if (d.structured_mask && !d.has_attention_mask) {
            reject(
                i,
                "structured attention mask has no dense Metal fallback");
            continue;
        }
        if (d.has_attention_mask &&
            (d.attention_mask_stride > in.attention_mask_stride ||
             d.attention_mask.size() !=
                 d.token_ids.size() *
                     static_cast<std::size_t>(
                         d.attention_mask_stride))) {
            reject(i, "dense attention mask shape is malformed");
            continue;
        }
        struct RequestSpan {
            std::uint32_t q0 = 0;
            std::uint32_t q1 = 0;
            std::uint32_t k0 = 0;
            std::uint32_t k1 = 0;
            std::uint32_t s0 = 0;
            std::uint32_t s1 = 0;
            std::uint32_t last = 0;
            std::uint32_t slot = 0;
            bool reset = false;
            bool read = false;
            bool write = false;
        };
        std::vector<RequestSpan> spans;
        spans.reserve(request_count);
        for (std::size_t request = 0; request < request_count;
             ++request) {
            RequestSpan span{
                .q0 = d.qo_indptr[request],
                .q1 = d.qo_indptr[request + 1],
                .k0 = d.kv_page_indptr[request],
                .k1 = d.kv_page_indptr[request + 1],
                .s0 = d.sampling_indptr[request],
                .s1 = d.sampling_indptr[request + 1],
                .last = d.kv_last_page_lens[request],
                .slot =
                    d.has_rs_slot
                        ? (legacy_single_rs
                               ? d.rs_slot_id
                               : d.request_rs_slot_ids[request])
                        : static_cast<std::uint32_t>(
                              accepted_requests.size() +
                              request),
                .reset =
                    d.has_rs_slot &&
                    (legacy_single_rs
                         ? d.rs_reset
                         : d.request_rs_reset[request] != 0),
                .read =
                    d.has_rs_slot &&
                    (legacy_single_rs
                         ? !d.rs_reset
                         : d.request_rs_read[request] != 0),
                .write =
                    d.has_rs_slot &&
                    (legacy_single_rs
                         ? true
                         : d.request_rs_write[request] != 0),
            };
            if (span.q1 <= span.q0 ||
                span.q1 > d.token_ids.size() ||
                span.k1 <= span.k0 ||
                span.k1 > d.kv_pages.size() ||
                span.s1 < span.s0 ||
                span.s1 >
                    d.readout_local_indices.size()) {
                reject(i, "paged request CSR span is invalid");
                break;
            }
            if (span.slot >= rs_slots()) {
                reject(i, "recurrent-state slot is out of range");
                break;
            }
            const std::uint64_t extent =
                std::uint64_t(span.k1 - span.k0 - 1) *
                    pool.page_size +
                span.last;
            if (span.last == 0 ||
                span.last > pool.page_size ||
                std::any_of(
                    d.position_ids.begin() + span.q0,
                    d.position_ids.begin() + span.q1,
                    [&](std::uint32_t position) {
                        return position >= extent;
                    })) {
                reject(
                    i,
                    "position is outside the request's paged KV extent");
                break;
            }
            if (std::any_of(
                    d.kv_pages.begin() + span.k0,
                    d.kv_pages.begin() + span.k1,
                    [&](std::uint32_t page) {
                        return page >= pool.total_pages;
                    })) {
                reject(i, "KV page id is outside the paged pool");
                break;
            }
            if (d.has_attention_mask &&
                d.attention_mask_stride < extent) {
                reject(
                    i,
                    "dense attention mask shape does not cover KV extent");
                break;
            }
            const std::uint32_t query_count =
                span.q1 - span.q0;
            for (std::uint32_t sample = span.s0;
                 sample < span.s1;
                 ++sample) {
                if (d.readout_local_indices[sample] >=
                    query_count) {
                    reject(
                        i,
                        "readout index exceeds its request token span");
                    break;
                }
            }
            if (!errors[i].empty()) break;
            if (d.has_rs_slot) {
                std::string state_error;
                if (!validate_paged_request_state(
                        slot_states_,
                        d,
                        request,
                        &state_error)) {
                    reject(i, state_error);
                    break;
                }
                if (slot_owner.find(span.slot) !=
                        slot_owner.end() ||
                    std::any_of(
                        spans.begin(),
                        spans.end(),
                        [&](const RequestSpan& prior) {
                            return prior.slot == span.slot;
                        })) {
                    reject(
                        i,
                        "two paged requests target the same recurrent-state slot in one fire");
                    break;
                }
            }
            spans.push_back(span);
        }
        if (!errors[i].empty()) continue;
        if (d.has_rs_slot) {
            for (const auto& span : spans) {
                slot_owner.emplace(span.slot, i);
            }
        }
        accepted_members.push_back(i);
        for (std::size_t request = 0; request < spans.size();
             ++request) {
            const RequestSpan& span = spans[request];
            const std::uint32_t token_base =
                static_cast<std::uint32_t>(
                    in.token_ids.size());
            in.qo_indptr.push_back(token_base);
            in.kv_page_indptr.push_back(
                static_cast<std::uint32_t>(
                    in.kv_page_indices.size()));
            in.token_ids.insert(
                in.token_ids.end(),
                d.token_ids.begin() + span.q0,
                d.token_ids.begin() + span.q1);
            in.position_ids.insert(
                in.position_ids.end(),
                d.position_ids.begin() + span.q0,
                d.position_ids.begin() + span.q1);
            in.kv_page_indices.insert(
                in.kv_page_indices.end(),
                d.kv_pages.begin() + span.k0,
                d.kv_pages.begin() + span.k1);
            in.kv_last_page_lens.push_back(span.last);
            in.rs_slot_ids.push_back(span.slot);
            in.rs_slot_flags.push_back(
                span.reset ? 1u : 0u);
            for (std::uint32_t token = span.q0;
                 token < span.q1;
                 ++token) {
                const std::uint32_t token_pos =
                    d.position_ids[token];
                const std::uint32_t csr_page =
                    d.kv_pages[
                        span.k0 +
                        token_pos / pool.page_size];
                in.w_page.push_back(
                    d.has_write_desc ? d.w_page[token]
                                     : csr_page);
                in.w_off.push_back(
                    d.has_write_desc
                        ? d.w_off[token]
                        : token_pos % pool.page_size);
                if (any_attention_mask) {
                    in.attention_mask_enabled.push_back(
                        d.has_attention_mask ? 1 : 0);
                    const std::size_t mask_base =
                        in.attention_mask.size();
                    in.attention_mask.resize(
                        mask_base + in.attention_mask_stride,
                        0);
                    if (d.has_attention_mask) {
                        const auto* source =
                            d.attention_mask.data() +
                            token * d.attention_mask_stride;
                        std::copy(
                            source,
                            source + d.attention_mask_stride,
                            in.attention_mask.begin() +
                                mask_base);
                    }
                }
            }
            for (std::uint32_t sample = span.s0;
                 sample < span.s1;
                 ++sample) {
                member_readout_rows[i].push_back(
                    token_base +
                    d.readout_local_indices[sample]);
            }
            accepted_requests.push_back({
                .member = i,
                .local_request =
                    static_cast<std::uint32_t>(request),
                .write = span.write,
            });
        }
    }
    if (in.token_ids.empty()) return false;
    in.row_needs_logits.assign(in.token_ids.size(), 0);
    for (const auto& rows : member_readout_rows)
        for (const std::uint32_t row : rows)
            if (row < in.row_needs_logits.size()) in.row_needs_logits[row] = 1;
    in.qo_indptr.push_back(static_cast<uint32_t>(in.token_ids.size()));
    in.kv_page_indptr.push_back(static_cast<uint32_t>(in.kv_page_indices.size()));

    const BatchSchedule schedule = build_batch_schedule(
        in.token_ids.data(), int(in.token_ids.size()), in.qo_indptr.data(),
        in.kv_page_indptr.data(), in.kv_last_page_lens.data(), in.rs_slot_ids.data(),
        in.rs_slot_flags.data(), int(in.qo_indptr.size()), int(pool.page_size));
    std::string batch_err;
    std::vector<PtirCommandCallbacks> compacted_callbacks;
    const std::vector<PtirCommandCallbacks>* dispatch_callbacks = ptir;
    if (ptir != nullptr) {
        compacted_callbacks = compact_ptir_member_callbacks(
            accepted_members,
            member_readout_rows,
            *ptir);
        dispatch_callbacks = &compacted_callbacks;
    }
    // Every member's rows go in ONE staging dispatch, and that dispatch rides
    // the forward's OWN command buffer. It used to be a second `run_step`:
    // another submit and another completion wait, per token, for a copy worth
    // about a microsecond of bandwidth. Nothing here depends on the forward's
    // result -- the destination rows are just a bump allocation -- so it can all
    // be decided first and encoded at the tail of the same buffer.
    impl_->pending_logits_stage_.clear();
    for (const std::size_t i : accepted_members) {
        LogitsOut& out = outs[i];
        out.vocab = vocab_;
        const auto& rows = member_readout_rows[i];
        out.rows = static_cast<uint32_t>(rows.size());
        out.device_row_offset =
            impl_->reserve_ptir_logits_rows(out.rows);
        impl_->attach_ptir_logits_view(out);
        if (ptir != nullptr && (*ptir)[i].consumes_logits_directly) continue;
        for (uint32_t row = 0; row < out.rows; ++row)
            impl_->pending_logits_stage_.emplace_back(
                rows[row], out.device_row_offset + row);
    }
    const bool ran = impl_->run_batch_step(
        schedule, in, &batch_err, dispatch_callbacks);
    impl_->pending_logits_stage_.clear();
    if (!ran) {
        // The only account of why a paged batch was refused. Without it the
        // caller sees a poison epoch, and any PTIR group riding this forward
        // reports lanes that never dispatched -- neither of which names the
        // reason. The simple-family path already prints its rejections.
        std::cerr << "[pie-driver-metal] paged batch forward rejected: "
                  << batch_err << "\n";
        for (const std::size_t member : accepted_members) {
            errors[member] = batch_err;
        }
        return false;
    }
    for (const std::size_t i : accepted_members) {
        if (!errors[i].empty()) continue;
        success[i] = 1;
    }
    for (const AcceptedRequest& request : accepted_requests) {
        if (!request.write || success[request.member] == 0) continue;
        const MemberForwardDesc& d = descs[request.member];
        commit_paged_request_state(
            slot_states_, d, request.local_request);
    }
    return true;
}

bool MetalExecutor::run_simple_batch_forward(const std::vector<MemberForwardDesc>& descs,
                                            std::vector<LogitsOut>& outs,
                                            std::vector<std::uint8_t>& success,
                                            std::vector<std::string>& errors,
                                            const std::vector<PtirCommandCallbacks>* ptir) {
    SimpleFamilyEngine* eng = impl_->simple_engine();
    if (eng == nullptr || !eng->paged()) {
        for (auto& e : errors) e = "this family has no paged batch path";
        return false;
    }
    const SlotHandle greedy_slot = eng->greedy_tokens_slot();
    const int page_size = eng->page_size();
    const std::uint32_t total_pages = std::uint32_t(eng->total_pages());

    // ── Build one fire out of every member that validates ──
    //
    // A member may carry several requests (its own CSR); each becomes a request
    // of the fire, so the fire's request ids are global and its `qo_indptr`
    // spans them all.
    SimpleFamilyEngine::FireCsr csr;
    csr.qo_indptr.push_back(0);
    csr.kv_page_indptr.push_back(0);
    struct Accepted {
        std::size_t member = 0;
        std::uint32_t row0 = 0;  // this member's first row in the fire
    };
    std::vector<Accepted> accepted;
    bool any_rejected = false;
    const auto reject = [&](std::size_t i, const std::string& why) {
        errors[i] = why;
        any_rejected = true;
    };

    for (std::size_t i = 0; i < descs.size(); ++i) {
        const MemberForwardDesc& d = descs[i];
        if (d.token_ids.empty() || d.token_ids.size() != d.position_ids.size()) {
            reject(i, "forward token/position count mismatch or empty span");
            continue;
        }
        if (d.qo_indptr.size() < 2 || d.qo_indptr.front() != 0 ||
            d.qo_indptr.back() != d.token_ids.size() ||
            d.kv_page_indptr.size() != d.qo_indptr.size() ||
            d.kv_page_indptr.front() != 0 ||
            d.kv_page_indptr.back() != d.kv_pages.size()) {
            reject(i, "member CSR is malformed");
            continue;
        }
        if (d.has_write_desc &&
            (d.w_page.size() != d.token_ids.size() || d.w_off.size() != d.token_ids.size())) {
            reject(i, "explicit w_page/w_off must have one entry per token");
            continue;
        }
        if (d.has_attention_mask || d.structured_mask) {
            reject(i, "this family's attention is bounded by position, and has no dense "
                      "mask port");
            continue;
        }
        bool bad = false;
        for (const std::uint32_t p : d.kv_pages) {
            if (p >= total_pages) {
                reject(i, "a KV page id (" + std::to_string(p) + ") is outside the pool's " +
                              std::to_string(total_pages) + " pages");
                bad = true;
                break;
            }
        }
        if (bad) continue;
        for (const std::uint32_t local : d.readout_local_indices) {
            if (local >= d.token_ids.size()) {
                reject(i, "readout index exceeds this fire's token span");
                bad = true;
                break;
            }
        }
        if (bad) continue;

        // Per-request: positions are absolute within the sequence, so the page
        // a token lands in is its position's page IN THIS REQUEST'S LIST.
        const std::size_t requests = d.qo_indptr.size() - 1;
        const std::uint32_t row0 = std::uint32_t(csr.token_ids.size());
        SimpleFamilyEngine::FireCsr add;
        for (std::size_t r = 0; r < requests && !bad; ++r) {
            const std::uint32_t q0 = d.qo_indptr[r], q1 = d.qo_indptr[r + 1];
            const std::uint32_t k0 = d.kv_page_indptr[r], k1 = d.kv_page_indptr[r + 1];
            const std::uint32_t req_id =
                std::uint32_t(csr.kv_page_indptr.size() - 1 + add.qo_indptr.size());
            for (std::uint32_t t = q0; t < q1; ++t) {
                const std::uint32_t pos = d.position_ids[t];
                const std::uint32_t page_ix = pos / std::uint32_t(page_size);
                if (k0 + page_ix >= k1) {
                    reject(i, "a token at position " + std::to_string(pos) +
                                  " has no page in its request's list");
                    bad = true;
                    break;
                }
                add.token_ids.push_back(d.token_ids[t]);
                add.position_ids.push_back(pos);
                add.req_of_token.push_back(req_id);
                add.w_page.push_back(d.has_write_desc ? d.w_page[t] : d.kv_pages[k0 + page_ix]);
                add.w_off.push_back(d.has_write_desc ? d.w_off[t]
                                                     : pos % std::uint32_t(page_size));
            }
            add.qo_indptr.push_back(std::uint32_t(add.token_ids.size()));
            for (std::uint32_t p = k0; p < k1; ++p) add.kv_page_indices.push_back(d.kv_pages[p]);
            add.kv_page_indptr.push_back(std::uint32_t(add.kv_page_indices.size()));
        }
        if (bad) continue;
        if (csr.token_ids.size() + add.token_ids.size() > std::size_t(eng->max_rows())) {
            reject(i, "this fire would exceed the driver's " +
                          std::to_string(eng->max_rows()) + "-row forward budget");
            continue;
        }
        // The tail is allocated per SAMPLED row, not per token, so it has a
        // bound of its own.
        // The staging copy rides the forward's own command buffer and is ONE
        // dispatch, so the sampled rows also have to fit the copy's parameter
        // buffer. Bounding acceptance here keeps that a rejection with a number
        // in it rather than a failure discovered mid-encode.
        const std::size_t sampled_cap =
            std::min<std::size_t>(std::size_t(eng->max_sampled_rows()), kPtirLogitsCopyMaxRows);
        if (csr.sample_rows.size() + d.readout_local_indices.size() > sampled_cap) {
            reject(i, "this fire would read more than the driver's " +
                          std::to_string(sampled_cap) + " logits rows");
            continue;
        }
        // `add`'s indptrs are this member's own cumulative counts, one entry
        // per request and no leading zero; the fire's carry a leading zero and
        // continue from where the previous member left off.
        const std::uint32_t token_base = csr.qo_indptr.back();
        const std::uint32_t page_base = csr.kv_page_indptr.back();
        for (std::size_t k = 0; k < add.qo_indptr.size(); ++k) {
            csr.qo_indptr.push_back(token_base + add.qo_indptr[k]);
            csr.kv_page_indptr.push_back(page_base + add.kv_page_indptr[k]);
        }
        csr.token_ids.insert(csr.token_ids.end(), add.token_ids.begin(), add.token_ids.end());
        csr.position_ids.insert(csr.position_ids.end(), add.position_ids.begin(),
                                add.position_ids.end());
        csr.req_of_token.insert(csr.req_of_token.end(), add.req_of_token.begin(),
                                add.req_of_token.end());
        csr.w_page.insert(csr.w_page.end(), add.w_page.begin(), add.w_page.end());
        csr.w_off.insert(csr.w_off.end(), add.w_off.begin(), add.w_off.end());
        csr.kv_page_indices.insert(csr.kv_page_indices.end(), add.kv_page_indices.begin(),
                                   add.kv_page_indices.end());
        // Only these rows reach the LM head. A prefill computes every row of the
        // body and reads one; the head is `hidden * vocab` per row, so this is
        // most of a prefill's cost and all of its logits memory.
        for (const std::uint32_t local : d.readout_local_indices) {
            csr.sample_rows.push_back(row0 + local);
        }
        if (d.greedy_token_only && greedy_slot.valid()) csr.run_argmax = true;
        accepted.push_back({i, row0});
    }
    if (accepted.empty()) return !any_rejected;

    // Reserve every accepted member's readout rows BEFORE the fire: the staging
    // copy runs after it, and the row a member's logits land in has to be
    // stable across the whole batch.
    for (const Accepted& a : accepted) {
        LogitsOut& out = outs[a.member];
        out.vocab = vocab_;
        out.rows = std::uint32_t(descs[a.member].readout_local_indices.size());
        if (!(descs[a.member].greedy_token_only && greedy_slot.valid() && ptir == nullptr)) {
            out.device_row_offset = impl_->reserve_ptir_logits_rows(out.rows);
            impl_->attach_ptir_logits_view(out);
        }
    }

    // PTIR is told which rows are whose, and its group is finalized, BEFORE the
    // fire: `post_forward` encodes a program whose descriptors those calls
    // wrote. Encoding first and describing after leaves the group's lanes
    // never dispatched -- which reports as `state=0`, a lane that did not run,
    // rather than as anything about ordering.
    std::vector<PtirCommandCallbacks> hooks;
    if (ptir != nullptr) {
        std::uint32_t ptir_sample = 0;
        for (const Accepted& a : accepted) {
            std::vector<std::uint32_t> rows;
            for (std::size_t r = 0; r < descs[a.member].readout_local_indices.size(); ++r) {
                // Sample indices, not token rows: the tail produced one row per
                // SAMPLED row, and PTIR reads the tail's buffer.
                rows.push_back(ptir_sample++);
            }
            PtirCommandCallbacks cb = (*ptir)[a.member];
            if (cb.set_logits_rows) cb.set_logits_rows(rows);
            if (cb.set_logits_row && rows.size() == 1) cb.set_logits_row(rows[0]);
            hooks.push_back(std::move(cb));
        }
        for (PtirCommandCallbacks& cb : hooks) {
            if (cb.finalize_group) cb.finalize_group();
        }
    }

    // The staging copy rides the forward's OWN command buffer, the way the
    // paged path already does it. It used to be a second `run_step` per fire:
    // another submit and another completion fence, at batch one once per token,
    // for a copy worth a microsecond of bandwidth. Nothing about it depends on
    // the forward's result -- the destination rows are a bump allocation made
    // above -- so it is all decided here and encoded at the tail of the buffer.
    impl_->pending_logits_stage_.clear();
    {
        std::uint32_t sample = 0;
        for (const Accepted& a : accepted) {
            const MemberForwardDesc& d = descs[a.member];
            const bool direct =
                (d.greedy_token_only && greedy_slot.valid() && ptir == nullptr) ||
                                (ptir != nullptr && (*ptir)[a.member].consumes_logits_directly);
            for (std::uint32_t r = 0; r < d.readout_local_indices.size(); ++r, ++sample) {
                if (direct) continue;
                impl_->pending_logits_stage_.push_back(
                    {sample, outs[a.member].device_row_offset + r});
            }
        }
    }

    // PTIR's device program is encoded into the SAME command buffer, before and
    // after the model, which is what makes a sampled token available without a
    // second submission.
    SimpleFamilyEngine::EncodeHook pre, post;
    const auto fire_t0 = std::chrono::steady_clock::now();
    std::string stage_err;
    if (!hooks.empty()) {
        pre = [&](StepEncoder& se) {
            for (const PtirCommandCallbacks& cb : hooks) {
                if (cb.pre_forward) cb.pre_forward(se);
            }
        };
    }
    post = [&](StepEncoder& se) {
        for (const PtirCommandCallbacks& cb : hooks) {
            if (cb.post_forward) cb.post_forward(se);
        }
        // A failure here leaves the buffer without the copy rather than
        // aborting the encode: the fire is already being built, and the
        // recorded error is what the members are failed with below.
        (void)impl_->encode_logits_stage(se, &stage_err);
    };
    const StepTiming timing = impl_->fire_simple(csr, pre, post);
    // Same meter as `run_batch_step`'s, on the path the simple families take.
    // Without it a question like "what bounds gpt-oss in a batch" can only be
    // answered from the outside, where the driver's time and the engine's are
    // added together.
    if constexpr (false) {
        static double gpu[33] = {};
        static double rep[33] = {};
        static double enc[33] = {};
        static double wall[33] = {};
        static int n[33] = {};
        const int rows = int(csr.token_ids.size());
        const int lanes = rows < 33 ? rows : 32;
        gpu[lanes] += timing.gpu_exec_ms;
        // What the GPU says it spent, as against `gpu_exec_ms`, which is
        // commit-to-fence measured by the host and so carries the wake-up. The
        // difference between the two is the only way to tell a slow kernel from
        // a slow round trip, and at batch one -- where a decode is a 3 ms fire
        // behind a fence the host has to be woken from -- that is most of the
        // question this meter exists to answer.
        rep[lanes] += timing.gpu_ms;
        enc[lanes] += timing.encode_ms;
        wall[lanes] +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - fire_t0)
                .count();
        if (++n[lanes] % 128 == 0) {
            std::fprintf(stderr,
                         "[gpu] lanes=%d n=%d gpu %.4f (reported %.4f) enc %.4f wall %.4f ms"
                         " | gpu/row %.4f ms\n",
                         lanes, n[lanes], gpu[lanes] / n[lanes], rep[lanes] / n[lanes],
                         enc[lanes] / n[lanes], wall[lanes] / n[lanes],
                         gpu[lanes] / n[lanes] / (rows > 0 ? rows : 1));
        }
    }
    if (!timing.succeeded()) {
        for (const Accepted& a : accepted) {
            errors[a.member] = "Metal forward timed out before its completion fence";
        }
        return false;
    }

    // The logits landed in the staging buffer inside the fire above.
    impl_->pending_logits_stage_.clear();
    if (!stage_err.empty()) {
        for (const Accepted& a : accepted) errors[a.member] = stage_err;
        return false;
    }

    if (greedy_slot.valid() && greedy_slot.contents() != nullptr) {
        const auto* tokens = static_cast<const std::uint32_t*>(greedy_slot.contents());
        std::uint32_t sample = 0;
        for (const Accepted& a : accepted) {
            LogitsOut& out = outs[a.member];
            if (descs[a.member].greedy_token_only) {
                out.greedy_contents = tokens;
                out.greedy_row_offset = sample;
            }
            sample += out.rows;
        }
    }

    for (const Accepted& a : accepted) {
        const MemberForwardDesc& d = descs[a.member];
        const std::uint32_t slot = d.has_rs_slot ? d.rs_slot_id : 0u;
        LinearSequenceState& state = slot_states_[slot];
        state.has_resident = true;
        state.resident_sequence_id = d.sequence_id;
        state.resident_slot = slot;
        state.resident_next_position = d.position_ids.back() + 1;
        state.resident_pages = d.kv_pages;
        // Paged, not ring-backed: several sequences are resident at once, each
        // in its own pages, so nothing here is exclusive.
        state.ring_backed = false;
        state.paged_backed = true;
        success[a.member] = 1;
    }
    return !any_rejected;
}

bool MetalExecutor::run_member_forward(const MemberForwardDesc& desc, LogitsOut& out,
                                       bool batch_serialized, std::string* err,
                                       const PtirCommandCallbacks* ptir) {
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
    out.device_row_offset = impl_->reserve_ptir_logits_rows(out.rows);
    impl_->attach_ptir_logits_view(out);

    for (std::size_t i = 0; i < desc.token_ids.size(); ++i) {
        PtirCommandCallbacks token_callbacks;
        const PtirCommandCallbacks* token_ptir = nullptr;
        if (ptir != nullptr) {
            if (i == 0) token_callbacks.pre_forward = ptir->pre_forward;
            if (i + 1 == desc.token_ids.size())
                token_callbacks.post_forward = ptir->post_forward;
            token_callbacks.consumes_logits_directly =
                ptir->consumes_logits_directly;
            token_ptir = &token_callbacks;
        }
        // Which of this token's rows -- at most one, since a step is one row
        // -- is read out, decided BEFORE the step so the copy can ride its
        // command buffer instead of costing a second submission and a second
        // fence. The step's row is always 0; only the destination varies.
        impl_->pending_logits_stage_.clear();
        if (ptir == nullptr || !ptir->consumes_logits_directly) {
            for (std::uint32_t r = 0; r < desc.readout_local_indices.size(); ++r) {
                if (desc.readout_local_indices[r] != static_cast<std::uint32_t>(i)) continue;
                impl_->pending_logits_stage_.push_back({0u, out.device_row_offset + r});
            }
        }
        impl_->step_stage_error_.clear();
        const StepTiming timing = impl_->step(
            desc.token_ids[i],
            desc.position_ids[i],
            slot,
            token_ptir);
        impl_->pending_logits_stage_.clear();
        if (!timing.succeeded()) {
            if (err != nullptr) {
                *err =
                    "Metal forward timed out before its completion fence";
            }
            return false;
        }
        if (!impl_->step_stage_error_.empty()) {
            if (err != nullptr) *err = impl_->step_stage_error_;
            return false;
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
std::uint64_t MetalExecutor::elastic_page_bytes() const { return 0; }
std::uint64_t MetalExecutor::elastic_budget_pages() const { return 0; }
std::uint64_t MetalExecutor::elastic_committed_pages() const { return 0; }

bool MetalExecutor::copy_state(std::uint32_t, std::uint32_t, std::string* err) {
    if (err != nullptr) *err = "Metal executor requires an Apple build";
    return false;
}

std::uint32_t MetalExecutor::kv_pool_total_pages() const { return 0; }
std::uint32_t MetalExecutor::kv_pool_committed_pages() const { return 0; }
std::uint32_t MetalExecutor::kv_pool_page_size() const { return 0; }
bool MetalExecutor::ensure_kv_pages(std::uint32_t, std::string* error) {
    if (error != nullptr) *error = "Metal executor requires an Apple build";
    return false;
}
bool MetalExecutor::ensure_launch_storage(
    std::uint32_t,
    std::uint32_t,
    std::uint32_t,
    std::string* error) {
    if (error != nullptr) *error = "Metal executor requires an Apple build";
    return false;
}

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
bool MetalExecutor::resize_elastic_pool(
    std::uint64_t,
    std::uint64_t,
    std::string* err) {
    if (err != nullptr) *err = "Metal executor requires an Apple build";
    return false;
}

bool MetalExecutor::forward(const MemberForwardDesc&, LogitsOut&, std::string* err) {
    if (err != nullptr) *err = "Metal executor requires an Apple build";
    return false;
}

bool MetalExecutor::run_member_forward(
    const MemberForwardDesc&,
    LogitsOut&,
    bool,
    std::string* err,
    const PtirCommandCallbacks*) {
    if (err != nullptr) *err = "Metal executor requires an Apple build";
    return false;
}

void MetalExecutor::forward_batch(const std::vector<MemberForwardDesc>& descs,
                                  std::vector<LogitsOut>& outs,
                                  std::vector<std::uint8_t>& success,
                                  std::vector<std::string>& errors,
                                  const std::vector<PtirCommandCallbacks>*) {
    outs.assign(descs.size(), LogitsOut{});
    success.assign(descs.size(), 0);
    errors.assign(descs.size(), std::string("Metal executor requires an Apple build"));
}

RawMetalContext* MetalExecutor::command_context() { return nullptr; }
SlotHandle MetalExecutor::logits_device_slot() const { return {}; }

#endif

}  // namespace pie::metal::batch

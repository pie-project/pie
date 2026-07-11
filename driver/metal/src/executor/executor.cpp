#include "executor.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>

#if defined(__APPLE__)
#include "decoder.hpp"
#endif

namespace pie_metal_driver::executor {

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
    raw_metal::RawMetalDecoder decoder;
};

MetalExecutor::MetalExecutor() = default;
MetalExecutor::~MetalExecutor() = default;

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
    raw_metal::DecodeGeometry geom{};  // shipped qwen3.6 defaults
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
    if (!impl->decoder.setup(cfg.checkpoint_dir, cfg.kernels_dir, geom, &derr)) {
        if (err != nullptr) *err = "RawMetalDecoder setup failed: " + derr;
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
        if (!impl->decoder.setup_kv_pool(cfg.total_pages, cfg.kv_page_size, &pool_err)) {
            std::cerr << "[pie-driver-metal] MetalExecutor::setup: KV page pool allocation "
                         "failed, copy_kv/resize_pool will be UNSUPPORTED: "
                      << pool_err << "\n";
        }
    }
    impl_ = std::move(impl);
    vocab_ = static_cast<std::uint32_t>(impl_->decoder.vocab());
    slot_states_.clear();
    return true;
}

bool MetalExecutor::ready() const { return impl_ != nullptr && impl_->decoder.ready(); }

std::uint32_t MetalExecutor::vocab() const { return vocab_; }

std::uint32_t MetalExecutor::rs_slots() const {
    return ready() ? static_cast<std::uint32_t>(impl_->decoder.geometry().max_slots) : 0u;
}

std::uint64_t MetalExecutor::rs_slot_bytes() const {
    return ready() ? impl_->decoder.rs_slot_bytes() : 0u;
}

std::uint32_t MetalExecutor::kv_pool_total_pages() const {
    return ready() && impl_->decoder.kv_pool().enabled ? impl_->decoder.kv_pool().total_pages : 0u;
}

std::uint32_t MetalExecutor::kv_pool_page_size() const {
    return ready() && impl_->decoder.kv_pool().enabled ? impl_->decoder.kv_pool().page_size : 0u;
}

bool MetalExecutor::copy_kv_pages(const std::vector<std::uint32_t>& src_pages,
                                  const std::vector<std::uint32_t>& dst_pages, std::string* err) {
    if (!ready()) {
        if (err != nullptr) *err = "Metal executor not initialized";
        return false;
    }
    return impl_->decoder.copy_kv_pages(src_pages, dst_pages, err);
}

bool MetalExecutor::copy_kv_cells(const std::vector<KvMoveCell>& cells, std::string* err) {
    if (!ready()) {
        if (err != nullptr) *err = "Metal executor not initialized";
        return false;
    }
    std::vector<raw_metal::RawMetalDecoder::KvMoveCell> mapped;
    mapped.reserve(cells.size());
    for (const auto& c : cells) {
        mapped.push_back({c.dst_page_id, c.dst_token_offset, c.src_page_id, c.src_token_offset});
    }
    return impl_->decoder.copy_kv_cells(mapped, err);
}

bool MetalExecutor::resize_kv_pool(std::uint32_t new_total_pages, bool unmapped_tail_pages,
                                   std::string* err) {
    if (!ready()) {
        if (err != nullptr) *err = "Metal executor not initialized";
        return false;
    }
    return impl_->decoder.resize_kv_pool(new_total_pages, unmapped_tail_pages, err);
}

bool MetalExecutor::copy_state(std::uint32_t src_slot, std::uint32_t dst_slot, std::string* err) {
    if (!ready()) {
        if (err != nullptr) *err = "Metal executor not initialized";
        return false;
    }
    if (!impl_->decoder.copy_state_slot(src_slot, dst_slot, err)) return false;
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
    const auto& pool = impl_->decoder.kv_pool();
    if (!pool.enabled) {
        for (auto& e : errors) e = "paged KV pool is not allocated";
        return false;
    }

    raw_metal::BatchStepInputs in;
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

    const raw_metal::BatchSchedule schedule = raw_metal::build_batch_schedule(
        in.token_ids.data(), int(in.token_ids.size()), in.qo_indptr.data(),
        in.kv_page_indptr.data(), in.kv_last_page_lens.data(), in.rs_slot_ids.data(),
        in.rs_slot_flags.data(), int(in.qo_indptr.size()), int(pool.page_size));
    std::string batch_err;
    if (!impl_->decoder.run_batch_step(schedule, in, &batch_err)) {
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
            impl_->decoder.copy_batch_logits_f32(
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
    if (is_fresh) impl_->decoder.reset_state(slot);

    out.vocab = vocab_;
    out.rows = static_cast<std::uint32_t>(desc.readout_local_indices.size());
    out.data.assign(static_cast<std::size_t>(out.rows) * out.vocab, 0.0f);

    for (std::size_t i = 0; i < desc.token_ids.size(); ++i) {
        impl_->decoder.step(desc.token_ids[i], desc.position_ids[i], slot);
        for (std::uint32_t r = 0; r < desc.readout_local_indices.size(); ++r) {
            if (desc.readout_local_indices[r] != static_cast<std::uint32_t>(i)) continue;
            impl_->decoder.copy_logits_f32(out.data.data() +
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

// Linux/CI stub build: the direct-ABI surface still validates (entry.cpp,
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

}  // namespace pie_metal_driver::executor

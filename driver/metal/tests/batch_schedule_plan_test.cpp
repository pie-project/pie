// Phase 3 (metal_ptir_plan.md §7) batch-scheduler gate — pure host unit test,
// no Metal/Apple/checkpoint dependency (`plan_batch_execution` is defined
// outside the `#if defined(__APPLE__)` split in executor.cpp, so this binary
// always builds and runs on every platform).
//
// A launch batch of forward-needing members shares the ONE M=1 KV ring. The
// scheduler decides which members the single-ring serial path can serve and in
// what order:
//   * the (at most one) member that CONTINUES the currently ring-backed
//     sequence runs first (before any fresh member clobbers the ring);
//   * every fresh member (reset+replay from position 0) is always serviceable
//     and runs after the leading continuation, in input order;
//   * a continuation of a NON-ring-backed sequence, a 2nd continuation, or an
//     explicit per-token KV write descriptor is honestly gated (needs the
//     per-request paged-KV path when exercising the legacy M=1 planner) — marked not-ok
//     with a precise reason instead of silently corrupting ring state.

#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#include "executor/executor.hpp"

using pie_metal_driver::executor::BatchExecPlan;
using pie_metal_driver::executor::LinearSequenceState;
using pie_metal_driver::executor::MemberForwardDesc;
using pie_metal_driver::executor::plan_batch_execution;

namespace {

int g_pass = 0, g_fail = 0;
void expect(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
}
bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

constexpr std::uint64_t kSeqA = 1001;
constexpr std::uint64_t kSeqB = 2002;
constexpr std::uint64_t kSeqC = 3003;

// A fresh member on `slot` for `seq` (rs_reset), single token at position 0.
MemberForwardDesc fresh(std::uint64_t seq, std::uint32_t slot, std::uint32_t page) {
    MemberForwardDesc d;
    d.sequence_id = seq;
    d.token_ids = {7};
    d.position_ids = {0};
    d.has_rs_slot = true;
    d.rs_slot_id = slot;
    d.rs_reset = true;
    d.kv_pages = {page};
    d.readout_local_indices = {0};
    return d;
}

// A continuation member on `slot` for `seq` at `pos`.
MemberForwardDesc cont(std::uint64_t seq, std::uint32_t slot, std::uint32_t pos,
                       std::uint32_t page) {
    MemberForwardDesc d;
    d.sequence_id = seq;
    d.token_ids = {9};
    d.position_ids = {pos};
    d.has_rs_slot = true;
    d.rs_slot_id = slot;
    d.rs_reset = false;
    d.kv_pages = {page};
    d.readout_local_indices = {0};
    return d;
}

// A per-slot residency map where `slot` is ring-backed for `seq`.
std::unordered_map<std::uint32_t, LinearSequenceState> ring_backed(std::uint32_t slot,
                                                                   std::uint64_t seq,
                                                                   std::uint32_t next_pos,
                                                                   std::uint32_t page) {
    LinearSequenceState st;
    st.has_resident = true;
    st.resident_sequence_id = seq;
    st.resident_slot = slot;
    st.resident_next_position = next_pos;
    st.resident_pages = {page};
    st.ring_backed = true;
    std::unordered_map<std::uint32_t, LinearSequenceState> m;
    m[slot] = st;
    return m;
}

}  // namespace

int main() {
    std::printf("[plan_batch_execution]\n");

    // Empty batch — empty order, no members.
    {
        const BatchExecPlan p = plan_batch_execution({}, {});
        expect(p.order.empty() && p.member_ok.empty(), "empty batch plans nothing");
    }

    // Single fresh member, nothing resident — serviceable, order [0].
    {
        std::vector<MemberForwardDesc> d = {fresh(kSeqA, 0, 5)};
        const BatchExecPlan p = plan_batch_execution({}, d);
        expect(p.order.size() == 1 && p.order[0] == 0 && p.member_ok[0] == 1,
               "single fresh member serviceable");
    }

    // Three fresh members (distinct slots), nothing resident — all serviceable,
    // order preserved (the case the OLD per-member loop failed on the 2nd+).
    {
        std::vector<MemberForwardDesc> d = {fresh(kSeqA, 0, 5), fresh(kSeqB, 1, 6),
                                            fresh(kSeqC, 2, 7)};
        const BatchExecPlan p = plan_batch_execution({}, d);
        const bool all_ok = p.member_ok[0] && p.member_ok[1] && p.member_ok[2];
        const bool ordered = p.order.size() == 3 && p.order[0] == 0 && p.order[1] == 1 &&
                             p.order[2] == 2;
        expect(all_ok && ordered, "three fresh members all serviceable, input order preserved");
    }

    // Leading continuation of the ring-backed sequence, then a fresh sibling —
    // the continuation must run FIRST (before the fresh member clobbers the
    // ring), so order is [continuation, fresh] regardless of input order.
    {
        auto slots = ring_backed(0, kSeqA, /*next_pos=*/3, /*page=*/5);
        std::vector<MemberForwardDesc> d = {fresh(kSeqB, 1, 6), cont(kSeqA, 0, 3, 5)};
        const BatchExecPlan p = plan_batch_execution(slots, d);
        const bool both_ok = p.member_ok[0] && p.member_ok[1];
        const bool leads = p.order.size() == 2 && p.order[0] == 1 && p.order[1] == 0;
        expect(both_ok && leads, "ring-backed continuation leads, fresh sibling follows");
    }

    // Continuation of a sequence with nothing resident — gated (needs a fresh
    // prefill / the paged path), with a precise reason. A fresh sibling (no
    // conflicting resident sequence) still runs.
    {
        std::vector<MemberForwardDesc> d = {cont(kSeqB, 1, 4, 6), fresh(kSeqC, 2, 7)};
        const BatchExecPlan p = plan_batch_execution({}, d);
        expect(p.member_ok[0] == 0 && contains(p.member_reason[0], "paged-KV path"),
               "continuation with nothing resident is gated (" + p.member_reason[0] + ")");
        expect(p.member_ok[1] == 1, "fresh sibling of a gated continuation still serviceable");
    }

    // Two continuations of DIFFERENT sequences: only the ring-backed one (A)
    // leads; the other (B) is gated. Concurrent multi-sequence decode is the
    // honest paged-path boundary.
    {
        auto slots = ring_backed(0, kSeqA, 3, 5);
        std::vector<MemberForwardDesc> d = {cont(kSeqA, 0, 3, 5), cont(kSeqB, 1, 9, 6)};
        const BatchExecPlan p = plan_batch_execution(slots, d);
        expect(p.member_ok[0] == 1 && p.order.size() == 1 && p.order[0] == 0,
               "the ring-backed continuation (A) is served and leads");
        expect(p.member_ok[1] == 0 && contains(p.member_reason[1], "at most one continuation"),
               "the 2nd, non-ring-backed continuation (B) is gated (" + p.member_reason[1] + ")");
    }

    // Continuation with nothing resident at all — gated with the "no resident
    // KV" reason (not the "at most one continuation" reason).
    {
        std::vector<MemberForwardDesc> d = {cont(kSeqA, 0, 4, 5)};
        const BatchExecPlan p = plan_batch_execution({}, d);
        expect(p.member_ok[0] == 0 && contains(p.member_reason[0], "no resident KV") &&
                   p.order.empty(),
               "continuation with nothing resident is gated (" + p.member_reason[0] + ")");
    }

    // Explicit per-token KV write descriptor is no longer dropped/gated: the
    // real paged path consumes it.  The legacy ring scheduler treats this
    // fresh member like any other serviceable fresh member; forward_batch
    // routes it to the paged encoder before this planner is used.
    {
        MemberForwardDesc w = fresh(kSeqA, 0, 5);
        w.has_write_desc = true;
        w.w_page = {5};
        w.w_off = {0};
        std::vector<MemberForwardDesc> d = {w};
        const BatchExecPlan p = plan_batch_execution({}, d);
        expect(p.member_ok[0] == 1 && p.order.size() == 1 && p.order[0] == 0,
               "explicit KV write descriptor is accepted for the paged encoder");
    }

    // A non-hybrid (no rs_slot) member is "fresh" iff position 0 — a position-0
    // member with nothing resident is serviceable; a non-zero position with
    // nothing resident is a continuation of nothing → gated.
    {
        MemberForwardDesc first;
        first.sequence_id = kSeqA;
        first.token_ids = {1, 2};
        first.position_ids = {0, 1};
        first.has_rs_slot = false;
        first.kv_pages = {5};
        first.readout_local_indices = {1};
        MemberForwardDesc later;
        later.sequence_id = kSeqB;
        later.token_ids = {3};
        later.position_ids = {4};  // not zero, nothing resident
        later.has_rs_slot = false;
        later.kv_pages = {6};
        later.readout_local_indices = {0};
        std::vector<MemberForwardDesc> d = {first, later};
        const BatchExecPlan p = plan_batch_execution({}, d);
        expect(p.member_ok[0] == 1, "non-hybrid position-0 member is fresh/serviceable");
        expect(p.member_ok[1] == 0, "non-hybrid non-zero-position member with nothing resident is gated");
    }

    // Sealed single-member protection: a lone FRESH member for a NEW sequence
    // while a DIFFERENT sequence is still ring-backed (and not referenced in
    // the batch) is gated — it must not silently steal the ring (byte-identical
    // to the sealed single-member forward's rejection).
    {
        auto slots = ring_backed(0, kSeqA, 3, 5);  // A resident, not in the batch
        std::vector<MemberForwardDesc> d = {fresh(kSeqB, 1, 6)};
        const BatchExecPlan p = plan_batch_execution(slots, d);
        expect(p.member_ok[0] == 0 && contains(p.member_reason[0], "different sequence is still resident"),
               "lone fresh B while different A resident is gated (" + p.member_reason[0] + ")");
    }

    // But the SAME instance re-resetting its own ring-backed sequence is
    // allowed (an explicit restart/re-prefill), matching the sealed path.
    {
        auto slots = ring_backed(0, kSeqA, 3, 5);
        std::vector<MemberForwardDesc> d = {fresh(kSeqA, 0, 9)};  // same seq A, fresh
        const BatchExecPlan p = plan_batch_execution(slots, d);
        expect(p.member_ok[0] == 1 && p.order.size() == 1,
               "same-instance fresh reset while it is resident is serviceable");
    }

    // A fresh member IS allowed to clobber the ring when the resident sequence
    // is being CONTINUED by a sibling in the same batch (served first).
    {
        auto slots = ring_backed(0, kSeqA, 3, 5);
        std::vector<MemberForwardDesc> d = {cont(kSeqA, 0, 3, 5), fresh(kSeqB, 1, 6)};
        const BatchExecPlan p = plan_batch_execution(slots, d);
        expect(p.member_ok[0] == 1 && p.member_ok[1] == 1 && p.order.size() == 2 &&
                   p.order[0] == 0,
               "fresh sibling allowed to clobber when resident seq is continued in-batch");
    }

    std::printf("\n==== batch_schedule_plan_test: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

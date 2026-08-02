// Phase 1a/1b linear-sequence geometry gate (metal_ptir_plan.md §5.4, Phase
// 1b state-slot fix) — pure host unit test, no Metal/Apple/checkpoint
// dependency (`validate_linear_sequence_geometry` / `close_linear_sequence`
// are defined outside the `#if defined(__APPLE__)` split in batch/forward.cpp,
// so this binary always builds and runs).
//
// MetalExecutor's shared M=1 KV ring holds exactly ONE resident linear
// KV/GDN sequence at a time; this gate is what keeps that single-stream
// assumption honest — accepting only a fresh sequence (for the same, or no,
// RING-BACKED resident sequence) or the exact continuation of the currently
// RING-BACKED one, and rejecting a second concurrent sequence / a page-list
// rewrite / duplicated pages / interleaved sequences / a "copied but not
// ring-backed" slot with a precise reason instead of quietly corrupting
// state.
//
// Physical KV page NUMBERING is reused across sequences by the runtime's
// free list and is not required to be arithmetically adjacent — {5, 9} is a
// perfectly valid two-page allocation; the gate tracks sequence IDENTITY
// (`sequence_id`, the engine's PTIR instance id) and a literal page-list
// PREFIX, not page-id arithmetic.
//
// Phase 1b state-slot fix: metadata is now tracked PER SLOT (a
// `LinearSequenceState` per rs_slot_id), because `MetalExecutor::copy_state`
// can populate a DIFFERENT slot's metadata without that slot backing the
// shared ring. `state.ring_backed` distinguishes "this slot's GDN state is
// what the ring's KV history actually matches" from "this slot merely holds
// a copy" — only a ring-backed slot may be CONTINUED (not-fresh). The
// caller (MetalExecutor::forward) precomputes `other_slot_ring_backed_
// different_sequence` by scanning its own slot map; this pure test drives
// that boolean directly instead of a live map.

#include <cstdio>
#include <string>

#include "batch/forward.hpp"

using pie::metal::batch::close_linear_sequence;
using pie::metal::batch::LinearSequenceState;
using pie::metal::batch::MemberForwardDesc;
using pie::metal::batch::validate_linear_sequence_geometry;
using pie::metal::batch::global_readout_rows;

namespace {

int g_pass = 0, g_fail = 0;
void expect(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

constexpr std::uint64_t kSeqA = 1001;
constexpr std::uint64_t kSeqB = 2002;

}  // namespace

int main() {
    std::printf("[validate_linear_sequence_geometry]\n");

    // Fresh sequence (RS_FLAG_RESET), no resident state yet — accepted, with
    // a NON-adjacent page pair {5, 9} as the baseline (the runtime free list
    // is not required to hand out arithmetically adjacent ids).
    {
        LinearSequenceState state;
        MemberForwardDesc desc;
        desc.sequence_id = kSeqA;
        desc.token_ids = {10, 11, 12};
        desc.position_ids = {0, 1, 2};
        desc.has_rs_slot = true;
        desc.rs_slot_id = 5;
        desc.rs_reset = true;
        desc.kv_pages = {5, 9};
        std::string reason;
        expect(validate_linear_sequence_geometry(state, /*other_ring_backed=*/false, desc,
                                                 &reason),
              "fresh sequence (rs_reset) with a valid non-adjacent {5,9} page pair accepts");
    }

    // Fresh sequence signalled by position 0 when the arch has no rs_slot
    // (non-hybrid) — accepted.
    {
        LinearSequenceState state;
        MemberForwardDesc desc;
        desc.sequence_id = kSeqA;
        desc.token_ids = {7};
        desc.position_ids = {0};
        desc.has_rs_slot = false;
        std::string reason;
        expect(validate_linear_sequence_geometry(state, false, desc, &reason),
              "fresh sequence via position 0 (no rs_slot) accepts");
    }

    // Exact continuation of the RING-BACKED resident sequence, EXTENDING the
    // {5,9} prefix with a newly appended, non-adjacent page (20) —
    // accepted: the prior pages are preserved as a literal prefix; the
    // appended page only needs to be unique, not arithmetically adjacent to 9.
    {
        LinearSequenceState state;
        state.has_resident = true;
        state.resident_sequence_id = kSeqA;
        state.resident_slot = 5;
        state.resident_next_position = 3;
        state.resident_pages = {5, 9};
        state.ring_backed = true;
        MemberForwardDesc desc;
        desc.sequence_id = kSeqA;
        desc.token_ids = {13};
        desc.position_ids = {3};
        desc.has_rs_slot = true;
        desc.rs_slot_id = 5;
        desc.rs_reset = false;
        desc.kv_pages = {5, 9, 20};  // preserves the {5,9} prefix, appends 20
        std::string reason;
        expect(validate_linear_sequence_geometry(state, false, desc, &reason),
              "prefix-preserving extension {5,9} -> {5,9,20} accepts (" + reason + ")");
    }

    // Continuation whose page list REWRITES an already-committed page (index
    // 1 changes from 9 to 7) — rejected: the prior prefix must survive
    // unmodified, regardless of whether 7 is itself a fresh-looking id.
    {
        LinearSequenceState state;
        state.has_resident = true;
        state.resident_sequence_id = kSeqA;
        state.resident_slot = 5;
        state.resident_next_position = 3;
        state.resident_pages = {5, 9};
        state.ring_backed = true;
        MemberForwardDesc desc;
        desc.sequence_id = kSeqA;
        desc.token_ids = {13};
        desc.position_ids = {3};
        desc.has_rs_slot = true;
        desc.rs_slot_id = 5;
        desc.rs_reset = false;
        desc.kv_pages = {5, 7, 20};  // index 1 rewrites 9 -> 7
        std::string reason;
        const bool ok = validate_linear_sequence_geometry(state, false, desc, &reason);
        expect(!ok && contains(reason, "does not preserve"),
              "page-list prefix rewrite rejects (" + reason + ")");
    }

    // A fresh fire for a DIFFERENT sequence while A is ring-backed elsewhere
    // — rejected precisely (the driver must not silently steal the shared
    // ring out from under A; the engine has to close_sequence(A) first).
    {
        LinearSequenceState state;  // this IS slot 6's own (empty) state
        MemberForwardDesc desc;
        desc.sequence_id = kSeqB;
        desc.token_ids = {1};
        desc.position_ids = {0};
        desc.has_rs_slot = true;
        desc.rs_slot_id = 6;
        desc.rs_reset = true;  // B claims to be fresh
        desc.kv_pages = {2};
        std::string reason;
        const bool ok = validate_linear_sequence_geometry(
            state, /*other_slot_ring_backed_different_sequence=*/true, desc, &reason);
        expect(!ok && contains(reason, "a different sequence is still resident"),
              "fresh B rejects while A is ring-backed elsewhere (" + reason + ")");
    }

    // The SAME resident sequence (A) going fresh again on its OWN slot (an
    // explicit restart/re-prefill, e.g. RS_FLAG_RESET re-issued) — accepted:
    // a same-instance reset must not be confused with "a different sequence
    // is resident" (there is no OTHER slot ring-backed for a DIFFERENT
    // sequence here, since A's own slot going fresh is not "other").
    {
        LinearSequenceState state;
        state.has_resident = true;
        state.resident_sequence_id = kSeqA;
        state.resident_slot = 5;
        state.resident_next_position = 8;
        state.resident_pages = {5, 9, 20};
        state.ring_backed = true;
        MemberForwardDesc desc;
        desc.sequence_id = kSeqA;  // same instance
        desc.token_ids = {1};
        desc.position_ids = {0};
        desc.has_rs_slot = true;
        desc.rs_slot_id = 5;
        desc.rs_reset = true;
        desc.kv_pages = {3};  // an entirely new baseline for the restarted sequence
        std::string reason;
        expect(validate_linear_sequence_geometry(state, false, desc, &reason),
              "same-instance (A) explicit reset while A is resident accepts (" + reason + ")");
    }

    // B accepted after `close_sequence(A)` (via the pure state-transition
    // helper `close_linear_sequence`, since this is the free-function unit
    // test — MetalExecutor::close_sequence delegates to it directly).
    {
        LinearSequenceState state;
        state.has_resident = true;
        state.resident_sequence_id = kSeqA;
        state.resident_slot = 5;
        state.resident_next_position = 3;
        state.resident_pages = {5, 9};
        state.ring_backed = true;

        // Closing an UNRELATED sequence id is a no-op — A stays resident.
        close_linear_sequence(state, kSeqB);
        expect(state.has_resident && state.resident_sequence_id == kSeqA,
              "close_linear_sequence(B) while A resident is a no-op");

        close_linear_sequence(state, kSeqA);
        expect(!state.has_resident, "close_linear_sequence(A) releases residency");

        // Now slot 5 is free; a fresh B fire elsewhere sees no OTHER
        // ring-backed sequence, so it is accepted.
        MemberForwardDesc desc;
        desc.sequence_id = kSeqB;
        desc.token_ids = {1};
        desc.position_ids = {0};
        desc.has_rs_slot = true;
        desc.rs_slot_id = 6;
        desc.rs_reset = true;
        desc.kv_pages = {2};
        std::string reason;
        expect(validate_linear_sequence_geometry(state, false, desc, &reason),
              "fresh B accepts after close_sequence(A) (" + reason + ")");
    }

    // Phase 1b: close_sequence(A) must NOT erase a DIFFERENT slot's
    // copy_state'd metadata for the SAME sequence id — close_linear_sequence
    // only touches an entry that is BOTH ring_backed AND matches
    // sequence_id; a non-ring-backed copy is untouched.
    {
        LinearSequenceState copied_slot_state;
        copied_slot_state.has_resident = true;
        copied_slot_state.resident_sequence_id = kSeqA;  // shares A's sequence id
        copied_slot_state.resident_slot = 7;             // a DIFFERENT slot than A's (5)
        copied_slot_state.resident_next_position = 3;
        copied_slot_state.resident_pages = {5, 9};
        copied_slot_state.ring_backed = false;  // copy_state'd, never forwarded through
        close_linear_sequence(copied_slot_state, kSeqA);
        expect(copied_slot_state.has_resident && copied_slot_state.resident_sequence_id == kSeqA &&
                  copied_slot_state.resident_next_position == 3,
              "close_linear_sequence(A) leaves a non-ring-backed copy's metadata for the "
              "SAME sequence id untouched (copied destination metadata survives)");
    }

    // Phase 1b: a slot whose metadata was copy_state'd here (has_resident,
    // correct sequence_id/next_position/pages) but is NOT ring_backed
    // cannot be CONTINUED — honestly rejected with a precise, distinct
    // reason (never silently treated as replay-ready).
    {
        LinearSequenceState state;
        state.has_resident = true;
        state.resident_sequence_id = kSeqA;
        state.resident_slot = 7;
        state.resident_next_position = 3;
        state.resident_pages = {5, 9};
        state.ring_backed = false;  // copied here, never forwarded through THIS slot
        MemberForwardDesc desc;
        desc.sequence_id = kSeqA;
        desc.token_ids = {13};
        desc.position_ids = {3};  // matches resident_next_position exactly
        desc.has_rs_slot = true;
        desc.rs_slot_id = 7;
        desc.rs_reset = false;
        desc.kv_pages = {5, 9};
        std::string reason;
        const bool ok = validate_linear_sequence_geometry(state, false, desc, &reason);
        expect(!ok && contains(reason, "not resident in the shared M=1 ring"),
              "continuing a copied-but-not-ring-backed slot rejects (" + reason + ")");
    }

    // Positions that do not extend the resident sequence (a gap, a rewind,
    // or a fork) — rejected.
    {
        LinearSequenceState state;
        state.has_resident = true;
        state.resident_sequence_id = kSeqA;
        state.resident_slot = 5;
        state.resident_next_position = 3;
        state.resident_pages = {5, 9};
        state.ring_backed = true;
        MemberForwardDesc desc;
        desc.sequence_id = kSeqA;
        desc.token_ids = {1, 2};
        desc.position_ids = {5, 6};  // gap: should have started at 3
        desc.has_rs_slot = true;
        desc.rs_slot_id = 5;
        desc.rs_reset = false;
        desc.kv_pages = {5, 9};
        std::string reason;
        const bool ok = validate_linear_sequence_geometry(state, false, desc, &reason);
        expect(!ok && contains(reason, "do not extend the currently resident sequence"),
              "position gap vs resident sequence rejects (" + reason + ")");
    }

    // Not fresh, and nothing resident yet — rejected (never silently
    // adopted as a continuation of "nothing").
    {
        LinearSequenceState state;  // has_resident = false
        MemberForwardDesc desc;
        desc.sequence_id = kSeqA;
        desc.token_ids = {1};
        desc.position_ids = {4};  // not 0, not marked fresh
        desc.has_rs_slot = true;
        desc.rs_slot_id = 0;
        desc.rs_reset = false;
        std::string reason;
        const bool ok = validate_linear_sequence_geometry(state, false, desc, &reason);
        expect(!ok && contains(reason, "no resident sequence to continue"),
              "non-fresh fire with nothing resident rejects (" + reason + ")");
    }

    // Non-contiguous positions WITHIN one fire — rejected ("in-order
    // positions").
    {
        LinearSequenceState state;
        MemberForwardDesc desc;
        desc.sequence_id = kSeqA;
        desc.token_ids = {1, 2, 3};
        desc.position_ids = {0, 1, 3};  // gap at the end
        desc.has_rs_slot = false;
        std::string reason;
        const bool ok = validate_linear_sequence_geometry(state, false, desc, &reason);
        expect(!ok && contains(reason, "in-order positions"),
              "non-contiguous positions within a fire rejects (" + reason + ")");
    }

    // Duplicated KV page id within one fire's list (a fork/share/corruption
    // signal) — rejected, even though {4, 7} alone would be a perfectly
    // valid non-adjacent pair.
    {
        LinearSequenceState state;
        MemberForwardDesc desc;
        desc.sequence_id = kSeqA;
        desc.token_ids = {1, 2};
        desc.position_ids = {0, 1};
        desc.has_rs_slot = false;
        desc.kv_pages = {4, 4, 7};  // repeated page id
        std::string reason;
        const bool ok = validate_linear_sequence_geometry(state, false, desc, &reason);
        expect(!ok && contains(reason, "duplicated physical page id"),
              "duplicated KV page id rejects (" + reason + ")");
    }

    // Empty token span — rejected.
    {
        LinearSequenceState state;
        MemberForwardDesc desc;
        desc.sequence_id = kSeqA;
        std::string reason;
        const bool ok = validate_linear_sequence_geometry(state, false, desc, &reason);
        expect(!ok && contains(reason, "no tokens"), "empty fire rejects (" + reason + ")");
    }

    // Mismatched token/position counts — rejected.
    {
        LinearSequenceState state;
        MemberForwardDesc desc;
        desc.sequence_id = kSeqA;
        desc.token_ids = {1, 2};
        desc.position_ids = {0};
        std::string reason;
        const bool ok = validate_linear_sequence_geometry(state, false, desc, &reason);
        expect(!ok && contains(reason, "count mismatch"),
              "token/position count mismatch rejects (" + reason + ")");
    }

    {
        const std::vector<std::uint32_t> last = global_readout_rows(7, {3});
        const std::vector<std::uint32_t> multiple = global_readout_rows(7, {0, 2, 3});
        expect(last == std::vector<std::uint32_t>({10}) &&
                   multiple == std::vector<std::uint32_t>({7, 9, 10}),
               "prefill local sampling rows [last] and multiple rows scatter to concatenated logits");
    }

    std::printf("\n==== executor_geometry_test: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

// Does the parser survive any reading of this group's tokens?
//
// The mask sweep's unit of work, and the other caller of `replay_chain`. In
// the Triton engine this is `_replay_group`, 330 lines, and it holds a second
// copy of the reduction chain because a Triton kernel cannot call another.
// Here it is a walk over readings and paths; the chain itself lives once, in
// `candidate.cuh`. The mask replays a reading exactly as the candidate does -
// what differs is that it throws the landing away and keeps only whether there
// was one.

#pragma once
#include "candidate.cuh"

namespace en {

/// What a group's replay found.
struct Verdict {
    bool admitted;   // some reading survived
    int32_t reach;   // deepest the window was pushed, for the compile-time bound
    int32_t read;    // the *lowest* stack entry any reading looked at
};

/// `read` is what the cross-step memo is keyed on: two configurations whose
/// stacks agree from `read` upward produce the same mask however they differ
/// below. A group that is refused, or whose readings die on their first
/// terminal, never looks below the top and leaves this at the depth it was
/// seeded with - which is most groups.
__device__ Verdict replay_group(
    const Tables& t,
    const int32_t* stack,
    int64_t base,
    int32_t* window,
    int32_t* probe_window,
    int32_t depth,
    int32_t group,
    int32_t max_reductions,
    int32_t paths,
    int32_t stack_stride,
    int32_t window_size,
    int32_t* overflow,
    int32_t sequence) {
    Verdict out{false, 0, depth};

    const int32_t use_at = t.reading_offsets[group];
    const int32_t use_end = use_at + 1 + t.reading_index[use_at];
    for (int32_t use = use_at + 1; use < use_end && !out.admitted; ++use) {
        int32_t reading = t.reading_index[use];
        int32_t span = 1;
        for (int32_t path = 0; path < paths && path < span && !out.admitted; ++path) {
            Replay r;
            r.rest = path;
            r.radix = 1;
            r.top = stack[base + depth - 1];
            r.depth = depth;
            r.floor = depth;
            r.alive = true;
            r.settled = false;

            int32_t term_end = t.reading_term_offsets[reading + 1];
            for (int32_t term = t.reading_term_offsets[reading]; term < term_end && r.alive;
                 ++term) {
                replay_chain(t, stack, base, window, r, t.reading_terminals[term],
                             max_reductions, paths, stack_stride, window_size, overflow,
                             sequence, LANDING);
            }
            out.read = min(out.read, r.floor);
            out.reach = max(out.reach, r.depth - min(r.floor, depth));

            if (r.alive) {
                int32_t next_state = t.reading_next_state[reading];
                int32_t pend = t.pending_offsets[next_state];
                int32_t pend_end = t.pending_offsets[next_state + 1];
                if (pend < pend_end) {
                    // Whatever lexeme is still pending has to be able to
                    // continue, or this reading is a dead end that only looks
                    // alive.
                    bool any = false;
                    for (; pend < pend_end && !any; ++pend) {
                        Replay p = r;
                        p.floor = r.depth;
                        p.alive = true;
                        p.settled = false;
                        replay_chain(t, stack, base, probe_window, p,
                                     t.pending_terminals[pend], max_reductions, paths,
                                     stack_stride, window_size, overflow, sequence,
                                     ADMISSIBLE, window, r.floor, r.depth);
                        r.radix = p.radix;
                        // Deliberately *not* `out.read`. The Triton engine
                        // updates it only from the reading replay, never from
                        // the probe, and a port that improves on its subject
                        // makes the comparison against it worthless. Noted
                        // rather than changed: the probe can read below the
                        // reading's floor - its index is clamped to
                        // `copy_depth - 1`, not to the floor - so whether the
                        // memo key is bounded by this is a question for the
                        // engine, not for the port.
                        out.reach = max(out.reach, p.depth - min(min(p.floor, r.floor), depth));
                        if (p.alive && p.settled) {
                            any = true;
                        }
                    }
                    r.alive = any;
                }
            }
            span = max(span, r.radix);
            if (r.alive) {
                out.admitted = true;
            }
        }
    }
    return out;
}

}  // namespace en

/// The mask sweep: a fixed number of threads draining a list of
/// (configuration, group).
///
/// Every `threads`-th item, not a contiguous run. A run would let a thread
/// resolve which configuration it is in once instead of searching per item,
/// which the Triton engine tried and measured worse - the groups of one state
/// cost wildly different amounts, most dying on their first terminal and a few
/// replaying whole readings, so a worker handed a run gets all of one state's
/// expensive ones. 37 us against 49.
extern "C" __global__ void en_mask(
    const en::Arena* arena,
    const en::BatchState* state,
    const int32_t* work_offsets,
    int32_t* scratch,
    int8_t* admitted,
    int32_t* high_water,
    int32_t* row_floor,
    int32_t rows,
    int32_t configs,
    int32_t stack_stride,
    int32_t max_reductions,
    int32_t window,
    int32_t paths,
    int32_t has_verdicts) {
    int32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t threads = gridDim.x * blockDim.x;
    int32_t total = work_offsets[rows];
    int32_t* mine = scratch + (int64_t)thread * 2 * window;
    int32_t* probe_window = mine + window;
    int32_t deepest = 0;

    for (int32_t item = thread; item < total; item += threads) {
        int32_t row = en::owner(work_offsets, rows, item);
        int32_t slot = item - work_offsets[row];
        int32_t sequence = row / configs;
        int32_t lexer = state->lexer_state[row];
        int32_t depth = state->depth[row];
        int32_t grammar = state->grammar_of[sequence];
        int64_t base = (int64_t)row * stack_stride;

        int32_t group_base = en::base_of(arena, grammar, en::B_GROUP_OFFSETS);
        int32_t group = arena->group_offsets[group_base + lexer] + slot;

        // Most of this answer does not depend on the stack. A group whose every
        // reading dies on a missing action dies for any stack, and that is 91%
        // of all replays on real grammars, so it is settled when the tables are
        // built and read here instead of run.
        int32_t settled = 0;
        if (has_verdicts) {
            int32_t stride =
                arena->verdict_stride[en::base_of(arena, grammar, en::B_VERDICT_STRIDE) + lexer];
            if (stride > 0) {
                int32_t top = state->stack[base + depth - 1];
                const int32_t* word = arena->verdicts
                    + en::base_of(arena, grammar, en::B_VERDICTS)
                    + arena->verdict_offsets[
                        en::base_of(arena, grammar, en::B_VERDICT_OFFSETS) + lexer]
                    + (int64_t)top * stride + (slot >> 4);
                settled = (*word >> (2 * (slot & 15))) & 3;
            }
        }

        // A settled group is refused without looking at the stack at all, so it
        // constrains nothing; only a replay can widen how much of the stack the
        // answer depends on.
        int32_t read = depth;
        int8_t got = 0;
        if (settled == 0) {
            en::Tables t = en::tables_of(arena, grammar);
            en::Verdict v = en::replay_group(t, state->stack, base, mine, probe_window,
                                             depth, group, max_reductions, paths,
                                             stack_stride, window, state->overflow,
                                             sequence);
            got = v.admitted ? 1 : 0;
            read = v.read;
            deepest = max(deepest, v.reach);
        }
        // Written whether or not the group is admitted, so the buffer never has
        // to be cleared - at batch 512 that clear was 13 MB a step.
        admitted[item] = got;
        // The sweep asks every group, so the deepest any of them looked is how
        // much of this configuration's stack the finished mask depends on.
        // Atomic because a row's groups are spread across workers - but only
        // when there is something to say, since doing it for every item cost
        // more than everything the suffix key buys: 40 to 146 us at batch 32.
        if (read < depth) {
            atomicMin(&row_floor[row], read);
        }
    }

    // How much of the window this worker actually needed. Once, not per push:
    // the point is to know how loose the compile-time bound is, and an atomic
    // in the reduce loop would be measuring the measurement.
    atomicMax(high_water, deepest);
}

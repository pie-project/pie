// Where each configuration lands if the sampled token is accepted.
//
// The largest kernel of the port, and the one the rewrite is for. Three things
// are different from the Triton version, and only the first is a translation.
//
// **The reduction chain is one function.** `_replay_group` and
// `_candidate_kernel` in the Triton engine contain the same seventy-six
// statements of LALR reduction, at 75% similarity, differing by one line and
// some naming - because a Triton kernel cannot call another. Here it is
// `replay_chain`, called from the reading walk, from the pending probe, and
// later from the fused step's mask sweep. A fix to one is a fix to all.
//
// **A thread owns a configuration, not a block.** Measured at 3.42x on this
// shape at our scale: a block scanning one configuration leaves 31 of 32 lanes
// idle, because the chain is serial per configuration and there is nothing to
// spread across lanes. The scratch is therefore per thread rather than per
// block, and the launch is sized so the total thread count matches what the
// Triton grid used - which keeps the memory identical.
//
// **The searches exit when they are done.** Triton runs a fixed
// `SEARCH_STEPS` because a masked tensor cannot stop early; a thread can.

#pragma once
#include "arena.cuh"

namespace en {

constexpr int32_t ACCEPT = -2147483648;

/// A grammar's tables, resolved once per configuration instead of per lookup.
/// Twelve base additions that would otherwise happen inside the chain.
struct Tables {
    const int32_t* reading_offsets;
    const int32_t* reading_index;
    const int32_t* reading_next_state;
    const int32_t* reading_term_offsets;
    const int32_t* reading_terminals;
    const int32_t* action_offsets;
    const int32_t* action_terminals;
    const int32_t* action_values;
    const int32_t* action_extra_offsets;
    const int32_t* action_extra;
    const int32_t* goto_offsets;
    const int32_t* goto_nonterminals;
    const int32_t* goto_targets;
    const int32_t* production_lhs;
    const int32_t* production_arity;
    const int32_t* pending_offsets;
    const int32_t* pending_terminals;
};

__device__ __forceinline__ Tables tables_of(const Arena* arena, int32_t grammar) {
    Tables t;
    t.reading_offsets = arena->reading_offsets + base_of(arena, grammar, B_READING_OFFSETS);
    t.reading_index = arena->reading_index + base_of(arena, grammar, B_READING_INDEX);
    t.reading_next_state = arena->reading_next_state + base_of(arena, grammar, B_READINGS);
    t.reading_term_offsets =
        arena->reading_term_offsets + base_of(arena, grammar, B_READING_TERM_OFFSETS);
    t.reading_terminals =
        arena->reading_terminals + base_of(arena, grammar, B_READING_TERMINALS);
    t.action_offsets = arena->action_offsets + base_of(arena, grammar, B_ACTION_OFFSETS);
    int32_t actions = base_of(arena, grammar, B_ACTIONS);
    t.action_terminals = arena->action_terminals + actions;
    t.action_values = arena->action_values + actions;
    t.action_extra_offsets =
        arena->action_extra_offsets + base_of(arena, grammar, B_ACTION_EXTRA_OFFSETS);
    t.action_extra = arena->action_extra + base_of(arena, grammar, B_ACTION_EXTRA);
    t.goto_offsets = arena->goto_offsets + base_of(arena, grammar, B_GOTO_OFFSETS);
    int32_t gotos = base_of(arena, grammar, B_GOTOS);
    t.goto_nonterminals = arena->goto_nonterminals + gotos;
    t.goto_targets = arena->goto_targets + gotos;
    int32_t productions = base_of(arena, grammar, B_PRODUCTIONS);
    t.production_lhs = arena->production_lhs + productions;
    t.production_arity = arena->production_arity + productions;
    t.pending_offsets = arena->pending_offsets + base_of(arena, grammar, B_PENDING_OFFSETS);
    t.pending_terminals =
        arena->pending_terminals + base_of(arena, grammar, B_PENDING_TERMINALS);
    return t;
}

/// Index of `needle` in the sorted run `[low, high)`, or -1.
__device__ __forceinline__ int32_t search(
    const int32_t* keys, int32_t low, int32_t high, int32_t needle) {
    while (low < high) {
        int32_t middle = (low + high) >> 1;
        int32_t value = keys[middle];
        if (value == needle) {
            return middle;
        }
        if (value < needle) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return -1;
}

/// Read stack entry `index` through up to two windows laid over the sequence's
/// own stack.
///
/// A replay never writes below where its own pushes started, so the stack can
/// stay where it is and be read in place; only what the replay adds is
/// private. The pending probe has *two* layers under it - its own pushes, then
/// the reading replay's, then the stack - because it continues from where that
/// replay left off rather than from the sequence's state.
///
/// Getting this wrong is invisible for a while: the probe only reads below its
/// own floor when a pending lexeme reduces further than the reading did, which
/// is a minority of a minority. It showed up as three wrong window entries in
/// 73,728 before it showed up as a wrong candidate count.
__device__ __forceinline__ int32_t peek(
    const int32_t* stack,
    int64_t base,
    const int32_t* window,
    int32_t floor,
    const int32_t* under,
    int32_t under_floor,
    int32_t under_depth,
    int32_t index) {
    if (index >= floor) {
        return window[index - floor];
    }
    if (under != nullptr) {
        int32_t at = min(index, under_depth - 1);
        if (at >= under_floor) {
            return under[at - under_floor];
        }
        return stack[base + at];
    }
    return stack[base + index];
}

/// One replay in progress. Passed by reference so the chain can be a function
/// rather than a block of statements pasted where it is needed.
struct Replay {
    int32_t top;    // parser state on top of the stack
    int32_t depth;  // how deep the stack is now
    int32_t floor;  // the lowest depth this replay has written
    int32_t rest;   // which derivation, as a mixed-radix selector
    int32_t radix;  // the product of the choices seen, so far
    bool alive;
    bool settled;
};

/// What a replay is being asked.
///
/// The distinction is between a *replay* and a *probe*, and not - as the first
/// version of this had it - between the mask and the candidate. Both of those
/// replay their reading the same way. It is what runs after them that differs.
///
/// A **replay** walks a reading's terminals and has to end up somewhere: a
/// shift is pushed onto the window and becomes the new top, and an accept
/// means the parse is complete and cannot read on, so it dies.
///
/// A **probe** asks only whether a pending lexeme could still continue from
/// where the replay left off. It never has to land, so a shift and an accept
/// both mean yes and neither is written down.
///
/// Getting this backwards is quiet in both directions: as `ADMISSIBLE` a
/// replay produced zero candidates where there should have been two, and as
/// `LANDING` a probe changed which groups the mask admitted.
enum Asking : int32_t {
    ADMISSIBLE = 0,  // a pending probe: can this continue?
    LANDING = 1,     // a reading replay: where does it end up?
};

/// Run the LALR automaton on one terminal until it shifts, accepts, or dies.
///
/// The heart of the parser, and in the Triton engine it is written out twice.
/// `window` is this replay's private scratch; `sequence` is only for the
/// overflow flag, which is how a narrowed mask is reported rather than hidden.
__device__ void replay_chain(
    const Tables& t,
    const int32_t* stack,
    int64_t base,
    int32_t* window,
    Replay& r,
    int32_t terminal,
    int32_t max_reductions,
    int32_t paths,
    int32_t stack_stride,
    int32_t window_size,
    int32_t* overflow,
    int32_t sequence,
    Asking asking = ADMISSIBLE,
    const int32_t* under = nullptr,
    int32_t under_floor = 0,
    int32_t under_depth = 0) {
    r.settled = false;
    // Bounded, but not fixed. A reduction chain ends at the first shift, and on
    // real documents that is two to four steps, while the bound has to cover
    // the deepest chain the grammar admits. The counter is a guard against a
    // grammar that never settles, not the schedule.
    for (int32_t spins = 0; !r.settled && r.alive && spins < max_reductions; ++spins) {
        int32_t low = t.action_offsets[r.top];
        int32_t high = t.action_offsets[r.top + 1];
        int32_t entry = search(t.action_terminals, low, high, terminal);
        if (entry < 0) {
            r.alive = false;
            return;
        }
        int32_t value = t.action_values[entry];
        if (paths > 1) {
            // A conflicted cell holds several actions. Which one this replay
            // takes is a digit of `rest` in a mixed radix, so that walking
            // `path` from zero enumerates every derivation exactly once.
            int32_t from = t.action_extra_offsets[entry];
            int32_t to = t.action_extra_offsets[entry + 1];
            int32_t count = 1 + to - from;
            if (count > 1) {
                r.radix *= count;
                int32_t pick = r.rest % count;
                r.rest /= count;
                if (pick > 0) {
                    value = t.action_extra[from + pick - 1];
                }
            }
        }
        if (value == ACCEPT) {
            // Admissible: yes, the terminal can be read. Landing: the parse is
            // complete, so there is nowhere for it to go on to.
            r.settled = asking == ADMISSIBLE;
            r.alive = asking == ADMISSIBLE;
            return;
        }
        if (value > 0) {
            if (asking == ADMISSIBLE) {
                r.settled = true;
                return;
            }
            if (r.depth >= stack_stride || r.depth - r.floor >= window_size) {
                r.alive = false;
                overflow[sequence] = 1;
                return;
            }
            r.top = value - 1;
            window[r.depth - r.floor] = r.top;
            r.depth += 1;
            r.settled = true;
            return;
        }
        int32_t production = -value - 1;
        int32_t arity = t.production_arity[production];
        if (r.depth <= arity) {
            r.alive = false;
            return;
        }
        r.depth -= arity;
        r.floor = min(r.floor, r.depth);
        int32_t exposed = peek(stack, base, window, r.floor, under, under_floor,
                               under_depth, r.depth - 1);
        int32_t lhs = t.production_lhs[production];
        int32_t grow = t.goto_offsets[exposed];
        int32_t grow_end = t.goto_offsets[exposed + 1];
        int32_t target = search(t.goto_nonterminals, grow, grow_end, lhs);
        if (target < 0) {
            r.alive = false;
            return;
        }
        if (r.depth >= stack_stride || r.depth - r.floor >= window_size) {
            // A ceiling, not a property of the grammar. Reaching it narrows
            // the mask, which is the one failure this engine must not do
            // quietly.
            r.alive = false;
            overflow[sequence] = 1;
            return;
        }
        r.top = t.goto_targets[target];
        window[r.depth - r.floor] = r.top;
        r.depth += 1;
    }
    if (!r.settled) {
        r.alive = false;
    }
}

}  // namespace en

/// One thread per configuration, grid-strided.
extern "C" __global__ void en_candidate(
    const en::Arena* arena,
    const en::BatchState* state,
    const int32_t* found,
    int32_t* scratch,
    int32_t* cand_count,
    int32_t* cand_lexer,
    int32_t* cand_depth,
    int32_t* cand_floor,
    int32_t* cand_window,
    int32_t* cand_at,
    int32_t* cand_used,
    int32_t rows,
    int32_t configs,
    int32_t max_readings,
    int32_t stack_stride,
    int32_t max_reductions,
    int32_t window,
    int32_t arena_slots,
    int32_t paths) {
    int32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t threads = gridDim.x * blockDim.x;
    // Two windows per thread: one for the reading being replayed, one for
    // probing what a pending lexeme could still become.
    int32_t* mine = scratch + (int64_t)thread * 2 * window;
    int32_t* probe_window = mine + window;

    for (int32_t row = thread; row < rows; row += threads) {
        int32_t group = found[row];
        // Written for every row, not only the ones that found a group: a row
        // skipped here would keep the count the previous step left, and the
        // commit would read candidates that are not there.
        if (group >= en::NO_GROUP) {
            cand_count[row] = 0;
            continue;
        }
        int32_t sequence = row / configs;
        int32_t grammar = state->grammar_of[sequence];
        en::Tables t = en::tables_of(arena, grammar);
        int32_t depth = state->depth[row];
        int64_t base = (int64_t)row * stack_stride;
        int64_t out_base = (int64_t)row * max_readings;

        int32_t index = 0;
        // Length-prefixed: a reading list is shared with every group in the
        // pool that wants the same one, so it cannot say how long it is by
        // where the next one starts.
        const int32_t use_at = t.reading_offsets[group];
        const int32_t use_end = use_at + 1 + t.reading_index[use_at];
        for (int32_t use = use_at + 1;
             use < use_end && index < max_readings; ++use) {
            int32_t reading = t.reading_index[use];
            // Every surviving derivation is kept, unlike the mask, which only
            // needed to know *whether* a token was admissible: two derivations
            // reach different stacks and both are states the next token may be
            // read from. A path past `radix` repeats one already taken.
            int32_t span = 1;
            for (int32_t path = 0; path < paths && index < max_readings && path < span;
                 ++path) {
                en::Replay r;
                r.rest = path;
                r.radix = 1;
                r.top = state->stack[base + depth - 1];
                r.depth = depth;
                r.floor = depth;
                r.alive = true;
                r.settled = false;

                int32_t term_end = t.reading_term_offsets[reading + 1];
                for (int32_t term = t.reading_term_offsets[reading];
                     term < term_end && r.alive; ++term) {
                    en::replay_chain(t, state->stack, base, mine, r,
                                     t.reading_terminals[term], max_reductions, paths,
                                     stack_stride, window, state->overflow, sequence,
                                     en::LANDING);
                }

                int32_t next_state = t.reading_next_state[reading];
                if (r.alive) {
                    // Whatever lexeme is still pending has to be able to
                    // continue from here, or this reading is a dead end that
                    // only looks alive.
                    int32_t pend_end = t.pending_offsets[next_state + 1];
                    int32_t pend = t.pending_offsets[next_state];
                    if (pend < pend_end) {
                        bool any = false;
                        for (; pend < pend_end && !any; ++pend) {
                            en::Replay p = r;
                            p.floor = r.depth;
                            p.settled = false;
                            p.alive = true;
                            // The path already spent its choices getting here;
                            // what is left of the radix is what this forks on.
                            en::replay_chain(t, state->stack, base, probe_window, p,
                                             t.pending_terminals[pend], max_reductions,
                                             paths, stack_stride, window,
                                             state->overflow, sequence,
                                             // The probe only asks whether the
                                             // pending lexeme could continue.
                                             en::ADMISSIBLE,
                                             // Over the reading's own window,
                                             // which it continues from.
                                             mine, r.floor, r.depth);
                            r.radix = p.radix;
                            if (p.alive && p.settled) {
                                any = true;
                            }
                        }
                        r.alive = any;
                    }
                }

                if (r.alive && path < r.radix) {
                    // A candidate outlives the step that made it, so unlike a
                    // replay it is written down - but not as a whole stack. It
                    // shares everything below its floor with the configuration
                    // it came from, so what is stored is the floor and the
                    // window. Whole stacks were 151 MB at batch 512.
                    // Packed, not placed. A bump per sequence costs one
                    // atomic on a counter two or three threads touch, and
                    // buys the difference between a budget and a product of
                    // ceilings. A sequence that runs out drops the candidate
                    // and says so, which is the narrowing signal the caller
                    // already refills from the reference matcher.
                    int32_t need = r.depth - r.floor;
                    int32_t at = need > 0
                        ? atomicAdd(&cand_used[sequence], need)
                        : 0;
                    if (at + need > arena_slots) {
                        state->overflow[sequence] = 1;
                        continue;
                    }
                    int32_t base_at = sequence * arena_slots + at;
                    cand_lexer[out_base + index] = next_state;
                    cand_depth[out_base + index] = r.depth;
                    cand_floor[out_base + index] = r.floor;
                    cand_at[out_base + index] = base_at;
                    for (int32_t k = 0; k < need; ++k) {
                        cand_window[base_at + k] = mine[k];
                    }
                    ++index;
                }
                span = max(span, r.radix);
            }
        }
        cand_count[row] = index;
        // More derivations than there is room for keeps a prefix of them,
        // which narrows the mask at the next token.
        if (index >= max_readings && use_at + 1 + index < use_end) {
            state->overflow[sequence] = 1;
        }
    }
}

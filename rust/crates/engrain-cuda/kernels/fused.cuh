// The whole advance as one kernel: locate, replay, commit.
//
// Four graph nodes become one. What makes it possible is dropping the global
// work list: the Triton engine counts every live configuration in the batch,
// prefix-sums them in a kernel of its own, and hands blocks contiguous slices
// so the load is balanced whatever the mixture. That scan is a *global*
// barrier, and a global barrier is a kernel boundary, and a kernel boundary is
// 1.10 us of graph-node dispatch.
//
// Here a block owns a sequence and its configurations are already its own -
// there is nothing to count and nothing to scan. The price is load imbalance
// across sequences, measured on the corpus at 1.28x at batch 128 and *0.90x*
// at batch 1 to 4, which is why this shape is chosen from the batch size.
//
// Between the phases is `__syncthreads`, which is free where a kernel boundary
// is not: measured at 4.4x on the mechanism at small batch, and grid-wide
// synchronisation - the other way to do this - is *slower* than launching.

#pragma once
#include "candidate.cuh"
#include "locate.cuh"

/// One block per sequence. A thread owns a configuration.
extern "C" __global__ void en_advance_fused(
    const en::Arena* arena,
    en::BatchState* state,
    const int32_t* token,
    int32_t* old_lexer,
    int32_t* old_count,
    int32_t* old_stack,
    int32_t* found,
    int32_t* scratch,
    int32_t* cand_count,
    int32_t* cand_lexer,
    int32_t* cand_depth,
    int32_t* cand_floor,
    int32_t* cand_window,
    int32_t* cand_at,
    int32_t* cand_used,
    int32_t* hist_slot,
    int32_t* hist_lexer,
    int32_t* hist_stack,
    int32_t* hist_depth,
    int32_t* hist_count,
    int32_t configs,
    int32_t max_readings,
    int32_t stack_stride,
    int32_t max_reductions,
    int32_t window,
    int32_t arena_slots,
    int32_t paths,
    int32_t has_verdicts,
    int32_t rollback,
    int32_t vocabulary) {
    int32_t sequence = blockIdx.x;
    int32_t lane = threadIdx.x;
    int32_t count = state->config_count[sequence];
    int32_t grammar = state->grammar_of[sequence];

    // Keep this step's parse state so a later one can be undone - before
    // anything below changes it. Speculative decoding advances through a draft
    // and then keeps only the prefix the model accepted, so the parser has to
    // go back, and going back by asking the host to replay the tokens is the
    // round trip this design exists not to make.
    //
    // The block already holds the sequence whose configurations these are, so
    // there is no live list to build, no prefix sum over it and no owner search
    // per item - which is what the unfused path spent two extra kernels and a
    // grid of one on. The slot is read from the device rather than passed in,
    // because a graph records the arguments it was given and a slot that
    // arrived as a scalar would be frozen at whatever it was when recorded.
    if (rollback > 0) {
        int32_t slot = *hist_slot % rollback;
        int64_t at = (int64_t)slot * gridDim.x * configs;
        if (lane == 0) {
            hist_count[slot * gridDim.x + sequence] = count;
        }
        for (int32_t config = 0; config < count; ++config) {
            int32_t row = sequence * configs + config;
            int32_t held = state->depth[row];
            if (lane == 0) {
                hist_lexer[at + row] = state->lexer_state[row];
                hist_depth[at + row] = held;
            }
            // Only as deep as the configuration goes. Writing every row to its
            // full stride is 67 MB a step at batch 512 with 128 configurations,
            // to preserve the one or two a sequence actually holds.
            for (int32_t up = lane; up < held; up += blockDim.x) {
                hist_stack[(at + row) * stack_stride + up] =
                    state->stack[(int64_t)row * stack_stride + up];
            }
        }
        __syncthreads();
    }
    en::Shape shape{0, configs, stack_stride, 0, 0};

    // Phase one: save the state the commit will read. The advance writes new
    // configurations while reading old ones, so this has to be complete before
    // anything overwrites - which is what the barrier at the end of the phase
    // is for, and what used to be a whole kernel.
    for (int32_t item = lane; item < count * stack_stride; item += blockDim.x) {
        int32_t config = item / stack_stride;
        int32_t row = sequence * configs + config;
        old_stack[(int64_t)row * stack_stride + item % stack_stride] =
            state->stack[(int64_t)row * stack_stride + item % stack_stride];
    }
    for (int32_t config = lane; config < count; config += blockDim.x) {
        int32_t row = sequence * configs + config;
        old_lexer[row] = state->lexer_state[row];
    }
    if (lane == 0) {
        old_count[sequence] = count;
        // The candidate arena is a bump per sequence, so the bump starts here.
        cand_used[sequence] = 0;
    }
    __syncthreads();

    // Phase two: which group holds the sampled token, for each configuration.
    // A warp to a configuration, since the scan is over the groups of a lexer
    // state and there are hundreds of them.
    int32_t warp = lane >> 5, warps = blockDim.x >> 5, in_warp = lane & 31;
    for (int32_t config = warp; config < count; config += warps) {
        int32_t row = sequence * configs + config;
        int32_t best = en::locate_one(arena, state, shape, token, sequence, row, in_warp,
                                      has_verdicts != 0, vocabulary);
        if (in_warp == 0) {
            found[row] = best;
        }
    }
    __syncthreads();

    // Phase three: where each configuration lands. A *thread* to a
    // configuration - the chain is serial per configuration and a warp doing
    // it leaves thirty-one lanes idle, which measured 3.42x at our scale.
    for (int32_t config = lane; config < count; config += blockDim.x) {
        int32_t row = sequence * configs + config;
        int32_t group = found[row];
        if (group >= en::NO_GROUP) {
            cand_count[row] = 0;
            continue;
        }
        // Indexed by (sequence, configuration), not by thread: at most
        // `configs` threads of a block ever replay, and indexing by thread
        // would size the buffer for `batch * blockDim` - eight times more at
        // batch 512, for slots no thread can reach.
        int32_t* mine =
            scratch + (int64_t)(sequence * configs + config) * 2 * window;
        int32_t* probe_window = mine + window;
        en::Tables t = en::tables_of(arena, grammar);
        int32_t depth = state->depth[row];
        int64_t base = (int64_t)row * stack_stride;
        int64_t out_base = (int64_t)row * max_readings;

        int32_t index = 0;
        int32_t use_end = t.reading_offsets[group + 1];
        for (int32_t use = t.reading_offsets[group];
             use < use_end && index < max_readings; ++use) {
            int32_t reading = t.reading_index[use];
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
                    int32_t pend_end = t.pending_offsets[next_state + 1];
                    int32_t pend = t.pending_offsets[next_state];
                    if (pend < pend_end) {
                        bool any = false;
                        for (; pend < pend_end && !any; ++pend) {
                            en::Replay p = r;
                            p.floor = r.depth;
                            p.settled = false;
                            p.alive = true;
                            en::replay_chain(t, state->stack, base, probe_window, p,
                                             t.pending_terminals[pend], max_reductions,
                                             paths, stack_stride, window,
                                             state->overflow, sequence, en::ADMISSIBLE,
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
        if (index >= max_readings && t.reading_offsets[group] + index < use_end) {
            state->overflow[sequence] = 1;
        }
    }
    __syncthreads();

    // Phase four: collect the survivors, serially and in the matcher's order.
    // Every thread walks the same control flow and they earn their keep on the
    // stack compares, exactly as the unfused commit does.
    int32_t written = 0;
    int32_t saturated = 0;
    for (int32_t state_slot = 0; state_slot < count && written < configs; ++state_slot) {
        int32_t lexer = old_lexer[sequence * configs + state_slot];
        bool seen = false;
        for (int32_t earlier = 0; earlier < state_slot; ++earlier) {
            if (old_lexer[sequence * configs + earlier] == lexer) {
                seen = true;
            }
        }
        if (seen) {
            continue;
        }
        for (int32_t source = 0; source < count && written < configs; ++source) {
            if (old_lexer[sequence * configs + source] != lexer) {
                continue;
            }
            int64_t base = (int64_t)(sequence * configs + source) * max_readings;
            int32_t made = cand_count[sequence * configs + source];
            for (int32_t index = 0; index < made; ++index) {
                if (written >= configs) {
                    saturated = 1;
                    break;
                }
                int32_t next_state = cand_lexer[base + index];
                int32_t depth = cand_depth[base + index];
                int32_t floor = cand_floor[base + index];
                int64_t source_row = (int64_t)(sequence * configs + source);
                int64_t candidate = base + index;
                bool duplicate = false;
                for (int32_t done = 0; done < written; ++done) {
                    int32_t out = sequence * configs + done;
                    if (state->lexer_state[out] != next_state
                        || state->depth[out] != depth) {
                        continue;
                    }
                    int32_t differs = 0;
                    for (int32_t slot = lane; slot < depth; slot += blockDim.x) {
                        if (state->stack[(int64_t)out * stack_stride + slot]
                            != en::stack_entry(old_stack, cand_window, source_row,
                                               cand_at[candidate], stack_stride, floor,
                                               slot)) {
                            differs = 1;
                        }
                    }
                    if (__syncthreads_or(differs) == 0) {
                        duplicate = true;
                    }
                }
                if (!duplicate) {
                    int32_t out = sequence * configs + written;
                    for (int32_t slot = lane; slot < depth; slot += blockDim.x) {
                        state->stack[(int64_t)out * stack_stride + slot] =
                            en::stack_entry(old_stack, cand_window, source_row,
                                            cand_at[candidate], stack_stride, floor,
                                            slot);
                    }
                    if (lane == 0) {
                        state->lexer_state[out] = next_state;
                        state->depth[out] = depth;
                    }
                    ++written;
                }
                __syncthreads();
            }
        }
    }
    if (lane == 0) {
        if (written == 0) {
            state->terminated[sequence] = 1;
        } else {
            state->config_count[sequence] = written;
        }
        if (saturated) {
            state->overflow[sequence] = 1;
        }
        atomicMax(state->widest, written);
    }
}

/// Phase one of the fill, on its own: has this answer been produced already?
///
/// Split from the sweep because the sweep must not be one block per sequence.
/// This half is O(configs x depth) - a table probe and a scan over earlier
/// sequences - so it is cheap and, more to the point, it is *balanced*: every
/// sequence costs about the same. The sweep is O(configs x groups) and is not.
///
/// It also does the clearing, which is the reason the split has to exist at
/// all rather than the sweep simply growing a second grid dimension: a block
/// that clears the row while another block is already OR-ing into it loses
/// bits, and losing bits is the one failure this engine must never make. In
/// Triton these were two kernels for exactly this reason.
extern "C" __global__ void en_fill_probe(
    en::BatchState* state,
    int32_t* mask,
    const int32_t* hash,
    const int32_t* suffix_hash,
    const int32_t* memo_hash,
    const int32_t* memo_lexer,
    const int32_t* memo_stack,
    const int32_t* memo_depth,
    const int32_t* memo_count,
    const int32_t* memo_grammar,
    const int32_t* memo_read,
    int32_t* memo_slot,
    int32_t* representative,
    int32_t* memo_store,
    int32_t* memo_want,
    int32_t* row_floor,
    int32_t* admitted,
    int32_t configs,
    int32_t stack_stride,
    int32_t slots,
    int32_t memo_configs,
    int32_t memo_stride,
    int32_t suffixes,
    int32_t mask_words,
    int32_t group_words) {
    int32_t sequence = blockIdx.x;
    int32_t lane = threadIdx.x;
    int32_t count = state->config_count[sequence];
    int32_t grammar = state->grammar_of[sequence];

    // Phase one: has this answer been produced, in an earlier step or by an
    // earlier sequence in this one?
    int32_t found = -1;
    for (int32_t attempt = 0; attempt <= suffixes && found < 0; ++attempt) {
        int32_t want = attempt == 0 ? -1 : attempt;
        int32_t digest = attempt == 0 ? hash[sequence]
                                      : suffix_hash[sequence * suffixes + attempt - 1];
        int32_t slot = (int32_t)(((uint32_t)digest & 0x7fffffffu) % (uint32_t)slots);
        if (count > memo_configs) {
            continue;
        }
        bool same = memo_hash[slot] == digest && memo_read[slot] == want
                    && memo_count[slot] == count && memo_grammar[slot] == grammar;
        for (int32_t config = 0; config < count && same; ++config) {
            int32_t row = sequence * configs + config;
            int64_t held = (int64_t)slot * memo_configs + config;
            int32_t depth = state->depth[row];
            int32_t kept = want > 0 ? min(depth, want) : depth;
            if (state->lexer_state[row] != memo_lexer[held] || kept != memo_depth[held]) {
                same = false;
            }
            int32_t differs = 0;
            for (int32_t at = lane; at < kept && at < memo_stride; at += blockDim.x) {
                if (state->stack[(int64_t)row * stack_stride + depth - kept + at]
                    != memo_stack[held * memo_stride + at]) {
                    differs = 1;
                }
            }
            if (__syncthreads_or(differs)) {
                same = false;
            }
        }
        if (same) {
            found = slot;
        }
    }

    int32_t neighbour = sequence;
    if (found < 0) {
        int32_t digest = hash[sequence];
        for (int32_t other = 0; other < sequence && neighbour == sequence; ++other) {
            if (hash[other] != digest || state->config_count[other] != count
                || state->grammar_of[other] != grammar) {
                continue;
            }
            bool same = true;
            for (int32_t config = 0; config < count && same; ++config) {
                int32_t a = sequence * configs + config;
                int32_t b = other * configs + config;
                int32_t depth = state->depth[a];
                if (state->lexer_state[a] != state->lexer_state[b]
                    || depth != state->depth[b]) {
                    same = false;
                }
                int32_t differs = 0;
                for (int32_t at = lane; at < depth; at += blockDim.x) {
                    if (state->stack[(int64_t)a * stack_stride + at]
                        != state->stack[(int64_t)b * stack_stride + at]) {
                        differs = 1;
                    }
                }
                if (__syncthreads_or(differs)) {
                    same = false;
                }
            }
            if (same) {
                neighbour = other;
            }
        }
    }
    if (lane == 0) {
        memo_slot[sequence] = found;
        representative[sequence] = neighbour;
    }

    bool computes = found < 0 && neighbour == sequence;
    if (lane == 0) {
        memo_store[sequence] = (computes && count <= memo_configs) ? 1 : -1;
    }
    if (!computes) {
        // Nothing to sweep: the copy will give this sequence its row. The
        // floors still have to be seeded, since the claim reads them.
        for (int32_t config = lane; config < count; config += blockDim.x) {
            row_floor[sequence * configs + config] = state->depth[sequence * configs + config];
        }
        if (lane == 0) {
            // -2, not -1: the claim's "this sequence may not store" is a
            // different answer from "it may, under the whole-stack key".
            memo_want[sequence] = -2;
        }
        return;
    }

    // Only what computes has to start empty; everything else is written whole
    // by the copy.
    for (int32_t at = lane; at < mask_words; at += blockDim.x) {
        mask[(int64_t)sequence * mask_words + at] = 0;
    }
    // And the record of which groups this row has already been given. A group
    // admitted by several configurations contributes the same tokens each
    // time, and the row is a union, so only the first write is work. Measured
    // on the corpus schema that forks hardest: sixty-four configurations
    // produce four distinct rows, and the bits written are sixty times the
    // bits that end up set.
    //
    // A small open-addressed set per row, not a bit per group. A bit per group
    // is `rows x num_groups / 8` and `num_groups` is the largest schema in the
    // pool, so a batch of 512 over sixty-four corpus schemas wanted 5.16 GiB
    // of it and a batch of 1,024 could not be built at all. What a row
    // actually admits is a handful, so a fixed set holds it - and when it does
    // not, the group is written twice, which is a union either way. This is a
    // filter, not a fact.
    for (int32_t at = lane; at < group_words; at += blockDim.x) {
        admitted[(int64_t)sequence * group_words + at] = 0;
    }
    for (int32_t config = lane; config < count; config += blockDim.x) {
        row_floor[sequence * configs + config] = state->depth[sequence * configs + config];
    }
}

/// Phase two of the fill: sweep every live configuration's groups.
///
/// A block no longer owns a sequence. It owns a *slice* of one, because the
/// cost of a sequence is its configurations times its groups and a serving
/// batch is skewed by construction - every request sits at its own point in
/// its own document. Measured with one block per sequence, batch 32: all rows
/// at one point 29.1 us, rows at random points 9883.5 us, on a schema whose
/// widest sequence held sixty-four configurations against the usual two.
extern "C" __global__ void en_fill_sweep(
    const en::Arena* arena,
    en::BatchState* state,
    int32_t* mask,
    const int32_t* memo_slot,
    const int32_t* representative,
    int32_t* row_floor,
    int32_t* scratch,
    int32_t* high_water,
    int32_t* admitted,
    int32_t configs,
    int32_t stack_stride,
    int32_t max_reductions,
    int32_t window,
    int32_t paths,
    int32_t has_verdicts,
    int32_t mask_words,
    int32_t group_words) {
    int32_t sequence = blockIdx.x;
    int32_t lane = threadIdx.x;
    int32_t count = state->config_count[sequence];
    int32_t grammar = state->grammar_of[sequence];
    if (memo_slot[sequence] >= 0 || representative[sequence] != sequence) {
        return;
    }
    int32_t deepest = 0;
    int32_t group_base = en::base_of(arena, grammar, en::B_GROUP_OFFSETS);
    int32_t groups = en::base_of(arena, grammar, en::B_GROUPS);
    int32_t payload_base = en::base_of(arena, grammar, en::B_SET_PAYLOAD);
    en::Tables t = en::tables_of(arena, grammar);

    // The iteration space is (configuration, group) flattened, so a block
    // takes a slice of a sequence's *whole* sweep rather than a slice of each
    // configuration in turn. A sequence holding sixty-four configurations at
    // one group each would otherwise put every block on the same one.
    //
    // The offsets are built here, per block, rather than by a counting kernel
    // and a prefix sum on a grid of one - which is what the work list cost.
    // A block redundantly scans its own sequence's configurations, which is at
    // most `configs` loads against the thousands of replays it is about to do.
    extern __shared__ int32_t offsets[];
    for (int32_t config = lane; config < count; config += blockDim.x) {
        int32_t lexer = state->lexer_state[sequence * configs + config];
        offsets[config] = arena->group_offsets[group_base + lexer + 1]
                          - arena->group_offsets[group_base + lexer];
    }
    __syncthreads();
    if (lane == 0) {
        int32_t running = 0;
        for (int32_t config = 0; config < count; ++config) {
            int32_t held = offsets[config];
            offsets[config] = running;
            running += held;
        }
        offsets[count] = running;
    }
    __syncthreads();
    int32_t total = offsets[count];

    {
        for (int32_t item = blockIdx.y * blockDim.x + lane; item < total;
             item += gridDim.y * blockDim.x) {
            int32_t low = 0;
            int32_t high = count;
            while (high - low > 1) {
                int32_t mid = (low + high) >> 1;
                if (offsets[mid] <= item) {
                    low = mid;
                } else {
                    high = mid;
                }
            }
            int32_t config = low;
            int32_t slot = item - offsets[config];
            int32_t row = sequence * configs + config;
            int32_t lexer = state->lexer_state[row];
            int32_t first = arena->group_offsets[group_base + lexer];
            int32_t depth = state->depth[row];
            int64_t base = (int64_t)row * stack_stride;
            int32_t group = first + slot;
            int32_t settled = 0;
            if (has_verdicts) {
                int32_t stride = arena->verdict_stride[
                    en::base_of(arena, grammar, en::B_VERDICT_STRIDE) + lexer];
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
            if (settled != 0) {
                continue;
            }
            // Indexed by the *thread*, not by the slot. A thread walks slots
            // `blockDim.x` apart, so `slot % configs` gave two of them the
            // same window whenever the block was wider than `configs` - a
            // race that only shows where a sequence holds many
            // configurations, which is where the corpus does and a unit test
            // does not.
            int32_t* mine =
                scratch
                + (int64_t)((blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + lane)
                      * 2 * window;
            en::Verdict v = en::replay_group(t, state->stack, base, mine, mine + window,
                                             depth, group, max_reductions, paths,
                                             stack_stride, window, state->overflow,
                                             sequence);
            if (v.read < depth) {
                atomicMin(&row_floor[row], v.read);
            }
            deepest = max(deepest, v.reach);
            if (!v.admitted) {
                continue;
            }
            // Phase three, inline: write this group's set - but only if no
            // other configuration of this row has written it already. The row
            // is a union and a group contributes the same tokens whichever
            // configuration admitted it, so every write after the first is
            // redundant bandwidth. `atomicOr` returns the word as it was, so
            // exactly one thread sees the bit clear and does the work.
            //
            // This is where a forked row's cost was. Admission still has to be
            // decided per configuration - that is the parse - but a DENSE
            // group is `mask_words` atomics, and paying that sixty times for
            // one answer is what made a schema that forks cost 3.6 ms against
            // 42 us for the same schema seeded where it does not.
            // Claim the group, or find it already claimed. Four probes and
            // then give up and write it again: a full filter costs work and
            // never an answer.
            bool given = false;
            uint32_t home = ((uint32_t)group * 2654435761u) & (uint32_t)(group_words - 1);
            for (int32_t probe = 0; probe < 4; ++probe) {
                int32_t* cell = &admitted[(int64_t)sequence * group_words
                                          + ((home + probe) & (group_words - 1))];
                int32_t was = atomicCAS(cell, 0, group + 1);
                if (was == 0) {
                    break;
                }
                if (was == group + 1) {
                    given = true;
                    break;
                }
            }
            if (given) {
                continue;
            }
            int32_t kind = arena->group_set_kind[groups + group];
            int32_t offset = arena->group_set_offset[groups + group];
            int32_t length = arena->group_set_length[groups + group];
            const int32_t* payload = arena->set_payload + payload_base + offset;
            int32_t* row_mask = mask + (int64_t)sequence * mask_words;
            if (kind == en::DENSE) {
                for (int32_t at = 0; at < mask_words; ++at) {
                    atomicOr(&row_mask[at], payload[at]);
                }
            } else if (kind == en::SPARSE) {
                for (int32_t at = 0; at < length; ++at) {
                    int32_t token = payload[at];
                    atomicOr(&row_mask[token >> 5], 1 << (token & 31));
                }
            } else {
                // Setting the row and punching the exclusions out is only
                // correct while there is one complement; a sequence at two
                // lexer states can have two, and the second erased what the
                // first admitted. So each word is decided before it is written.
                int32_t cursor = 0;
                for (int32_t at = 0; at < mask_words; ++at) {
                    int32_t value = -1;
                    while (cursor < length && (payload[cursor] >> 5) == at) {
                        value &= ~(1 << (payload[cursor] & 31));
                        ++cursor;
                    }
                    atomicOr(&row_mask[at], value);
                }
            }
        }
    }
    if (lane == 0) {
        atomicMax(high_water, deepest);
    }
}

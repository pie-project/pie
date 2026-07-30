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
extern "C" __global__ void gg_advance_fused(
    const gg::Arena* arena,
    gg::BatchState* state,
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
    int32_t configs,
    int32_t max_readings,
    int32_t stack_stride,
    int32_t max_reductions,
    int32_t window,
    int32_t paths,
    int32_t has_verdicts) {
    int32_t sequence = blockIdx.x;
    int32_t lane = threadIdx.x;
    int32_t count = state->config_count[sequence];
    int32_t grammar = state->grammar_of[sequence];
    gg::Shape shape{0, configs, stack_stride, 0, 0};

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
    }
    __syncthreads();

    // Phase two: which group holds the sampled token, for each configuration.
    // A warp to a configuration, since the scan is over the groups of a lexer
    // state and there are hundreds of them.
    int32_t warp = lane >> 5, warps = blockDim.x >> 5, in_warp = lane & 31;
    for (int32_t config = warp; config < count; config += warps) {
        int32_t row = sequence * configs + config;
        int32_t best = gg::locate_one(arena, state, shape, token, sequence, row, in_warp,
                                      has_verdicts != 0);
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
        if (group >= gg::NO_GROUP) {
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
        gg::Tables t = gg::tables_of(arena, grammar);
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
                gg::Replay r;
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
                    gg::replay_chain(t, state->stack, base, mine, r,
                                     t.reading_terminals[term], max_reductions, paths,
                                     stack_stride, window, state->overflow, sequence,
                                     gg::LANDING);
                }

                int32_t next_state = t.reading_next_state[reading];
                if (r.alive) {
                    int32_t pend_end = t.pending_offsets[next_state + 1];
                    int32_t pend = t.pending_offsets[next_state];
                    if (pend < pend_end) {
                        bool any = false;
                        for (; pend < pend_end && !any; ++pend) {
                            gg::Replay p = r;
                            p.floor = r.depth;
                            p.settled = false;
                            p.alive = true;
                            gg::replay_chain(t, state->stack, base, probe_window, p,
                                             t.pending_terminals[pend], max_reductions,
                                             paths, stack_stride, window,
                                             state->overflow, sequence, gg::ADMISSIBLE,
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
                    cand_lexer[out_base + index] = next_state;
                    cand_depth[out_base + index] = r.depth;
                    cand_floor[out_base + index] = r.floor;
                    int64_t at = (out_base + index) * (int64_t)window;
                    for (int32_t k = 0; k < r.depth - r.floor; ++k) {
                        cand_window[at + k] = mine[k];
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
                int32_t value = 0;
                if (lane < depth) {
                    value = lane < floor
                        ? old_stack[(int64_t)(sequence * configs + source) * stack_stride
                                    + lane]
                        : cand_window[(base + index) * (int64_t)window + (lane - floor)];
                }
                bool duplicate = false;
                for (int32_t done = 0; done < written; ++done) {
                    int32_t out = sequence * configs + done;
                    if (state->lexer_state[out] != next_state
                        || state->depth[out] != depth) {
                        continue;
                    }
                    int32_t differs = (lane < depth)
                        && (state->stack[(int64_t)out * stack_stride + lane] != value);
                    if (__syncthreads_or(differs) == 0) {
                        duplicate = true;
                    }
                }
                if (!duplicate) {
                    int32_t out = sequence * configs + written;
                    if (lane < depth) {
                        state->stack[(int64_t)out * stack_stride + lane] = value;
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

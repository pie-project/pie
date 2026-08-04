// Collect the surviving candidates into the next configuration set.
//
// **Serial, and deliberately.** The reference matcher deduplicates in a
// particular order and stops at its configuration ceiling, so a parallel
// collection that produced the same *set* could still produce a different
// *prefix* once the ceiling bites. Reproducing the order is what lets the two
// be compared for equality rather than for similarity - and equality is the
// whole verification strategy.
//
// So the control flow is scalar and every thread walks it identically, which
// is what Triton generates too. The threads earn their keep on the vector
// parts: comparing a stack, and writing one.
//
// The two block-wide predicates use `__syncthreads_or`, which is the barrier
// and the reduction in one instruction. Triton spells the same thing
// `tl.sum(tl.where(...)) == 0` and pays a full add-reduce for a question that
// is really "did any lane disagree".

#pragma once
#include "arena.cuh"

/// One block per sequence. Any block width: a thread walks the stack in a
/// strided loop rather than owning one entry, so the depth a batch allows is
/// not also a launch parameter.
extern "C" __global__ void en_commit(
    en::BatchState* state,
    const int32_t* old_lexer,
    const int32_t* old_count,
    const int32_t* old_stack,
    const int32_t* cand_count,
    const int32_t* cand_lexer,
    const int32_t* cand_depth,
    const int32_t* cand_floor,
    const int32_t* cand_window,
    const int32_t* cand_at,
    int32_t configs,
    int32_t max_readings,
    int32_t stack_stride) {
    int32_t sequence = blockIdx.x;
    int32_t lane = threadIdx.x;
    int32_t count = old_count[sequence];

    int32_t written = 0;
    // Set when a surviving candidate has to be dropped for want of room. The
    // configuration ceiling is a policy, not a property of the grammar, and a
    // parse that outgrows it keeps a prefix of its states - which narrows the
    // mask. Narrowing is the failure this engine must never do quietly.
    int32_t saturated = 0;

    // Bounded by the count, not by the ceiling: the loops are nested, so the
    // ceiling would enter squared. Measured in the Triton engine as 88 us
    // against 27 when it ran to 128 for a parse holding one configuration.
    for (int32_t state_slot = 0; state_slot < count && written < configs; ++state_slot) {
        int32_t lexer = old_lexer[sequence * configs + state_slot];

        // Only the first configuration carrying a lexer state introduces it; a
        // later one would repeat every candidate the first produced.
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

                // A thread owns every `blockDim.x`-th entry rather than one
                // each, so a block of any width covers a stack of any depth.
                int64_t source_row = (int64_t)(sequence * configs + source);
                int64_t candidate = base + index;

                bool duplicate = false;
                for (int32_t done = 0; done < written; ++done) {
                    int32_t out = sequence * configs + done;
                    // Cheap scalar test first, and every thread agrees on it,
                    // so the block-wide compare below is only paid when two
                    // configurations are the same shape.
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
                    // Uniform across the block: the loop above varies per
                    // thread, this does not.
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
                // The next candidate reads what this one wrote, through the
                // duplicate check.
                __syncthreads();
            }
        }
    }

    if (lane == 0) {
        // No candidate survived: the token was refused. The set is left as it
        // was and the sequence is marked, because a mask filled from an empty
        // set would silently allow everything.
        if (written == 0) {
            state->terminated[sequence] = 1;
        } else {
            state->config_count[sequence] = written;
        }
        if (saturated) {
            state->overflow[sequence] = 1;
        }
        // The widest set in the batch, kept on the device. The fill's grid is
        // sized for the ceiling because the host may not ask, but every
        // program can read this and return at once - which turns the ceiling
        // from work into a launch.
        atomicMax(state->widest, written);
    }
}

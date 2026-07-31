// The cross-step mask memo: a hash table keyed on a sequence's parse state.
//
// Not a parser. Nothing here replays anything - it is a fingerprint, a table
// lookup, a search for a sequence in the same batch that will compute the same
// answer, and two copies. It is worth 92-96% hit rates at every batch size
// including one, because dedup within a step only sees the step it is in.
//
// The keys are the interesting part. A mask depends on the stack only as far
// down as its replay looked, which the sweep measures into `row_floor`, so an
// entry saved under the *suffix* it actually needs is found again by whatever
// agrees on that much however different the two stacks are underneath. On a
// grammar that nests, the stack grows with the document and the answer does
// not.

#pragma once
#include "arena.cuh"

namespace en {

constexpr int32_t MEMO_EMPTY = -1;
constexpr uint32_t FNV_OFFSET = 2166136261u;
constexpr uint32_t FNV_PRIME = 16777619u;

__device__ __forceinline__ int32_t fold(int32_t digest, int32_t with) {
    return (int32_t)((((uint32_t)digest) ^ (uint32_t)with) * FNV_PRIME);
}

/// Every thread in the block computes the same scalar answer from the same
/// loads, which is what Triton generates too; the threads earn their keep on
/// the stack folds and comparisons below.
struct Sequence {
    int32_t count;
    int32_t grammar;
};

}  // namespace en

/// A fingerprint of a sequence's whole parse state, and one per suffix width.
///
/// One *warp* per sequence, not a block. The fold is a sum over a stack whose
/// depth is single digits on real documents, and a block-wide tree reduction
/// pays seven `__syncthreads` for it - which measured slower than the Triton
/// kernel at batch 1 and 8, where there is nothing else to hide the barriers
/// behind. A warp shuffle needs none.
///
/// Both digests in one pass over the configurations: a pass apiece reads the
/// stack `suffixes` times over, and a sequence can hold sixty-four
/// configurations - which cost more than the suffix key buys, 40 to 146 us at
/// batch 32 on the widest schema.
__device__ __forceinline__ int32_t warp_sum(int32_t value) {
    for (int32_t half = 16; half > 0; half >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, half);
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

extern "C" __global__ void en_hash(
    const en::BatchState* state,
    int32_t* hash,
    int32_t* suffix_hash,
    int32_t configs,
    int32_t stack_stride,
    int32_t suffixes) {
    int32_t sequence = blockIdx.x;
    int32_t lane = threadIdx.x & 31;
    int32_t warp = threadIdx.x >> 5;
    int32_t warps = blockDim.x >> 5;
    int32_t count = state->config_count[sequence];
    int32_t grammar = state->grammar_of[sequence];

    // The fold is sequential in configuration order - it has to be, the digest
    // of one feeds the next - but the stack sums it folds are not. A sequence
    // can hold sixty-four configurations, and one warp doing sixty-four
    // reductions in a row was 65.8 us against Triton's 32.8 at batch 32 on the
    // widest schema. So the warps compute the sums in parallel and one thread
    // folds the results, which is the only part that must be in order.
    extern __shared__ int32_t parts[];

    for (int32_t config = warp; config < count; config += warps) {
        int32_t row = sequence * configs + config;
        int32_t depth = state->depth[row];
        int32_t part = 0;
        for (int32_t at = lane; at < depth; at += 32) {
            part += state->stack[(int64_t)row * stack_stride + at] * (at + 1);
        }
        int32_t folded = warp_sum(part);
        if (lane == 0) {
            parts[config] = folded;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // The grammar is part of the state: two sequences under different
        // schemas can sit at the same parser state with the same stack and
        // still admit different tokens, so sharing a mask between them would
        // be wrong.
        int32_t digest = en::fold(en::fold((int32_t)en::FNV_OFFSET, grammar), count);
        for (int32_t config = 0; config < count; ++config) {
            int32_t row = sequence * configs + config;
            digest = en::fold(digest, state->lexer_state[row]);
            digest = en::fold(digest, state->depth[row]);
            digest = en::fold(digest, parts[config]);
        }
        hash[sequence] = digest;
    }

    for (int32_t width = 1; width <= suffixes; ++width) {
        __syncthreads();
        for (int32_t config = warp; config < count; config += warps) {
            int32_t row = sequence * configs + config;
            int32_t depth = state->depth[row];
            int32_t floor = depth - min(depth, width);
            int32_t part = 0;
            for (int32_t at = floor + lane; at < depth; at += 32) {
                part += state->stack[(int64_t)row * stack_stride + at] * (at - floor + 1);
            }
            int32_t folded = warp_sum(part);
            if (lane == 0) {
                parts[config] = folded;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int32_t d = en::fold((int32_t)en::FNV_OFFSET, grammar);
            d = en::fold(en::fold(d, count), width);
            for (int32_t config = 0; config < count; ++config) {
                int32_t row = sequence * configs + config;
                int32_t depth = state->depth[row];
                d = en::fold(d, state->lexer_state[row]);
                d = en::fold(d, min(depth, width));
                // A stack shorter than `k` is folded whole, so it can never
                // look like a longer one that happens to end the same way.
                d = en::fold(d, depth <= width ? 1 : 0);
                d = en::fold(d, parts[config]);
            }
            suffix_hash[sequence * suffixes + width - 1] = d;
        }
    }
}

/// Look the state up, then look for a neighbour that will compute it anyway.
///
/// One warp, for the reason `en_hash` is one warp: the comparisons are over a
/// stack whose depth is single digits, and `__syncthreads_or` is a barrier
/// where `__any_sync` is an instruction.
///
/// One lookup and two places, because the question is the same one: has this
/// answer been produced, in an earlier step or by an earlier sequence in this
/// one? Equal states share a fingerprint and probe the same slot, so a
/// neighbour that matches is one that missed the table too - nothing ends up
/// following a follower.
extern "C" __global__ void en_probe(
    const en::BatchState* state,
    const int32_t* hash,
    const int32_t* suffix_hash,
    const int32_t* memo_hash,
    const int32_t* memo_lexer,
    const int32_t* memo_stack,
    const int32_t* memo_depth,
    const int32_t* memo_count,
    const int32_t* memo_grammar,
    const int32_t* memo_read,
    int32_t* mask,
    int32_t* memo_slot,
    int32_t* representative,
    int32_t* row_floor,
    int32_t* memo_store,
    int32_t batch,
    int32_t configs,
    int32_t stack_stride,
    int32_t slots,
    int32_t memo_configs,
    int32_t memo_stride,
    int32_t suffixes,
    int32_t mask_words) {
    int32_t sequence = blockIdx.x;
    int32_t lane = threadIdx.x;
    int32_t count = state->config_count[sequence];
    int32_t mine = state->grammar_of[sequence];

    int32_t found = -1;
    for (int32_t attempt = 0; attempt <= suffixes && found < 0; ++attempt) {
        int32_t want = attempt == 0 ? -1 : attempt;
        int32_t digest = attempt == 0
            ? hash[sequence]
            : suffix_hash[sequence * suffixes + attempt - 1];
        int32_t slot = (int32_t)(((uint32_t)digest & 0x7fffffffu) % (uint32_t)slots);
        if (count > memo_configs) {
            continue;
        }
        bool same = memo_hash[slot] == digest && memo_read[slot] == want
                    && memo_count[slot] == count && memo_grammar[slot] == mine;
        for (int32_t config = 0; config < count && same; ++config) {
            int32_t row = sequence * configs + config;
            int64_t held = (int64_t)slot * memo_configs + config;
            int32_t depth = state->depth[row];
            int32_t kept = want > 0 ? min(depth, want) : depth;
            if (state->lexer_state[row] != memo_lexer[held]
                || kept != memo_depth[held]) {
                same = false;
            }
            int32_t differs = 0;
            for (int32_t at = lane; at < kept && at < memo_stride; at += 32) {
                if (state->stack[(int64_t)row * stack_stride + depth - kept + at]
                    != memo_stack[held * memo_stride + at]) {
                    differs = 1;
                }
            }
            if (__any_sync(0xffffffffu, differs)) {
                same = false;
            }
        }
        if (same) {
            found = slot;
        }
    }
    if (lane == 0) {
        memo_slot[sequence] = found;
    }

    // Every sequence walks every earlier one, so this is quadratic - which is
    // why the candidates are screened by fingerprint before any stack is read.
    int32_t neighbour = sequence;
    if (found < 0) {
        int32_t digest = hash[sequence];
        for (int32_t other = 0; other < sequence && neighbour == sequence; ++other) {
            if (hash[other] != digest || state->config_count[other] != count
                || state->grammar_of[other] != mine) {
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
                for (int32_t at = lane; at < depth; at += 32) {
                    if (state->stack[(int64_t)a * stack_stride + at]
                        != state->stack[(int64_t)b * stack_stride + at]) {
                        differs = 1;
                    }
                }
                if (__any_sync(0xffffffffu, differs)) {
                    same = false;
                }
            }
            if (same) {
                neighbour = other;
            }
        }
    }
    if (lane == 0) {
        representative[sequence] = neighbour;
    }

    // Only what computes has to start empty; everything else is written whole
    // by the copy. Clearing all of them was 9.7 MB a step to make room for
    // 0.6 MB of answers.
    if (found < 0 && neighbour == sequence) {
        for (int32_t at = lane; at < mask_words; at += 32) {
            mask[(int64_t)sequence * mask_words + at] = 0;
        }
    }

    // Seed the floors the sweep reduces.
    for (int32_t config = lane; config < count; config += 32) {
        int32_t row = sequence * configs + config;
        row_floor[row] = state->depth[row];
    }

    // Whether this sequence may put its answer in the table. Only one that
    // computed it may - a sequence that copied would store what it was given,
    // and a chain of followers is how an entry outlives the state it describes.
    //
    // Leaving this out is silent and total: the CUDA memo stored nothing at
    // all, so every step recomputed every mask, and it showed up two kernels
    // downstream as a *Triton* scatter costing 2.5x - because with nothing to
    // copy, every row was one the sweep had to build.
    if (lane == 0) {
        memo_store[sequence] =
            (found < 0 && neighbour == sequence && count <= memo_configs) ? 1 : -1;
    }
}

/// Give every sequence that did not compute the mask it was promised.
///
/// Two sources and one pass, because the probe already decided which: a table
/// slot for a state an earlier step masked, or a neighbour's row for one this
/// step is masking anyway.
///
/// **Before the store**, so a sequence reading a slot reads the entry it
/// matched rather than one a provider has since replaced.
extern "C" __global__ void en_copy(
    int32_t* mask,
    const int32_t* memo_slot,
    const int32_t* representative,
    const int32_t* memo_mask,
    const en::BatchState* state,
    const int32_t* row_floor,
    int32_t* memo_want,
    int32_t mask_words,
    int32_t configs,
    int32_t memo_stride,
    int32_t suffixes) {
    int32_t sequence = blockIdx.y;
    int32_t slot = memo_slot[sequence];
    int32_t source = representative[sequence];

    // How much of the stack the answer turned out to need - the fill's last
    // phase, which has to see every `row_floor` the sweep wrote. The sweep now
    // spreads one sequence over several blocks, so it cannot decide this
    // itself, and this is the next kernel in the chain rather than a node of
    // its own. `en_store` is too late: its rival scan reads *other* sequences'
    // answers, and nothing orders those against a block that has not run.
    if (blockIdx.x == 0 && threadIdx.x == 0 && slot < 0 && source == sequence) {
        int32_t count = state->config_count[sequence];
        int32_t need = 1;
        bool keep = true;
        for (int32_t config = 0; config < count; ++config) {
            int32_t row = sequence * configs + config;
            int32_t depth = state->depth[row];
            need = max(need, depth - row_floor[row] + 1);
            if (depth > memo_stride) {
                keep = false;
            }
        }
        memo_want[sequence] = (keep && need <= suffixes) ? need : (keep ? -1 : -2);
    }
    if (slot < 0 && source == sequence) {
        return;
    }
    for (int32_t at = blockIdx.x * blockDim.x + threadIdx.x; at < mask_words;
         at += gridDim.x * blockDim.x) {
        mask[(int64_t)sequence * mask_words + at] =
            slot >= 0 ? memo_mask[(int64_t)slot * mask_words + at]
                      : mask[(int64_t)source * mask_words + at];
    }
}

/// Remember a mask this step had to compute.
///
/// Written after the scatter, so what is stored is the finished row. Only a
/// representative that missed is stored - a hit is already there, and a
/// duplicate has not computed anything.
///
/// A state too wide or too deep for an entry is simply not remembered. The
/// bound keeps the table small enough to be worth having, and the states that
/// exceed it are rare; leaving them out costs a recomputation rather than a
/// wrong answer.
extern "C" __global__ void en_store(
    const int32_t* lexer_state,
    const int32_t* stack,
    const int32_t* stack_depth,
    const int32_t* config_count,
    const int32_t* grammar_of,
    const int32_t* state_hash,
    const int32_t* memo_want,
    const int32_t* suffix_hash,
    int32_t* memo_read,
    int32_t* memo_hash,
    int32_t* memo_lexer,
    int32_t* memo_stack,
    int32_t* memo_depth,
    int32_t* memo_count,
    int32_t* memo_grammar,
    int32_t* memo_mask,
    const int32_t* mask,
    int32_t mask_words,
    int32_t configs,
    int32_t stack_stride,
    int32_t slots,
    int32_t memo_configs,
    int32_t memo_stride,
    int32_t suffixes) {
    int32_t sequence = blockIdx.y;
    int32_t want = memo_want[sequence];
    if (want == -2) {
        return;
    }
    int32_t digest =
        want > 0 ? suffix_hash[sequence * suffixes + want - 1] : state_hash[sequence];
    int32_t slot = (digest & 0x7FFFFFFF) % slots;

    // One writer per slot. Two sequences whose fingerprints collide would
    // otherwise interleave and leave an entry holding one state and the other's
    // mask, which a later probe matches and hands back. Decided by a scan rather
    // than an atomic so the answer does not depend on block order - and the scan
    // is over the batch, which is why the whole block joins it.
    int rival = 0;
    for (int32_t other = threadIdx.x; other < sequence; other += blockDim.x) {
        int32_t theirs_k = memo_want[other];
        if (theirs_k == -2) {
            continue;
        }
        int32_t theirs = theirs_k > 0 ? suffix_hash[other * suffixes + theirs_k - 1]
                                      : state_hash[other];
        if (((theirs & 0x7FFFFFFF) % slots) == slot) {
            rival = 1;
        }
    }
    if (__syncthreads_or(rival)) {
        return;
    }

    if (blockIdx.x == 0) {
        int32_t count = config_count[sequence];
        for (int32_t config = threadIdx.x; config < count; config += blockDim.x) {
            int32_t row = sequence * configs + config;
            int32_t depth = stack_depth[row];
            int32_t kept = want > 0 && want < depth ? want : depth;
            memo_lexer[slot * memo_configs + config] = lexer_state[row];
            memo_depth[slot * memo_configs + config] = kept;
        }
        for (int32_t at = threadIdx.x; at < count * memo_stride; at += blockDim.x) {
            int32_t config = at / memo_stride;
            int32_t lane = at - config * memo_stride;
            int32_t row = sequence * configs + config;
            int32_t depth = stack_depth[row];
            int32_t kept = want > 0 && want < depth ? want : depth;
            if (lane < kept) {
                memo_stack[((int64_t)slot * memo_configs + config) * memo_stride + lane] =
                    stack[(int64_t)row * stack_stride + depth - kept + lane];
            }
        }
        if (threadIdx.x == 0) {
            memo_count[slot] = count;
            memo_grammar[slot] = grammar_of[sequence];
            memo_read[slot] = want;
        }
        // The fingerprint last, so a reader that sees it finds the rest of the
        // entry already written.
        __syncthreads();
        __threadfence();
        if (threadIdx.x == 0) {
            memo_hash[slot] = digest;
        }
    }

    for (int32_t at = blockIdx.x * blockDim.x + threadIdx.x; at < mask_words;
         at += gridDim.x * blockDim.x) {
        memo_mask[(int64_t)slot * mask_words + at] = mask[(int64_t)sequence * mask_words + at];
    }
}

/// Put a kept parse state back. The slot is known on the host here.
///
/// Unlike the advance, a rollback is not part of a captured decode step - it
/// happens when a draft is rejected, which the host already knows about - so
/// the slot may be an argument.
extern "C" __global__ void en_restore(
    int32_t* lexer_state,
    int32_t* stack,
    int32_t* stack_depth,
    int32_t* config_count,
    int32_t* widest,
    const int32_t* hist_lexer,
    const int32_t* hist_stack,
    const int32_t* hist_depth,
    const int32_t* hist_count,
    int32_t slot,
    int32_t rows,
    int32_t configs,
    int32_t stack_stride) {
    int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    int64_t at = (int64_t)slot * rows;
    lexer_state[row] = hist_lexer[at + row];
    int32_t depth = hist_depth[at + row];
    stack_depth[row] = depth;
    if (row < rows / configs) {
        int32_t count = hist_count[slot * (rows / configs) + row];
        config_count[row] = count;
        atomicMax(widest, count);
    }
    for (int32_t up = 0; up < stack_stride; ++up) {
        stack[(int64_t)row * stack_stride + up] = hist_stack[(at + row) * stack_stride + up];
    }
}

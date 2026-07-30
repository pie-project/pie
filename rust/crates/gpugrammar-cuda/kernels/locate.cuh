// Which group holds the sampled token, for each live configuration.
//
// The first kernel of the port that does real work. It has the two things
// every later one needs - arena lookups through a grammar's bases, and CSR
// traversal - and not the third, the reduction chain, which is why it is
// first.
//
// **The mapping is not the Triton one.** Triton scans `GROUP_BLOCK` groups as
// a 64-wide tensor and reduces with `tl.min`. Here a *warp* owns a
// configuration and its lanes take groups strided by 32, reduced with
// `__shfl_down_sync`. Same shape, but the lanes are real lanes: a lane that
// decides its group early stops, where a masked tensor operation still costs
// every element. That is the difference the whole rewrite is for.

#pragma once
#include "arena.cuh"

namespace gg {

// How a group's token set is stored. Three shapes because one does not fit:
// a set of four tokens is a list, a set of forty thousand is a bitset, and a
// set that is nearly everything is cheaper as the handful it excludes.
enum SetKind : int32_t { SPARSE = 0, COMPLEMENT = 1, DENSE = 2 };

// No group holds it. The maximum, so that a minimum over "found" answers can
// start here and need no separate "did anything find it" flag.
constexpr int32_t NO_GROUP = 2147483647;

/// Does this group's set hold `token`?
__device__ __forceinline__ bool set_holds(
    const Arena* arena,
    int32_t payload_base,
    int32_t kind,
    int32_t offset,
    int32_t length,
    int32_t token) {
    const int32_t* payload = arena->set_payload + payload_base + offset;
    if (kind == DENSE) {
        return ((payload[token >> 5] >> (token & 31)) & 1) == 1;
    }
    // A sorted list's ends are its bounds, so most groups are decided without
    // searching at all - which is what this kernel is mostly doing, since the
    // groups of a state are disjoint and at most one of them can hold a token.
    if (length <= 0 || token < payload[0] || token > payload[length - 1]) {
        return kind == COMPLEMENT;
    }
    int32_t low = 0, high = length;
    bool found = false;
    while (low < high) {
        int32_t middle = (low + high) >> 1;
        int32_t value = payload[middle];
        if (value == token) {
            found = true;
            break;
        }
        if (value < token) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return kind == COMPLEMENT ? !found : found;
}

/// The verdict two-bit code for one (parser state, group slot), or 0 when the
/// grammar carries no verdict table.
///
/// Settled at compile time for 92.5% of replays, and 91% of those are refusals
/// - a group the tables already refused for this parser state cannot be the
/// one that advances, so this is most of the rejection done before the device
/// sees it.
__device__ __forceinline__ int32_t verdict_of(
    const Arena* arena,
    const int32_t* row,
    int32_t stride,
    int32_t slot) {
    if (row == nullptr || stride <= 0) {
        return 0;
    }
    return (row[slot >> 4] >> (2 * (slot & 15))) & 3;
}

/// Which row owns work item `item`, given that row's CSR offsets. The last row
/// whose offset is at or below it, since rows contributing nothing have equal
/// offsets and taking the last steps past them.
__device__ __forceinline__ int32_t owner(const int32_t* offsets, int32_t rows, int32_t item) {
    int32_t low = 0, high = rows - 1;
    while (low < high) {
        int32_t middle = (low + high + 1) >> 1;
        if (offsets[middle] <= item) { low = middle; } else { high = middle - 1; }
    }
    return low;
}

/// Search one configuration's groups. One warp, lanes strided over the groups.
__device__ __forceinline__ int32_t locate_one(
    const Arena* arena,
    const BatchState* state,
    const Shape shape,
    const int32_t* token_of,
    int32_t sequence,
    int32_t row,
    int32_t lane,
    bool has_verdicts) {
    int32_t grammar = state->grammar_of[sequence];
    int32_t lexer = state->lexer_state[row];
    int32_t token = token_of[sequence];

    int32_t group_base = base_of(arena, grammar, B_GROUP_OFFSETS);
    int32_t groups = base_of(arena, grammar, B_GROUPS);
    int32_t payload_base = base_of(arena, grammar, B_SET_PAYLOAD);
    int32_t first = arena->group_offsets[group_base + lexer];
    int32_t last = arena->group_offsets[group_base + lexer + 1];

    // The verdict row for this (lexer state, parser state), resolved once
    // rather than per group.
    const int32_t* verdict_row = nullptr;
    int32_t stride = 0;
    // Whether the *pool* carries verdicts at all, which is a property of the
    // mixture: one grammar too large for the table turns the shortcut off for
    // every grammar, because the kernel is compiled once for the pool.
    if (has_verdicts) {
        stride = arena->verdict_stride[base_of(arena, grammar, B_VERDICT_STRIDE) + lexer];
        int32_t depth = state->depth[row];
        if (stride > 0 && depth > 0) {
            int32_t top = state->stack[(int64_t)row * shape.stack_stride + depth - 1];
            verdict_row = arena->verdicts + base_of(arena, grammar, B_VERDICTS)
                + arena->verdict_offsets[base_of(arena, grammar, B_VERDICT_OFFSETS) + lexer]
                + (int64_t)top * stride;
        }
    }

    // The earliest group that holds it. The reference matcher takes the
    // earliest, and the groups are disjoint, so this is a minimum rather than
    // a choice - but two lanes can still both find one when a grammar is
    // ambiguous, and then the rule has to be the same as the matcher's.
    int32_t best = NO_GROUP;
    for (int32_t group = first + lane; group < last; group += 32) {
        int32_t kind = arena->group_set_kind[groups + group];
        int32_t offset = arena->group_set_offset[groups + group];
        int32_t length = arena->group_set_length[groups + group];
        if (!set_holds(arena, payload_base, kind, offset, length, token)) {
            continue;
        }
        if (verdict_of(arena, verdict_row, stride, group - first) == 1) {
            continue;  // refused at compile time
        }
        best = min(best, group);
    }
    // Warp reduction rather than a shared-memory pass: the answer is one
    // number and the lanes already hold it between them.
    for (int32_t half = 16; half > 0; half >>= 1) {
        best = min(best, __shfl_down_sync(0xffffffffu, best, half));
    }
    return best;
}

}  // namespace gg

/// A warp per live configuration, grid-strided over the work list.
///
/// The state copy inside it is `int4` rather than scalar, which is worth
/// saying because it was measured rather than assumed: at four `int32` to a
/// vector a warp moves a 1,024-byte row in two steps instead of eight, and
/// that halved this kernel's cost. Giving the copy to the whole block instead
/// was tried and is *worse* - every chunk then needs its own owner search, and
/// that costs more than the extra threads recover.
extern "C" __global__ void gg_locate(
    const gg::Arena* arena,
    const gg::BatchState* state,
    const int32_t* token,
    const int32_t* live_offsets,
    int32_t* found,
    int32_t* old_lexer,
    int32_t* old_count,
    int32_t* old_stack,
    int32_t batch,
    int32_t configs,
    int32_t stack_stride,
    int32_t rows,
    int32_t has_verdicts) {
    gg::Shape shape{batch, configs, stack_stride, 0, rows};
    int32_t lane = threadIdx.x & 31;
    int32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int32_t warps = (gridDim.x * blockDim.x) >> 5;
    int32_t total = live_offsets[rows];

    for (int32_t slot = warp; slot < total; slot += warps) {
        int32_t row = gg::owner(live_offsets, rows, slot);
        int32_t sequence = row / configs;

        // The old state, saved here rather than in a launch of its own: the
        // advance writes new configurations while reading old ones, so the
        // copy has to happen before the candidates are built - and this warp
        // already owns the row.
        if (lane == 0) {
            old_lexer[row] = state->lexer_state[row];
            old_count[sequence] = state->config_count[sequence];
        }
        if ((stack_stride & 3) == 0) {
            const int4* from =
                reinterpret_cast<const int4*>(state->stack + (int64_t)row * stack_stride);
            int4* to = reinterpret_cast<int4*>(old_stack + (int64_t)row * stack_stride);
            for (int32_t at = lane; at < (stack_stride >> 2); at += 32) {
                to[at] = from[at];
            }
        } else {
            for (int32_t at = lane; at < stack_stride; at += 32) {
                old_stack[(int64_t)row * stack_stride + at] =
                    state->stack[(int64_t)row * stack_stride + at];
            }
        }

        int32_t best =
            gg::locate_one(arena, state, shape, token, sequence, row, lane, has_verdicts != 0);
        if (lane == 0) {
            found[row] = best;
        }
    }
}

/// The same kernel with the state copy removed, to attribute its cost. Not
/// used by the engine; kept because "which half is slow" is a question that
/// comes back every time a kernel is changed.
extern "C" __global__ void gg_locate_no_copy(
    const gg::Arena* arena,
    const gg::BatchState* state,
    const int32_t* token,
    const int32_t* live_offsets,
    int32_t* found,
    int32_t* old_lexer,
    int32_t* old_count,
    int32_t* old_stack,
    int32_t batch,
    int32_t configs,
    int32_t stack_stride,
    int32_t rows,
    int32_t has_verdicts) {
    gg::Shape shape{batch, configs, stack_stride, 0, rows};
    int32_t lane = threadIdx.x & 31;
    int32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int32_t warps = (gridDim.x * blockDim.x) >> 5;
    int32_t total = live_offsets[rows];
    for (int32_t slot = warp; slot < total; slot += warps) {
        int32_t row = gg::owner(live_offsets, rows, slot);
        int32_t sequence = row / configs;
        int32_t best = gg::locate_one(arena, state, shape, token, sequence, row, lane, has_verdicts != 0);
        if (lane == 0) { found[row] = best; }
    }
}

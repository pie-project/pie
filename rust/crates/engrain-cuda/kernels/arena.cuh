// The tables and the batch, as two structs instead of fifty kernel arguments.
//
// Triton takes one argument per array, so `_fill` and `_advance` pass 246
// argument slots across eleven launches and the same seven pointers appear in
// five of them. That is not a style complaint: every one of those is a chance
// to pass the right pointer in the wrong position, and the compiler cannot
// see it because they are all `int32*`.
//
// Here the pool's twenty-five arrays are one `Arena`, uploaded when the pool's
// arrays move and not again, and a kernel takes `const Arena*`. The field
// order is generated from `_ARENA` in `_engine.py`, which is the same list the
// Triton launches are built from, so the two cannot drift.

#pragma once
#include <stdint.h>

namespace en {

// Per-grammar base offsets into the arena, `NBASES` of them per grammar. A
// pool is one flat set of arrays holding every admitted grammar end to end, so
// a table lookup is `array[base + index]` and the base comes from here.
enum Base : int32_t {
    B_GROUP_OFFSETS = 0,
    B_GROUPS = 1,
    B_SET_PAYLOAD = 2,
    B_READING_OFFSETS = 3,
    B_READING_INDEX = 4,
    B_READINGS = 5,
    B_READING_TERM_OFFSETS = 6,
    B_READING_TERMINALS = 7,
    B_ACTION_OFFSETS = 8,
    B_ACTIONS = 9,
    B_GOTO_OFFSETS = 10,
    B_GOTOS = 11,
    B_PRODUCTIONS = 12,
    B_PENDING_OFFSETS = 13,
    B_PENDING_TERMINALS = 14,
    B_ACTION_EXTRA_OFFSETS = 15,
    B_ACTION_EXTRA = 16,
    B_VERDICT_OFFSETS = 17,
    B_VERDICTS = 18,
    B_VERDICT_STRIDE = 19,
    NBASES = 20,
};

// Every array the pool holds, in the order `_ARENA` declares them, plus the
// bases. Uploaded once per pool revision.
//
// `const` throughout: a kernel never writes the tables. That is worth having
// the compiler enforce, because a table written during a step would be a
// corruption every later sequence inherits.
struct Arena {
    const int32_t* group_offsets;
    const int32_t* group_set_kind;
    const int32_t* group_set_offset;
    const int32_t* group_set_length;
    const int32_t* set_payload;
    const int32_t* reading_offsets;
    const int32_t* reading_index;
    const int32_t* reading_next_state;
    const int32_t* reading_term_offsets;
    const int32_t* reading_terminals;
    const int32_t* action_offsets;
    const int32_t* action_terminals;
    const int32_t* action_values;
    const int32_t* goto_offsets;
    const int32_t* goto_nonterminals;
    const int32_t* goto_targets;
    const int32_t* production_lhs;
    const int32_t* production_arity;
    const int32_t* pending_offsets;
    const int32_t* pending_terminals;
    const int32_t* action_extra_offsets;
    const int32_t* action_extra;
    const int32_t* verdict_offsets;
    const int32_t* verdicts;
    const int32_t* verdict_stride;
    // Not in `_ARENA` because it is not a table but the index into them.
    const int32_t* bases;
};

// How many pointer-sized slots `Arena` occupies. Python packs that many and
// asserts this number, so adding a field to one side and not the other is a
// failure at the first launch rather than a wrong table at the tenth.
constexpr int ARENA_SLOTS = sizeof(Arena) / sizeof(const int32_t*);

// A grammar's base for one table.
__device__ __forceinline__ int32_t base_of(const Arena* arena, int32_t grammar, Base which) {
    return arena->bases[grammar * NBASES + static_cast<int32_t>(which)];
}

// The per-sequence parse state a step reads and writes. Uploaded when the
// batch's buffers are made, which is once - they do not move afterwards,
// which is the same property that lets a graph hold their addresses.
struct BatchState {
    int32_t* lexer_state;
    int32_t* stack;
    int32_t* depth;
    int32_t* config_count;
    int32_t* widest;
    int32_t* grammar_of;
    int32_t* terminated;
    int32_t* overflow;
};

// `token` and `mask` are deliberately *not* in here. Everything above is a
// buffer made once whose address a recorded graph holds; those two are
// rebound - the draft walk points them at a row of its own arrays for each
// position of the speculative walk. A struct cached past that aims a kernel at
// the wrong tensor, and rebuilding it inside a capture is a host-to-device
// copy, which a capture forbids. So they are passed per launch, where they
// belong: they are what a step is *about*, not what a batch is made of.

constexpr int BATCH_SLOTS = sizeof(BatchState) / sizeof(int32_t*);

// The shapes. Passed by value rather than through the struct: they are what a
// C++ template would specialise on, and a kernel that reads them from memory
// cannot have its loops unrolled against them.
struct Shape {
    int32_t batch;
    int32_t configs;
    int32_t stack_stride;
    int32_t mask_words;
    int32_t rows;  // batch * configs
};

/// One entry of a candidate's reassembled stack.
///
/// Everything below the candidate's floor belongs to the source configuration
/// and is read from the copy taken before this pass began overwriting the live
/// one; everything at or above it is what the replay produced.
///
/// A function of the slot rather than a value held in a register, because the
/// commit's threads walk the stack in a strided loop rather than owning one
/// entry each. Owning one entry each meant the block had to be as wide as the
/// deepest stack a batch allowed, which put the depth ceiling into the launch:
/// a stack of 512 asked for 512 threads and the fused kernel could not be
/// launched with them at all - "too many resources requested for launch" -
/// while the parse that needed it was a JSON array with no maxItems, which is
/// an ordinary schema rather than a pathological one.
/// `at` is where this candidate's window was packed, not where it would sit in
/// a grid of worst cases. The grid was `rows x readings x window`, a product of
/// four independent ceilings that never co-occur: measured over one real step
/// at batch 512, 8.00 GiB was reserved and 0.01 MiB written.
__device__ inline int32_t stack_entry(
    const int32_t* old_stack,
    const int32_t* cand_window,
    int64_t source_row,
    int32_t at,
    int32_t stack_stride,
    int32_t floor,
    int32_t slot) {
    return slot < floor
        ? old_stack[source_row * stack_stride + slot]
        : cand_window[at + (slot - floor)];
}

}  // namespace en

// Does the arena struct actually describe the pool?
//
// The plumbing under test is that Python packed twenty-six pointers in the
// order `arena.cuh` declares them and that a grammar's base offsets index the
// tables the way `_engine.py` believes. Nothing about the parser, but every
// later kernel is wrong in an invisible way if this is - a table read through
// the wrong field is still a valid `int32*` and still returns numbers.
//
// So this reads a value the host already knows and writes it back to be
// compared. A field pair swapped in the packing changes the answer.

#include "arena.cuh"

extern "C" __global__ void gg_arena_readback(
    const gg::Arena* arena,
    const int32_t* grammar_of,
    int32_t* out,
    int32_t sequences,
    int32_t slots) {
    int32_t sequence = blockIdx.x * blockDim.x + threadIdx.x;
    if (sequence >= sequences) {
        return;
    }
    int32_t grammar = grammar_of[sequence];
    int32_t at = sequence * slots;

    // One reading per table, through the base that table is supposed to use.
    // Every one of these is a number the host can derive independently.
    out[at + 0] = gg::base_of(arena, grammar, gg::B_GROUP_OFFSETS);
    out[at + 1] = gg::base_of(arena, grammar, gg::B_ACTION_OFFSETS);
    out[at + 2] = gg::base_of(arena, grammar, gg::B_GOTO_OFFSETS);

    // The first group offset of this grammar's start lexer state, which is the
    // shape every sweep depends on.
    int32_t group_base = gg::base_of(arena, grammar, gg::B_GROUP_OFFSETS);
    out[at + 3] = arena->group_offsets[group_base];
    out[at + 4] = arena->group_offsets[group_base + 1];

    // And one from each of the three CSR families, so a swap between them
    // shows up rather than cancelling.
    int32_t action_base = gg::base_of(arena, grammar, gg::B_ACTION_OFFSETS);
    out[at + 5] = arena->action_offsets[action_base];
    int32_t goto_base = gg::base_of(arena, grammar, gg::B_GOTO_OFFSETS);
    out[at + 6] = arena->goto_offsets[goto_base];
    int32_t reading_base = gg::base_of(arena, grammar, gg::B_READING_OFFSETS);
    out[at + 7] = arena->reading_offsets[reading_base];

    // The struct's own size, so a mismatch between what Python packed and what
    // the kernel expects is reported rather than read past.
    out[at + 8] = gg::ARENA_SLOTS;
    out[at + 9] = gg::NBASES;
}

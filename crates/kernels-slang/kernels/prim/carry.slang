#pragma once

// A buffering fire's staged rs table (`kernels_slang::ssm::Carry`), as a
// recurrent kernel reads a lane's buffered rows in the pool where they lie:
// lane `l` of the fire has `PIE_CARRY_HEADER` ints from int `l *
// PIE_CARRY_HEADER`:
//
//   at      the buffer token its first own row lands at (-1: it lands none)
//   n       the buffered rows it carries ahead of its own (its replay)
//   tok     the int its carried rows' buffer tokens start at; negative, its
//           carried rows are one run of buffer tokens from `-tok - 1`
//   page    the int its buffer pages start at (page `p` of its buffer holds
//           tokens `p * page_tokens ..`); negative, a beam lane's: its page
//           is resolved through its list (`BEAM-PAGE-LISTS.md` §11.1) from
//           the resolve record at int `-page - 1`, which the group's head
//           step writes ahead of the fire: its list's first pool word, the
//           group's `f0`, the kv page's slots, the buffer's base slot, the
//           pool word of the set's base pages (its resident state) and how
//           many it holds
//   commit  how many rows of its run (carried, then own) the state is
//           committed at
//
// A page is a slot of the pool's rs list: plane `p` of token `t` lies at
// virtual word `page * stride_words + plane_at + (t % page_tokens) *
// plane_row` (`prim/pool.slang`). Needs `carry` (an int buffer), the pool's
// `pie_pool_get` and `params.page_tokens`, `params.stride_words` declared
// before.
//
// A beam lane's buffer token `t` counts from the set's base slot, as any
// lane's does; the head step has checked every page its list resolves to
// lies in the set's buffer (else the lane faults and no fire reads it).

#define PIE_CARRY_HEADER 5u

// Virtual word `v` of the pool: an includer whose pool reads vectors states
// its own word read.
#ifndef PIE_CARRY_POOL_WORD
#define PIE_CARRY_POOL_WORD(v) pie_pool_get(v)
#endif

struct CarryLane {
    int at;
    int n;
    uint tok;
    uint page;
    int commit;
};

CarryLane carry_lane(uint lane) {
    uint h = lane * PIE_CARRY_HEADER;
    CarryLane l;
    l.at = carry[h];
    l.n = carry[h + 1u];
    l.tok = uint(carry[h + 2u]);
    l.page = uint(carry[h + 3u]);
    l.commit = carry[h + 4u];
    return l;
}

// The buffer token of the lane's `j`-th carried row.
uint carry_tok(CarryLane l, uint j) {
    if (int(l.tok) < 0) {
        return uint(-int(l.tok) - 1) + j;
    }
    return uint(carry[l.tok + j]);
}

// The rs page (a slot of the pool's rs list) buffer token `t` lies in: the
// lane's own page list, or a beam lane's list resolved slot by slot as the
// head step resolves it (the set's entry holding the rows of slot
// `(f0 + list[j]) S + s mod S`, `s` the page's first slot, `j` its column).
uint carry_page(CarryLane l, uint t) {
    uint p = t / params.page_tokens;
    if (int(l.page) >= 0) {
        return uint(carry[l.page + p]);
    }
    uint r = uint(-int(l.page) - 1);
    uint list = uint(carry[r]);
    uint f0 = uint(carry[r + 1u]);
    uint kv = max(uint(carry[r + 2u]), 1u);
    uint base = uint(carry[r + 3u]);
    uint bases = uint(carry[r + 4u]);
    uint slot = base + p * params.page_tokens;
    // A slot below the group's first pool page is the prompt's own, the
    // same in every beam.
    uint sigma = slot < f0 * kv ? slot : (f0 + PIE_CARRY_POOL_WORD(list + slot / kv - f0)) * kv + slot % kv;
    return PIE_CARRY_POOL_WORD(bases + (sigma - base) / params.page_tokens);
}

// The virtual word buffer token `t`'s row of a plane starts at.
uint carry_word(CarryLane l, uint t, uint plane_at, uint plane_row) {
    return carry_page(l, t) * params.stride_words + plane_at + (t % params.page_tokens) * plane_row;
}

#pragma once

// A buffering fire's staged rs table (`kernels_slang::ssm::Carry`), as a
// recurrent kernel reads a lane's buffered rows in the pool where they lie:
// lane `l` of the fire has `PIE_CARRY_HEADER` ints from int `l *
// PIE_CARRY_HEADER`:
//
//   at      the buffer token its first own row lands at (-1: it lands none)
//   n       the buffered rows it carries ahead of its own (its replay)
//   tok     the int its carried rows' buffer tokens start at
//   page    the int its buffer pages start at (page `p` of its buffer holds
//           tokens `p * page_tokens ..`)
//   commit  how many rows of its run (carried, then own) the state is
//           committed at
//
// A page is a slot of the pool's rs list: plane `p` of token `t` lies at
// virtual word `page * stride_words + plane_at + (t % page_tokens) *
// plane_row` (`prim/pool.slang`). Needs `carry` (an int buffer), the pool's
// `pie_pool_get` and `params.page_tokens`, `params.stride_words` declared
// before.

#define PIE_CARRY_HEADER 5u

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
    return uint(carry[l.tok + j]);
}

// The virtual word buffer token `t`'s row of a plane starts at.
uint carry_word(CarryLane l, uint t, uint plane_at, uint plane_row) {
    uint page = uint(carry[l.page + t / params.page_tokens]);
    return page * params.stride_words + plane_at + (t % params.page_tokens) * plane_row;
}

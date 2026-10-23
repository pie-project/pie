#pragma once

// Unsigned 64-bit arithmetic on two 32-bit halves: WGSL has no 64-bit
// integer, so one source carries its own. Wrapping, as Rust's `wrapping_*`.

struct U64 {
    uint lo;
    uint hi;
};

U64 u64_of(uint lo, uint hi) {
    U64 r;
    r.lo = lo;
    r.hi = hi;
    return r;
}

// `v` sign-extended, as `v as i64 as u64`.
U64 u64_from_i32(int v) {
    return u64_of(uint(v), v < 0 ? 0xffffffffu : 0u);
}

U64 u64_mul32(uint a, uint b) {
    uint al = a & 0xffffu;
    uint ah = a >> 16;
    uint bl = b & 0xffffu;
    uint bh = b >> 16;
    uint p0 = al * bl;
    uint p1 = al * bh;
    uint p2 = ah * bl;
    uint p3 = ah * bh;
    uint mid = (p0 >> 16) + (p1 & 0xffffu) + (p2 & 0xffffu);
    return u64_of((p0 & 0xffffu) | (mid << 16), p3 + (p1 >> 16) + (p2 >> 16) + (mid >> 16));
}

U64 u64_mul(U64 a, U64 b) {
    U64 low = u64_mul32(a.lo, b.lo);
    return u64_of(low.lo, low.hi + a.lo * b.hi + a.hi * b.lo);
}

U64 u64_xor(U64 a, U64 b) {
    return u64_of(a.lo ^ b.lo, a.hi ^ b.hi);
}

U64 u64_add(U64 a, U64 b) {
    uint lo = a.lo + b.lo;
    return u64_of(lo, a.hi + b.hi + (lo < a.lo ? 1u : 0u));
}

bool u64_ge(U64 a, U64 b) {
    return a.hi > b.hi || (a.hi == b.hi && a.lo >= b.lo);
}

U64 u64_sub(U64 a, U64 b) {
    return u64_of(a.lo - b.lo, a.hi - b.hi - (a.lo < b.lo ? 1u : 0u));
}

// `x % m` by shift and subtract, a bit at a time.
U64 u64_mod(U64 x, U64 m) {
    U64 r = u64_of(0u, 0u);
    for (uint i = 0u; i < 64u; i++) {
        uint bit = i < 32u ? (x.hi >> (31u - i)) & 1u : (x.lo >> (63u - i)) & 1u;
        r = u64_of((r.lo << 1) | bit, (r.hi << 1) | (r.lo >> 31));
        if (u64_ge(r, m)) {
            r = u64_sub(r, m);
        }
    }
    return r;
}

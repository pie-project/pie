struct U64 {
    lo: u32,
    hi: u32,
}

fn u64_at(lo: u32, hi: u32) -> U64 {
    return U64(lo, hi);
}

fn u64_from_i32(v: i32) -> U64 {
    return U64(u32(v), select(0u, 0xffffffffu, v < 0));
}

fn u64_mul32(a: u32, b: u32) -> U64 {
    let al = a & 0xffffu;
    let ah = a >> 16u;
    let bl = b & 0xffffu;
    let bh = b >> 16u;
    let p0 = al * bl;
    let p1 = al * bh;
    let p2 = ah * bl;
    let p3 = ah * bh;
    let mid = (p0 >> 16u) + (p1 & 0xffffu) + (p2 & 0xffffu);
    let lo = (p0 & 0xffffu) | (mid << 16u);
    let hi = p3 + (p1 >> 16u) + (p2 >> 16u) + (mid >> 16u);
    return U64(lo, hi);
}

fn u64_mul(a: U64, b: U64) -> U64 {
    let low = u64_mul32(a.lo, b.lo);
    let hi = low.hi + a.lo * b.hi + a.hi * b.lo;
    return U64(low.lo, hi);
}

fn u64_xor(a: U64, b: U64) -> U64 {
    return U64(a.lo ^ b.lo, a.hi ^ b.hi);
}

fn u64_add(a: U64, b: U64) -> U64 {
    let lo = a.lo + b.lo;
    let carry = select(0u, 1u, lo < a.lo);
    return U64(lo, a.hi + b.hi + carry);
}

fn u64_ge(a: U64, b: U64) -> bool {
    return a.hi > b.hi || (a.hi == b.hi && a.lo >= b.lo);
}

fn u64_sub(a: U64, b: U64) -> U64 {
    let lo = a.lo - b.lo;
    let borrow = select(0u, 1u, a.lo < b.lo);
    return U64(lo, a.hi - b.hi - borrow);
}

fn u64_mod(x: U64, m: U64) -> U64 {
    var r = U64(0u, 0u);
    for (var i = 0u; i < 64u; i = i + 1u) {
        var bit = 0u;
        if (i < 32u) {
            bit = (x.hi >> (31u - i)) & 1u;
        } else {
            bit = (x.lo >> (63u - i)) & 1u;
        }
        r = U64((r.lo << 1u) | bit, (r.hi << 1u) | (r.lo >> 31u));
        if (u64_ge(r, m)) {
            r = u64_sub(r, m);
        }
    }
    return r;
}

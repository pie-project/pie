#pragma once

// The chunk pool as a kernel reads it (SPEC 10, 11): up to four segments,
// each its own buffer (Vulkan and WebGPU bind no buffer past their binding
// limit, so a pool is cut into segments no larger), a chunk never
// straddling two. A kernel names a pool element by its address, a u32 that
// carries its segment: the segment in bits 30-31, the element's index in
// that segment in bits 0-29. A segment is no larger than one binding (at
// most 4 GiB, the host states it so), so an index of 4-byte elements fits
// and no address passes a u32 however large the pool. The element is the
// segments' type (`uint`, `float`, `int`): a word. A kernel reading bf16
// halves (`uint16_t` segments) forms word addresses and doubles the index
// once it has the segment ([`pie_pool_half`]); one reading `uint4` divides
// it by four ([`pie_pool_quarter`]).
//
// A slot as the host stages it (a page table entry, a record's piece, a
// weight slot) is a unit id: bit 31 clear (a negative entry names none),
// its segment in bits 27-30, the unit within the segment in bits 0-26.
// [`pie_unit_at`] turns it into the address of its first word, which a
// kernel then adds within-slot words to (an origin, a row, a column): a
// slot lies in one chunk, so every sum stays in its segment.
//
// The segments sit at four consecutive bindings after the kernel's own
// buffers (a segment the pool does not have is bound to the engine's dummy);
// one binding a segment, chosen by comparisons on the address's segment,
// since WGSL has no arrays of buffers. Strides and origins come from the
// host in u32 words.
//
//   PIE_POOL_SEGMENTS(StructuredBuffer, uint, 9)
//   ... Params with `uint k_origin; uint page_stride; ...`, PIE_PARAMS ...
//   PIE_POOL_READ(uint)
//   uint w = params.k_origin + pie_unit_at(page, params.page_stride) + row_words;
//   uint word = pie_pool_get(w + d);

#define PIE_POOL_SEG_SHIFT 30u
#define PIE_POOL_LOCAL 0x3fffffffu
#define PIE_UNIT_SEG_SHIFT 27u
#define PIE_UNIT_LOCAL 0x07ffffffu

// The address of unit id `id`'s first word, `stride` words a unit.
uint pie_unit_at(uint id, uint stride) {
    return ((id >> PIE_UNIT_SEG_SHIFT) << PIE_POOL_SEG_SHIFT) | ((id & PIE_UNIT_LOCAL) * stride);
}

// Word address `w` as an address of `uint4` elements (its index a multiple
// of four words).
uint pie_pool_quarter(uint w) {
    return (w & ~PIE_POOL_LOCAL) | ((w & PIE_POOL_LOCAL) >> 2);
}

#define PIE_POOL_SEGMENTS(access_, type_, binding_) \
    [[vk::binding(binding_)]] access_<type_> pie_seg0; \
    [[vk::binding(binding_ + 1)]] access_<type_> pie_seg1; \
    [[vk::binding(binding_ + 2)]] access_<type_> pie_seg2; \
    [[vk::binding(binding_ + 3)]] access_<type_> pie_seg3;

// The segment address `v` lies in and its index there.
void pie_pool_find(uint v, out uint seg, out uint local) {
    seg = v >> PIE_POOL_SEG_SHIFT;
    local = v & PIE_POOL_LOCAL;
}

// [`pie_pool_find`] for `uint16_t` segments: word address `w`'s segment and
// the index of its low half there.
void pie_pool_half(uint w, out uint seg, out uint local) {
    seg = w >> PIE_POOL_SEG_SHIFT;
    local = 2u * (w & PIE_POOL_LOCAL);
}

#define PIE_POOL_READ(type_) \
    type_ pie_pool_get(uint v) { \
        uint s = v >> PIE_POOL_SEG_SHIFT; \
        uint l = v & PIE_POOL_LOCAL; \
        if (s == 0u) { return pie_seg0[l]; } \
        if (s == 1u) { return pie_seg1[l]; } \
        if (s == 2u) { return pie_seg2[l]; } \
        return pie_seg3[l]; \
    } \
    /* The segment address `v` lies in and its index there, for a span */ \
    /* inside one segment read through `PIE_POOL_ON`; on WGSL the address */ \
    /* itself, which `PIE_POOL_ON` reads through the chain. */ \
    void pie_pool_at(uint v, out uint seg, out uint local) { \
        if (PIE_POOL_CHAINED) { \
            seg = 0u; \
            local = v; \
            return; \
        } \
        pie_pool_find(v, seg, local); \
    } \
    struct pie_pool_chain_ro { \
        __subscript(uint i)->type_ { \
            get { return pie_pool_get(i); } \
        } \
    }; \
    static pie_pool_chain_ro pie_pool_chain;

// `body_(buffer)` with the segment `seg_` names as `buffer`: a span that
// lies in one segment (a page never straddles a chunk, nor a chunk two
// segments) reads it with no comparison an element. WGSL reads the four
// bindings slower through a branch than through the chain an element, so
// there `pie_pool_at` keeps the address and the body reads the chain
// (`PIE_POOL_ON_RW` for a body that writes too).
#if defined(PIE_WGSL)
#define PIE_POOL_CHAINED true
#define PIE_POOL_ON(seg_, body_) { body_(pie_pool_chain) }
#define PIE_POOL_ON_RW(seg_, body_) { body_(pie_pool_chain_rw) }
#else
#define PIE_POOL_CHAINED false
#define PIE_POOL_ON(seg_, body_) \
    if ((seg_) == 0u) { body_(pie_seg0) } \
    else if ((seg_) == 1u) { body_(pie_seg1) } \
    else if ((seg_) == 2u) { body_(pie_seg2) } \
    else { body_(pie_seg3) }
#define PIE_POOL_ON_RW(seg_, body_) PIE_POOL_ON(seg_, body_)
#endif

// `PIE_POOL_ON` by a branch on every target, its segment from
// `pie_pool_find`: a decode row, a few elements a lane, reads faster so on
// WGSL too (the branch once a row, not an element).
#define PIE_POOL_BRANCH(seg_, body_) \
    if ((seg_) == 0u) { body_(pie_seg0) } \
    else if ((seg_) == 1u) { body_(pie_seg1) } \
    else if ((seg_) == 2u) { body_(pie_seg2) } \
    else { body_(pie_seg3) }

#define PIE_POOL_WRITE(type_) \
    void pie_pool_set(uint v, type_ x) { \
        uint s = v >> PIE_POOL_SEG_SHIFT; \
        uint l = v & PIE_POOL_LOCAL; \
        if (s == 0u) { pie_seg0[l] = x; return; } \
        if (s == 1u) { pie_seg1[l] = x; return; } \
        if (s == 2u) { pie_seg2[l] = x; return; } \
        pie_seg3[l] = x; \
    } \
    struct pie_pool_chain_rw_t { \
        __subscript(uint i)->type_ { \
            get { return pie_pool_get(i); } \
            set { pie_pool_set(i, newValue); } \
        } \
    }; \
    static pie_pool_chain_rw_t pie_pool_chain_rw;

// A record cut into pieces (SPEC 4: a fixed row larger than a slot is cut on
// its atom axis into equal pieces, each in a slot of its own at one common
// offset). A state kernel's `slots` table holds `params.pieces` units a lane
// or a row, the record's slots in order; this row's piece `p` lies in slot
// `params.piece_first + p`, `params.piece_atoms` atoms a piece. Needs
// `slots` and `params` (`pieces`, `piece_first`, `piece_atoms`,
// `slot_stride`) declared before.
#define PIE_STATE_PIECES \
    uint pie_piece_unit(uint at, uint atom) { \
        return uint(slots[at * params.pieces + params.piece_first + atom / params.piece_atoms]); \
    } \
    /* Atom `atom` (each `elems` long, atoms outermost) of entry `at`'s record. */ \
    uint pie_state_atom(uint at, uint atom, uint elems) { \
        return pie_unit_at(pie_piece_unit(at, atom), params.slot_stride) + (atom % params.piece_atoms) * elems; \
    } \
    /* Whether entry `at` names no record (a negative unit). */ \
    bool pie_state_none(uint at) { \
        return int(slots[at * params.pieces]) < 0; \
    }

// Snapshots (CONTRACT-BC 16): a lane folding its rows lands the state as
// of up to `PIE_RS_SNAPSHOTS` rows inside it in records of their own.
// `pie_snaps` holds, a lane (from `params.snap_lane0`; -1: no snapshots this
// fire), `PIE_RS_SNAPSHOTS` rows into the lane (ascending, a row `<= 0`
// ends them), then each snapshot's record as `params.pieces` units, as a
// `slots` entry holds a record. Needs `pie_snaps` and `params` (`pieces`,
// `piece_first`, `piece_atoms`, `slot_stride`, `snap_lane0`) declared
// before.
#define PIE_RS_SNAPSHOTS 4u
#define PIE_STATE_SNAPSHOTS \
    /* The snapshot lane of request `r`, or -1 where this fire lands none. */ \
    int pie_snap_lane(uint r) { \
        return params.snap_lane0 >= 0 ? params.snap_lane0 + int(r) : -1; \
    } \
    uint pie_snap_entry(uint lane) { \
        return lane * PIE_RS_SNAPSHOTS * (1u + params.pieces); \
    } \
    /* Snapshot `n`'s row into lane `lane`, `<= 0` past its last. */ \
    int pie_snap_row(int lane, uint n) { \
        if (lane < 0 || n >= PIE_RS_SNAPSHOTS) { \
            return 0; \
        } \
        return pie_snaps[pie_snap_entry(uint(lane)) + n]; \
    } \
    /* The unit of snapshot `n`'s record that holds atom `atom`. */ \
    uint pie_snap_unit(uint lane, uint n, uint atom) { \
        return uint(pie_snaps[pie_snap_entry(lane) + PIE_RS_SNAPSHOTS + n * params.pieces \
            + params.piece_first + atom / params.piece_atoms]); \
    } \
    /* Atom `atom` (each `elems` long) of snapshot `n`'s record. */ \
    uint pie_snap_atom(uint lane, uint n, uint atom, uint elems) { \
        return pie_unit_at(pie_snap_unit(lane, n, atom), params.slot_stride) + (atom % params.piece_atoms) * elems; \
    }

// A read-only window on the pool from `origin_` on.
#define PIE_POOL_VIEW_RO(name_, type_, origin_) \
    struct name_##_pool_view { \
        __subscript(uint i)->type_ { \
            get { return pie_pool_get((origin_) + i); } \
        } \
    }; \
    static name_##_pool_view name_;

// A read-write window on the pool from `origin_` on.
#define PIE_POOL_VIEW(name_, type_, origin_) \
    struct name_##_pool_view { \
        __subscript(uint i)->type_ { \
            get { return pie_pool_get((origin_) + i); } \
            set { pie_pool_set((origin_) + i, newValue); } \
        } \
    }; \
    static name_##_pool_view name_;

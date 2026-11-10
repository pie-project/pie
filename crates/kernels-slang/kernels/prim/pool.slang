#pragma once

// The chunk pool as a kernel reads it (SPEC 10, 11): one address space of
// up to four segments, each its own buffer (Vulkan and WebGPU bind no buffer
// past their binding limit, so a pool is cut into segments no larger), a
// chunk never straddling two. A kernel names a pool element by its virtual
// index, `segment * seg + local`: `base + slot * stride + offset` with the
// slot's chunk anywhere in the pool.
//
// The segments sit at four consecutive bindings after the kernel's own
// buffers (a segment the pool does not have is bound to the engine's dummy);
// one binding a segment, chosen by comparisons, since WGSL has no arrays of
// buffers. Offsets and strides come from the host in u32 words; a kernel
// reading bf16 halves scales them by 2.
//
//   PIE_POOL_SEGMENTS(StructuredBuffer, uint16_t, 9)
//   ... Params with `uint seg_words; uint k_origin; ...`, PIE_PARAMS ...
//   PIE_POOL_READ(uint16_t, 2u * params.seg_words)
//   PIE_POOL_VIEW_RO(k_pages, uint16_t, 2u * params.k_origin)
//
// after which `k_pages[i]` reads virtual element `k_origin + i`.

#define PIE_POOL_SEGMENTS(access_, type_, binding_) \
    [[vk::binding(binding_)]] access_<type_> pie_seg0; \
    [[vk::binding(binding_ + 1)]] access_<type_> pie_seg1; \
    [[vk::binding(binding_ + 2)]] access_<type_> pie_seg2; \
    [[vk::binding(binding_ + 3)]] access_<type_> pie_seg3;

#define PIE_POOL_READ(type_, seg_) \
    type_ pie_pool_get(uint v) { \
        uint s = (seg_); \
        if (v < s) { return pie_seg0[v]; } \
        v -= s; \
        if (v < s) { return pie_seg1[v]; } \
        v -= s; \
        if (v < s) { return pie_seg2[v]; } \
        return pie_seg3[v - s]; \
    }

#define PIE_POOL_WRITE(type_, seg_) \
    void pie_pool_set(uint v, type_ x) { \
        uint s = (seg_); \
        if (v < s) { pie_seg0[v] = x; return; } \
        v -= s; \
        if (v < s) { pie_seg1[v] = x; return; } \
        v -= s; \
        if (v < s) { pie_seg2[v] = x; return; } \
        pie_seg3[v - s] = x; \
    }

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
        return pie_piece_unit(at, atom) * params.slot_stride + (atom % params.piece_atoms) * elems; \
    } \
    /* Whether entry `at` names no record (a negative unit). */ \
    bool pie_state_none(uint at) { \
        return int(slots[at * params.pieces]) < 0; \
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

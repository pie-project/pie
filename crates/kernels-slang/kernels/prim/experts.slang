#pragma once

#include "prim/bf16.slang"
#include "prim/pool.slang"

// A routed bank read through the weight cache's expert table (SPEC "Weight
// cache"): an expert of a MoE layer is one atom, its plane of every bank
// of the layer back to back in one slot. `pie_xt_units[atom_first + e]`
// names the pool unit expert `e`'s slot starts at, or `PIE_XT_NOT_HELD`;
// an expert no slot holds was staged for this fire, `pie_xt_staged[..]`
// its seat of `pie_xt_stage` (a slot's words a seat). A bank's plane starts
// `origin` words into the atom, laid as one expert of the contiguous bank,
// so a kernel reads the same element wherever the expert lies.
//
// A slot lies in one chunk and so in one segment: the segment (or the
// stage) is chosen once an expert (`pie_xt_at`), the reads after it walk
// one buffer.
//
//   PIE_EXPERT_TABLE(3)          // eight bindings from 3: units, staged,
//                                // stage, hits, the four pool segments
//   ... Params with seg_words, unit_words, atom_words, atom_first ...
//   pie_xt_at(e, params.atom_first, params.seg_words, params.unit_words,
//             params.atom_words);
//   float w = pie_xt_bf16(origin, i);   // element i of the expert's plane

#define PIE_XT_NOT_HELD 0xffffffffu

#define PIE_EXPERT_TABLE(first_) \
    [[vk::binding(first_)]] StructuredBuffer<uint> pie_xt_units; \
    [[vk::binding(first_ + 1)]] StructuredBuffer<uint> pie_xt_staged; \
    [[vk::binding(first_ + 2)]] StructuredBuffer<uint> pie_xt_stage; \
    [[vk::binding(first_ + 3)]] RWStructuredBuffer<Atomic<uint>> pie_xt_hits; \
    PIE_POOL_SEGMENTS(StructuredBuffer, uint, first_ + 4)

// The buffer the current expert lies in (0..3 a segment, 4 the stage, 5
// nowhere: reads land 0), and its slot's first word there.
static uint pie_xt_which;
static uint pie_xt_base;

void pie_xt_at(uint e, uint atom_first, uint seg_words, uint unit_words, uint atom_words) {
    uint a = atom_first + e;
    uint unit = pie_xt_units[a];
    if (unit != PIE_XT_NOT_HELD) {
        uint v = unit * unit_words;
        uint s = min(v / max(seg_words, 1u), 3u);
        pie_xt_which = s;
        pie_xt_base = v - s * seg_words;
    } else {
        uint seat = pie_xt_staged[a];
        pie_xt_which = seat != PIE_XT_NOT_HELD ? 4u : 5u;
        pie_xt_base = seat != PIE_XT_NOT_HELD ? seat * atom_words : 0u;
    }
}

// One use of expert `e`.
void pie_xt_count(uint e, uint atom_first) {
    pie_xt_hits[atom_first + e].add(1u);
}

// Word `i` of the current expert's slot.
uint pie_xt_word(uint i) {
    uint at = pie_xt_base + i;
    if (pie_xt_which == 0u) {
        return pie_seg0[at];
    }
    if (pie_xt_which == 1u) {
        return pie_seg1[at];
    }
    if (pie_xt_which == 2u) {
        return pie_seg2[at];
    }
    if (pie_xt_which == 3u) {
        return pie_seg3[at];
    }
    if (pie_xt_which == 4u) {
        return pie_xt_stage[at];
    }
    return 0u;
}

// bf16 element `i` of the plane `origin` words into the current expert.
float pie_xt_bf16(uint origin, uint i) {
    uint word = pie_xt_word(origin + (i >> 1));
    return bf16_to_f32((i & 1u) != 0u ? (word >> 16) : (word & 0xffffu));
}

// Byte `i` of the plane `origin` words into the current expert.
uint pie_xt_byte(uint origin, uint i) {
    return (pie_xt_word(origin + (i >> 2)) >> ((i & 3u) * 8u)) & 0xffu;
}

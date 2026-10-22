#pragma once

#include "prim/bf16.slang"

// Elements two at a time: element `i` and, when `both`, element `i + 1`. A
// bf16 buffer holds the pair in one word, so `i` is even; an f32 buffer holds
// two floats. A run may be odd long, its last pair half there: the missing
// half reads as zero and is never written (a bf16 word keeps its high half).
// `PIE_ACT_F32` picks the f32 form; a kernel declares its activations
// `PIE_ACT` and walks them a pair a lane.

#ifdef PIE_ACT_F32
#define PIE_ACT float

float2 pair_at(StructuredBuffer<float> buf, uint i, bool both) {
    return float2(buf[i], both ? buf[i + 1u] : 0.0);
}

float2 pair_at(RWStructuredBuffer<float> buf, uint i, bool both) {
    return float2(buf[i], both ? buf[i + 1u] : 0.0);
}

void pair_put(RWStructuredBuffer<float> buf, uint i, bool both, float2 v) {
    buf[i] = v.x;
    if (both) {
        buf[i + 1u] = v.y;
    }
}
#else
#define PIE_ACT uint

float2 pair_at(StructuredBuffer<uint> buf, uint i, bool both) {
    uint word = buf[i >> 1];
    return float2(bf16_to_f32(word & 0xffffu), both ? bf16_to_f32(word >> 16) : 0.0);
}

float2 pair_at(RWStructuredBuffer<uint> buf, uint i, bool both) {
    uint word = buf[i >> 1];
    return float2(bf16_to_f32(word & 0xffffu), both ? bf16_to_f32(word >> 16) : 0.0);
}

void pair_put(RWStructuredBuffer<uint> buf, uint i, bool both, float2 v) {
    if (both) {
        buf[i >> 1] = pack_bf16(v.x, v.y);
    } else {
        buf[i >> 1] = (buf[i >> 1] & 0xffff0000u) | f32_to_bf16(v.x);
    }
}
#endif

// `v` as the activation holds it: a stated scalar rounds like a stored one.
float act_round(float v) {
#ifdef PIE_ACT_F32
    return v;
#else
    return bf16_to_f32(f32_to_bf16(v));
#endif
}

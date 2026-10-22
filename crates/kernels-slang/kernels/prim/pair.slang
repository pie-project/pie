#pragma once

#include "prim/bf16.slang"

// Elements two at a time: element `i` and, when `both`, element `i + 1`. A
// bf16 buffer holds the pair in one word, so `i` is even and `both` holds; an
// f32 buffer holds two floats, and a run may be odd long, its last pair half
// there. `PIE_ACT_F32` picks the f32 form; a kernel declares its activations
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
    return float2(bf16_to_f32(word & 0xffffu), bf16_to_f32(word >> 16));
}

float2 pair_at(RWStructuredBuffer<uint> buf, uint i, bool both) {
    uint word = buf[i >> 1];
    return float2(bf16_to_f32(word & 0xffffu), bf16_to_f32(word >> 16));
}

void pair_put(RWStructuredBuffer<uint> buf, uint i, bool both, float2 v) {
    buf[i >> 1] = pack_bf16(v.x, v.y);
}
#endif

#pragma once

#include "prim/bf16.slang"

// Rotary turns on bf16 rows two a word. A lane turns two neighbouring pairs,
// so it owns every word it writes: pair 0 at angle `cs.xy` (cos, sin), pair 1
// at `cs.zw`.

// Halves: pair k is (`i1 + k`, `i2 + k`), `i1` and `i2` even, so the lane
// reads one word of each half.
void rotate_halves(RWStructuredBuffer<uint> buf, uint i1, uint i2, float4 cs) {
    uint a = buf[i1 >> 1];
    uint b = buf[i2 >> 1];
    float2 x1 = float2(bf16_to_f32(a & 0xffffu), bf16_to_f32(a >> 16));
    float2 x2 = float2(bf16_to_f32(b & 0xffffu), bf16_to_f32(b >> 16));
    float2 c = cs.xz;
    float2 s = cs.yw;
    float2 lo = x1 * c - x2 * s;
    float2 hi = x1 * s + x2 * c;
    buf[i1 >> 1] = pack_bf16(lo.x, lo.y);
    buf[i2 >> 1] = pack_bf16(hi.x, hi.y);
}

// Interleaved: pair k is (`j + 2k`, `j + 2k + 1`), `j` even, so each pair is
// one word.
void rotate_adjacent(RWStructuredBuffer<uint> buf, uint j, float4 cs) {
    uint w = j >> 1;
    for (uint k = 0u; k < 2u; k++) {
        uint word = buf[w + k];
        float a = bf16_to_f32(word & 0xffffu);
        float b = bf16_to_f32(word >> 16);
        float c = k == 0u ? cs.x : cs.z;
        float s = k == 0u ? cs.y : cs.w;
        buf[w + k] = pack_bf16(a * c - b * s, a * s + b * c);
    }
}

float4 turns(float t0, float t1) {
    return float4(cos(t0), sin(t0), cos(t1), sin(t1));
}

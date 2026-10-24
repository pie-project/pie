#pragma once

// bf16 values two to a 32-bit word, the even element in the low half: the
// one layout a SPIR-V and a WGSL storage buffer both hold, byte for byte the
// layout of a plain bf16 array.

float bf16_to_f32(uint h) {
    return asfloat(h << 16);
}

// Round to nearest even; a NaN stays a quiet NaN.
uint f32_to_bf16(float f) {
    uint bits = asuint(f);
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u) {
        return 0x7fc0u;
    }
    return (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
}

// Element `i` of a packed bf16 buffer.
float bf16_at(StructuredBuffer<uint> buf, uint i) {
    uint word = buf[i >> 1];
    return bf16_to_f32((i & 1u) != 0u ? (word >> 16) : (word & 0xffffu));
}

// Two bf16 values as the word that holds them, `lo` the even element.
uint pack_bf16(float lo, float hi) {
    return f32_to_bf16(lo) | (f32_to_bf16(hi) << 16);
}

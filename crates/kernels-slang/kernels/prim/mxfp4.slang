#pragma once

// MXFP4 (OCP microscaling): four-bit E2M1 codes, eight a word low first, and
// an E8M0 exponent byte a block of `PIE_MXFP4_BLOCK` codes.

#define PIE_MXFP4_BLOCK 32

// An E2M1 code's value.
float mxfp4_code(uint nibble) {
    uint e = (nibble >> 1u) & 3u;
    float mant = float(nibble & 1u);
    float mag = e == 0u ? 0.5 * mant : ldexp(1.0 + 0.5 * mant, int(e) - 1);
    return (nibble & 8u) != 0u ? -mag : mag;
}

// Code `i` (0 to 7) of `word`.
float mxfp4_code_of(uint word, uint i) {
    return mxfp4_code((word >> ((i & 7u) * 4u)) & 0xfu);
}

// An E8M0 block scale; 0xff is NaN.
float mxfp4_block_scale(uint code) {
    if (code == 0xffu) {
        return asfloat(0x7fc00000u);
    }
    return ldexp(1.0, int(code) - 127);
}

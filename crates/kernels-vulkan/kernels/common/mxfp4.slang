#ifndef PIE_VULKAN_MXFP4_SLANG
#define PIE_VULKAN_MXFP4_SLANG

static const float kMxfp4Lut[16] = {
    0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};

float mxfp4_code(uint nibble) {
    uint e = (nibble >> 1u) & 3u;
    float mant = float(nibble & 1u);
    float mag = e == 0u ? 0.5 * mant : ldexp(1.0 + 0.5 * mant, int(e) - 1);
    return (nibble & 8u) != 0u ? -mag : mag;
}

float mxfp4_lo(uint byte_) { return kMxfp4Lut[byte_ & 0xfu]; }
float mxfp4_hi(uint byte_) { return kMxfp4Lut[(byte_ >> 4) & 0xfu]; }

float mxfp4_block_scale(uint code) {
    if (code == 0xffu) {
        return asfloat(0x7fc00000u);
    }
    return ldexp(1.0, int(code) - 127);
}

#define PIE_MXFP4_BLOCK 32

#endif

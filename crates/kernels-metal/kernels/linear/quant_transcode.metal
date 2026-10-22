#include <metal_stdlib>
using namespace metal;

constant float kMxfp4Lut[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

inline float mxfp4_lo(uint8_t byte) { return kMxfp4Lut[byte & 0xf]; }
inline float mxfp4_hi(uint8_t byte) { return kMxfp4Lut[byte >> 4]; }

inline float mxfp4_block_scale(uint8_t code) {
  return code == 0xff ? NAN : metal::ldexp(1.0f, int(code) - 127);
}

kernel void mxfp4_dequant_bf16(
    device const uchar* payload [[buffer(0)]],
    device const uchar* exponents [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    const constant uint& blocks [[buffer(3)]],
    const constant uint& block_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= blocks) return;
    const uint pb = block_size;
    const float factor = mxfp4_block_scale(exponents[gid]);
    const uint first = gid * pb;
    for (uint i = 0; i < pb; i += 2) {
        const uchar byte = payload[(first + i) / 2];
        out[first + i] = bfloat(mxfp4_lo(byte) * factor);
        out[first + i + 1] = bfloat(mxfp4_hi(byte) * factor);
    }
}

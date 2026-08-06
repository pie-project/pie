// MXFP4, in the one place both kernels that read it can see.
//
// Included by `quantized_qmv.metal` and `quantized_qmm_t.metal`. Metal's
// runtime compiler hands each `.metal` file to `newLibraryWithSource:` on its
// own and resolves no local includes, so the driver splices these in itself --
// see `read_metal_source`. Without that this file would be two copies, and the
// three things below are exactly the ones that fail silently when the copies
// drift: the decode path would be right on a prefill and wrong on a decode, or
// the other way round, and the output would still look like text.
//
// E2M1: one sign bit, two exponent bits, one mantissa bit, with the all-zero
// exponent giving the ±0.5 denormals. The values are NOT linear in the code --
// they step by 0.5 up to 2 and then by 1, 2, 2 -- which is the whole reason
// MXFP4 cannot borrow the affine dot, whose trick is that `scale * code + bias`
// is linear in the code.
//
// E8M0: an unsigned power of two, 127-biased, one per 32-element block, with
// 0xff reserved for NaN.

#ifndef PIE_METAL_MXFP4_CODEC_H
#define PIE_METAL_MXFP4_CODEC_H

constant float kMxfp4Lut[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

/// The two codes in one byte, low nibble first. Swapping them transposes every
/// adjacent pair of weights, which is wrong by a plausible amount.
inline float mxfp4_lo(uint8_t byte) { return kMxfp4Lut[byte & 0xf]; }
inline float mxfp4_hi(uint8_t byte) { return kMxfp4Lut[byte >> 4]; }

/// One block's E8M0 exponent as a multiplier. 0xff is the NaN encoding; every
/// other code is an exact power of two.
inline float mxfp4_block_scale(uint8_t code) {
  return code == 0xff ? NAN : metal::ldexp(1.0f, int(code) - 127);
}

#endif

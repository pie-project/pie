// MXFP4, in the one place every kernel that reads it can see.
//
// E2M1: one sign bit, two exponent bits, one mantissa bit, with the all-zero
// exponent giving the +/-0.5 denormals. The values are NOT linear in the code --
// they step by 0.5 up to 2 and then by 1, 2, 2 -- which is the whole reason
// MXFP4 cannot borrow the affine dot, whose trick is that `scale * code + bias`
// is linear in the code.
//
// E8M0: an unsigned power of two, 127-biased, one per 32-element block, with
// 0xff reserved for NaN.
//
// The Metal sibling (`quant/mxfp4_codec.h`) says the same three things and says
// why they have to live in one file: the decode path would be right on a
// prefill and wrong on a decode, or the other way round, and the output would
// still look like text.
//
// `GL_EXT_float_e2m1` exists and llama.cpp uses it behind `USE_OCP_FP4`. This
// tree does not, for `bf16.glsl`'s reason: an optional extension is a hard
// failure on the drivers a Vulkan shell exists to reach, and a sixteen-entry
// lookup is not the cost worth taking it for.

#ifndef PIE_VULKAN_MXFP4_GLSL
#define PIE_VULKAN_MXFP4_GLSL

const float kMxfp4Lut[16] = float[16](
    0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0);

/// The two codes in one byte, low nibble first. Swapping them transposes every
/// adjacent pair of weights, which is wrong by a plausible amount.
float mxfp4_lo(uint byte_) { return kMxfp4Lut[byte_ & 0xfu]; }
float mxfp4_hi(uint byte_) { return kMxfp4Lut[(byte_ >> 4) & 0xfu]; }

/// One block's E8M0 exponent as a multiplier. 0xff is the NaN encoding; every
/// other code is an exact power of two.
///
/// `ldexp` is the spelling rather than `pow(2.0, e)`: the exponent is an
/// integer and `pow` is neither exact nor cheap for one.
float mxfp4_block_scale(uint code) {
    if (code == 0xffu) {
        return uintBitsToFloat(0x7fc00000u);  // quiet NaN
    }
    return ldexp(1.0, int(code) - 127);
}

/// Elements per E8M0 block. Fixed by the format, not a deployment's choice --
/// which is why the routed MXFP4 rows are compiled at `_gs_32` alone.
#define PIE_MXFP4_BLOCK 32

#endif  // PIE_VULKAN_MXFP4_GLSL

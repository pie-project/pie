//! E8M0: a power-of-two scale factor in one byte, bias 127.
//!
//! The shared exponent of an MX block, kept apart from [`super::mxfp4`]
//! because FP8 blocks carry one too and a second copy of the bias would be a
//! second place for it to be wrong.

/// The E8M0 byte for one group: the smallest `b` with `6 · 2^(b-127) ≥ absmax`
/// (`quant_bf16_to_mxfp4.cu::encode_e8m0`), and `0` for a group of zeros — or
/// of NaNs, which the kernel's `!(absmax > 0)` spelling covers.
///
/// The `log2` is where this can drift from the device: CUDA's `log2f` is within
/// 1 ulp and the host's is correctly rounded, so a disagreement flips the
/// `ceil` only when `absmax / 6` is within an ulp of a power of two.
// `!(absmax > 0.0)` is false for NaN where `absmax <= 0.0` is not.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub fn encode_e8m0(absmax: f32) -> u8 {
    if !(absmax > 0.0) {
        return 0;
    }
    // The `+ 127.0` stays in `f32` so an infinite absmax saturates through the
    // int conversion instead of overflowing after it.
    let b = ((absmax / 6.0).log2().ceil() + 127.0) as i32;
    b.clamp(0, 254) as u8
}

/// `2^(sb - 127)` as the kernel's `ldexpf` builds it: an exact power of two,
/// subnormal at `sb == 0`.
pub fn exp2_e8m0(sb: u8) -> f32 {
    if sb == 0 {
        f32::from_bits(1 << 22)
    } else {
        f32::from_bits(u32::from(sb) << 23)
    }
}

/// `2^e` for a normal-range exponent, built rather than computed.
pub fn exp2i(e: i32) -> f32 {
    f32::from_bits(((e + 127) as u32) << 23)
}

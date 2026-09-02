//! OCP FP8 E4M3: one byte, four exponent bits, three mantissa bits. Decode
//! and encode are kept in one file since a round-trip that doesn't close is
//! the bug this arrangement is meant to make visible.
use super::e8m0::exp2i;

/// One `f64` per `Fp8E4M3` byte: sign, four exponent bits, three mantissa
/// bits, bias 7.
///
/// OCP `E4M3` with no infinity: the all-ones exponent carries ordinary
/// values up to 448 and only `S.1111.111` is NaN. A subnormal is
/// `mantissa/8 * 2^-6`.
pub fn decode_fp8_e4m3_elements(bytes: &[u8]) -> Vec<f64> {
    bytes
        .iter()
        .map(|&byte| {
            let sign = if byte & 0x80 != 0 { -1.0f64 } else { 1.0 };
            let exponent = i32::from((byte >> 3) & 0x0F);
            let mantissa = f64::from(byte & 0x07);
            if exponent == 0x0F && mantissa == 7.0 {
                return f64::NAN;
            }
            let magnitude = if exponent == 0 {
                mantissa / 8.0 * (-6.0f64).exp2()
            } else {
                (1.0 + mantissa / 8.0) * f64::from(exponent - 7).exp2()
            };
            sign * magnitude
        })
        .collect()
}

/// One E4M3 byte as the `f32` it denotes, `__nv_cvt_fp8_to_halfraw` widened:
/// 1-4-3, bias 7, no infinities, `S.1111.111` the one NaN.
pub fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = (byte >> 3) & 0xF;
    let mant = (byte & 0x7) as f32;
    if exp == 0xF && byte & 0x7 == 0x7 {
        return f32::NAN;
    }
    let value = if exp == 0 {
        // Subnormal: units of 2^-9.
        mant * exp2i(-9)
    } else {
        (1.0 + mant / 8.0) * exp2i(i32::from(exp) - 7)
    };
    sign * value
}

/// `__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3)`: round to nearest
/// even, saturate finite overflow — and infinity — to ±448, NaN to the
/// scheme's NaN byte with the sign kept.
pub fn f32_to_fp8_e4m3(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    if x.is_nan() {
        return sign | 0x7F;
    }
    let a = x.abs();
    if a >= 448.0 {
        // Everything past the last code rounds or saturates onto it: the
        // next magnitude up the grid is the NaN slot, which SATFINITE never
        // produces.
        return sign | 0x7E;
    }
    if a < 0.015625 {
        // Subnormal: quantize in units of 2^-9. The multiply is by a power
        // of two, so it is exact and the tie is the true tie; 8 rolls over
        // into exactly the first normal code.
        let q = (a * 512.0).round_ties_even() as u32;
        return sign | q as u8;
    }
    let bits = a.to_bits();
    let mut e = ((bits >> 23) as i32) - 127;
    let m = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000);
    let mut q = (m * 8.0).round_ties_even() as u32;
    if q == 16 {
        e += 1;
        q = 8;
    }
    sign | (((e + 7) as u8) << 3) | (q as u8 - 8)
}


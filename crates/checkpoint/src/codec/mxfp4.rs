//! OCP MX FP4 E2M1: sixteen codepoints in a nibble, thirty-two to a block.
//!
//! The block's shared exponent is [`super::e8m0`]'s. What is here is the
//! nibble table and the group encoder, plus the AVX2 implementation of the
//! same function — which is in this file, not a sibling, because a vector
//! path that disagrees with its scalar path is the defect it exists to be
//! checked against.

use super::e8m0::{encode_e8m0, exp2_e8m0};

/// One `f64` per E2M1 code, low nibble first.
///
/// The nibble order and the codepoint table are the OCP MX FP4 spec's, and
/// have to stay the CUDA kernel's: `kFp4Lut` in `kernels/dequant_fp4.cu` is
/// the same sixteen values in the same order, and the two executors are
/// compared element for element.
pub fn decode_mxfp4_elements(bytes: &[u8]) -> Vec<f64> {
    const LUT: [f64; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    let mut values = Vec::with_capacity(bytes.len() * 2);
    for byte in bytes {
        values.push(LUT[(byte & 0xF) as usize]);
        values.push(LUT[(byte >> 4) as usize]);
    }
    values
}

/// One 32-element MXFP4 group: absmax → E8M0 byte, elements → 16 packed
/// nibble pairs in `out`. Returns the scale byte.
///
/// Dispatches to the AVX2 restatement when the CPU has it; the scalar body
/// below is the reference the vector path is checked against
/// (`avx2_group_encode_matches_the_scalar_reference`), and both are the
/// kernel's arithmetic.
pub fn encode_mxfp4_group(group: &[f32], out: &mut [u8]) -> u8 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // Sound: the feature was just detected, and the slices are checked by
        // the callee's debug asserts and the caller's slicing.
        return unsafe { avx2::encode_mxfp4_group(group, out) };
    }
    encode_mxfp4_group_scalar(group, out)
}

pub fn encode_mxfp4_group_scalar(group: &[f32], out: &mut [u8]) -> u8 {
    // `f32::max` ignores a NaN operand, which is the kernel's
    // `if (a > absmax)` by another spelling.
    let mut absmax = 0.0f32;
    for &v in group {
        absmax = absmax.max(v.abs());
    }
    let sb = encode_e8m0(absmax);
    // An exact power of two, so the reciprocal is exact too and multiplying
    // by it is the kernel's divide.
    let inv_s = 1.0 / exp2_e8m0(sb);
    let mut codes = [0u8; 32];
    for (v, code) in group.iter().zip(codes.iter_mut()) {
        *code = encode_fp4_e2m1(v * inv_s);
    }
    for (k, pair) in codes.chunks_exact(2).enumerate() {
        out[k] = (pair[1] << 4) | pair[0];
    }
    sb
}

/// One FP4 E2M1 codepoint: `quant_bf16_to_mxfp4.cu::encode_fp4_e2m1`, its
/// midpoint table verbatim.
///
/// Every comparison is false for a NaN operand, which falls through to the
/// largest magnitude with a positive sign — not a choice here, the port of
/// the kernel's. A signed zero rounds to `+0`.
// The negated comparisons are the port: `!(a < t)` is true for NaN where
// `a >= t` is not, and NaN-rounds-to-the-top-magnitude is the kernel's
// behaviour.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub fn encode_fp4_e2m1(x: f32) -> u8 {
    let a = x.abs();
    // Magnitudes 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0; boundaries are the
    // midpoints between neighbours, so this is round-to-nearest with ties up.
    //
    // The kernel's ladder, summed instead of chained: the magnitude is how
    // many midpoints `a` is *not below*, which is the same arithmetic — a NaN
    // fails every comparison and lands on 7, exactly as it falls through the
    // ladder — but branch-free, which is what lets the per-element encode
    // loop vectorize.
    let mag = u8::from(!(a < 0.25))
        + u8::from(!(a < 0.75))
        + u8::from(!(a < 1.25))
        + u8::from(!(a < 1.75))
        + u8::from(!(a < 2.5))
        + u8::from(!(a < 3.5))
        + u8::from(!(a < 5.0));
    // Signed zero keeps rounding to +0: the sign lands only on a non-zero
    // magnitude.
    let sign = u8::from(x < 0.0) << 3;
    (mag | sign) * u8::from(mag != 0)
}

/// The encode hot loops as AVX2 lanes, lane-for-lane the scalar arithmetic.
///
/// Nothing here is a different algorithm — every intrinsic was chosen to
/// reproduce one scalar expression bit for bit, NaN cases included:
///
/// * `_CMP_NLT_UQ` is `!(a < t)` — true on NaN — which is the E2M1 ladder's
///   test and how a NaN element lands on magnitude 7;
/// * `_mm256_max_ps(x, acc)` returns `acc` when `x` is NaN, which is
///   `acc.max(x)`'s NaN-ignoring absmax;
/// * `_CMP_LT_OQ` against zero is `x < 0.0` — false for NaN and `-0.0` — so
///   the sign bit lands exactly where the scalar puts it.
///
/// The multiply is `vmulps`, IEEE round-to-nearest like the scalar's, and the
/// BF16 widening is the same `<< 16`. Everything is `unsafe` only for the
/// `target_feature` contract; callers dispatch behind runtime detection.
#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    use std::arch::x86_64::*;

    /// Widen one BF16 row to `f32`, eight lanes per step.
    ///
    /// # Safety
    /// The caller detected `avx2`. `row` holds `out.len()` little-endian
    /// BF16 values.
    #[target_feature(enable = "avx2")]
    pub unsafe fn decode_bf16_row(row: &[u8], out: &mut [f32]) {
        debug_assert!(row.len() == out.len() * 2);
        let n = out.len();
        let mut at = 0;
        unsafe {
            while at + 8 <= n {
                let half = _mm_loadu_si128(row.as_ptr().add(at * 2).cast());
                let wide = _mm256_cvtepu16_epi32(half);
                let bits = _mm256_slli_epi32::<16>(wide);
                _mm256_storeu_ps(out.as_mut_ptr().add(at), _mm256_castsi256_ps(bits));
                at += 8;
            }
        }
        for k in at..n {
            let bits = u16::from_le_bytes([row[2 * k], row[2 * k + 1]]);
            out[k] = f32::from_bits(u32::from(bits) << 16);
        }
    }

    /// One 32-element MXFP4 group; see [`super::encode_mxfp4_group_scalar`]
    /// for the reference this restates.
    ///
    /// # Safety
    /// The caller detected `avx2`. `group` has 32 elements, `out` has 16.
    #[target_feature(enable = "avx2")]
    pub unsafe fn encode_mxfp4_group(group: &[f32], out: &mut [u8]) -> u8 {
        debug_assert!(group.len() == 32 && out.len() == 16);
        unsafe {
            let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));

            let mut lanes = [_mm256_setzero_ps(); 4];
            let mut acc = _mm256_setzero_ps();
            for (i, chunk) in lanes.iter_mut().enumerate() {
                let v = _mm256_loadu_ps(group.as_ptr().add(i * 8));
                *chunk = v;
                // NaN in the first operand yields the second: `acc.max(|v|)`.
                acc = _mm256_max_ps(_mm256_and_ps(v, abs_mask), acc);
            }
            // The accumulator lanes are NaN-free, so the horizontal order is
            // free to be anything.
            let quad = _mm_max_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps::<1>(acc));
            let pair = _mm_max_ps(quad, _mm_movehl_ps(quad, quad));
            let one = _mm_max_ss(pair, _mm_shuffle_ps::<1>(pair, pair));
            let absmax = _mm_cvtss_f32(one);

            let sb = super::encode_e8m0(absmax);
            let inv_s = _mm256_set1_ps(1.0 / super::exp2_e8m0(sb));

            let thresholds = [0.25f32, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];
            let zero_ps = _mm256_setzero_ps();
            let zero_si = _mm256_setzero_si256();
            let low_bit = _mm256_set1_epi32(1);
            let mut codes = [0i32; 32];
            for (i, &chunk) in lanes.iter().enumerate() {
                let x = _mm256_mul_ps(chunk, inv_s);
                let a = _mm256_and_ps(x, abs_mask);
                let mut mag = zero_si;
                for t in thresholds {
                    // A true lane is -1; subtracting counts the midpoints
                    // `a` is not below, NaN counting all seven.
                    let not_below = _mm256_cmp_ps::<_CMP_NLT_UQ>(a, _mm256_set1_ps(t));
                    mag = _mm256_sub_epi32(mag, _mm256_castps_si256(not_below));
                }
                let negative = _mm256_cmp_ps::<_CMP_LT_OQ>(x, zero_ps);
                let sign = _mm256_slli_epi32::<3>(_mm256_and_si256(
                    _mm256_castps_si256(negative),
                    low_bit,
                ));
                let nonzero = _mm256_cmpgt_epi32(mag, zero_si);
                let code = _mm256_and_si256(_mm256_or_si256(mag, sign), nonzero);
                _mm256_storeu_si256(codes.as_mut_ptr().add(i * 8).cast(), code);
            }
            for (k, pair) in codes.chunks_exact(2).enumerate() {
                out[k] = ((pair[1] as u8) << 4) | pair[0] as u8;
            }
            sb
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::e8m0::{encode_e8m0, exp2_e8m0};
    use super::*;

    /// The two encode primitives at their edges, against values computed by
    /// hand from the kernel's comments.
    #[test]
    fn mxfp4_encode_primitives_match_the_kernel_at_the_edges() {
        // Signed zero rounds to +0; NaN falls through every comparison to the
        // top magnitude, positive.
        assert_eq!(encode_fp4_e2m1(0.0), 0);
        assert_eq!(encode_fp4_e2m1(-0.0), 0);
        assert_eq!(encode_fp4_e2m1(f32::NAN), 0x7);
        assert_eq!(encode_fp4_e2m1(-5.0), 0xF);
        assert_eq!(encode_fp4_e2m1(f32::INFINITY), 0x7);
        assert_eq!(encode_fp4_e2m1(f32::NEG_INFINITY), 0xF);
        // Each boundary is closed on its upper side.
        assert_eq!(encode_fp4_e2m1(0.25), 1);
        assert_eq!(encode_fp4_e2m1(-1.75), 0xC);

        // b is the smallest byte with 6·2^(b-127) ≥ absmax.
        assert_eq!(encode_e8m0(0.0), 0);
        assert_eq!(encode_e8m0(f32::NAN), 0);
        assert_eq!(encode_e8m0(6.0), 127);
        assert_eq!(encode_e8m0(6.1), 128);
        assert_eq!(encode_e8m0(3.0), 126);
        assert_eq!(encode_e8m0(12.0), 128);
        assert_eq!(encode_e8m0(f32::INFINITY), 254);
        assert_eq!(encode_e8m0(f32::MIN_POSITIVE), 0);

        assert_eq!(exp2_e8m0(127), 1.0);
        assert_eq!(exp2_e8m0(128), 2.0);
        assert_eq!(exp2_e8m0(0), f32::from_bits(1 << 22));
        assert_eq!(exp2_e8m0(254), f32::from_bits(254 << 23));
    }

    /// The AVX2 group encode against the scalar reference, over inputs built
    /// to visit every lane behaviour: all magnitudes both sides of each
    /// midpoint, both signs, signed zeros, NaN, infinity, subnormals, and a
    /// spread of group absmaxes — on any machine without AVX2 this reduces to
    /// scalar-vs-scalar and passes vacuously.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_group_encode_matches_the_scalar_reference() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // A deterministic mix: an LCG over interesting bf16-representable
        // values plus injected specials.
        let specials = [
            f32::NAN,
            -f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
        ];
        let mut state = 0x243F_6A88u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        };
        for round in 0..256 {
            let mut group = [0.0f32; 32];
            for (at, value) in group.iter_mut().enumerate() {
                let roll = next();
                *value = if roll % 11 == 0 {
                    specials[(roll / 11) as usize % specials.len()]
                } else {
                    // A finite bf16 value with a bounded exponent, signed.
                    let mantissa = (roll >> 8) & 0xFF;
                    let exponent = 118 + (roll % 21); // 2^-9 .. 2^11
                    let sign = (roll & 1) << 31;
                    f32::from_bits(sign | (exponent << 23) | (mantissa << 15))
                };
                if round == 0 && at == 0 {
                    // One all-zero-leading group exercises the dead-group arm.
                    *value = 0.0;
                }
            }
            let mut scalar_out = [0u8; 16];
            let mut avx2_out = [0u8; 16];
            let scalar_sb = encode_mxfp4_group_scalar(&group, &mut scalar_out);
            let avx2_sb = unsafe { avx2::encode_mxfp4_group(&group, &mut avx2_out) };
            assert_eq!(scalar_sb, avx2_sb, "scale byte diverged on {group:?}");
            assert_eq!(scalar_out, avx2_out, "codes diverged on {group:?}");
        }

        // The decode too, odd tail included.
        let mut bytes = Vec::new();
        for _ in 0..77 {
            bytes.extend_from_slice(&(next() as u16).to_le_bytes());
        }
        let mut scalar = vec![0.0f32; 77];
        for (le, out) in bytes.chunks_exact(2).zip(scalar.iter_mut()) {
            *out = f32::from_bits(u32::from(u16::from_le_bytes(le.try_into().unwrap())) << 16);
        }
        let mut vector = vec![0.0f32; 77];
        unsafe { avx2::decode_bf16_row(&bytes, &mut vector) };
        assert_eq!(
            scalar.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            vector.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}

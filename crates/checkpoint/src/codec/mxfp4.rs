//! OCP MX FP4 E2M1 codec: nibble table, group encoder, AVX2 variant.

use super::e8m0::{encode_e8m0, exp2_e8m0};

/// One `f64` per E2M1 code, low nibble first.
/// Order must match `kFp4Lut` in `kernels/dequant_fp4.cu`.
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
pub fn encode_mxfp4_group(group: &[f32], out: &mut [u8]) -> u8 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: avx2 just detected; slice lengths checked by callee's debug_asserts.
        return unsafe { avx2::encode_mxfp4_group(group, out) };
    }
    encode_mxfp4_group_scalar(group, out)
}

pub fn encode_mxfp4_group_scalar(group: &[f32], out: &mut [u8]) -> u8 {
    // f32::max ignores a NaN operand, matching the kernel's `if (a > absmax)`.
    let mut absmax = 0.0f32;
    for &v in group {
        absmax = absmax.max(v.abs());
    }
    let sb = encode_e8m0(absmax);
    // Exact power of two: reciprocal is exact, so multiply == kernel's divide.
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

/// One FP4 E2M1 codepoint: `quant_bf16_to_mxfp4.cu::encode_fp4_e2m1` ported.
/// NaN falls through to the top magnitude, positive sign; `-0.0` rounds to `+0`.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub fn encode_fp4_e2m1(x: f32) -> u8 {
    let a = x.abs();
    // Magnitude is the count of midpoints `a` is not below (branch-free,
    // vectorizable); NaN fails every comparison and lands on 7.
    let mag = u8::from(!(a < 0.25))
        + u8::from(!(a < 0.75))
        + u8::from(!(a < 1.25))
        + u8::from(!(a < 1.75))
        + u8::from(!(a < 2.5))
        + u8::from(!(a < 3.5))
        + u8::from(!(a < 5.0));
    // Sign applies only to a non-zero magnitude, so signed zero rounds to +0.
    let sign = u8::from(x < 0.0) << 3;
    (mag | sign) * u8::from(mag != 0)
}

/// AVX2 restatement of the scalar encode, lane-for-lane, NaN cases included.
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

    /// AVX2 form of [`super::encode_mxfp4_group_scalar`].
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


//! Reading a bf16 logits row back as `f32`.
//!
//! `logits_convert.hpp`, which is thirty lines and one idea: the device writes
//! logits as bf16 and every host-side consumer — the sampler, the golden tap,
//! the interpreter oracle — wants `f32`. The conversion is exact and cheap, and
//! the only thing worth being careful about is that there is one of it.
//!
//! # bf16 to f32 is a shift, not a rounding
//!
//! bf16 is the top sixteen bits of an f32 with the same exponent range, so
//! widening is `bits << 16` and nothing else: every bf16 has an exact f32, and
//! `NaN` payloads and signed zeros survive. That matters here because these are
//! the values an argmax then chooses between, and the interpreter oracle
//! compares them by `to_bits` — a conversion that rounded, or that flushed a
//! subnormal, would put a difference into the comparison that the device never
//! computed.
//!
//! The C++ writes it with a `memcpy` through a `uint32_t`, which is the correct
//! way to type-pun in C++ and is what `f32::from_bits` is.
//!
//! # Why this is its own module rather than a helper where it is used
//!
//! It already had two homes. `batch/golden.rs` carries a private copy for the
//! tap dumps, and the logits readback needs the same function; a third caller
//! would have made a third. The C++ avoided that by giving the conversion a
//! file of its own, and this is that file — the golden tap now calls it rather
//! than spelling it again.

/// Widen one bf16 to the `f32` it exactly represents.
#[must_use]
pub const fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Widen a bf16 row into an `f32` buffer.
///
/// The C++ takes a raw pointer pair and a count, so a caller that gets the
/// count wrong walks off one end or the other. This takes slices and refuses a
/// mismatch, because the two lengths are the same fact written twice.
///
/// # Errors
///
/// [`LengthMismatch`] when the destination is not the source's length.
pub fn widen_into(src: &[u16], dst: &mut [f32]) -> Result<(), LengthMismatch> {
    if src.len() != dst.len() {
        return Err(LengthMismatch {
            src: src.len(),
            dst: dst.len(),
        });
    }
    for (out, &bits) in dst.iter_mut().zip(src) {
        *out = bf16_to_f32(bits);
    }
    Ok(())
}

/// Widen a bf16 row into a fresh `f32` vector.
#[must_use]
pub fn widen(src: &[u16]) -> Vec<f32> {
    src.iter().copied().map(bf16_to_f32).collect()
}

/// The destination did not match the source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LengthMismatch {
    /// Values to widen.
    pub src: usize,
    /// Slots to widen them into.
    pub dst: usize,
}

impl core::fmt::Display for LengthMismatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} bf16 values into {} f32 slots", self.src, self.dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_widened_bf16_is_the_top_half_of_the_f32_it_names() {
        // 0x3F80 is bf16 1.0; its f32 is 0x3F800000.
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xBF80), -1.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        assert_eq!(bf16_to_f32(0x4000), 2.0);
    }

    #[test]
    fn a_negative_zero_stays_negative_and_a_nan_stays_a_nan() {
        // Both are invisible to `==` and both reach an argmax, which is why
        // the oracle compares logits by `to_bits`. A conversion that lost
        // either would put a difference into that comparison the device never
        // computed.
        assert!(bf16_to_f32(0x8000).is_sign_negative());
        assert_eq!(bf16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
        assert!(bf16_to_f32(0x7FC0).is_nan());
        assert!(bf16_to_f32(0x7F80).is_infinite() && bf16_to_f32(0x7F80) > 0.0);
        assert!(bf16_to_f32(0xFF80).is_infinite() && bf16_to_f32(0xFF80) < 0.0);
    }

    #[test]
    fn a_subnormal_survives_rather_than_flushing_to_zero() {
        // The smallest positive bf16. Widening is a shift, so it lands on a
        // perfectly ordinary f32 rather than on zero.
        let tiny = bf16_to_f32(0x0001);
        assert!(tiny > 0.0, "a shift cannot flush");
        assert_eq!(tiny.to_bits(), 1 << 16);
    }

    #[test]
    fn a_destination_of_the_wrong_length_is_refused_rather_than_partly_filled() {
        let src = [0x3F80u16; 4];
        let mut dst = [0.0f32; 3];
        assert_eq!(
            widen_into(&src, &mut dst),
            Err(LengthMismatch { src: 4, dst: 3 })
        );
        assert_eq!(dst, [0.0; 3], "nothing was written");
    }

    #[test]
    fn widening_in_place_and_into_a_fresh_vector_agree() {
        let src = [0x3F80u16, 0x4000, 0xBF80, 0x0000];
        let mut dst = [0.0f32; 4];
        widen_into(&src, &mut dst).expect("same length");
        assert_eq!(dst.to_vec(), widen(&src));
    }
}

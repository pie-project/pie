//! Packed-bitmask logit-mask semantics — the single source of truth for cut #2
//! grammar masking, byte-identical to the driver's `0x65 MaskApply` op and the
//! host-side bit-AND composition.
//!
//! A logit mask is one bit per vocabulary token, packed into `ceil(vocab/32)`
//! `u32` words: **bit `1` = ALLOWED, bit `0` = DISALLOWED**. Token `j`'s bit is
//! word `j >> 5`, bit `j & 31`. The driver's mask-apply sets the bf16 logit to
//! `−∞` wherever the bit is `0`, then argmax picks the highest survivor.
//!
//! These helpers let host code (the SDK constraint composition and the grammar
//! verify inferlet's CPU reference) reproduce that exact behaviour, so a
//! conformance check can't silently degenerate on a host↔device semantics drift.

/// Number of `u32` words a packed mask for `vocab` tokens occupies.
#[inline]
pub fn mask_words(vocab: usize) -> usize {
    vocab.div_ceil(32)
}

/// An all-allowed packed mask for `vocab` tokens (every bit `1`). The identity
/// for [`and_into`]; tail bits past `vocab` in the last word are don't-care
/// (consumers index only `[0, vocab)`).
#[inline]
pub fn all_allowed(vocab: usize) -> Vec<u32> {
    vec![u32::MAX; mask_words(vocab)]
}

/// Whether token `j` is allowed (bit `1`) in the packed mask.
#[inline]
pub fn bit_allowed(mask: &[u32], j: usize) -> bool {
    (mask[j >> 5] >> (j & 31)) & 1 == 1
}

/// Pack an allowed-token id list into a `[ceil(vocab/32)]` u32 bitmask (bit `1`
/// = allowed, all others disallowed). The constructive inverse of
/// [`bit_allowed`]; ids `>= vocab` are ignored.
pub fn pack_allowed(vocab: usize, allowed: &[u32]) -> Vec<u32> {
    let mut mask = vec![0u32; mask_words(vocab)];
    for &id in allowed {
        let j = id as usize;
        if j < vocab {
            mask[j >> 5] |= 1 << (j & 31);
        }
    }
    mask
}

/// Compose two packed masks by word-wise bitwise-AND (`acc &= other`): the
/// intersection of the allowed sets — the AND of multiple constraints, and the
/// packed-bit analog of the old `brle_and`. Both must be the same word count.
#[inline]
pub fn and_into(acc: &mut [u32], other: &[u32]) {
    debug_assert_eq!(acc.len(), other.len(), "mask word-count mismatch");
    for (a, b) in acc.iter_mut().zip(other) {
        *a &= *b;
    }
}

/// Convert a bf16 bit pattern to `f32` exactly as the driver does
/// (`__uint_as_float(h << 16)`): bf16 is the high 16 bits of an `f32`.
#[inline]
pub fn bf16_hi_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Argmax over `logits` with the packed mask applied — the host reference for
/// the driver's `0x65 MaskApply` + argmax: a disallowed token (bit `0`) is
/// treated as `−∞`, so the result is the highest-logit *allowed* token. Ties go
/// to the lowest index (matching a forward argmax scan). If every token is
/// disallowed, returns `0`.
pub fn apply_mask_argmax(logits: &[f32], mask: &[u32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (j, &logit) in logits.iter().enumerate() {
        let v = if bit_allowed(mask, j) {
            logit
        } else {
            f32::NEG_INFINITY
        };
        if v > best_val {
            best_val = v;
            best_idx = j as u32;
        }
    }
    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_words_rounds_up() {
        assert_eq!(mask_words(32), 1);
        assert_eq!(mask_words(33), 2);
        assert_eq!(mask_words(151_936), 4748);
    }

    #[test]
    fn bit_allowed_indexes_word_and_bit() {
        // bits 0 and 2 allowed, bit 1 disallowed -> 0b101 = 5.
        let mask = [0b101u32];
        assert!(bit_allowed(&mask, 0));
        assert!(!bit_allowed(&mask, 1));
        assert!(bit_allowed(&mask, 2));
        // bit 33 lives in word 1, bit 1.
        let mask2 = [0u32, 0b10u32];
        assert!(bit_allowed(&mask2, 33));
        assert!(!bit_allowed(&mask2, 32));
    }

    #[test]
    fn all_allowed_passes_through_to_global_argmax() {
        let logits = [1.0f32, 5.0, 3.0];
        let mask = all_allowed(logits.len());
        // No restriction -> the natural argmax (index 1).
        assert_eq!(apply_mask_argmax(&logits, &mask), 1);
    }

    #[test]
    fn disallowed_argmax_is_forced_out() {
        // The natural argmax (index 1, value 5.0) is disallowed (bit 1 = 0);
        // the result must be the highest *allowed* token (index 2). This is the
        // assert-#3 core: it proves -inf actually fired, not passthrough.
        let logits = [1.0f32, 5.0, 3.0];
        let mask = [0b101u32]; // allow 0 and 2, disallow 1
        assert_eq!(apply_mask_argmax(&logits, &mask), 2);
    }

    #[test]
    fn and_into_is_set_intersection() {
        // {0,1,2} AND {0,2} = {0,2}.
        let mut acc = [0b111u32];
        and_into(&mut acc, &[0b101u32]);
        assert_eq!(acc, [0b101u32]);
        // {1,2} AND {0,1} = {1}.
        let mut acc2 = [0b110u32];
        and_into(&mut acc2, &[0b011u32]);
        assert_eq!(acc2, [0b010u32]);
    }

    #[test]
    fn composed_mask_forces_out_via_intersection() {
        // Compose all-allowed with a ban on the natural argmax -> forced out.
        let logits = [1.0f32, 5.0, 3.0];
        let mut mask = all_allowed(logits.len());
        and_into(&mut mask, &[0b101u32]); // intersect: disallow index 1
        assert_eq!(apply_mask_argmax(&logits, &mask), 2);
    }

    #[test]
    fn bf16_hi_round_trips_high_bits() {
        // 1.0f32 = 0x3F800000; its bf16 high half is 0x3F80.
        assert_eq!(bf16_hi_to_f32(0x3F80), 1.0);
        // bf16 -inf = 0xFF80 -> f32 -inf.
        assert_eq!(bf16_hi_to_f32(0xFF80), f32::NEG_INFINITY);
    }

    #[test]
    fn pack_allowed_round_trips_bits() {
        // allow {0, 2, 33} over a 40-token vocab (2 words).
        let mask = pack_allowed(40, &[0, 2, 33]);
        assert_eq!(mask.len(), 2);
        assert!(bit_allowed(&mask, 0));
        assert!(!bit_allowed(&mask, 1));
        assert!(bit_allowed(&mask, 2));
        assert!(bit_allowed(&mask, 33));
        assert!(!bit_allowed(&mask, 32));
        // out-of-range ids are ignored (no panic, not set).
        let m2 = pack_allowed(8, &[3, 99]);
        assert!(bit_allowed(&m2, 3));
    }

    #[test]
    fn packed_allowed_forces_out_disallowed_argmax() {
        // logits favor token 1, but only {0,2} are packed allowed -> forced to 2.
        let logits = [1.0f32, 5.0, 3.0];
        let mask = pack_allowed(3, &[0, 2]);
        assert_eq!(apply_mask_argmax(&logits, &mask), 2);
    }
}

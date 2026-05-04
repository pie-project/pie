//! Token bitmask utilities.
//!
//! A bitmask is a `Vec<u32>` where bit `i` indicates whether token `i` is allowed.
//! Bit 1 = allowed, bit 0 = rejected.

/// Compute the number of `u32` words needed for a bitmask of `vocab_size` tokens.
pub fn bitmask_size(vocab_size: usize) -> usize {
    (vocab_size + 31) / 32
}

/// Set bit `i` in the bitmask (mark token as allowed).
#[inline]
pub fn set_bit(bitmask: &mut [u32], i: usize) {
    bitmask[i / 32] |= 1 << (i % 32);
}

/// Clear bit `i` in the bitmask (mark token as rejected).
#[inline]
pub fn clear_bit(bitmask: &mut [u32], i: usize) {
    bitmask[i / 32] &= !(1 << (i % 32));
}

/// Get bit `i` from the bitmask. Returns true if the token is allowed.
#[inline]
pub fn get_bit(bitmask: &[u32], i: usize) -> bool {
    (bitmask[i / 32] >> (i % 32)) & 1 == 1
}

/// Reset the bitmask to all-ones (all tokens allowed).
pub fn reset_bitmask(bitmask: &mut [u32], vocab_size: usize) {
    let full_words = vocab_size / 32;
    let remainder = vocab_size % 32;

    for word in bitmask[..full_words].iter_mut() {
        *word = u32::MAX;
    }

    if remainder > 0 && full_words < bitmask.len() {
        bitmask[full_words] = (1u32 << remainder) - 1;
    }

    // Zero out any trailing words beyond vocab_size
    for word in bitmask[full_words + if remainder > 0 { 1 } else { 0 }..].iter_mut() {
        *word = 0;
    }
}

/// Reset the bitmask to all-zeros (all tokens rejected).
pub fn clear_bitmask(bitmask: &mut [u32]) {
    for word in bitmask.iter_mut() {
        *word = 0;
    }
}

/// Apply the token bitmask to logits in-place.
/// Sets logits to `-inf` for rejected tokens (bit = 0).
pub fn apply_token_bitmask_inplace(logits: &mut [f32], bitmask: &[u32]) {
    for (i, logit) in logits.iter_mut().enumerate() {
        if !get_bit(bitmask, i) {
            *logit = f32::NEG_INFINITY;
        }
    }
}

/// Check if all bits in the bitmask are set (all tokens allowed).
pub fn is_bitmask_all_ones(bitmask: &[u32], vocab_size: usize) -> bool {
    let full_words = vocab_size / 32;
    let remainder = vocab_size % 32;

    for &word in &bitmask[..full_words] {
        if word != u32::MAX {
            return false;
        }
    }

    if remainder > 0 {
        let mask = (1u32 << remainder) - 1;
        if bitmask[full_words] & mask != mask {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitmask_size() {
        assert_eq!(bitmask_size(0), 0);
        assert_eq!(bitmask_size(1), 1);
        assert_eq!(bitmask_size(31), 1);
        assert_eq!(bitmask_size(32), 1);
        assert_eq!(bitmask_size(33), 2);
        assert_eq!(bitmask_size(64), 2);
        assert_eq!(bitmask_size(65), 3);
    }

    #[test]
    fn test_set_get_clear_bit() {
        let mut bm = vec![0u32; 2];

        set_bit(&mut bm, 0);
        assert!(get_bit(&bm, 0));
        assert!(!get_bit(&bm, 1));

        set_bit(&mut bm, 31);
        assert!(get_bit(&bm, 31));

        set_bit(&mut bm, 32);
        assert!(get_bit(&bm, 32));

        clear_bit(&mut bm, 0);
        assert!(!get_bit(&bm, 0));
        assert!(get_bit(&bm, 31));
    }

    #[test]
    fn test_reset_bitmask() {
        let vocab_size = 50;
        let mut bm = vec![0u32; bitmask_size(vocab_size)];
        reset_bitmask(&mut bm, vocab_size);

        for i in 0..vocab_size {
            assert!(get_bit(&bm, i), "bit {} should be set", i);
        }
        // Bits beyond vocab_size should not be set
        for i in vocab_size..bm.len() * 32 {
            assert!(!get_bit(&bm, i), "bit {} should not be set", i);
        }
    }

    #[test]
    fn test_clear_bitmask() {
        let mut bm = vec![u32::MAX; 3];
        clear_bitmask(&mut bm);
        for &word in &bm {
            assert_eq!(word, 0);
        }
    }

    #[test]
    fn test_apply_token_bitmask() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        let mut bm = vec![0u32; bitmask_size(4)];
        set_bit(&mut bm, 0);
        set_bit(&mut bm, 2);

        apply_token_bitmask_inplace(&mut logits, &bm);

        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[2], 3.0);
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    #[test]
    fn test_is_bitmask_all_ones() {
        let vocab_size = 50;
        let mut bm = vec![0u32; bitmask_size(vocab_size)];
        reset_bitmask(&mut bm, vocab_size);
        assert!(is_bitmask_all_ones(&bm, vocab_size));

        clear_bit(&mut bm, 25);
        assert!(!is_bitmask_all_ones(&bm, vocab_size));
    }
}

//! Token bitmask utilities.
//!
//! A bitmask is a `Vec<u32>` where bit `i` indicates whether token `i` is allowed.
//! Bit 1 = allowed, bit 0 = rejected.

/// Compute the number of `u32` words needed for a bitmask of `vocab_size` tokens.
pub fn bitmask_size(vocab_size: usize) -> usize {
    vocab_size.div_ceil(32)
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

/// Reset the bitmask to all-zeros (all tokens rejected).
pub fn clear_bitmask(bitmask: &mut [u32]) {
    for word in bitmask.iter_mut() {
        *word = 0;
    }
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
    fn test_clear_bitmask() {
        let mut bm = vec![u32::MAX; 3];
        clear_bitmask(&mut bm);
        for &word in &bm {
            assert_eq!(word, 0);
        }
    }
}

#[inline]
pub fn mask_words(vocab: usize) -> usize {
    vocab.div_ceil(32)
}

#[inline]
pub fn all_allowed(vocab: usize) -> Vec<u32> {
    vec![u32::MAX; mask_words(vocab)]
}

#[inline]
pub fn bit_allowed(mask: &[u32], j: usize) -> bool {
    let word = j >> 5;
    word < mask.len() && (mask[word] >> (j & 31)) & 1 == 1
}

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

pub fn unpack_mask(packed: &[u32], vocab: u32) -> Vec<bool> {
    if packed.is_empty() {
        return vec![true; vocab as usize];
    }
    (0..vocab as usize)
        .map(|j| bit_allowed(packed, j))
        .collect()
}

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
    fn mask_every_case() {
        bit_allowed_indexes_word_and_bit();
        bit_allowed_refuses_tokens_past_the_mask();
        pack_allowed_round_trips_bits();
    }

    fn bit_allowed_indexes_word_and_bit() {
        let mask = [0b101u32];
        assert!(bit_allowed(&mask, 0));
        assert!(!bit_allowed(&mask, 1));
        assert!(bit_allowed(&mask, 2));
        let mask2 = [0u32, 0b10u32];
        assert!(bit_allowed(&mask2, 33));
        assert!(!bit_allowed(&mask2, 32));
    }

    fn bit_allowed_refuses_tokens_past_the_mask() {
        let mask = pack_allowed(151_669, &[7, 151_668]);
        assert_eq!(mask.len(), 4740);
        assert!(bit_allowed(&mask, 7));
        assert!(bit_allowed(&mask, 151_668));
        for j in [151_680, 151_935, usize::MAX / 64] {
            assert!(!bit_allowed(&mask, j), "token {j} should be disallowed");
        }
    }

    fn pack_allowed_round_trips_bits() {
        let mask = pack_allowed(40, &[0, 2, 33]);
        assert_eq!(mask.len(), 2);
        assert!(bit_allowed(&mask, 0));
        assert!(!bit_allowed(&mask, 1));
        assert!(bit_allowed(&mask, 2));
        assert!(bit_allowed(&mask, 33));
        assert!(!bit_allowed(&mask, 32));
        let m2 = pack_allowed(8, &[3, 99]);
        assert!(bit_allowed(&m2, 3));
    }
}

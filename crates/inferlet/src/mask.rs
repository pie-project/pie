//! Packed-bitmask logit-mask semantics, byte-identical to the engine's `0x65 MaskApply` op.
//! A logit mask is one bit per vocabulary token, packed into `ceil(vocab/32)` `u32` words: bit `1` = allowed. Token `j`'s bit is word `j >> 5`, bit `j & 31`.

/// Number of `u32` words a packed mask for `vocab` tokens occupies.
#[inline]
pub fn mask_words(vocab: usize) -> usize {
    vocab.div_ceil(32)
}

/// An all-allowed packed mask for `vocab` tokens (every bit `1`). The identity
/// under the word-wise AND that composes two constraints; tail bits past
/// `vocab` in the last word are don't-care (consumers index only
/// `[0, vocab)`).
#[inline]
pub fn all_allowed(vocab: usize) -> Vec<u32> {
    vec![u32::MAX; mask_words(vocab)]
}

/// Whether token `j` is allowed (bit `1`) in the packed mask.
/// Tokens past the mask's word coverage read as disallowed — a model's output vocabulary is routinely padded above its tokenizer's, and those slots decode to no token at all.
#[inline]
pub fn bit_allowed(mask: &[u32], j: usize) -> bool {
    let word = j >> 5;
    word < mask.len() && (mask[word] >> (j & 31)) & 1 == 1
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

/// Expand a packed mask into one `bool` per token — the allocating inverse of
/// [`bit_allowed`], for callers that scan the whole vocabulary anyway. An empty
/// `packed` means the constraint is inactive, so everything is allowed.
pub fn unpack_mask(packed: &[u32], vocab: u32) -> Vec<bool> {
    if packed.is_empty() {
        return vec![true; vocab as usize];
    }
    (0..vocab as usize)
        .map(|j| bit_allowed(packed, j))
        .collect()
}

/// Argmax over `logits` with the packed mask applied: a disallowed token is treated as `-inf`. Ties go to the lowest index. If every token is disallowed, returns `0`.
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

    /// A model's padded output vocabulary can outrun the tokenizer's, so the
    /// top logit slots have no bit in a constraint mask packed for the
    /// tokenizer. They decode to no token and must read as disallowed rather
    /// than panicking. Qwen3 is the live case: 151936 declared against a mask
    /// covering 151680.
    #[test]
    fn bit_allowed_refuses_tokens_past_the_mask() {
        let mask = pack_allowed(151_669, &[7, 151_668]);
        assert_eq!(mask.len(), 4740);
        assert!(bit_allowed(&mask, 7));
        assert!(bit_allowed(&mask, 151_668));
        for j in [151_680, 151_935, usize::MAX / 64] {
            assert!(!bit_allowed(&mask, j), "token {j} should be disallowed");
        }
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

}

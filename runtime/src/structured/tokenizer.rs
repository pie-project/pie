//! Tokenizer vocabulary handling.
//!
//! `TokenizerInfo` encapsulates the vocabulary of an LLM tokenizer, providing:
//! - Decoded vocabulary (raw token bytes → decoded strings)
//! - Lexicographically sorted vocabulary for efficient prefix-based iteration
//! - Trie subtree ranges for batch prefix rejection during token mask generation

use anyhow::{Result, bail};

/// The type of vocabulary encoding used by the tokenizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VocabType {
    /// Tokens are used as-is, no decoding needed.
    Raw,
    /// Byte fallback encoding: `<0xAB>` → byte 0xAB, `▁` (U+2581) → space.
    /// Used by SentencePiece tokenizers (e.g., Llama).
    ByteFallback,
    /// Byte-level BPE: each byte is mapped to a unique Unicode character.
    /// Used by GPT-2/GPT-3 style tokenizers.
    ByteLevel,
}

/// Tokenizer vocabulary information for grammar-guided generation.
#[derive(Debug, Clone)]
pub struct TokenizerInfo {
    /// Decoded vocabulary: decoded_vocab[token_id] = decoded string.
    decoded_vocab: Vec<String>,
    /// Vocabulary sorted lexicographically by decoded string: (original_token_id, decoded_string).
    sorted_vocab: Vec<(u32, String)>,
    /// Total vocabulary size (may be larger than decoded_vocab.len() due to padding tokens).
    vocab_size: usize,
    /// Vocabulary encoding type.
    vocab_type: VocabType,
    /// Token IDs for special tokens (empty decoded string).
    special_token_ids: Vec<u32>,
    /// Trie subtree ranges: for sorted_vocab[i], trie_subtree_end[i] is the index of the first
    /// token whose decoded string does NOT start with sorted_vocab[i].second.
    /// Used to skip entire subtrees of tokens during mask generation.
    trie_subtree_end: Vec<usize>,
}

impl TokenizerInfo {
    /// Create a new TokenizerInfo from an encoded vocabulary.
    ///
    /// - `encoded_vocab`: The raw token strings from the tokenizer.
    /// - `vocab_type`: How to decode token strings.
    /// - `vocab_size`: Total vocabulary size (if larger than encoded_vocab.len(), extra tokens are
    ///   treated as special).
    pub fn new(
        encoded_vocab: &[String],
        vocab_type: VocabType,
        vocab_size: Option<usize>,
    ) -> Result<Self> {
        let vocab_size = vocab_size.unwrap_or(encoded_vocab.len());
        if vocab_size < encoded_vocab.len() {
            bail!(
                "vocab_size ({}) must be >= encoded_vocab.len() ({})",
                vocab_size,
                encoded_vocab.len()
            );
        }

        // Decode each token
        let decoded_vocab: Vec<String> = encoded_vocab
            .iter()
            .map(|tok| decode_token(tok, vocab_type))
            .collect::<Result<Vec<_>>>()?;

        // Build sorted vocab + special tokens
        let mut sorted_vocab = Vec::new();
        let mut special_token_ids = Vec::new();

        for (id, decoded) in decoded_vocab.iter().enumerate() {
            let id = id as u32;
            if decoded.is_empty() {
                special_token_ids.push(id);
            } else {
                sorted_vocab.push((id, decoded.clone()));
            }
        }

        // Sort lexicographically by decoded string
        sorted_vocab.sort_by(|a, b| a.1.cmp(&b.1));

        // Build trie subtree ranges
        let trie_subtree_end = build_trie_subtree_ranges(&sorted_vocab);

        Ok(Self {
            decoded_vocab,
            sorted_vocab,
            vocab_size,
            vocab_type,
            special_token_ids,
            trie_subtree_end,
        })
    }

    pub fn vocab_type(&self) -> VocabType {
        self.vocab_type
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn decoded_vocab(&self) -> &[String] {
        &self.decoded_vocab
    }

    pub fn sorted_vocab(&self) -> &[(u32, String)] {
        &self.sorted_vocab
    }

    pub fn special_token_ids(&self) -> &[u32] {
        &self.special_token_ids
    }

    pub fn trie_subtree_end(&self) -> &[usize] {
        &self.trie_subtree_end
    }

    /// Get the decoded string for a token ID.
    pub fn decode_token(&self, token_id: u32) -> Option<&str> {
        self.decoded_vocab.get(token_id as usize).map(|s| s.as_str())
    }
}

/// Decode a single token according to the vocabulary type.
fn decode_token(encoded: &str, vocab_type: VocabType) -> Result<String> {
    match vocab_type {
        VocabType::Raw => Ok(encoded.to_string()),
        VocabType::ByteFallback => decode_byte_fallback(encoded),
        VocabType::ByteLevel => decode_byte_level(encoded),
    }
}

/// Decode a byte-fallback token (SentencePiece style).
/// `<0xAB>` → byte 0xAB, `▁` (U+2581) → space.
fn decode_byte_fallback(encoded: &str) -> Result<String> {
    // Check for <0xHH> pattern
    if encoded.len() == 6
        && encoded.starts_with("<0x")
        && encoded.ends_with('>')
    {
        let hex = &encoded[3..5];
        let byte = u8::from_str_radix(hex, 16)
            .map_err(|_| anyhow::anyhow!("invalid byte fallback token: {}", encoded))?;
        // Return the raw byte as a string (may not be valid UTF-8 on its own,
        // but we need to preserve it for byte-level matching)
        return Ok(String::from(byte as char));
    }

    // Replace ▁ (U+2581, "LOWER ONE EIGHTH BLOCK") with space
    Ok(encoded.replace('\u{2581}', " "))
}

/// Decode a byte-level BPE token (GPT-2 style).
/// Maps each Unicode character back to its original byte using the inverse of
/// the `bytes_to_unicode()` mapping.
fn decode_byte_level(encoded: &str) -> Result<String> {
    let mut bytes = Vec::new();
    for c in encoded.chars() {
        let cp = c as u32;
        match byte_level_char_to_byte(cp) {
            Some(b) => bytes.push(b),
            None => {
                // If the character is not in the mapping, preserve as UTF-8
                let mut buf = [0u8; 4];
                let s = c.encode_utf8(&mut buf);
                bytes.extend_from_slice(s.as_bytes());
            }
        }
    }
    // The result may be arbitrary bytes; try to interpret as UTF-8 but fall back to lossy
    Ok(String::from_utf8_lossy(&bytes).to_string())
}

/// Inverse of GPT-2's `bytes_to_unicode()` mapping.
/// Maps a Unicode codepoint back to the original byte value.
fn byte_level_char_to_byte(cp: u32) -> Option<u8> {
    // The GPT-2 bytes_to_unicode mapping:
    // For bytes 0x21..=0x7E, 0xA1..=0xAC, 0xAE..=0xFF: identity mapping (byte == codepoint)
    // For remaining bytes (0x00..=0x20, 0x7F..=0xA0, 0xAD): mapped to 256..=511 range
    match cp {
        0x21..=0x7E => Some(cp as u8),
        0xA1..=0xAC => Some(cp as u8),
        0xAE..=0xFF => Some(cp as u8),
        // Mapped bytes: codepoints 256+ map back to the "gap" bytes
        _ => {
            // Build the inverse: for each byte not in the identity ranges,
            // it was mapped to 256 + index
            let gap_bytes: Vec<u8> = (0u16..=255)
                .filter(|&b| {
                    !((0x21..=0x7E).contains(&b)
                        || (0xA1..=0xAC).contains(&b)
                        || (0xAE..=0xFF).contains(&b))
                })
                .map(|b| b as u8)
                .collect();

            let offset = cp.checked_sub(256)?;
            gap_bytes.get(offset as usize).copied()
        }
    }
}

/// Build trie subtree ranges for the sorted vocabulary.
///
/// For each entry `sorted_vocab[i]`, compute the index of the first entry that does NOT
/// have `sorted_vocab[i].1` as a prefix. This allows skipping entire subtrees during
/// token mask generation when a prefix is rejected.
fn build_trie_subtree_ranges(sorted_vocab: &[(u32, String)]) -> Vec<usize> {
    let n = sorted_vocab.len();
    let mut ranges = vec![n; n];
    let mut stack: Vec<(usize, &str)> = Vec::new(); // (index, prefix)

    for i in 0..n {
        let s = &sorted_vocab[i].1;
        // Pop entries whose prefix does not match the current string
        while let Some(&(idx, prefix)) = stack.last() {
            if s.starts_with(prefix) {
                break;
            }
            ranges[idx] = i;
            stack.pop();
        }
        stack.push((i, s));
    }

    // Pop remaining entries
    while let Some((idx, _)) = stack.pop() {
        ranges[idx] = n;
    }

    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_vocab() {
        let vocab: Vec<String> = vec!["hello".into(), "world".into(), "!".into()];
        let info = TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap();
        assert_eq!(info.vocab_size(), 3);
        assert_eq!(info.decoded_vocab(), &["hello", "world", "!"]);
    }

    #[test]
    fn test_byte_fallback_decode() {
        assert_eq!(decode_byte_fallback("<0x41>").unwrap(), "A");
        assert_eq!(decode_byte_fallback("<0x0A>").unwrap(), "\n");
        assert_eq!(decode_byte_fallback("▁hello").unwrap(), " hello");
        assert_eq!(decode_byte_fallback("normal").unwrap(), "normal");
    }

    #[test]
    fn test_byte_level_decode() {
        // 'A' (0x41) is in the identity range, maps to itself
        assert_eq!(decode_byte_level("A").unwrap(), "A");
        // 'Ġ' (U+0120) maps to byte 0x20 (space) in GPT-2
        assert_eq!(decode_byte_level("Ġ").unwrap(), " ");
    }

    #[test]
    fn test_sorted_vocab() {
        let vocab: Vec<String> = vec!["banana".into(), "apple".into(), "cherry".into()];
        let info = TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap();
        let sorted: Vec<&str> = info.sorted_vocab().iter().map(|(_, s)| s.as_str()).collect();
        assert_eq!(sorted, &["apple", "banana", "cherry"]);
    }

    #[test]
    fn test_special_tokens() {
        let vocab: Vec<String> = vec!["hello".into(), "".into(), "world".into()];
        let info = TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap();
        assert_eq!(info.special_token_ids(), &[1]);
    }

    #[test]
    fn test_trie_subtree_ranges() {
        let vocab: Vec<String> = vec![
            "a".into(),
            "ab".into(),
            "abc".into(),
            "b".into(),
            "bc".into(),
        ];
        let info = TokenizerInfo::new(&vocab, VocabType::Raw, None).unwrap();

        // sorted_vocab should be: ["a", "ab", "abc", "b", "bc"]
        let sorted: Vec<&str> = info.sorted_vocab().iter().map(|(_, s)| s.as_str()).collect();
        assert_eq!(sorted, &["a", "ab", "abc", "b", "bc"]);

        let ranges = info.trie_subtree_end();
        // "a" prefixes "ab" and "abc", so range should end at index 3 (which is "b")
        assert_eq!(ranges[0], 3);
        // "ab" prefixes "abc", range ends at 3
        assert_eq!(ranges[1], 3);
        // "abc" has no further prefixes, range ends at 3
        assert_eq!(ranges[2], 3);
        // "b" prefixes "bc", range ends at 5
        assert_eq!(ranges[3], 5);
        // "bc" is last in its subtree, range ends at 5
        assert_eq!(ranges[4], 5);
    }

    #[test]
    fn test_vocab_size_larger_than_vocab() {
        let vocab: Vec<String> = vec!["a".into(), "b".into()];
        let info = TokenizerInfo::new(&vocab, VocabType::Raw, Some(100)).unwrap();
        assert_eq!(info.vocab_size(), 100);
        assert_eq!(info.decoded_vocab().len(), 2);
    }

    #[test]
    fn test_vocab_size_too_small() {
        let vocab: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let result = TokenizerInfo::new(&vocab, VocabType::Raw, Some(2));
        assert!(result.is_err());
    }
}

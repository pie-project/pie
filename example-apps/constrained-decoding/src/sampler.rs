//! A custom sampler that constrains token generation to match a Lark grammar.
//!
//! Uses the `llguidance` library to compute token masks based on grammar state,
//! ensuring outputs are always syntactically valid.

use fancy_regex::Regex;
use inferlet::sampler::Sample;
use inferlet::{Result, bail};
use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};
use llguidance::{Matcher, ParserFactory};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub type Rank = u32;

pub struct ConstrainedSampler {
    inner: RefCell<Inner>,
}

struct Inner {
    constraint: Matcher,
    eos_token_id: u32,
}

type Vocab = (Vec<u32>, Vec<Vec<u8>>);

impl ConstrainedSampler {
    pub fn new(
        vocab: Vocab,
        special_tokens: Vocab,
        split_regex: String,
        grammar: String,
        eos_token_id: u32,
        escape_non_printable: bool,
    ) -> Self {
        let (ranks, words) = vocab;

        let rank_map: HashMap<u32, Vec<u8>> = ranks.into_iter().zip(words).collect();

        let (special_ranks, special_words) = special_tokens;
        let special_tokens: HashMap<String, u32> = special_words
            .into_iter()
            .map(|w| String::from_utf8(w).unwrap())
            .zip(special_ranks)
            .collect();

        let tokenizer = BytePairEncoder::new(
            rank_map,
            special_tokens,
            &split_regex,
            eos_token_id,
            escape_non_printable,
        );
        let tokenizer_env = tokenizer.unwrap().to_env();

        let grammar = TopLevelGrammar::from_lark(grammar);

        let factory = ParserFactory::new_simple(&tokenizer_env).unwrap();
        let parser = factory.create_parser(grammar);

        let constraint = Matcher::new(parser);
        ConstrainedSampler {
            inner: RefCell::new(Inner {
                constraint,
                eos_token_id,
            }),
        }
    }
}

impl Sample for ConstrainedSampler {
    fn sample(&self, token_ids: &[u32], probs: &[f32]) -> u32 {
        let mut inner = self.inner.borrow_mut();
        let res = inner.constraint.compute_mask();

        if let Err(_) = res {
            return inner.eos_token_id;
        }

        let res = res.unwrap();

        if res.is_empty() {
            return inner.eos_token_id;
        }

        let mut max_prob = f32::NEG_INFINITY;
        let mut best_token = None;

        // Find the highest-probability token allowed by the grammar mask
        for (i, &token_id) in token_ids.iter().enumerate() {
            if res.is_allowed(token_id) && probs[i] > max_prob {
                max_prob = probs[i];
                best_token = Some(token_id);
            }
        }

        let sampled_token_id = if let Some(token) = best_token {
            token
        } else {
            return res.first_bit_set().unwrap_or(0) as u32;
        };

        // Commit the chosen token to advance the parser state
        inner.constraint.consume_token(sampled_token_id).unwrap();

        sampled_token_id
    }
}

// Byte-pair encoding adapted from tiktoken-rs:
// https://github.com/openai/tiktoken/blob/main/src/lib.rs

/// Performs the byte pair merging operation.
fn _byte_pair_merge(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<(usize, Rank)> {
    let mut parts = Vec::with_capacity(piece.len() + 1);

    // Creates initial parts and finds the first best merge.
    let mut min_rank: (Rank, usize) = (Rank::MAX, usize::MAX);
    for i in 0..piece.len() - 1 {
        let rank = *ranks.get(&piece[i..i + 2]).unwrap_or(&Rank::MAX);
        if rank < min_rank.0 {
            min_rank = (rank, i);
        }
        parts.push((i, rank));
    }
    parts.push((piece.len() - 1, Rank::MAX));
    parts.push((piece.len(), Rank::MAX));

    let get_rank = |parts: &Vec<(usize, Rank)>, i: usize| {
        if (i + 3) < parts.len() {
            *ranks
                .get(&piece[parts[i].0..parts[i + 3].0])
                .unwrap_or(&Rank::MAX)
        } else {
            Rank::MAX
        }
    };

    // Greedily merges the best pairs until no more merges are possible.
    while min_rank.0 != Rank::MAX {
        let i = min_rank.1;

        if i > 0 {
            parts[i - 1].1 = get_rank(&parts, i - 1);
        }
        parts[i].1 = get_rank(&parts, i);
        parts.remove(i + 1);

        min_rank = (Rank::MAX, usize::MAX);
        for (i, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < min_rank.0 {
                min_rank = (rank, i);
            }
        }
    }
    parts
}

/// Encodes a byte slice into a sequence of token ranks using byte-pair encoding.
pub fn byte_pair_encode(piece: &[u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<Rank> {
    if piece.len() == 1 {
        return vec![ranks[piece]];
    }
    _byte_pair_merge(ranks, piece)
        .windows(2)
        .map(|part| ranks[&piece[part[0].0..part[1].0]])
        .collect()
}

/// An encoder that uses byte-pair encoding to tokenize text.
#[derive(Clone)]
pub struct BytePairEncoder {
    encoder: HashMap<Vec<u8>, Rank>,
    special_tokens: HashSet<String>,
    special_tokens_encoder: HashMap<String, Rank>,
    regex: Regex,
    special_regex: Regex,
    tok_trie: TokTrie,
    escape_non_printable: bool,
}

impl BytePairEncoder {
    pub fn new(
        decoder: HashMap<Rank, Vec<u8>>,
        special_tokens_encoder: HashMap<String, Rank>,
        pattern: &str,
        eos_token: u32,
        escape_non_printable: bool,
    ) -> Result<Self> {
        let regex = Regex::new(pattern)?;
        let special_regex = {
            let parts = special_tokens_encoder
                .keys()
                .map(|s| fancy_regex::escape(s))
                .collect::<Vec<_>>();
            Regex::new(&parts.join("|"))?
        };

        let encoder: HashMap<Vec<u8>, Rank> =
            decoder.iter().map(|(k, v)| (v.clone(), *k)).collect();

        if encoder.len() != decoder.len() {
            bail!(
                "Encoder and decoder must be of equal length; maybe you had duplicate token indices in your encoder?"
            );
        }

        // Build the token vocabulary for TokTrie
        let mut max_rank = 0;
        for rank in decoder.keys() {
            max_rank = max_rank.max(*rank);
        }
        for rank in special_tokens_encoder.values() {
            max_rank = max_rank.max(*rank);
        }
        let n_vocab = (max_rank + 1) as usize;

        let mut tokens = vec![vec![]; n_vocab];

        for (rank, bytes) in decoder.iter() {
            // If escape_non_printable is enabled, the vocab contains escaped bytes
            // (e.g., "Ġ" for space). We need to unescape them for the TokTrie
            // so that the grammar parser sees raw bytes (e.g., 0x20 for space).
            let raw_bytes = if escape_non_printable {
                unescape_non_printable(bytes)
            } else {
                bytes.clone()
            };
            tokens[*rank as usize] = raw_bytes;
        }

        for (name, rank) in special_tokens_encoder.iter() {
            let mut spec_bytes = Vec::with_capacity(name.len() + 1);
            spec_bytes.push(TokTrie::SPECIAL_TOKEN_MARKER);
            spec_bytes.extend_from_slice(name.as_bytes());
            tokens[*rank as usize] = spec_bytes;
        }

        let special_tokens = special_tokens_encoder.keys().cloned().collect();

        // Fill gaps in the vocabulary with placeholder tokens
        for (i, token) in tokens.iter_mut().enumerate() {
            if token.is_empty() {
                let mut name = format!(".<[{i}]>").into_bytes();
                name[0] = TokTrie::SPECIAL_TOKEN_MARKER;
                *token = name;
            }
        }

        let tok_trie = TokTrie::from(
            &TokRxInfo {
                vocab_size: n_vocab as u32,
                tok_eos: eos_token,
                tok_end_of_turn: None,
                tok_unk: None,
                tok_pad: None,
                tok_bos: None,
            },
            &tokens,
        );

        Ok(Self {
            encoder,
            special_tokens,
            special_tokens_encoder,
            regex,
            special_regex,
            tok_trie,
            escape_non_printable,
        })
    }

    /// Encodes a string into tokens, respecting a set of allowed special tokens.
    pub fn encode(&self, text: &str) -> Vec<Rank> {
        let mut ret = vec![];

        let mut start = 0;
        loop {
            let mut next_special;
            let mut start_find = start;
            loop {
                next_special = self.special_regex.find_from_pos(text, start_find).unwrap();
                match next_special {
                    Some(m) => {
                        if self.special_tokens.contains(&text[m.start()..m.end()]) {
                            break;
                        }
                        start_find = m.start() + 1;
                    }
                    None => break,
                }
            }
            let end = next_special.map_or(text.len(), |m| m.start());

            for mat in self.regex.find_iter(&text[start..end]) {
                let mut piece = mat.unwrap().as_str().as_bytes();

                let escaped_piece = escape_non_printable(piece);
                if self.escape_non_printable {
                    piece = escaped_piece.as_bytes();
                }

                if let Some(token) = self.encoder.get(piece) {
                    ret.push(*token);
                    continue;
                }
                let tokens = byte_pair_encode(piece, &self.encoder);
                ret.extend(&tokens);
            }

            match next_special {
                // And here we push the special token
                Some(m) => {
                    let piece = m.as_str();
                    let token = self.special_tokens_encoder[piece];
                    ret.push(token);
                    start = m.end();
                }
                None => break,
            }
        }

        ret
    }

    /// Wraps the tokenizer in an `Arc` to create a `TokEnv`.
    pub fn to_env(self) -> TokEnv {
        Arc::new(self)
    }
}

impl TokenizerEnv for BytePairEncoder {
    fn tok_trie(&self) -> &TokTrie {
        &self.tok_trie
    }

    /// Tokenizes a byte slice, using BPE as a fallback for unknown sequences.
    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        self.tok_trie
            .tokenize_with_greedy_fallback(s, |s| self.encode(s))
    }

    /// Tokenizes a byte slice, handling special tokens correctly within fallback sequences.
    fn tokenize_bytes_special(&self, s: &[u8]) -> Vec<TokenId> {
        self.tok_trie.tokenize_with_greedy_fallback(s, |s| {
            self.tok_trie.tokenize_with_special(s, |s| self.encode(s))
        })
    }
}

/// Generate the 256-entry “byte-level” maps.
///
///  * `enc[byte] -> char`  (stage-2 encoding)
///  * `dec[char] -> byte`  (stage-2 decoding)
///
/// The algorithm is identical to OpenAI-tiktoken’s `bytes_to_unicode()`.
///
/// Printable ranges kept as-is:
///   1.  '!' (0x21) .. '~' (0x7E)
///   2.  '¡' (0xA1) .. '¬' (0xAC)
///   3.  '®' (0xAE) .. 'ÿ' (0xFF)
///
/// Everything else (control bytes, space, TAB, …) is
/// remapped to the BMP starting at U+0100.
fn build_tables() -> ([char; 256], HashMap<char, u8>) {
    // Step 1: collect the “safe” byte values we keep unchanged
    let mut bs: Vec<u8> = (b'!'..=b'~').collect(); // 0x21–0x7E
    bs.extend(0xA1..=0xAC); // 0xA1–0xAC
    bs.extend(0xAE..=0xFF); // 0xAE–0xFF

    // cs will hold the *Unicode code points* corresponding to bs
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();

    // Step 2: assign code points ≥ 0x100 to the remaining bytes
    let mut n = 0u32;
    for b in 0u8..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n); // U+0100, U+0101, …
            n += 1;
        }
    }

    // Convert to char
    let cs: Vec<char> = cs.into_iter().map(|u| char::from_u32(u).unwrap()).collect();

    // Zip into the forward & reverse tables
    let mut enc = ['\0'; 256];
    let mut dec = HashMap::with_capacity(256);
    for (b, ch) in bs.into_iter().zip(cs.into_iter()) {
        enc[b as usize] = ch;
        dec.insert(ch, b);
    }
    (enc, dec)
}

/// Encode a byte slice with the Qwen/GPT byte-level mapping.
fn escape_non_printable(bytes: &[u8]) -> String {
    static TABLES: once_cell::sync::Lazy<([char; 256], HashMap<char, u8>)> =
        once_cell::sync::Lazy::new(build_tables);

    bytes.iter().map(|&b| TABLES.0[b as usize]).collect()
}

/// Decode an escaped string back to raw bytes (reverse of escape_non_printable).
/// This handles UTF-8 encoded strings where non-printable bytes were escaped to
/// Unicode code points >= U+0100.
fn unescape_non_printable(bytes: &[u8]) -> Vec<u8> {
    static TABLES: once_cell::sync::Lazy<([char; 256], HashMap<char, u8>)> =
        once_cell::sync::Lazy::new(build_tables);

    // Try to interpret as UTF-8 and decode each char
    match std::str::from_utf8(bytes) {
        Ok(s) => s
            .chars()
            .filter_map(|c| TABLES.1.get(&c).copied())
            .collect(),
        // If not valid UTF-8, return as-is (might be raw bytes already)
        Err(_) => bytes.to_vec(),
    }
}

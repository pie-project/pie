//! Top-level Tokenizer struct.
//!
//! Provides `encode` / `decode` over a BPE vocabulary. Construction happens
//! via [`loader`] (Hugging Face `tokenizer.json` and native tiktoken formats).
//!
//! External formats compile into one of a small number of supported modern
//! pipelines. Unsupported legacy combinations are rejected at load time.

mod bpe;
pub mod loader;

use std::borrow::Cow;
use std::sync::{Arc, OnceLock};

use aho_corasick::AhoCorasick;
use anyhow::Context;
use smallvec::SmallVec;
use unicode_normalization::{IsNormalized, UnicodeNormalization, is_nfc_quick};

use bpe::BpeTable;

// ---------------------------------------------------------------------------
// Added tokens
// ---------------------------------------------------------------------------

/// Representation of a token added on top of the base vocabulary.
#[derive(Debug, Clone)]
pub struct AddedToken {
    pub id: u32,
    pub content: String,
    pub special: bool,
}

/// Whether an already-known whole piece bypasses BPE merging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BpeMode {
    Merge,
    PreferWholeToken,
}

impl BpeMode {
    #[inline]
    fn prefer_whole_token(self) -> bool {
        matches!(self, Self::PreferWholeToken)
    }
}

/// Compiled tokenizer behavior for the modern model families supported by Pie.
#[derive(Debug)]
pub(crate) enum Pipeline {
    /// Optional NFC, one or more isolated regex splitters, then byte-level BPE.
    ByteLevelRegex {
        nfc: bool,
        splitters: Vec<fancy_regex::Regex>,
        bpe_mode: BpeMode,
    },
    /// Gemma-style space marker normalization with byte fallback on decode.
    ByteFallbackReplace {
        normalizer_from: String,
        normalizer_to: String,
        unk_token_id: Option<u32>,
    },
    /// Minimal char-level path used by grammar fixtures and `from_vocab`.
    RawChar,
}

impl Pipeline {
    fn grammar_token_bytes(&self, raw: Arc<[u8]>) -> Arc<[u8]> {
        match self {
            Self::ByteLevelRegex { .. } | Self::RawChar => raw,
            Self::ByteFallbackReplace {
                normalizer_from,
                normalizer_to,
                ..
            } => byte_fallback_value(raw.as_ref())
                .map(|byte| Arc::from([byte]))
                .unwrap_or_else(|| {
                    replace_bytes(
                        raw.as_ref(),
                        normalizer_to.as_bytes(),
                        normalizer_from.as_bytes(),
                    )
                    .into()
                }),
        }
    }
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

/// A BPE tokenizer for the modern model profiles supported by Pie.
///
/// # Example
///
/// ```no_run
/// use std::path::Path;
/// use pie_tokenizer::Tokenizer;
///
/// let tokenizer = Tokenizer::from_file(Path::new("tokenizer.json")).unwrap();
/// let ids = tokenizer.encode("Hello, world!");
/// let text = tokenizer.decode(&ids, false);
/// ```
pub struct Tokenizer {
    // Core BPE
    bpe: BpeTable,

    pipeline: Pipeline,

    // Added / special tokens (sorted for binary_search)
    special_token_ids: Vec<u32>,
    added_token_matcher: Option<AhoCorasick>,
    added_token_ids: Vec<u32>,

    grammar: OnceLock<GrammarVocabulary>,
}

struct GrammarVocabulary {
    token_bytes: Vec<Option<Arc<[u8]>>>,
    sorted_token_ids: Vec<u32>,
    trie_subtree_end: Vec<usize>,
}

/// Stateful incremental decoder for streaming generation.
pub struct TokenizerDecoder {
    tokenizer: Arc<Tokenizer>,
    skip_special: bool,
    pending_utf8: Vec<u8>,
    fallback_run: Vec<u8>,
}

impl Tokenizer {
    /// Construct a `Tokenizer` from its components.
    ///
    /// Prefer [`from_file`] for loading an external tokenizer artifact.
    /// This constructor is for use by format-specific loaders
    /// (e.g. [`loader::huggingface`]).
    pub(crate) fn new(
        bpe: BpeTable,
        pipeline: Pipeline,
        added_tokens: Vec<AddedToken>,
    ) -> anyhow::Result<Self> {
        let mut special_token_ids: Vec<u32> = added_tokens
            .iter()
            .filter(|at| at.special)
            .map(|at| at.id)
            .collect();
        special_token_ids.sort_unstable();

        // Build an Aho-Corasick matcher for added/special tokens.
        let (added_token_matcher, added_token_ids) = if !added_tokens.is_empty() {
            let patterns: Vec<&str> = added_tokens.iter().map(|t| t.content.as_str()).collect();
            let matcher = AhoCorasick::builder()
                .match_kind(aho_corasick::MatchKind::LeftmostLongest)
                .build(&patterns)
                .context("building added-token matcher")?;
            let ids: Vec<u32> = added_tokens.iter().map(|t| t.id).collect();
            (Some(matcher), ids)
        } else {
            (None, vec![])
        };

        Ok(Tokenizer {
            bpe,
            pipeline,
            special_token_ids,
            added_token_matcher,
            added_token_ids,
            grammar: OnceLock::new(),
        })
    }

    // -----------------------------------------------------------------------
    // Loading (convenience delegates to loader)
    // -----------------------------------------------------------------------

    /// Build a minimal tokenizer from raw token strings.
    ///
    /// Each string becomes a token with ID = its index. No BPE merges, no
    /// normalization, no special tokens. Uses the raw char fixture pipeline.
    pub fn from_vocab(vocab: &[String]) -> Self {
        use std::collections::HashMap;
        let map: HashMap<u32, Vec<u8>> = vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (i as u32, s.as_bytes().to_vec()))
            .collect();
        let bpe = bpe::BpeTable::from_decoder_map(map)
            .expect("enumerated vocabulary must have contiguous IDs");
        Self::new(bpe, Pipeline::RawChar, vec![])
            .expect("raw vocabulary must produce a valid tokenizer")
    }

    /// Load a tokenizer from a supported external format.
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        loader::from_file(path)
    }

    /// Load a Kimi K2/K2.5 tiktoken rank file (`base64(token_bytes) rank`).
    ///
    /// A rank file does not contain its required split regex. The sibling
    /// `tokenizer_config.json` must identify the official
    /// `tokenization_kimi.TikTokenTokenizer`; unknown tiktoken formats are
    /// rejected rather than encoded without pre-tokenization.
    pub fn from_tiktoken_file(path: &std::path::Path) -> anyhow::Result<Self> {
        loader::tiktoken::from_file(path)
    }

    /// Create an incremental decoder sharing this tokenizer.
    pub fn decoder(self: &Arc<Self>, skip_special: bool) -> TokenizerDecoder {
        TokenizerDecoder {
            tokenizer: self.clone(),
            skip_special,
            pending_utf8: Vec::new(),
            fallback_run: Vec::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Encoding
    // -----------------------------------------------------------------------

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::with_capacity(text.len() / 3 + 1);
        self.encode_into(text, &mut ids);
        ids
    }

    /// Append encoded token IDs to an existing output buffer.
    pub fn encode_into(&self, text: &str, ids: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let Some(matcher) = &self.added_token_matcher else {
            self.encode_text(text, ids);
            return;
        };

        let mut last_end = 0;
        for matched in matcher.find_iter(text) {
            if matched.start() > last_end {
                self.encode_text(&text[last_end..matched.start()], ids);
            }
            ids.push(self.added_token_ids[matched.pattern().as_usize()]);
            last_end = matched.end();
        }
        if last_end < text.len() {
            self.encode_text(&text[last_end..], ids);
        }
    }

    /// Encode a single piece of text using the appropriate BPE atom mode.
    #[inline]
    fn encode_piece(&self, piece: &str, ids: &mut Vec<u32>) {
        if piece.is_empty() {
            return;
        }
        match &self.pipeline {
            Pipeline::ByteLevelRegex { bpe_mode, .. } => bpe::bpe_encode_bytes(
                piece.as_bytes(),
                &self.bpe,
                bpe_mode.prefer_whole_token(),
                false,
                None,
                ids,
            ),
            Pipeline::ByteFallbackReplace { unk_token_id, .. } => {
                bpe::bpe_encode_chars(piece, &self.bpe, false, true, *unk_token_id, ids)
            }
            Pipeline::RawChar => bpe::bpe_encode_chars(piece, &self.bpe, true, false, None, ids),
        }
    }

    fn encode_text(&self, text: &str, ids: &mut Vec<u32>) {
        match &self.pipeline {
            Pipeline::ByteLevelRegex { nfc, splitters, .. } => {
                let text = if *nfc && is_nfc_quick(text.chars()) != IsNormalized::Yes {
                    Cow::Owned(text.nfc().collect())
                } else {
                    Cow::Borrowed(text)
                };
                self.split_regex_sequence(&text, splitters, ids);
            }
            Pipeline::ByteFallbackReplace {
                normalizer_from,
                normalizer_to,
                ..
            } => {
                let text = if text.contains(normalizer_from.as_str()) {
                    Cow::Owned(text.replace(normalizer_from.as_str(), normalizer_to.as_str()))
                } else {
                    Cow::Borrowed(text)
                };
                self.encode_piece(&text, ids);
            }
            Pipeline::RawChar => self.encode_piece(text, ids),
        }
    }

    fn split_regex_sequence(
        &self,
        text: &str,
        splitters: &[fancy_regex::Regex],
        ids: &mut Vec<u32>,
    ) {
        if let [regex] = splitters {
            let output_start = ids.len();
            let mut last_end = 0;
            for result in regex.find_iter(text) {
                let Ok(matched) = result else {
                    ids.truncate(output_start);
                    self.encode_piece(text, ids);
                    return;
                };
                if matched.start() > last_end {
                    self.encode_piece(&text[last_end..matched.start()], ids);
                }
                if matched.start() < matched.end() {
                    self.encode_piece(matched.as_str(), ids);
                }
                last_end = matched.end();
            }
            if last_end < text.len() {
                self.encode_piece(&text[last_end..], ids);
            }
            return;
        }

        let mut pieces: SmallVec<[&str; 32]> = SmallVec::new();
        pieces.push(text);

        for regex in splitters {
            let mut next: SmallVec<[&str; 32]> = SmallVec::new();
            for piece in pieces.iter().copied() {
                let mut last_end = 0;
                for result in regex.find_iter(piece) {
                    let Ok(matched) = result else {
                        // Profiles contain trusted static/model regexes. Preserve
                        // input rather than returning a partially encoded result.
                        self.encode_piece(text, ids);
                        return;
                    };
                    if matched.start() > last_end {
                        next.push(&piece[last_end..matched.start()]);
                    }
                    if matched.start() < matched.end() {
                        next.push(matched.as_str());
                    }
                    last_end = matched.end();
                }
                if last_end < piece.len() {
                    next.push(&piece[last_end..]);
                }
            }
            pieces = next;
        }

        for piece in pieces {
            self.encode_piece(piece, ids);
        }
    }

    // -----------------------------------------------------------------------
    // Decoding
    // -----------------------------------------------------------------------

    /// Decode token IDs back into text.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        match &self.pipeline {
            Pipeline::ByteFallbackReplace {
                normalizer_from,
                normalizer_to,
                ..
            } => self.decode_byte_fallback(
                ids,
                skip_special,
                normalizer_to.as_bytes(),
                normalizer_from.as_bytes(),
            ),
            Pipeline::ByteLevelRegex { .. } | Pipeline::RawChar => {
                self.decode_raw(ids, skip_special)
            }
        }
    }

    fn decode_raw(&self, ids: &[u32], skip_special: bool) -> String {
        let mut bytes = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            if skip_special && self.special_token_ids.binary_search(&id).is_ok() {
                continue;
            }
            if let Some(raw) = self.bpe.id_to_bytes(id) {
                bytes.extend_from_slice(raw);
            }
        }
        bytes_to_string(bytes)
    }

    fn decode_byte_fallback(
        &self,
        ids: &[u32],
        skip_special: bool,
        decoder_pattern: &[u8],
        decoder_content: &[u8],
    ) -> String {
        let mut output = Vec::with_capacity(ids.len() * 4);
        let mut fallback_bytes = Vec::new();

        for &id in ids {
            if skip_special && self.special_token_ids.binary_search(&id).is_ok() {
                continue;
            }
            let Some(raw) = self.bpe.id_to_bytes(id) else {
                continue;
            };
            if let Some(byte) = byte_fallback_value(raw) {
                fallback_bytes.push(byte);
            } else {
                flush_byte_fallback(&mut fallback_bytes, &mut output);
                append_replaced(&mut output, raw, decoder_pattern, decoder_content);
            }
        }
        flush_byte_fallback(&mut fallback_bytes, &mut output);
        bytes_to_string(output)
    }

    // -----------------------------------------------------------------------
    // Vocabulary access
    // -----------------------------------------------------------------------

    /// Get the vocabulary size (including added tokens).
    pub fn vocab_size(&self) -> usize {
        self.bpe.vocab_size()
    }

    /// Look up a token string → ID.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.bpe.bytes_to_id(token.as_bytes())
    }

    /// Look up an ID → token bytes.
    pub fn id_to_token(&self, id: u32) -> Option<Vec<u8>> {
        self.bpe.id_to_bytes(id).map(|s| s.to_vec())
    }

    /// Look up an ID → token string (lossy UTF-8 conversion).
    pub fn id_to_token_str(&self, id: u32) -> Option<String> {
        self.bpe
            .id_to_bytes(id)
            .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
    }

    /// Get the split regex when the pipeline has exactly one splitter.
    ///
    /// Returns an empty string for zero or multiple splitters. Use
    /// [`split_regexes`](Self::split_regexes) when sequence semantics matter.
    pub fn get_split_regex(&self) -> String {
        match &self.pipeline {
            Pipeline::ByteLevelRegex { splitters, .. } if splitters.len() == 1 => {
                splitters[0].as_str().to_string()
            }
            _ => String::new(),
        }
    }

    /// Regex splitters in the order they are applied.
    pub fn split_regexes(&self) -> Vec<&str> {
        match &self.pipeline {
            Pipeline::ByteLevelRegex { splitters, .. } => {
                splitters.iter().map(fancy_regex::Regex::as_str).collect()
            }
            _ => Vec::new(),
        }
    }

    /// Get the special token IDs and their byte representations.
    pub fn get_special_tokens(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        let mut ids = Vec::with_capacity(self.special_token_ids.len());
        let mut bytes = Vec::with_capacity(self.special_token_ids.len());
        for &id in &self.special_token_ids {
            if let Some(tok_bytes) = self.bpe.id_to_bytes(id) {
                ids.push(id);
                bytes.push(tok_bytes.to_vec());
            }
        }
        (ids, bytes)
    }

    // -----------------------------------------------------------------------
    // Trie (for grammar-guided generation)
    // -----------------------------------------------------------------------

    /// Decoder-aware bytes contributed by one token.
    ///
    /// Returns `None` for special or unmapped tokens.
    pub fn decoded_token_bytes(&self, token_id: u32) -> Option<&[u8]> {
        self.grammar()
            .token_bytes
            .get(token_id as usize)
            .and_then(Option::as_deref)
            .filter(|bytes| !bytes.is_empty())
    }

    /// Non-special token IDs sorted lexicographically by decoded bytes.
    pub fn sorted_token_ids(&self) -> &[u32] {
        &self.grammar().sorted_token_ids
    }

    /// Trie subtree ranges over [`sorted_token_ids`](Self::sorted_token_ids).
    ///
    /// `trie_subtree_end[i]` is the index of the first entry whose decoded
    /// bytes do **not** start with the bytes for entry `i`.
    /// Enables O(1) subtree skipping during token mask generation.
    pub fn trie_subtree_end(&self) -> &[usize] {
        &self.grammar().trie_subtree_end
    }

    /// Sorted list of special token IDs.
    pub fn special_token_ids(&self) -> &[u32] {
        &self.special_token_ids
    }

    fn grammar(&self) -> &GrammarVocabulary {
        self.grammar.get_or_init(|| {
            let vocab_size = self.bpe.vocab_size();
            let mut token_bytes = Vec::with_capacity(vocab_size);
            for id in 0..vocab_size as u32 {
                let bytes = if self.special_token_ids.binary_search(&id).is_ok() {
                    None
                } else {
                    self.bpe
                        .id_to_shared_bytes(id)
                        .map(|raw| self.pipeline.grammar_token_bytes(raw))
                };
                token_bytes.push(bytes);
            }

            let mut sorted_token_ids = (0..vocab_size as u32)
                .filter(|id| {
                    token_bytes[*id as usize]
                        .as_ref()
                        .is_some_and(|bytes| !bytes.is_empty())
                })
                .collect::<Vec<_>>();
            sorted_token_ids.sort_by(|left, right| {
                token_bytes[*left as usize].cmp(&token_bytes[*right as usize])
            });
            let trie_subtree_end = build_subtree_ranges(&sorted_token_ids, &token_bytes);
            GrammarVocabulary {
                token_bytes,
                sorted_token_ids,
                trie_subtree_end,
            }
        })
    }
}

impl TokenizerDecoder {
    /// Decode newly arrived token IDs and return only the new text.
    pub fn feed(&mut self, ids: &[u32]) -> String {
        let mut output = Vec::with_capacity(ids.len() * 4);
        match &self.tokenizer.pipeline {
            Pipeline::ByteLevelRegex { .. } | Pipeline::RawChar => {
                for &id in ids {
                    if self.skip_special
                        && self.tokenizer.special_token_ids.binary_search(&id).is_ok()
                    {
                        continue;
                    }
                    if let Some(raw) = self.tokenizer.bpe.id_to_bytes(id) {
                        self.pending_utf8.extend_from_slice(raw);
                        drain_utf8(&mut self.pending_utf8, &mut output, false);
                    }
                }
            }
            Pipeline::ByteFallbackReplace {
                normalizer_from,
                normalizer_to,
                ..
            } => {
                for &id in ids {
                    if self.skip_special
                        && self.tokenizer.special_token_ids.binary_search(&id).is_ok()
                    {
                        continue;
                    }
                    let Some(raw) = self.tokenizer.bpe.id_to_bytes(id) else {
                        continue;
                    };
                    if let Some(byte) = byte_fallback_value(raw) {
                        self.fallback_run.push(byte);
                    } else {
                        flush_byte_fallback(&mut self.fallback_run, &mut output);
                        append_replaced(
                            &mut output,
                            raw,
                            normalizer_to.as_bytes(),
                            normalizer_from.as_bytes(),
                        );
                    }
                }
            }
        }
        bytes_to_string(output)
    }

    /// Flush an incomplete trailing byte sequence.
    pub fn finish(&mut self) -> String {
        let mut output = Vec::new();
        match &self.tokenizer.pipeline {
            Pipeline::ByteLevelRegex { .. } | Pipeline::RawChar => {
                drain_utf8(&mut self.pending_utf8, &mut output, true);
            }
            Pipeline::ByteFallbackReplace { .. } => {
                flush_byte_fallback(&mut self.fallback_run, &mut output);
            }
        }
        bytes_to_string(output)
    }

    /// Reset decoder state for a new stream.
    pub fn reset(&mut self) {
        self.pending_utf8.clear();
        self.fallback_run.clear();
    }
}

/// Implement FromStr so `"json".parse::<Tokenizer>()` works idiomatically.
impl std::str::FromStr for Tokenizer {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        loader::huggingface::from_slice(s.as_bytes())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build trie subtree ranges for a sorted vocabulary.
///
/// For each entry `sorted[i]`, returns an array where `result[i]` is the
/// index of the first entry whose string does **not** start with `sorted[i].1`.
fn build_subtree_ranges(sorted_ids: &[u32], vocab: &[Option<Arc<[u8]>>]) -> Vec<usize> {
    let n = sorted_ids.len();
    let mut ranges = vec![n; n];
    let mut stack: Vec<(usize, &[u8])> = Vec::new();

    for (i, &token_id) in sorted_ids.iter().enumerate() {
        let bytes = vocab[token_id as usize]
            .as_deref()
            .expect("sorted token IDs have grammar bytes");
        while let Some(&(idx, prefix)) = stack.last() {
            if bytes.starts_with(prefix) {
                break;
            }
            ranges[idx] = i;
            stack.pop();
        }
        stack.push((i, bytes));
    }

    while let Some((idx, _)) = stack.pop() {
        ranges[idx] = n;
    }

    ranges
}

fn replace_bytes(haystack: &[u8], needle: &[u8], replacement: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(haystack.len());
    append_replaced(&mut result, haystack, needle, replacement);
    result
}

/// Append `haystack`, replacing every `needle` with `replacement`.
fn append_replaced(output: &mut Vec<u8>, haystack: &[u8], needle: &[u8], replacement: &[u8]) {
    if needle.is_empty() {
        output.extend_from_slice(haystack);
        return;
    }

    let finder = memchr::memmem::Finder::new(needle);
    let mut start = 0;
    for pos in finder.find_iter(haystack) {
        output.extend_from_slice(&haystack[start..pos]);
        output.extend_from_slice(replacement);
        start = pos + needle.len();
    }
    output.extend_from_slice(&haystack[start..]);
}

fn byte_fallback_value(token: &[u8]) -> Option<u8> {
    if token.len() != 6 || !token.starts_with(b"<0x") || token[5] != b'>' {
        return None;
    }
    let hex = std::str::from_utf8(&token[3..5]).ok()?;
    u8::from_str_radix(hex, 16).ok()
}

fn flush_byte_fallback(bytes: &mut Vec<u8>, output: &mut Vec<u8>) {
    if bytes.is_empty() {
        return;
    }
    if std::str::from_utf8(bytes).is_ok() {
        output.extend_from_slice(bytes);
    } else {
        for _ in 0..bytes.len() {
            output.extend_from_slice("\u{FFFD}".as_bytes());
        }
    }
    bytes.clear();
}

fn bytes_to_string(bytes: Vec<u8>) -> String {
    String::from_utf8(bytes).unwrap_or_else(|error| {
        let bytes = error.into_bytes();
        String::from_utf8_lossy(&bytes).into_owned()
    })
}

fn drain_utf8(pending: &mut Vec<u8>, output: &mut Vec<u8>, finish: bool) {
    let mut consumed = 0;
    while consumed < pending.len() {
        match std::str::from_utf8(&pending[consumed..]) {
            Ok(_) => {
                output.extend_from_slice(&pending[consumed..]);
                consumed = pending.len();
            }
            Err(error) => {
                let valid_end = consumed + error.valid_up_to();
                output.extend_from_slice(&pending[consumed..valid_end]);
                consumed = valid_end;
                match error.error_len() {
                    Some(error_len) => {
                        output.extend_from_slice("\u{FFFD}".as_bytes());
                        consumed += error_len;
                    }
                    None => break,
                }
            }
        }
    }

    if consumed > 0 {
        pending.drain(..consumed);
    }
    if finish && !pending.is_empty() {
        output.extend_from_slice(String::from_utf8_lossy(pending).as_bytes());
        pending.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_byte_tokenizer(
        vocab: &[(&str, u32)],
        merges: &[(&str, &str)],
        nfc: bool,
        splitters: &[&str],
        bpe_mode: BpeMode,
        added_tokens: Vec<AddedToken>,
    ) -> Tokenizer {
        let vocab_map: HashMap<String, u32> =
            vocab.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        let merge_pairs: Vec<(String, String)> = merges
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect();
        let mut bpe =
            bpe::BpeTable::from_vocab_and_merges(&vocab_map, &merge_pairs, false).unwrap();
        for at in &added_tokens {
            bpe.insert_added(at.content.as_bytes().to_vec(), at.id)
                .unwrap();
        }
        Tokenizer::new(
            bpe,
            Pipeline::ByteLevelRegex {
                nfc,
                splitters: splitters
                    .iter()
                    .map(|pattern| fancy_regex::Regex::new(pattern).unwrap())
                    .collect(),
                bpe_mode,
            },
            added_tokens,
        )
        .unwrap()
    }

    fn make_byte_fallback_tokenizer(vocab: &[(&str, u32)], merges: &[(&str, &str)]) -> Tokenizer {
        let vocab_map: HashMap<String, u32> =
            vocab.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        let merge_pairs: Vec<(String, String)> = merges
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect();
        let bpe = bpe::BpeTable::from_vocab_and_merges(&vocab_map, &merge_pairs, false).unwrap();
        Tokenizer::new(
            bpe,
            Pipeline::ByteFallbackReplace {
                normalizer_from: " ".into(),
                normalizer_to: "▁".into(),
                unk_token_id: None,
            },
            vec![],
        )
        .unwrap()
    }

    #[test]
    fn byte_level_encode_decode_roundtrip() {
        let tok = make_byte_tokenizer(
            &[("h", 0), ("i", 1), ("hi", 2)],
            &[("h", "i")],
            false,
            &[],
            BpeMode::Merge,
            vec![],
        );
        let ids = tok.encode("hi");
        assert_eq!(ids, vec![2]);
        assert_eq!(tok.decode(&ids, false), "hi");
    }

    #[test]
    fn byte_level_nfc_normalization() {
        let tok = make_byte_tokenizer(
            &[("\u{00E9}", 0)],
            &[],
            true,
            &[],
            BpeMode::PreferWholeToken,
            vec![],
        );
        assert_eq!(tok.encode("\u{0065}\u{0301}"), vec![0]);
    }

    #[test]
    fn byte_level_regex_sequence_splits_in_order() {
        let tok = make_byte_tokenizer(
            &[("a", 0), ("1", 1), ("2", 2), ("b", 3), ("12", 4)],
            &[("1", "2")],
            false,
            &[r"\d+", r"[a-z]+"],
            BpeMode::Merge,
            vec![],
        );
        assert_eq!(tok.encode("a12b"), vec![0, 4, 3]);
        assert!(tok.get_split_regex().is_empty());
        assert_eq!(tok.split_regexes(), vec![r"\d+", r"[a-z]+"]);
    }

    #[test]
    fn byte_fallback_replace_roundtrip() {
        let tok = make_byte_fallback_tokenizer(
            &[
                ("a", 0),
                ("▁", 1),
                ("b", 2),
                ("a▁", 3),
                ("a▁b", 4),
                ("<0xE5>", 5),
                ("<0x8F>", 6),
                ("<0xAB>", 7),
                ("<0xFF>", 8),
            ],
            &[("a", "▁"), ("a▁", "b")],
        );
        let ids = tok.encode("a b");
        assert_eq!(ids, vec![4]);
        assert_eq!(tok.decode(&ids, false), "a b");
        assert_eq!(tok.decode(&[5, 6, 7], false), "叫");
        assert_eq!(tok.decode(&[5, 6], false), "��");
    }

    #[test]
    fn added_tokens_and_skip_special() {
        let tok = make_byte_tokenizer(
            &[("h", 0), ("i", 1), ("hi", 2)],
            &[("h", "i")],
            false,
            &[],
            BpeMode::Merge,
            vec![
                AddedToken {
                    id: 3,
                    content: "<s>".into(),
                    special: true,
                },
                AddedToken {
                    id: 4,
                    content: "</s>".into(),
                    special: true,
                },
            ],
        );
        assert_eq!(tok.encode("<s>hi</s>"), vec![3, 2, 4]);
        assert_eq!(tok.decode(&[3, 2], true), "hi");
        assert_eq!(tok.decode(&[3, 2], false), "<s>hi");
    }

    #[test]
    fn encode_into_matches_encode() {
        let tok = make_byte_tokenizer(
            &[("h", 0), ("i", 1), ("hi", 2)],
            &[("h", "i")],
            false,
            &[],
            BpeMode::Merge,
            vec![],
        );
        let mut output = vec![99];
        tok.encode_into("hi", &mut output);
        assert_eq!(output, vec![99, 2]);
    }

    #[test]
    fn incremental_byte_level_holds_partial_utf8() {
        let bpe = BpeTable::from_decoder_map(HashMap::from([
            (0, vec![0xC3]),
            (1, vec![0xA9]),
            (2, b"!".to_vec()),
        ]))
        .unwrap();
        let tokenizer = Arc::new(
            Tokenizer::new(
                bpe,
                Pipeline::ByteLevelRegex {
                    nfc: false,
                    splitters: Vec::new(),
                    bpe_mode: BpeMode::PreferWholeToken,
                },
                vec![],
            )
            .unwrap(),
        );
        let mut decoder = tokenizer.decoder(false);
        assert_eq!(decoder.feed(&[0]), "");
        assert_eq!(decoder.feed(&[1]), "é");
        assert_eq!(decoder.feed(&[2]), "!");
        assert_eq!(decoder.finish(), "");
    }

    #[test]
    fn incremental_byte_fallback_matches_full_decode() {
        let tokenizer = Arc::new(make_byte_fallback_tokenizer(
            &[
                ("a", 0),
                ("▁", 1),
                ("b", 2),
                ("a▁", 3),
                ("a▁b", 4),
                ("<0xE5>", 5),
                ("<0x8F>", 6),
                ("<0xAB>", 7),
                ("<0xFF>", 8),
            ],
            &[("a", "▁"), ("a▁", "b")],
        ));
        let mut decoder = tokenizer.decoder(false);
        assert_eq!(decoder.feed(&[5]), "");
        assert_eq!(decoder.feed(&[6]), "");
        assert_eq!(decoder.feed(&[7]), "");
        assert_eq!(decoder.finish(), "叫");

        decoder.reset();
        assert_eq!(decoder.feed(&[5, 6]), "");
        assert_eq!(decoder.finish(), "��");
        assert_eq!(tokenizer.decode(&[5, 6], false), "��");

        decoder.reset();
        assert_eq!(decoder.feed(&[5, 6, 7, 0]), "叫a");
        assert_eq!(decoder.finish(), "");

        decoder.reset();
        assert_eq!(decoder.feed(&[5, 6, 7, 8]), "");
        assert_eq!(decoder.finish(), "����");
        assert_eq!(tokenizer.decode(&[5, 6, 7, 8], false), "����");
    }
}

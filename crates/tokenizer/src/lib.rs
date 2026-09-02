//! Top-level Tokenizer struct.
//!
//! Provides `encode` / `decode` over a BPE vocabulary. Construction happens
//! via [`loader`] (Hugging Face `tokenizer.json` and native tiktoken formats).
//!
//! External formats compile into one of a small number of supported modern
//! pipelines. Unsupported legacy combinations are rejected at load time.

mod bpe;
pub mod canonical;
pub mod contract;
pub mod loader;

use std::borrow::Cow;
use std::sync::{Arc, OnceLock};

use aho_corasick::AhoCorasick;
use anyhow::Context;
use smallvec::SmallVec;
use unicode_normalization::{IsNormalized, UnicodeNormalization, is_nfc_quick};

use bpe::BpeTable;

/// Representation of a token added on top of the base vocabulary.
#[derive(Debug, Clone)]
pub struct AddedToken {
    pub id: u32,
    pub content: String,
    pub special: bool,
    /// The whitespace run BEFORE this token is consumed by the match
    /// (Hugging Face `lstrip`). The consumed whitespace is not encoded.
    pub lstrip: bool,
    /// The whitespace run AFTER this token is consumed by the match
    /// (Hugging Face `rstrip`). The consumed whitespace is not encoded.
    pub rstrip: bool,
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

/// How the sentencepiece dummy prefix (`normalizer_to` marker) is injected
/// while encoding a `ByteFallbackReplace` pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum DummyPrefix {
    /// No marker is prepended (Gemma).
    #[default]
    None,
    /// Prepend one marker to every encoded text segment (legacy Llama-2/Phi-3).
    EverySegment,
    /// Prepend one marker only to the segment starting the input, and only
    /// if it isn't already there after space replacement (Mistral).
    FirstSegment,
}

/// One pre-tokenizer split stage: the regex whose matches become pieces, and
/// whether the text between matches survives as pieces too (`keep_gaps`).
#[derive(Debug)]
pub(crate) struct Splitter {
    pub(crate) regex: fancy_regex::Regex,
    pub(crate) keep_gaps: bool,
}

/// Compiled tokenizer behavior for the modern model families supported by Pie.
#[derive(Debug)]
pub(crate) enum Pipeline {
    /// Optional NFC, one or more regex splitters, then byte-level BPE.
    ByteLevelRegex {
        nfc: bool,
        splitters: Vec<Splitter>,
        bpe_mode: BpeMode,
    },
    /// Sentencepiece-style space marker normalization with byte fallback on
    /// decode.
    ByteFallbackReplace {
        normalizer_from: String,
        normalizer_to: String,
        unk_token_id: Option<u32>,
        dummy_prefix: DummyPrefix,
        /// Remove one leading `normalizer_from` from the decoded stream,
        /// undoing the dummy prefix.
        strip_decoder_marker: bool,
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

/// A BPE tokenizer for the modern model profiles supported by Pie.
///
/// # Example
///
/// ```no_run
/// use std::path::Path;
/// use tokenizer::Tokenizer;
///
/// let tokenizer = Tokenizer::from_file(Path::new("tokenizer.json")).unwrap();
/// let ids = tokenizer.encode("Hello, world!");
/// let text = tokenizer.decode(&ids, false);
/// ```
pub struct Tokenizer {
    bpe: BpeTable,

    pipeline: Pipeline,

    /// Sorted for binary_search.
    special_token_ids: Vec<u32>,
    added_token_matcher: Option<AhoCorasick>,
    /// Parallel to the matcher's patterns.
    added_tokens: Vec<AddedToken>,

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
    /// Whether the decoder `Strip` still must inspect this stream's first
    /// output; disarms after the first non-empty chunk.
    strip_armed: bool,
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

        let added_token_matcher = if !added_tokens.is_empty() {
            let patterns: Vec<&str> = added_tokens.iter().map(|t| t.content.as_str()).collect();
            Some(
                AhoCorasick::builder()
                    .match_kind(aho_corasick::MatchKind::LeftmostLongest)
                    .build(&patterns)
                    .context("building added-token matcher")?,
            )
        } else {
            None
        };

        Ok(Tokenizer {
            bpe,
            pipeline,
            special_token_ids,
            added_token_matcher,
            added_tokens,
            grammar: OnceLock::new(),
        })
    }

    /// Build a minimal tokenizer from raw token strings.
    ///
    /// Each string becomes a token with ID = its index; the 256 single-byte
    /// tokens are appended after them so stated ids keep their values, and
    /// so BPE has base symbols to fall back on for any string that isn't
    /// exactly one whole-word entry. No normalization, no special tokens.
    pub fn from_vocab(vocab: &[String]) -> Self {
        use std::collections::HashMap;
        let mut map: HashMap<u32, Vec<u8>> = vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (i as u32, s.as_bytes().to_vec()))
            .collect();
        let stated = u32::try_from(vocab.len()).expect("fixture vocabularies are small");
        for byte in 0u16..256 {
            map.insert(stated + u32::from(byte), vec![byte as u8]);
        }
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
    /// A rank file carries no split regex; the sibling `tokenizer_config.json`
    /// must identify a known tiktoken tokenizer, or loading is rejected.
    pub fn from_tiktoken_file(path: &std::path::Path) -> anyhow::Result<Self> {
        loader::tiktoken::from_file(path)
    }

    /// Create an incremental decoder sharing this tokenizer.
    pub fn decoder(self: &Arc<Self>, skip_special: bool) -> TokenizerDecoder {
        TokenizerDecoder {
            tokenizer: self.clone(),
            skip_special,
            strip_armed: self.strips_decoder_marker(),
            pending_utf8: Vec::new(),
            fallback_run: Vec::new(),
        }
    }

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
            self.encode_text(text, true, ids);
            return;
        };

        let mut last_end = 0;
        for matched in matcher.find_iter(text) {
            let token = &self.added_tokens[matched.pattern().as_usize()];
            let mut segment_end = matched.start().max(last_end);
            if token.lstrip {
                let segment = &text[last_end..segment_end];
                segment_end = last_end + segment.trim_end_matches(char::is_whitespace).len();
            }
            if segment_end > last_end {
                // A segment starts the input iff it begins at byte 0.
                self.encode_text(&text[last_end..segment_end], last_end == 0, ids);
            }
            ids.push(token.id);
            last_end = last_end.max(matched.end());
            if token.rstrip {
                let rest = &text[last_end..];
                last_end += rest.len() - rest.trim_start_matches(char::is_whitespace).len();
            }
        }
        if last_end < text.len() {
            self.encode_text(&text[last_end..], last_end == 0, ids);
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

    fn encode_text(&self, text: &str, starts_input: bool, ids: &mut Vec<u32>) {
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
                dummy_prefix,
                ..
            } => {
                let text = if text.contains(normalizer_from.as_str()) {
                    Cow::Owned(text.replace(normalizer_from.as_str(), normalizer_to.as_str()))
                } else {
                    Cow::Borrowed(text)
                };
                let prepend = match dummy_prefix {
                    DummyPrefix::None => false,
                    DummyPrefix::EverySegment => true,
                    DummyPrefix::FirstSegment => {
                        starts_input && !text.starts_with(normalizer_to.as_str())
                    }
                };
                if prepend {
                    let mut prefixed = String::with_capacity(normalizer_to.len() + text.len());
                    prefixed.push_str(normalizer_to);
                    prefixed.push_str(&text);
                    self.encode_piece(&prefixed, ids);
                } else {
                    self.encode_piece(&text, ids);
                }
            }
            Pipeline::RawChar => self.encode_piece(text, ids),
        }
    }

    fn split_regex_sequence(&self, text: &str, splitters: &[Splitter], ids: &mut Vec<u32>) {
        if let [splitter] = splitters {
            let output_start = ids.len();
            let mut last_end = 0;
            for result in splitter.regex.find_iter(text) {
                let Ok(matched) = result else {
                    ids.truncate(output_start);
                    self.encode_piece(text, ids);
                    return;
                };
                if splitter.keep_gaps && matched.start() > last_end {
                    self.encode_piece(&text[last_end..matched.start()], ids);
                }
                if matched.start() < matched.end() {
                    self.encode_piece(matched.as_str(), ids);
                }
                last_end = matched.end();
            }
            if splitter.keep_gaps && last_end < text.len() {
                self.encode_piece(&text[last_end..], ids);
            }
            return;
        }

        let mut pieces: SmallVec<[&str; 32]> = SmallVec::new();
        pieces.push(text);

        for splitter in splitters {
            let mut next: SmallVec<[&str; 32]> = SmallVec::new();
            for piece in pieces.iter().copied() {
                let mut last_end = 0;
                for result in splitter.regex.find_iter(piece) {
                    let Ok(matched) = result else {
                        // Preserve input rather than a partially encoded result.
                        self.encode_piece(text, ids);
                        return;
                    };
                    if splitter.keep_gaps && matched.start() > last_end {
                        next.push(&piece[last_end..matched.start()]);
                    }
                    if matched.start() < matched.end() {
                        next.push(matched.as_str());
                    }
                    last_end = matched.end();
                }
                if splitter.keep_gaps && last_end < piece.len() {
                    next.push(&piece[last_end..]);
                }
            }
            pieces = next;
        }

        for piece in pieces {
            self.encode_piece(piece, ids);
        }
    }

    fn strips_decoder_marker(&self) -> bool {
        matches!(
            self.pipeline,
            Pipeline::ByteFallbackReplace {
                strip_decoder_marker: true,
                ..
            }
        )
    }

    /// Decode token IDs back into text.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        match &self.pipeline {
            Pipeline::ByteFallbackReplace {
                normalizer_from,
                normalizer_to,
                strip_decoder_marker,
                ..
            } => {
                let strip_prefix = strip_decoder_marker.then_some(normalizer_from.as_bytes());
                self.decode_byte_fallback(
                    ids,
                    skip_special,
                    normalizer_to.as_bytes(),
                    normalizer_from.as_bytes(),
                    strip_prefix,
                )
            }
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
        strip_prefix: Option<&[u8]>,
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
        if let Some(prefix) = strip_prefix
            && output.starts_with(prefix)
        {
            output.drain(..prefix.len());
        }
        bytes_to_string(output)
    }

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

    /// Every token id whose bytes begin with `prefix`, ascending. An empty
    /// prefix matches the whole vocabulary.
    pub fn ids_with_prefix(&self, prefix: &[u8]) -> Vec<u32> {
        (0..self.bpe.vocab_size() as u32)
            .filter(|id| {
                self.bpe
                    .id_to_bytes(*id)
                    .is_some_and(|bytes| bytes.starts_with(prefix))
            })
            .collect()
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
                splitters[0].regex.as_str().to_string()
            }
            _ => String::new(),
        }
    }

    /// Regex splitters in the order they are applied.
    pub fn split_regexes(&self) -> Vec<&str> {
        match &self.pipeline {
            Pipeline::ByteLevelRegex { splitters, .. } => {
                splitters.iter().map(|s| s.regex.as_str()).collect()
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
        self.apply_stream_strip(&mut output);
        bytes_to_string(output)
    }

    /// Applies the decoder's leading-marker strip once, to the stream's
    /// first non-empty chunk.
    fn apply_stream_strip(&mut self, output: &mut Vec<u8>) {
        if !self.strip_armed || output.is_empty() {
            return;
        }
        self.strip_armed = false;
        if let Pipeline::ByteFallbackReplace {
            normalizer_from, ..
        } = &self.tokenizer.pipeline
            && output.starts_with(normalizer_from.as_bytes())
        {
            output.drain(..normalizer_from.len());
        }
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
        self.strip_armed = self.tokenizer.strips_decoder_marker();
    }
}

/// Implement FromStr so `"json".parse::<Tokenizer>()` works idiomatically.
impl std::str::FromStr for Tokenizer {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        loader::huggingface::from_slice(s.as_bytes())
    }
}

/// Build trie subtree ranges for a sorted vocabulary: `result[i]` is the
/// index of the first entry whose string does not start with `sorted[i]`'s.
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


//! Top-level Tokenizer struct.
//!
//! Provides `encode` / `decode` over a BPE vocabulary. Construction happens
//! via [`hf_loader`](crate::hf_loader) (HuggingFace `tokenizer.json` format).
//!
//! The encode path is a **two-phase pipeline**:
//!   1. **Normalize**: `norm_steps` transforms input text (NFC, prepend space, replace).
//!   2. **Split + encode**: `split_step` splits text into pieces and BPE-encodes each.
//!
//! The decode path is a simple ordered list of `DecodeStep` transforms.

pub mod bpe;
pub mod hf_loader;

use std::borrow::Cow;

use aho_corasick::AhoCorasick;
use unicode_normalization::{is_nfc_quick, IsNormalized, UnicodeNormalization};

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

// ---------------------------------------------------------------------------
// Segment (internal: added-token split result)
// ---------------------------------------------------------------------------

/// A segment produced by splitting input text on added/special tokens.
enum Segment<'a> {
    Text(&'a str),
    AddedToken(u32),
}

// ---------------------------------------------------------------------------
// Vocabulary type
// ---------------------------------------------------------------------------

/// How the BPE vocabulary is encoded and how atoms are defined.
///
/// This single enum replaces the previous combination of `AtomMode`,
/// `byte_fallback: bool`, and `raw_byte_keys: bool` — only three valid
/// combinations of those flags exist, and this enum makes that explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VocabType {
    /// Byte-level BPE (GPT-2 style).
    ///
    /// Atoms are bytes. Vocab keys are GPT-2 unicode → raw bytes.
    /// Used by GPT-2, LLaMA-3, Qwen3, DeepSeek.
    ByteLevel,
    /// Char-level BPE with byte fallback (SentencePiece style).
    ///
    /// Atoms are Unicode chars. Unknown chars → `<0xHH>` byte tokens.
    /// Used by Mistral, Llama-2.
    ByteFallback,
    /// Char-level BPE without byte fallback.
    ///
    /// Atoms are Unicode chars. Unknown chars → UNK token.
    /// Used by Gemma-2, Gemma-3.
    CharLevel,
}

impl VocabType {
    /// Whether atoms are individual bytes (vs. Unicode chars).
    #[inline]
    pub fn is_byte_level(self) -> bool {
        matches!(self, VocabType::ByteLevel)
    }

    /// Whether unknown atoms are encoded as `<0xHH>` byte-fallback tokens.
    #[inline]
    pub fn byte_fallback(self) -> bool {
        matches!(self, VocabType::ByteFallback)
    }
}

// ---------------------------------------------------------------------------
// Encode pipeline: normalize + split
// ---------------------------------------------------------------------------

/// A text normalization step (phase 1 of encoding).
///
/// Normalizers transform text in-place without splitting.
/// Applied in order before the split step.
#[derive(Debug)]
pub enum NormStep {
    /// NFC unicode normalization.
    Nfc,
    /// Prepend a space before text if it doesn't start with one.
    PrependSpace,
    /// String replacement in text.
    Replace { from: String, to: String },
}

/// The text splitting step (phase 2 of encoding).
///
/// Exactly one split step runs per encode call. It consumes the normalized
/// text, splits it into pieces, and BPE-encodes each piece.
#[derive(Debug)]
pub enum SplitStep {
    /// Remap bytes to GPT-2 unicode, split via regex, encode each match.
    Gpt2RegexSplit(fancy_regex::Regex),
    /// Split on regex matches (HF `Split` with `behavior: Isolated`).
    /// Each match AND each gap become separate pieces.
    RegexSplitIsolated(fancy_regex::Regex),
    /// Metaspace: replace spaces with replacement char, then split + prepend.
    MetaspaceSplit {
        replacement: char,
        replacement_str: String,
        /// None = never, Some(false) = first only, Some(true) = all pieces.
        prepend: Option<bool>,
        split: bool,
    },
    /// No splitting — encode the whole string as one piece.
    None,
}

// ---------------------------------------------------------------------------
// Decode step
// ---------------------------------------------------------------------------

/// A single decode transform step.
#[derive(Debug, Clone)]
pub enum DecodeStep {
    /// Replace `<0xNN>` tokens with raw bytes, coalescing runs.
    ByteFallback,
    /// Replace pattern bytes → replacement bytes (per token).
    Replace { pattern: Vec<u8>, content: Vec<u8> },
    /// Strip `content` from start/end of each token N times.
    Strip { content: Vec<u8>, start: usize, stop: usize },
    /// Strip `content` from start of the FIRST token only (Metaspace add_prefix_space).
    StripFirst { content: Vec<u8> },
    /// Fuse all tokens into a single buffer.
    Fuse,
}

// ---------------------------------------------------------------------------
// Encode scratch buffers (reused across pieces within one encode call)
// ---------------------------------------------------------------------------

/// Reusable buffers for the encode hot path.
///
/// Created once per `encode()` call and passed through to avoid
/// per-piece allocations (especially for the GPT-2 byte remap).
#[derive(Default)]
struct EncodeScratch {
    /// GPT-2 byte→unicode remapped string.
    remap_buf: String,
    /// Char byte offsets within `remap_buf`.
    char_offsets: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

/// A minimalistic BPE tokenizer compatible with HuggingFace `tokenizer.json`.
///
/// # Example
///
/// ```no_run
/// use std::path::Path;
/// use tokenizer_mini::Tokenizer;
///
/// let tokenizer = Tokenizer::from_file(Path::new("tokenizer.json")).unwrap();
/// let ids = tokenizer.encode("Hello, world!");
/// let text = tokenizer.decode(&ids, false);
/// ```
pub struct Tokenizer {
    // Core BPE
    bpe: BpeTable,

    // Pipeline
    vocab_type: VocabType,
    norm_steps: Vec<NormStep>,
    split_step: SplitStep,
    decode_steps: Vec<DecodeStep>,

    // BPE model options
    unk_token_id: Option<u32>,

    // Added / special tokens (sorted for binary_search)
    special_token_ids: Vec<u32>,
    added_token_matcher: Option<AhoCorasick>,
    added_token_ids: Vec<u32>,

    // Decoded & sorted vocabulary with trie subtree ranges
    // (built once at construction for grammar-guided generation)
    decoded_vocab: Vec<String>,
    sorted_vocab: Vec<(u32, String)>,
    trie_subtree_end: Vec<usize>,
}

impl Tokenizer {
    /// Construct a `Tokenizer` from its components.
    ///
    /// Prefer [`from_file`] for loading from HuggingFace JSON.
    /// This constructor is for use by format-specific loaders
    /// (e.g. [`hf_loader`]).
    pub fn new(
        bpe: BpeTable,
        vocab_type: VocabType,
        norm_steps: Vec<NormStep>,
        split_step: SplitStep,
        decode_steps: Vec<DecodeStep>,
        unk_token_id: Option<u32>,
        added_tokens: Vec<AddedToken>,
    ) -> Self {
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
                .ok();
            let ids: Vec<u32> = added_tokens.iter().map(|t| t.id).collect();
            (matcher, ids)
        } else {
            (None, vec![])
        };

        // Build decoded vocabulary + sorted vocab + trie subtree ranges.
        let vocab_size = bpe.vocab_size();
        let mut decoded_vocab = Vec::with_capacity(vocab_size);
        let mut sorted_vocab = Vec::with_capacity(vocab_size);

        for id in 0..vocab_size as u32 {
            let decoded = match bpe.id_to_bytes(id) {
                Some(raw) => String::from_utf8_lossy(raw).into_owned(),
                None => String::new(),
            };
            if !decoded.is_empty() {
                sorted_vocab.push((id, decoded.clone()));
            }
            decoded_vocab.push(decoded);
        }

        sorted_vocab.sort_by(|a, b| a.1.cmp(&b.1));
        let trie_subtree_end = build_subtree_ranges(&sorted_vocab);

        Tokenizer {
            bpe,
            vocab_type,
            norm_steps,
            split_step,
            decode_steps,
            unk_token_id,
            special_token_ids,
            added_token_matcher,
            added_token_ids,
            decoded_vocab,
            sorted_vocab,
            trie_subtree_end,
        }
    }

    // -----------------------------------------------------------------------
    // Loading (convenience delegates to hf_loader)
    // -----------------------------------------------------------------------

    /// Load a tokenizer from an HF `tokenizer.json` file.
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        hf_loader::from_file(path)
    }

    // -----------------------------------------------------------------------
    // Encoding
    // -----------------------------------------------------------------------

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        let segments = self.split_added_tokens(text);
        let mut ids = Vec::new();
        let mut scratch = EncodeScratch::default();
        let mut is_first_text = true;

        for segment in segments {
            match segment {
                Segment::Text(t) => {
                    self.encode_text(t, is_first_text, &mut ids, &mut scratch);
                    is_first_text = false;
                }
                Segment::AddedToken(id) => ids.push(id),
            }
        }

        ids
    }

    /// Encode a single piece of text using the appropriate BPE atom mode.
    #[inline]
    fn encode_piece(&self, piece: &str, ids: &mut Vec<u32>) {
        if piece.is_empty() {
            return;
        }
        let bf = self.vocab_type.byte_fallback();
        if self.vocab_type.is_byte_level() {
            bpe::bpe_encode_bytes(
                piece.as_bytes(),
                &self.bpe,
                bf,
                self.unk_token_id,
                ids,
            );
        } else {
            bpe::bpe_encode_chars(
                piece,
                &self.bpe,
                bf,
                self.unk_token_id,
                ids,
            );
        }
    }

    /// Encode a single text segment (no added tokens).
    ///
    /// Phase 1: normalize the text (Cow — no allocation if unchanged).
    /// Phase 2: split into pieces and BPE-encode each.
    fn encode_text(
        &self,
        text: &str,
        is_first: bool,
        ids: &mut Vec<u32>,
        scratch: &mut EncodeScratch,
    ) {
        let text = self.normalize(text);
        self.split_and_encode(&text, is_first, ids, scratch);
    }

    /// Phase 1: apply normalization steps.
    ///
    /// Returns `Cow::Borrowed` when no step modifies the text (common for
    /// ByteLevel models), avoiding allocation entirely.
    fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str> {
        let mut text = Cow::Borrowed(text);
        for step in &self.norm_steps {
            match step {
                NormStep::Nfc => {
                    if is_nfc_quick(text.chars()) != IsNormalized::Yes {
                        text = Cow::Owned(text.nfc().collect());
                    }
                }
                NormStep::PrependSpace => {
                    if !text.starts_with(' ') {
                        text.to_mut().insert(0, ' ');
                    }
                }
                NormStep::Replace { from, to } => {
                    if text.contains(from.as_str()) {
                        text = Cow::Owned(text.replace(from.as_str(), to.as_str()));
                    }
                }
            }
        }
        text
    }

    /// Phase 2: split text into pieces and BPE-encode each.
    fn split_and_encode(
        &self,
        text: &str,
        is_first: bool,
        ids: &mut Vec<u32>,
        scratch: &mut EncodeScratch,
    ) {
        match &self.split_step {
            SplitStep::Gpt2RegexSplit(regex) => {
                let bytes = text.as_bytes();
                let lut = bpe::bytes_to_unicode();

                // Remap bytes → GPT-2 unicode (reuse scratch buffer).
                scratch.remap_buf.clear();
                scratch.remap_buf.extend(bytes.iter().map(|&b| lut[b as usize]));

                // Build char→byte offset table (reuse scratch buffer).
                scratch.char_offsets.clear();
                scratch.char_offsets.extend(
                    scratch.remap_buf.char_indices().map(|(i, _)| i),
                );
                let n_chars = scratch.char_offsets.len();

                for m in regex.find_iter(&scratch.remap_buf) {
                    let m = match m {
                        Ok(m) => m,
                        Err(_) => continue,
                    };
                    let cs = scratch.char_offsets.partition_point(|&o| o < m.start());
                    let ce = scratch.char_offsets.partition_point(|&o| o < m.end());
                    if cs >= ce || ce > n_chars {
                        continue;
                    }
                    bpe::bpe_encode_bytes(
                        &bytes[cs..ce],
                        &self.bpe,
                        self.vocab_type.byte_fallback(),
                        self.unk_token_id,
                        ids,
                    );
                }
            }
            SplitStep::RegexSplitIsolated(regex) => {
                let mut last_end = 0;
                for m in regex.find_iter(text) {
                    let m = match m {
                        Ok(m) => m,
                        Err(_) => continue,
                    };
                    if m.start() > last_end {
                        self.encode_piece(&text[last_end..m.start()], ids);
                    }
                    self.encode_piece(m.as_str(), ids);
                    last_end = m.end();
                }
                if last_end < text.len() {
                    self.encode_piece(&text[last_end..], ids);
                }
            }
            SplitStep::MetaspaceSplit {
                replacement,
                replacement_str,
                prepend,
                split,
            } => {
                let replaced = text.replace(' ', replacement_str);

                let mut pieces: Vec<String> = if *split {
                    let mut result = Vec::new();
                    for (i, s) in replaced.split(*replacement).enumerate() {
                        if s.is_empty() && i == 0 {
                            continue;
                        }
                        let mut piece = String::with_capacity(
                            replacement_str.len() + s.len(),
                        );
                        piece.push_str(replacement_str);
                        piece.push_str(s);
                        result.push(piece);
                    }
                    if result.is_empty() {
                        vec![replacement_str.clone()]
                    } else {
                        result
                    }
                } else {
                    vec![replaced]
                };

                match prepend {
                    Some(true) => {
                        for p in pieces.iter_mut() {
                            if !p.starts_with(*replacement) {
                                p.insert_str(0, replacement_str);
                            }
                        }
                    }
                    Some(false) => {
                        if is_first {
                            if let Some(first) = pieces.first_mut() {
                                if !first.starts_with(*replacement) {
                                    first.insert_str(0, replacement_str);
                                }
                            }
                        }
                    }
                    None => {}
                }

                for piece in &pieces {
                    self.encode_piece(piece, ids);
                }
            }
            SplitStep::None => {
                self.encode_piece(text, ids);
            }
        }
    }

    /// Split text on added/special tokens using the Aho-Corasick matcher.
    fn split_added_tokens<'a>(&self, text: &'a str) -> Vec<Segment<'a>> {
        let matcher = match &self.added_token_matcher {
            Some(m) => m,
            None => return vec![Segment::Text(text)],
        };

        let mut segments = Vec::new();
        let mut last_end = 0;

        for mat in matcher.find_iter(text) {
            let start = mat.start();
            let end = mat.end();
            if start > last_end {
                segments.push(Segment::Text(&text[last_end..start]));
            }
            let id = self.added_token_ids[mat.pattern().as_usize()];
            segments.push(Segment::AddedToken(id));
            last_end = end;
        }

        if last_end < text.len() {
            segments.push(Segment::Text(&text[last_end..]));
        }

        if segments.is_empty() {
            segments.push(Segment::Text(text));
        }

        segments
    }

    // -----------------------------------------------------------------------
    // Decoding
    // -----------------------------------------------------------------------

    /// Decode token IDs back into text.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        if self.decode_steps.is_empty() {
            // ByteLevel: raw-byte keyed — just concatenate.
            let mut buf = Vec::with_capacity(ids.len() * 4);
            for &id in ids {
                if skip_special
                    && self.special_token_ids.binary_search(&id).is_ok()
                {
                    continue;
                }
                if let Some(bytes) = self.bpe.id_to_bytes(id) {
                    buf.extend_from_slice(bytes);
                }
            }
            return String::from_utf8_lossy(&buf).into_owned();
        }

        // CharLevel: collect per-token raw bytes, apply decode steps.
        let mut tokens: Vec<Vec<u8>> = Vec::with_capacity(ids.len());
        for &id in ids {
            if skip_special
                && self.special_token_ids.binary_search(&id).is_ok()
            {
                continue;
            }
            if let Some(raw) = self.bpe.id_to_bytes(id) {
                tokens.push(raw.to_vec());
            }
        }

        // Apply decode steps in order.
        for step in &self.decode_steps {
            match step {
                DecodeStep::ByteFallback => {
                    let old = std::mem::take(&mut tokens);
                    let mut byte_buf: Vec<u8> = Vec::new();
                    for token in old {
                        if token.len() == 6
                            && token.starts_with(b"<0x")
                            && token[5] == b'>'
                        {
                            if let Ok(hex) = std::str::from_utf8(&token[3..5]) {
                                if let Ok(byte) = u8::from_str_radix(hex, 16) {
                                    byte_buf.push(byte);
                                    continue;
                                }
                            }
                        }
                        if !byte_buf.is_empty() {
                            tokens.push(std::mem::take(&mut byte_buf));
                        }
                        tokens.push(token);
                    }
                    if !byte_buf.is_empty() {
                        tokens.push(byte_buf);
                    }
                }
                DecodeStep::Replace { pattern, content } => {
                    for token in tokens.iter_mut() {
                        *token = replace_bytes_owned(
                            std::mem::take(token),
                            pattern,
                            content,
                        );
                    }
                }
                DecodeStep::Strip { content, start, stop } => {
                    for token in tokens.iter_mut() {
                        for _ in 0..*start {
                            if token.starts_with(content.as_slice()) {
                                token.drain(..content.len());
                            } else {
                                break;
                            }
                        }
                        for _ in 0..*stop {
                            if token.ends_with(content.as_slice()) {
                                token.truncate(token.len() - content.len());
                            } else {
                                break;
                            }
                        }
                    }
                }
                DecodeStep::StripFirst { content } => {
                    if let Some(first) = tokens.first_mut() {
                        if first.starts_with(content.as_slice()) {
                            first.drain(..content.len());
                        }
                    }
                }
                DecodeStep::Fuse => {
                    let fused: Vec<u8> = tokens.drain(..).flatten().collect();
                    if !fused.is_empty() {
                        tokens.push(fused);
                    }
                }
            }
        }

        // Concat remaining tokens.
        let mut buf = Vec::with_capacity(tokens.iter().map(|t| t.len()).sum());
        for token in &tokens {
            buf.extend_from_slice(token);
        }
        String::from_utf8_lossy(&buf).into_owned()
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

    /// Get the split regex pattern (from the split step).
    ///
    /// Returns an empty string if no regex-based split step is configured.
    pub fn get_split_regex(&self) -> String {
        match &self.split_step {
            SplitStep::Gpt2RegexSplit(re)
            | SplitStep::RegexSplitIsolated(re) => re.as_str().to_string(),
            _ => String::new(),
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

    /// The decoded vocabulary: `decoded_vocab[token_id]` = decoded string.
    ///
    /// Empty strings indicate special/unmapped tokens.
    pub fn decoded_vocab(&self) -> &[String] {
        &self.decoded_vocab
    }

    /// Vocabulary sorted lexicographically by decoded string: `(token_id, decoded_string)`.
    ///
    /// Only contains tokens with non-empty decoded strings (special tokens excluded).
    pub fn sorted_vocab(&self) -> &[(u32, String)] {
        &self.sorted_vocab
    }

    /// Trie subtree ranges over [`sorted_vocab`](Self::sorted_vocab).
    ///
    /// `trie_subtree_end[i]` is the index of the first entry in `sorted_vocab`
    /// whose decoded string does **not** start with `sorted_vocab[i].1`.
    /// Enables O(1) subtree skipping during token mask generation.
    pub fn trie_subtree_end(&self) -> &[usize] {
        &self.trie_subtree_end
    }
}

/// Implement FromStr so `"json".parse::<Tokenizer>()` works idiomatically.
impl std::str::FromStr for Tokenizer {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        hf_loader::from_slice(s.as_bytes())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build trie subtree ranges for a sorted vocabulary.
///
/// For each entry `sorted[i]`, returns an array where `result[i]` is the
/// index of the first entry whose string does **not** start with `sorted[i].1`.
fn build_subtree_ranges(sorted: &[(u32, String)]) -> Vec<usize> {
    let n = sorted.len();
    let mut ranges = vec![n; n];
    let mut stack: Vec<(usize, &str)> = Vec::new();

    for i in 0..n {
        let s = sorted[i].1.as_str();
        while let Some(&(idx, prefix)) = stack.last() {
            if s.starts_with(prefix) {
                break;
            }
            ranges[idx] = i;
            stack.pop();
        }
        stack.push((i, s));
    }

    while let Some((idx, _)) = stack.pop() {
        ranges[idx] = n;
    }

    ranges
}

/// Replace all occurrences of `needle` with `replacement` in `haystack`.
///
/// Uses SIMD-accelerated `memchr::memmem` for searching. Returns the original
/// `haystack` untouched (no allocation) if `needle` is not found.
fn replace_bytes_owned(haystack: Vec<u8>, needle: &[u8], replacement: &[u8]) -> Vec<u8> {
    if needle.is_empty() {
        return haystack;
    }

    let finder = memchr::memmem::Finder::new(needle);
    let mut iter = finder.find_iter(&haystack).peekable();

    // Fast path: no match → return original without allocation.
    if iter.peek().is_none() {
        return haystack;
    }

    let mut result = Vec::with_capacity(haystack.len());
    let mut start = 0;
    for pos in iter {
        result.extend_from_slice(&haystack[start..pos]);
        result.extend_from_slice(replacement);
        start = pos + needle.len();
    }
    result.extend_from_slice(&haystack[start..]);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_byte_tokenizer(
        vocab: &[(&str, u32)],
        merges: &[(&str, &str)],
        norm_steps: Vec<NormStep>,
        split_step: SplitStep,
        decode_steps: Vec<DecodeStep>,
        added_tokens: Vec<AddedToken>,
    ) -> Tokenizer {
        let vocab_map: HashMap<String, u32> =
            vocab.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        let merge_pairs: Vec<(String, String)> = merges
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect();
        let mut bpe = bpe::BpeTable::from_vocab_and_merges(
            &vocab_map, &merge_pairs, "", false,
        );
        for at in &added_tokens {
            bpe.insert(at.content.as_bytes().to_vec(), at.id);
        }
        Tokenizer::new(bpe, VocabType::ByteLevel, norm_steps, split_step, decode_steps, None, added_tokens)
    }

    fn make_char_tokenizer(
        vocab: &[(&str, u32)],
        merges: &[(&str, &str)],
        norm_steps: Vec<NormStep>,
        split_step: SplitStep,
        decode_steps: Vec<DecodeStep>,
    ) -> Tokenizer {
        let vocab_map: HashMap<String, u32> =
            vocab.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        let merge_pairs: Vec<(String, String)> = merges
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect();
        let bpe = bpe::BpeTable::from_vocab_and_merges(
            &vocab_map, &merge_pairs, "", false,
        );
        Tokenizer::new(bpe, VocabType::CharLevel, norm_steps, split_step, decode_steps, None, vec![])
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let tok = make_byte_tokenizer(
            &[("h", 0), ("i", 1), ("hi", 2)],
            &[("h", "i")],
            vec![], SplitStep::None, vec![], vec![],
        );
        let ids = tok.encode("hi");
        assert_eq!(ids, vec![2]);
        assert_eq!(tok.decode(&ids, false), "hi");
    }

    #[test]
    fn test_normstep_nfc() {
        let tok = make_byte_tokenizer(
            &[("\u{00E9}", 0)], // NFC form
            &[],
            vec![NormStep::Nfc], SplitStep::None,
            vec![], vec![],
        );
        assert_eq!(tok.encode("\u{0065}\u{0301}"), vec![0]); // NFD → NFC
    }

    #[test]
    fn test_normstep_prepend_space() {
        let tok = make_byte_tokenizer(
            &[(" ", 0), ("h", 1), ("i", 2)],
            &[],
            vec![NormStep::PrependSpace], SplitStep::None,
            vec![], vec![],
        );
        assert_eq!(tok.encode("hi"), vec![0, 1, 2]);
        assert_eq!(tok.encode(" hi"), vec![0, 1, 2]); // no double-prepend
    }

    #[test]
    fn test_normstep_replace() {
        let tok = make_byte_tokenizer(
            &[("X", 0), ("!", 1)],
            &[],
            vec![NormStep::Replace { from: "hello".into(), to: "X".into() }],
            SplitStep::None,
            vec![], vec![],
        );
        assert_eq!(tok.encode("hello!"), vec![0, 1]);
    }

    #[test]
    fn test_splitstep_metaspace_split() {
        let tok = make_char_tokenizer(
            &[("▁", 0), ("▁a", 1), ("▁b", 2), ("a", 3), ("b", 4)],
            &[("▁", "a"), ("▁", "b")],
            vec![],
            SplitStep::MetaspaceSplit {
                replacement: '▁',
                replacement_str: "▁".into(),
                prepend: Some(true),
                split: true,
            },
            vec![],
        );
        assert_eq!(tok.encode("a b"), vec![1, 2]); // ▁a, ▁b
    }

    #[test]
    fn test_splitstep_metaspace_no_split() {
        let tok = make_char_tokenizer(
            &[("a", 0), ("▁", 1), ("b", 2), ("a▁b", 3)],
            &[("a", "▁"), ("a▁", "b")],
            vec![],
            SplitStep::MetaspaceSplit {
                replacement: '▁',
                replacement_str: "▁".into(),
                prepend: None,
                split: false,
            },
            vec![],
        );
        assert_eq!(tok.encode("a b"), vec![3]); // a▁b as single token
    }

    #[test]
    fn test_splitstep_regex_split() {
        let re = fancy_regex::Regex::new(r"\d+").unwrap();
        let tok = make_byte_tokenizer(
            &[("a", 0), ("1", 1), ("2", 2), ("b", 3), ("12", 4)],
            &[("1", "2")],
            vec![], SplitStep::RegexSplitIsolated(re),
            vec![], vec![],
        );
        assert_eq!(tok.encode("a12b"), vec![0, 4, 3]);
    }

    #[test]
    fn test_decode_metaspace_pipeline() {
        let tok = make_char_tokenizer(
            &[("▁hi", 0)],
            &[],
            vec![],
            SplitStep::MetaspaceSplit {
                replacement: '▁',
                replacement_str: "▁".into(),
                prepend: Some(true),
                split: false,
            },
            vec![
                DecodeStep::Replace { pattern: "▁".as_bytes().to_vec(), content: b" ".to_vec() },
                DecodeStep::StripFirst { content: b" ".to_vec() },
            ],
        );
        let ids = tok.encode("hi");
        assert_eq!(tok.decode(&ids, false), "hi"); // ▁→space then strip leading
    }

    #[test]
    fn test_decode_byte_fallback() {
        let tok = make_byte_tokenizer(
            &[("a", 0), ("<0xE5>", 1), ("<0x8F>", 2), ("<0xAB>", 3)],
            &[],
            vec![], SplitStep::None,
            vec![DecodeStep::ByteFallback],
            vec![],
        );
        assert_eq!(tok.decode(&[1, 2, 3], false), "叫"); // E5 8F AB → 叫
    }

    #[test]
    fn test_added_tokens_and_skip_special() {
        let tok = make_byte_tokenizer(
            &[("h", 0), ("i", 1), ("hi", 2)],
            &[("h", "i")],
            vec![], SplitStep::None, vec![],
            vec![
                AddedToken { id: 100, content: "<s>".into(), special: true },
                AddedToken { id: 101, content: "</s>".into(), special: true },
            ],
        );
        assert_eq!(tok.encode("<s>hi</s>"), vec![100, 2, 101]);
        assert_eq!(tok.decode(&[100, 2], true), "hi");     // skip special
        assert_eq!(tok.decode(&[100, 2], false), "<s>hi");  // include special
    }
}

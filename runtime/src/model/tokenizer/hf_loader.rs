//! HuggingFace `tokenizer.json` loader.
//!
//! # HF Tokenizer Architecture (reference: `tokenizers/tokenizers/src/`)
//!
//! An HF `tokenizer.json` encodes a four-stage pipeline:
//!
//! ```text
//! ┌──────────────┐    ┌────────────────┐    ┌───────┐    ┌──────────┐
//! │  Normalizer   │ → │ Pre-tokenizer  │ →  │ Model │ →  │ Decoder  │
//! │  (text→text)  │   │ (text→pieces)  │    │ (BPE) │    │ (ids→text│
//! └──────────────┘    └────────────────┘    └───────┘    └──────────┘
//! ```
//!
//! ## How Pie tokenizer maps this
//!
//! Pie tokenizer collapses the first two stages into `Vec<NormStep>` (text
//! normalization) + `SplitStep` (splitting strategy) and the last stage into
//! `Vec<DecodeStep>`.  The `Model` is always BPE, stored as a [`BpeTable`].
//!
//! There are two atom modes:
//!
//! | Mode            | HF source type     | How vocab is keyed        |
//! |-----------------|--------------------|---------------------------|
//! | `VocabType::ByteLevel`    | `ByteLevel`        | Raw bytes (GPT-2 remap)   |
//! | `VocabType::ByteFallback` | `Metaspace` + others| UTF-8 strings            |
//! | `VocabType::CharLevel`    | `Metaspace` + others| UTF-8 strings            |
//!
//! ## JSON structure overview
//!
//! ```text
//! {
//!   "model": { "type":"BPE", "vocab":{...}, "merges":[...], ... },
//!   "normalizer": { "type":"Sequence"|"NFC"|"Replace"|..., ... },
//!   "pre_tokenizer": { "type":"Sequence"|"ByteLevel"|"Metaspace"|"Split"|..., ... },
//!   "decoder": { "type":"Sequence"|"ByteFallback"|"Metaspace"|"Replace"|..., ... },
//!   "added_tokens": [{ "id":0, "content":"<s>", "special":true }, ...]
//! }
//! ```
//!
//! All of `normalizer`, `pre_tokenizer`, and `decoder` can be wrapped in a
//! `"type":"Sequence"` node with a children array keyed by `"normalizers"`,
//! `"pretokenizers"`, or `"decoders"` respectively.
//! [`flatten_sequence`] handles this recursively.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use super::bpe::BpeTable;
use super::{
    AddedToken, DecodeStep, NormStep, SplitStep, Tokenizer, VocabType,
};

/// The classic GPT-2 regex.
///
/// HF ref: `pre_tokenizers/byte_level.rs` static `RE`.
/// Note: the HF version has an extra `(?!\S)` lookahead on the trailing `\s+`
/// group. We omit it — the difference is negligible for most inputs and
/// `fancy_regex` handles `\s+` without the lookahead.
const GPT2_REGEX: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+";

// ---------------------------------------------------------------------------
// HuggingFace JSON schema (typed deserialization)
// ---------------------------------------------------------------------------
//
// We type-deserialize the stable top-level fields (`model`, `added_tokens`)
// and leave the polymorphic subtrees (`normalizer`, `pre_tokenizer`,
// `decoder`) as `serde_json::Value` so that [`build_pipeline`] and
// [`build_decode_steps`] can walk them generically.

/// Root of `tokenizer.json`.
#[derive(Deserialize)]
struct HfTokenizerJson {
    /// The BPE model: vocab, merges, and model-level options.
    model: HfModel,

    /// Normalizer subtree (NFC, Replace, Sequence, etc).
    /// HF ref: `normalizers/mod.rs` `NormalizerWrapper` enum.
    #[serde(default)]
    normalizer: Option<serde_json::Value>,

    /// Pre-tokenizer subtree (ByteLevel, Metaspace, Split, Sequence, etc).
    /// HF ref: `pre_tokenizers/mod.rs` `PreTokenizerWrapper` enum.
    #[serde(default)]
    pre_tokenizer: Option<serde_json::Value>,

    /// Decoder subtree (ByteFallback, Metaspace, Replace, Strip, Fuse, etc).
    /// HF ref: `decoders/mod.rs` `DecoderWrapper` enum.
    #[serde(default)]
    decoder: Option<serde_json::Value>,

    /// Special / added tokens with explicit IDs.
    #[serde(default)]
    added_tokens: Vec<HfAddedToken>,
}

/// `model` section of `tokenizer.json`.
///
/// HF ref: `models/bpe/serialization.rs` `BPEVisitor`.
#[derive(Deserialize)]
struct HfModel {
    /// Only `"BPE"` is supported.
    #[serde(rename = "type", default = "default_model_type")]
    model_type: String,

    /// Token string → token ID mapping.
    /// For ByteLevel models the keys are GPT-2-remapped unicode strings;
    /// `BpeTable::from_vocab_and_merges` re-keys them to raw bytes.
    #[serde(default)]
    vocab: HashMap<String, u32>,

    /// Merge rules. Two formats exist:
    ///   - Legacy: `"a b"` (space-separated string)
    ///   - Tuple: `["a", "b"]`
    ///
    /// HF ref: `models/bpe/serialization.rs` `MergeType` enum.
    #[serde(default)]
    merges: Vec<serde_json::Value>,

    /// Whether the model has byte-fallback tokens like `<0xFF>`.
    /// HF ref: `models/bpe/model.rs` `BPE.byte_fallback`.
    #[serde(default)]
    byte_fallback: bool,

    /// Unknown-token string (e.g. `"<unk>"`). Looked up in `vocab` at load
    /// time to get the numeric ID.
    #[serde(default)]
    unk_token: Option<String>,

    /// Prefix prepended to non-initial subwords during BPE (e.g. `"##"` for
    /// BERT-style, rare for modern BPE models).
    /// HF ref: `models/bpe/model.rs` `BPE.continuing_subword_prefix`.
    #[serde(default)]
    continuing_subword_prefix: Option<String>,
}

fn default_model_type() -> String {
    "BPE".to_string()
}

#[derive(Deserialize)]
struct HfAddedToken {
    id: u32,
    content: String,
    #[serde(default)]
    special: bool,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Load a tokenizer from an HF `tokenizer.json` file on disk.
pub fn from_file(path: &Path) -> Result<Tokenizer> {
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    from_slice(&data)
}

/// Load a tokenizer from raw JSON bytes.
pub fn from_slice(json: &[u8]) -> Result<Tokenizer> {
    let hf: HfTokenizerJson = serde_json::from_slice(json).context("parsing tokenizer JSON")?;
    from_hf(hf)
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

fn from_hf(hf: HfTokenizerJson) -> Result<Tokenizer> {
    if hf.model.model_type != "BPE" {
        bail!("unsupported model type: {}", hf.model.model_type);
    }

    // Parse merges (handles both `"a b"` legacy and `["a","b"]` tuple formats).
    let mut merge_pairs = Vec::with_capacity(hf.model.merges.len());
    for merge_val in &hf.model.merges {
        merge_pairs.push(parse_merge_entry(merge_val)?);
    }

    let continuing_subword_prefix = hf
        .model
        .continuing_subword_prefix
        .as_deref()
        .unwrap_or("");

    let pre_tokenizer_val = hf
        .pre_tokenizer
        .as_ref()
        .unwrap_or(&serde_json::Value::Null);
    let normalizer_val = hf.normalizer.as_ref().unwrap_or(&serde_json::Value::Null);
    let decoder_val = hf.decoder.as_ref().unwrap_or(&serde_json::Value::Null);

    // Build encode pipeline (normalizer + pre-tokenizer → NormStep + SplitStep).
    let (is_byte_level, norm_steps, split_step) =
        build_pipeline(pre_tokenizer_val, normalizer_val)?;

    // Build BPE table.
    // For ByteLevel models, vocab keys are GPT-2-remapped unicode (e.g. "Ġ" = 0x20).
    // `raw_byte_keys=true` tells BpeTable to invert the GPT-2 mapping so that
    // internal lookups work on raw bytes directly.
    let mut bpe = BpeTable::from_vocab_and_merges(
        &hf.model.vocab,
        &merge_pairs,
        continuing_subword_prefix,
        is_byte_level, // raw_byte_keys
    );

    let byte_fallback = hf.model.byte_fallback;
    let unk_token_id = hf
        .model
        .unk_token
        .as_ref()
        .and_then(|t| hf.model.vocab.get(t).copied());

    // Derive VocabType from the two independent flags.
    let vocab_type = if is_byte_level {
        VocabType::ByteLevel
    } else if byte_fallback {
        VocabType::ByteFallback
    } else {
        VocabType::CharLevel
    };

    // Build decode pipeline (decoder → Vec<DecodeStep>).
    // ByteLevel models need no decode steps — the raw-byte keying means
    // concatenating decoded byte sequences is sufficient.
    let decode_steps = if is_byte_level {
        vec![]
    } else {
        build_decode_steps(decoder_val)?
    };

    // Insert added tokens into the BPE vocab so they can be encoded/decoded.
    let mut added_tokens = Vec::with_capacity(hf.added_tokens.len());
    for ht in &hf.added_tokens {
        bpe.insert(ht.content.as_bytes().to_vec(), ht.id);
        added_tokens.push(AddedToken {
            id: ht.id,
            content: ht.content.clone(),
            special: ht.special,
        });
    }

    Ok(Tokenizer::new(
        bpe,
        vocab_type,
        norm_steps,
        split_step,
        decode_steps,
        unk_token_id,
        added_tokens,
    ))
}

// ---------------------------------------------------------------------------
// Pipeline builder (normalizer → NormStep, pre-tokenizer → SplitStep)
// ---------------------------------------------------------------------------
//
// HF splits text processing into separate Normalizer and PreTokenizer traits,
// each with many implementors (NFC, Replace, ByteLevel, Metaspace, Split, …).
//
// Pie tokenizer separates these into two phases:
//   - `Vec<NormStep>`: text normalization (NFC, prepend space, replace)
//   - `SplitStep`: exactly one splitting strategy (GPT-2 regex, isolated
//     regex, metaspace, or none)

fn build_pipeline(
    pre_tokenizer: &serde_json::Value,
    normalizer: &serde_json::Value,
) -> Result<(bool, Vec<NormStep>, SplitStep)> {
    let mut norm_steps = Vec::new();
    let mut split_step = SplitStep::None;
    let mut is_byte_level = false;

    // ── Normalizer passes ──────────────────────────────────────────────
    //
    // HF ref: `normalizers/unicode.rs` `NFC` — applies Unicode NFC.
    if has_type_in_tree(normalizer, "NFC") {
        norm_steps.push(NormStep::Nfc);
    }

    // HF ref: `normalizers/replace.rs` `Replace`.
    // Only `String` patterns are supported (sufficient for Gemma, LLaMA,
    // Mistral normalizers).
    for (from, to) in collect_norm_replacements(normalizer) {
        norm_steps.push(NormStep::Replace { from, to });
    }

    // HF ref: `normalizers/prepend.rs` `Prepend`.
    // Llama-2 uses `{"type":"Prepend","prepend":"▁"}` in its normalizer.
    let normalizer_prepend = has_prepend_in_tree(normalizer);

    // ── Pre-tokenizer passes ───────────────────────────────────────────
    //
    // HF pre-tokenizers can be wrapped in `{"type":"Sequence","pretokenizers":[...]}`.
    // We flatten that recursion into a list of leaf nodes.
    let nodes = flatten_sequence(pre_tokenizer, "pretokenizers");

    for node in &nodes {
        match node_type(node) {
            // ── ByteLevel ──────────────────────────────────────────
            //
            // HF ref: `pre_tokenizers/byte_level.rs` `ByteLevel`.
            //
            // Fields:
            //   - `add_prefix_space` → NormStep::PrependSpace
            //   - `use_regex` (default true) → SplitStep::Gpt2RegexSplit
            //     When false, a separate Split node handles splitting.
            "ByteLevel" => {
                is_byte_level = true;

                let add_prefix_space = node
                    .get("add_prefix_space")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                let use_regex = node
                    .get("use_regex")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);

                if add_prefix_space {
                    norm_steps.push(NormStep::PrependSpace);
                }

                if use_regex {
                    let re = fancy_regex::Regex::new(GPT2_REGEX)
                        .context("compiling GPT-2 regex")?;
                    split_step = SplitStep::Gpt2RegexSplit(re);
                }
            }

            // ── Metaspace ──────────────────────────────────────────
            //
            // HF ref: `pre_tokenizers/metaspace.rs` `Metaspace`.
            //
            // Fields:
            //   - `replacement` (char, default '▁')
            //   - `prepend_scheme`: "always" | "first" | "never"
            //   - `add_prefix_space` (legacy): false → force never
            //   - `split` (bool, default true)
            "Metaspace" => {
                let replacement = node
                    .get("replacement")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.chars().next())
                    .unwrap_or('▁');

                let scheme = node
                    .get("prepend_scheme")
                    .and_then(|v| v.as_str())
                    .unwrap_or("always");

                let prepend =
                    if node.get("add_prefix_space").and_then(|v| v.as_bool()) == Some(false) {
                        None
                    } else {
                        match scheme {
                            "first" => Some(false),
                            "never" => None,
                            _ => Some(true),
                        }
                    };

                let split = node
                    .get("split")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);

                split_step = SplitStep::MetaspaceSplit {
                    replacement,
                    replacement_str: replacement.to_string(),
                    prepend,
                    split,
                };
            }

            // ── Split ──────────────────────────────────────────────
            //
            // HF ref: `pre_tokenizers/split.rs` `Split`.
            //
            // Only Regex patterns with Isolated behavior are supported.
            // String-based Split nodes are silently skipped.
            "Split" => {
                if let Some(regex_str) = node
                    .get("pattern")
                    .and_then(|p| p.get("Regex"))
                    .and_then(|r| r.as_str())
                {
                    // Only set if no split step has been set yet (first wins).
                    if matches!(split_step, SplitStep::None) {
                        let re = fancy_regex::Regex::new(regex_str)
                            .with_context(|| format!("compiling Split regex: {regex_str}"))?;
                        split_step = SplitStep::RegexSplitIsolated(re);
                    }
                }
            }

            _ => {} // Unknown pre-tokenizer types are silently skipped.
        }
    }

    // If no splitter was found for a char-level model, add a MetaspaceSplit
    // passthrough (e.g. Gemma2 encodes spaces in the normalizer via Replace,
    // not via Metaspace pre-tokenizer).
    if !is_byte_level && matches!(split_step, SplitStep::None) {
        split_step = SplitStep::MetaspaceSplit {
            replacement: '▁',
            replacement_str: "▁".to_string(),
            prepend: if normalizer_prepend { Some(false) } else { None },
            split: false,
        };
    }

    Ok((is_byte_level, norm_steps, split_step))
}

// ---------------------------------------------------------------------------
// Decode pipeline builder (decoder JSON → DecodeStep)
// ---------------------------------------------------------------------------
//
// HF decoders form a `decode_chain(tokens: Vec<String>) → Vec<String>`
// pipeline. Each decoder transforms the string list.  We flatten Sequence
// wrappers and map each leaf to a `DecodeStep`.
//
// Only used for char-level (non-ByteLevel) models.  ByteLevel models decode
// by simply concatenating raw bytes — no decode chain needed.

fn build_decode_steps(decoder: &serde_json::Value) -> Result<Vec<DecodeStep>> {
    let nodes = flatten_sequence(decoder, "decoders");
    let mut steps = Vec::new();

    for node in &nodes {
        match node_type(node) {
            // HF ref: `decoders/byte_fallback.rs` `ByteFallback`.
            //
            // Converts tokens matching `<0xHH>` to their byte value, then
            // accumulates consecutive byte tokens and attempts UTF-8 decode.
            // Non-byte tokens flush the accumulator.
            //
            // Used by LLaMA-family models with `model.byte_fallback: true`.
            "ByteFallback" => {
                steps.push(DecodeStep::ByteFallback);
            }

            // HF ref: `pre_tokenizers/metaspace.rs` `Metaspace` (impl Decoder).
            //
            // As a decoder, Metaspace does two things:
            //   1. Replace all `replacement` chars → spaces
            //   2. If `add_prefix_space`, strip the leading space from
            //      the first token
            //
            // Pie tokenizer maps this to:
            //   - `DecodeStep::Replace { pattern: "▁", content: " " }`
            //   - `DecodeStep::StripFirst { content: " " }` (conditional)
            "Metaspace" => {
                let replacement = node
                    .get("replacement")
                    .and_then(|v| v.as_str())
                    .unwrap_or("▁");

                steps.push(DecodeStep::Replace {
                    pattern: replacement.as_bytes().to_vec(),
                    content: b" ".to_vec(),
                });

                // HF default is `add_prefix_space: true`.
                let add_prefix_space = node
                    .get("add_prefix_space")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);

                if add_prefix_space {
                    steps.push(DecodeStep::StripFirst {
                        content: b" ".to_vec(),
                    });
                }
            }

            // HF ref: `normalizers/replace.rs` `Replace` (impl Decoder).
            //
            // Replace also implements the Decoder trait:
            //   pattern.String matches → replaced with content.
            //
            // We only handle `String` patterns here.  `Regex` patterns in
            // decoders are rare and unsupported.
            "Replace" => {
                let pattern = node
                    .get("pattern")
                    .and_then(|p| p.get("String").and_then(|s| s.as_str()))
                    .unwrap_or("");
                let content = node
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                if !pattern.is_empty() {
                    steps.push(DecodeStep::Replace {
                        pattern: pattern.as_bytes().to_vec(),
                        content: content.as_bytes().to_vec(),
                    });
                }
            }

            // HF ref: `decoders/strip.rs` `Strip`.
            //
            // Strips up to `start` occurrences of `content` from the
            // beginning and `stop` occurrences from the end of each token.
            "Strip" => {
                let content = node
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or(" ");
                let start = node
                    .get("start")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let stop = node
                    .get("stop")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;

                steps.push(DecodeStep::Strip {
                    content: content.as_bytes().to_vec(),
                    start,
                    stop,
                });
            }

            // HF ref: `decoders/fuse.rs` `Fuse`.
            //
            // Joins all tokens into a single string.  Normally implicit
            // (decode always concatenates at the end), but Fuse exists for
            // decoders that need to run *after* concatenation.
            "Fuse" => {
                steps.push(DecodeStep::Fuse);
            }

            _ => {} // Unknown decoder types are silently skipped.
        }
    }

    Ok(steps)
}

// ---------------------------------------------------------------------------
// JSON tree helpers
// ---------------------------------------------------------------------------

/// Get the `"type"` field of a JSON node as `&str`.
///
/// Nearly every HF component node has `"type": "SomeTypeName"`.
fn node_type(v: &serde_json::Value) -> &str {
    v.get("type").and_then(|v| v.as_str()).unwrap_or("")
}

/// Check if any node in a JSON tree has the given `type` field.
fn has_type_in_tree(value: &serde_json::Value, target: &str) -> bool {
    find_type_in_tree(value, target).is_some()
}

/// Check if a normalizer tree contains a `Prepend` node with `"▁"`.
///
/// Llama-2 uses `{"type":"Prepend","prepend":"▁"}` in the normalizer instead
/// of a Metaspace pre-tokenizer.
fn has_prepend_in_tree(normalizer: &serde_json::Value) -> bool {
    if let Some(node) = find_type_in_tree(normalizer, "Prepend") {
        node.get("prepend")
            .and_then(|v| v.as_str())
            .is_some_and(|s| s.contains('▁'))
    } else {
        false
    }
}

/// Find the first node in a JSON tree with the given `type` field.
///
/// Recursively descends into `Sequence` children via the standard HF
/// array keys: `"pretokenizers"`, `"normalizers"`, `"decoders"`.
fn find_type_in_tree<'a>(
    value: &'a serde_json::Value,
    target: &str,
) -> Option<&'a serde_json::Value> {
    if value.is_null() {
        return None;
    }
    if node_type(value) == target {
        return Some(value);
    }
    for key in &["pretokenizers", "normalizers", "decoders"] {
        if let Some(arr) = value.get(key).and_then(|v| v.as_array()) {
            for item in arr {
                if let Some(found) = find_type_in_tree(item, target) {
                    return Some(found);
                }
            }
        }
    }
    None
}

/// Flatten a `Sequence` node into leaf children.
///
/// HF wraps multiple components of the same kind in:
/// ```json
/// { "type": "Sequence", "<key>": [ child1, child2, ... ] }
/// ```
/// where `<key>` is `"pretokenizers"`, `"decoders"`, or `"normalizers"`.
///
/// This recursively unwraps Sequence wrappers and returns the leaf nodes.
fn flatten_sequence<'a>(value: &'a serde_json::Value, key: &str) -> Vec<&'a serde_json::Value> {
    if value.is_null() {
        return vec![];
    }
    if node_type(value) == "Sequence" {
        value
            .get(key)
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().flat_map(|v| flatten_sequence(v, key)).collect())
            .unwrap_or_default()
    } else {
        vec![value]
    }
}

/// Collect normalizer `Replace` patterns from the normalizer tree.
///
/// HF ref: `normalizers/replace.rs` `Replace`.
///
/// Only `{"pattern": {"String": "..."}}` patterns are extracted.
/// Regex-based normalizer replacements are not supported.
fn collect_norm_replacements(normalizer: &serde_json::Value) -> Vec<(String, String)> {
    flatten_sequence(normalizer, "normalizers")
        .into_iter()
        .filter(|node| node_type(node) == "Replace")
        .filter_map(|node| {
            let pattern = node
                .get("pattern")
                .and_then(|p| p.get("String").and_then(|s| s.as_str()))
                .unwrap_or("");
            let content = node
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if pattern.is_empty() {
                None
            } else {
                Some((pattern.to_string(), content.to_string()))
            }
        })
        .collect()
}

/// Parse a merge entry from the `model.merges` array.
///
/// HF ref: `models/bpe/serialization.rs` `MergeType` enum.
///
/// Two formats exist:
/// - **Legacy** (HF `MergeType::Legacy`): `"token_a token_b"` — a single
///   space-separated string.
/// - **Tuple** (HF `MergeType::Tuple`): `["token_a", "token_b"]` — a
///   two-element JSON array (avoids ambiguity when tokens contain spaces).
fn parse_merge_entry(value: &serde_json::Value) -> Result<(String, String)> {
    if let Some(s) = value.as_str() {
        // Legacy format: "token_a token_b"
        let parts: Vec<&str> = s.splitn(2, ' ').collect();
        if parts.len() != 2 {
            bail!("bad merge entry: {s:?}");
        }
        Ok((parts[0].to_string(), parts[1].to_string()))
    } else if let Some(arr) = value.as_array() {
        // Tuple format: ["token_a", "token_b"]
        if arr.len() != 2 {
            bail!("bad merge tuple: expected 2 elements, got {}", arr.len());
        }
        let a = arr[0]
            .as_str()
            .context("merge tuple element not string")?
            .to_string();
        let b = arr[1]
            .as_str()
            .context("merge tuple element not string")?
            .to_string();
        Ok((a, b))
    } else {
        bail!("unexpected merge format: {value}");
    }
}

//! Strict loader for modern Hugging Face `tokenizer.json` BPE pipelines.
//!
//! Pie supports two structural profiles:
//! - Byte-level BPE with optional NFC and one or more isolated regex splitters
//!   (Qwen 3+, DeepSeek V4, GLM 5.2, Nemotron 3).
//! - String replacement plus byte-fallback BPE (Gemma 4).
//!
//! Other component combinations are rejected instead of being partially
//! interpreted.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

use crate::bpe::BpeTable;
use crate::{AddedToken, BpeMode, Pipeline, Tokenizer};

#[derive(Deserialize)]
struct HfTokenizerJson {
    model: HfModel,
    #[serde(default)]
    normalizer: Option<serde_json::Value>,
    #[serde(default)]
    pre_tokenizer: Option<serde_json::Value>,
    #[serde(default)]
    decoder: Option<serde_json::Value>,
    #[serde(default)]
    added_tokens: Vec<HfAddedToken>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HfModel {
    #[serde(rename = "type")]
    model_type: String,
    vocab: HashMap<String, u32>,
    #[serde(default)]
    merges: Vec<serde_json::Value>,
    #[serde(default)]
    dropout: Option<f64>,
    #[serde(default)]
    unk_token: Option<String>,
    #[serde(default)]
    continuing_subword_prefix: Option<String>,
    #[serde(default)]
    end_of_word_suffix: Option<String>,
    #[serde(default)]
    fuse_unk: bool,
    #[serde(default)]
    byte_fallback: bool,
    #[serde(default)]
    ignore_merges: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HfAddedToken {
    id: u32,
    content: String,
    #[serde(default)]
    special: bool,
    #[serde(default)]
    single_word: bool,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
    #[serde(default)]
    normalized: bool,
}

struct CompiledProfile {
    pipeline: Pipeline,
    raw_byte_keys: bool,
    normalizes_text: bool,
}

/// Load a tokenizer from an HF `tokenizer.json` file.
pub fn from_file(path: &Path) -> Result<Tokenizer> {
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    from_slice(&data)
}

/// Load a tokenizer from raw HF `tokenizer.json` bytes.
pub fn from_slice(json: &[u8]) -> Result<Tokenizer> {
    let hf: HfTokenizerJson = serde_json::from_slice(json).context("parsing tokenizer JSON")?;
    from_hf(hf)
}

fn from_hf(hf: HfTokenizerJson) -> Result<Tokenizer> {
    validate_model_basics(&hf.model)?;
    let profile = compile_profile(&hf)?;

    let merge_pairs = hf
        .model
        .merges
        .iter()
        .map(parse_merge_entry)
        .collect::<Result<Vec<_>>>()?;
    let mut bpe =
        BpeTable::from_vocab_and_merges(&hf.model.vocab, &merge_pairs, profile.raw_byte_keys)?;
    if matches!(&profile.pipeline, Pipeline::ByteFallbackReplace { .. }) {
        ensure!(
            bpe.has_complete_byte_fallback(),
            "byte-fallback profile requires all 256 <0xNN> tokens"
        );
    } else {
        ensure!(
            bpe.has_all_byte_atoms(),
            "byte-level profile requires all 256 byte atoms"
        );
    }

    let mut hf_added_tokens = hf.added_tokens;
    hf_added_tokens.sort_by_key(|token| token.id);
    let mut added_tokens = Vec::with_capacity(hf_added_tokens.len());
    let mut added_ids = HashSet::with_capacity(hf_added_tokens.len());
    let mut added_contents = HashSet::with_capacity(hf_added_tokens.len());
    for token in hf_added_tokens {
        ensure!(
            !token.content.is_empty(),
            "added-token content cannot be empty"
        );
        ensure!(
            added_ids.insert(token.id),
            "duplicate added-token ID {}",
            token.id
        );
        ensure!(
            added_contents.insert(token.content.clone()),
            "duplicate added-token content {:?}",
            token.content
        );
        ensure!(
            !token.single_word && !token.lstrip && !token.rstrip,
            "unsupported added-token boundary flags for {:?}",
            token.content
        );
        ensure!(
            !token.normalized || !profile.normalizes_text,
            "normalized added token {:?} is unsupported with an active normalizer",
            token.content
        );
        bpe.insert_added(token.content.as_bytes().to_vec(), token.id)?;
        added_tokens.push(AddedToken {
            id: token.id,
            content: token.content,
            special: token.special,
        });
    }

    Tokenizer::new(bpe, profile.pipeline, added_tokens)
}

fn validate_model_basics(model: &HfModel) -> Result<()> {
    ensure!(
        model.model_type == "BPE",
        "unsupported model type: {}",
        model.model_type
    );
    ensure!(!model.vocab.is_empty(), "BPE vocabulary is empty");
    ensure!(model.dropout.is_none(), "BPE dropout is unsupported");
    ensure!(
        model
            .continuing_subword_prefix
            .as_deref()
            .is_none_or(str::is_empty),
        "continuing_subword_prefix is unsupported"
    );
    ensure!(
        model
            .end_of_word_suffix
            .as_deref()
            .is_none_or(str::is_empty),
        "end_of_word_suffix is unsupported"
    );
    Ok(())
}

fn compile_profile(hf: &HfTokenizerJson) -> Result<CompiledProfile> {
    let pre_tokenizer = hf.pre_tokenizer.as_ref().context("missing pre_tokenizer")?;
    if is_byte_level_sequence(pre_tokenizer) {
        compile_byte_level_profile(hf)
    } else if node_type(pre_tokenizer) == "Split" {
        compile_byte_fallback_profile(hf)
    } else {
        bail!(
            "unsupported pre_tokenizer profile: {}",
            node_type(pre_tokenizer)
        )
    }
}

fn compile_byte_level_profile(hf: &HfTokenizerJson) -> Result<CompiledProfile> {
    ensure!(
        !hf.model.byte_fallback,
        "byte-level profile cannot enable byte_fallback"
    );
    ensure!(
        !hf.model.fuse_unk,
        "byte-level profile cannot enable fuse_unk"
    );
    ensure!(
        hf.model.unk_token.is_none(),
        "byte-level profile cannot define unk_token"
    );

    let nfc = match hf.normalizer.as_ref() {
        None => false,
        Some(value) if value.is_null() => false,
        Some(value) if node_type(value) == "NFC" => true,
        Some(value) if is_empty_sequence(value, "normalizers") => false,
        Some(value) => bail!("unsupported byte-level normalizer: {}", node_type(value)),
    };

    let pre_tokenizer = hf.pre_tokenizer.as_ref().context("missing pre_tokenizer")?;
    let nodes = sequence_children(pre_tokenizer, "pretokenizers")?;
    ensure!(
        nodes.len() >= 2,
        "byte-level profile requires Split + ByteLevel"
    );
    let byte_level = nodes
        .last()
        .context("missing final ByteLevel pre-tokenizer")?;
    ensure!(
        node_type(byte_level) == "ByteLevel",
        "ByteLevel must be the final pre-tokenizer"
    );
    ensure!(
        byte_level
            .get("use_regex")
            .and_then(serde_json::Value::as_bool)
            == Some(false),
        "ByteLevel.use_regex must be false"
    );
    ensure!(
        byte_level
            .get("add_prefix_space")
            .and_then(serde_json::Value::as_bool)
            == Some(false),
        "ByteLevel.add_prefix_space must be false"
    );

    let mut splitters = Vec::with_capacity(nodes.len() - 1);
    for node in &nodes[..nodes.len() - 1] {
        ensure!(
            node_type(node) == "Split",
            "unsupported pre-tokenizer: {}",
            node_type(node)
        );
        ensure!(
            node.get("behavior").and_then(serde_json::Value::as_str) == Some("Isolated"),
            "only isolated regex splits are supported"
        );
        ensure!(
            node.get("invert").and_then(serde_json::Value::as_bool) == Some(false),
            "inverted regex splits are unsupported"
        );
        let pattern = regex_pattern(node).context("Split must contain a Regex pattern")?;
        splitters.push(
            fancy_regex::Regex::new(pattern)
                .with_context(|| format!("compiling Split regex: {pattern}"))?,
        );
    }

    let decoder = hf.decoder.as_ref().context("missing byte-level decoder")?;
    ensure!(
        node_type(decoder) == "ByteLevel",
        "byte-level profile requires ByteLevel decoder"
    );

    Ok(CompiledProfile {
        pipeline: Pipeline::ByteLevelRegex {
            nfc,
            splitters,
            bpe_mode: if hf.model.ignore_merges {
                BpeMode::PreferWholeToken
            } else {
                BpeMode::Merge
            },
        },
        raw_byte_keys: true,
        normalizes_text: nfc,
    })
}

fn compile_byte_fallback_profile(hf: &HfTokenizerJson) -> Result<CompiledProfile> {
    ensure!(
        hf.model.byte_fallback,
        "byte-fallback profile requires byte_fallback=true"
    );
    ensure!(
        hf.model.fuse_unk,
        "byte-fallback profile requires fuse_unk=true"
    );
    ensure!(
        !hf.model.ignore_merges,
        "byte-fallback profile cannot ignore merges"
    );

    let normalizer = hf
        .normalizer
        .as_ref()
        .context("missing Replace normalizer")?;
    ensure!(
        node_type(normalizer) == "Replace",
        "byte-fallback profile requires Replace normalizer"
    );
    let normalizer_from =
        string_pattern(normalizer).context("Replace normalizer requires a String pattern")?;
    let normalizer_to = normalizer
        .get("content")
        .and_then(serde_json::Value::as_str)
        .context("Replace normalizer requires content")?;
    ensure!(
        !normalizer_from.is_empty() && normalizer_from != normalizer_to,
        "invalid Replace normalizer"
    );

    let pre_tokenizer = hf.pre_tokenizer.as_ref().context("missing pre_tokenizer")?;
    ensure!(
        string_pattern(pre_tokenizer) == Some(normalizer_from),
        "Gemma Split pattern must match the normalized source string"
    );
    ensure!(
        pre_tokenizer
            .get("behavior")
            .and_then(serde_json::Value::as_str)
            == Some("MergedWithPrevious"),
        "Gemma Split behavior must be MergedWithPrevious"
    );
    ensure!(
        pre_tokenizer
            .get("invert")
            .and_then(serde_json::Value::as_bool)
            == Some(false),
        "Gemma Split.invert must be false"
    );

    let decoder = hf
        .decoder
        .as_ref()
        .context("missing byte-fallback decoder")?;
    let decoders = sequence_children(decoder, "decoders")?;
    ensure!(
        decoders.len() == 3,
        "Gemma decoder must contain Replace, ByteFallback, Fuse"
    );
    ensure!(
        node_type(decoders[0]) == "Replace",
        "first Gemma decoder must be Replace"
    );
    ensure!(
        node_type(decoders[1]) == "ByteFallback",
        "second Gemma decoder must be ByteFallback"
    );
    ensure!(
        node_type(decoders[2]) == "Fuse",
        "third Gemma decoder must be Fuse"
    );
    let decoder_pattern =
        string_pattern(decoders[0]).context("decoder Replace requires a String pattern")?;
    let decoder_content = decoders[0]
        .get("content")
        .and_then(serde_json::Value::as_str)
        .context("decoder Replace requires content")?;
    ensure!(
        decoder_pattern == normalizer_to && decoder_content == normalizer_from,
        "decoder Replace must reverse the normalizer"
    );

    let unk_token_id = hf
        .model
        .unk_token
        .as_ref()
        .context("byte-fallback profile requires unk_token")
        .and_then(|token| {
            hf.model
                .vocab
                .get(token)
                .copied()
                .with_context(|| format!("unknown unk_token {token:?}"))
        })?;

    Ok(CompiledProfile {
        pipeline: Pipeline::ByteFallbackReplace {
            normalizer_from: normalizer_from.to_string(),
            normalizer_to: normalizer_to.to_string(),
            unk_token_id: Some(unk_token_id),
        },
        raw_byte_keys: false,
        normalizes_text: true,
    })
}

fn is_byte_level_sequence(value: &serde_json::Value) -> bool {
    node_type(value) == "Sequence"
        && value
            .get("pretokenizers")
            .and_then(serde_json::Value::as_array)
            .and_then(|nodes| nodes.last())
            .is_some_and(|node| node_type(node) == "ByteLevel")
}

fn is_empty_sequence(value: &serde_json::Value, key: &str) -> bool {
    node_type(value) == "Sequence"
        && value
            .get(key)
            .and_then(serde_json::Value::as_array)
            .is_some_and(Vec::is_empty)
}

fn sequence_children<'a>(
    value: &'a serde_json::Value,
    key: &str,
) -> Result<Vec<&'a serde_json::Value>> {
    ensure!(
        node_type(value) == "Sequence",
        "expected Sequence, got {}",
        node_type(value)
    );
    value
        .get(key)
        .and_then(serde_json::Value::as_array)
        .map(|nodes| nodes.iter().collect())
        .with_context(|| format!("Sequence is missing {key}"))
}

fn node_type(value: &serde_json::Value) -> &str {
    value
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
}

fn regex_pattern(value: &serde_json::Value) -> Option<&str> {
    value.get("pattern")?.get("Regex")?.as_str()
}

fn string_pattern(value: &serde_json::Value) -> Option<&str> {
    value.get("pattern")?.get("String")?.as_str()
}

fn parse_merge_entry(value: &serde_json::Value) -> Result<(String, String)> {
    if let Some(legacy) = value.as_str() {
        let (left, right) = legacy
            .split_once(' ')
            .with_context(|| format!("bad merge entry: {legacy:?}"))?;
        Ok((left.to_string(), right.to_string()))
    } else if let Some(tuple) = value.as_array() {
        ensure!(tuple.len() == 2, "bad merge tuple: expected 2 elements");
        let left = tuple[0]
            .as_str()
            .context("merge tuple element is not a string")?;
        let right = tuple[1]
            .as_str()
            .context("merge tuple element is not a string")?;
        Ok((left.to_string(), right.to_string()))
    } else {
        bail!("unexpected merge format: {value}")
    }
}

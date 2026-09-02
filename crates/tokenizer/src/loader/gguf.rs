//! Compiles GGUF's `tokenizer.ggml.*` tables into a `Tokenizer`.

use std::collections::HashMap;

use anyhow::{Result, bail, ensure};

use crate::bpe::BpeTable;
use crate::{AddedToken, BpeMode, Pipeline, Splitter, Tokenizer};

/// ggml's `llama_token_type`, for the values this reads.
const NORMAL: i64 = 1;
const CONTROL: i64 = 3;
const USER_DEFINED: i64 = 4;
const UNUSED: i64 = 5;

/// The tables as they come off the file, borrowed from the mmap.
#[derive(Clone, Copy, Debug)]
pub struct Tables<'a> {
    /// `tokenizer.ggml.model`.
    pub model: &'a str,
    /// `tokenizer.ggml.pre`.
    pub pre: Option<&'a str>,
    /// `tokenizer.ggml.tokens`, in id order.
    pub tokens: &'a [String],
    /// `tokenizer.ggml.token_type`, parallel to `tokens`.
    pub token_types: &'a [i64],
    /// `tokenizer.ggml.merges`, `"left right"`, in rank order.
    pub merges: &'a [String],
}

/// Compiles a GGUF's tokenizer tables.
///
/// Refuses tables that are absent, inconsistent, or name a model or
/// pre-tokenizer this crate does not resolve, rather than approximating.
pub fn from_tables(tables: &Tables) -> Result<Tokenizer> {
    ensure!(
        !tables.tokens.is_empty(),
        "this GGUF carries no tokenizer.ggml.tokens"
    );
    ensure!(
        tables.token_types.len() == tables.tokens.len(),
        "tokenizer.ggml.token_type has {} entries for {} tokens; \
         the two are parallel arrays and a short one cannot be aligned",
        tables.token_types.len(),
        tables.tokens.len()
    );
    ensure!(
        tables.model == "gpt2",
        "unsupported tokenizer.ggml.model `{}`; this reads byte-level BPE \
         (`gpt2`) and refuses the rest rather than compiling a SentencePiece \
         vocabulary into a byte-level pipeline",
        tables.model
    );
    let pipeline = pipeline_named(tables.pre)?;

    // Ids index the embedding matrix and must stay where the file put them;
    // split by type, never renumbered.
    let mut vocab = HashMap::with_capacity(tables.tokens.len());
    let mut added = Vec::new();
    let mut trailing_unused = 0usize;
    for (id, (token, kind)) in tables.tokens.iter().zip(tables.token_types).enumerate() {
        let id = u32::try_from(id).expect("a vocabulary that size cannot be indexed");
        match *kind {
            NORMAL => {
                ensure!(
                    vocab.insert(token.clone(), id).is_none(),
                    "duplicate token {token:?} in tokenizer.ggml.tokens"
                );
            }
            // Padding: an id with no text behind it, dropped.
            UNUSED => trailing_unused += 1,
            CONTROL | USER_DEFINED => added.push(AddedToken {
                id,
                content: token.clone(),
                // CONTROL tokens are model-emitted, not user-typed: `special`.
                special: *kind == CONTROL,
                lstrip: false,
                rstrip: false,
            }),
            other => bail!(
                "token {id} ({token:?}) has ggml token type {other}, which \
                 this build does not read"
            ),
        }
    }

    // `BpeTable` needs ids contiguous from zero; `insert_added` needs each
    // added id appended at the end. Checked rather than assumed.
    ensure!(
        vocab.len() + added.len() + trailing_unused == tables.tokens.len(),
        "token types do not partition the vocabulary"
    );
    let merge_pairs = tables
        .merges
        .iter()
        .map(|merge| {
            let (left, right) = merge
                .split_once(' ')
                .ok_or_else(|| anyhow::anyhow!("merge {merge:?} is not a `left right` pair"))?;
            Ok((left.to_string(), right.to_string()))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut bpe = BpeTable::from_vocab_and_merges(&vocab, &merge_pairs, true)?;
    ensure!(
        bpe.has_all_byte_atoms(),
        "byte-level profile requires all 256 byte atoms"
    );
    added.sort_by_key(|token| token.id);
    for token in &added {
        bpe.insert_added(token.content.as_bytes().to_vec(), token.id)?;
    }

    Tokenizer::new(bpe, pipeline, added)
}

/// llama.cpp's pre-tokenizer name, resolved to the pipeline it stands for.
///
/// Patterns are `tokenizer.json`'s own spelling, not llama.cpp's. An unknown
/// name is refused rather than defaulted, since it can't be guessed.
fn pipeline_named(pre: Option<&str>) -> Result<Pipeline> {
    let pre = pre.unwrap_or_default();
    let (pattern, nfc) = match pre {
        "qwen2" => (
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            true,
        ),
        "" => bail!(
            "this GGUF names no tokenizer.ggml.pre, so its pre-tokenizer is \
             not recoverable -- llama.cpp writes the name of a hard-coded \
             splitter rather than its pattern"
        ),
        other => bail!(
            "unknown tokenizer.ggml.pre `{other}`; the pattern it stands for \
             is not in the file, and defaulting would silently pick a \
             different model's splitter"
        ),
    };
    Ok(Pipeline::ByteLevelRegex {
        nfc,
        splitters: vec![Splitter {
            regex: fancy_regex::Regex::new(pattern)?,
            // matches become pieces; text between them survives.
            keep_gaps: true,
        }],
        bpe_mode: BpeMode::Merge,
    })
}


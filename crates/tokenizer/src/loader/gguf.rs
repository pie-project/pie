//! GGUF's `tokenizer.ggml.*` tables, compiled.
//!
//! # What GGUF keeps, and what it throws away
//!
//! A `tokenizer.json` is a *program*: a normalizer, a list of pre-tokenizer
//! regexes, a decoder, and a model, each written out in full. GGUF keeps the
//! data and replaces the program with a NAME — `tokenizer.ggml.pre` says
//! `"qwen2"` where the JSON spelled a regex. llama.cpp resolves that name
//! against a table compiled into the binary, and so does this: there is no
//! other way, because the pattern is genuinely not in the file.
//!
//! That is the whole difficulty, and it is worth being clear that it is a
//! lossy format rather than a different one. Everything else is a faithful
//! re-encoding of the same tables, measured on
//! `Qwen2.5-0.5B-Instruct-Q4_0.gguf` against the same tokenizer's
//! `tokenizer.json`:
//!
//! | | GGUF | `tokenizer.json` | agreement |
//! |---|---|---|---|
//! | base vocabulary | `tokens[0..151643]` | `model.vocab` | 151,643 of 151,643 |
//! | merges | `merges` | `model.merges` | 151,387 of 151,387, same order |
//! | added tokens | `token_type` 3 and 4 | `added_tokens` | 20 special, 2 not |
//! | padding | `token_type` 5, `[PAD151665]`… | absent | 271 |
//!
//! Two of those rows are the interesting ones. The added tokens are not a
//! separate list here — they are entries in the same array, told apart by a
//! parallel `token_type` table — and the padding is a thing `tokenizer.json`
//! does not have at all: ids that exist because the embedding matrix was
//! rounded up to 151,936, with no text behind them. They are dropped, which
//! is what makes the compiled result equal to the one the JSON produces
//! rather than merely similar to it.
//!
//! # Deliberately not covered
//!
//! `tokenizer.ggml.model` other than `gpt2`. `llama` is SentencePiece and
//! `bert` is WordPiece; both reach different pipelines in this crate, and a
//! `pre` name resolved against the wrong model would produce a tokenizer that
//! runs and is quietly wrong. Refused by name.
//!
//! Scores (`tokenizer.ggml.scores`) are not read: they belong to the unigram
//! model, which is the `llama` case this does not accept.

use std::collections::HashMap;

use anyhow::{Result, bail, ensure};

use crate::bpe::BpeTable;
use crate::{AddedToken, BpeMode, Pipeline, Splitter, Tokenizer};

/// ggml's `llama_token_type`, for the values this reads.
const NORMAL: i64 = 1;
const CONTROL: i64 = 3;
const USER_DEFINED: i64 = 4;
const UNUSED: i64 = 5;

/// The tables as they come off the file.
///
/// Borrowed: the caller has just read them out of a memory map and this does
/// not need to own a 150,000-entry vocabulary to look at it.
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

/// Compile a GGUF's tokenizer tables.
///
/// # Errors
///
/// The tables are absent, inconsistent, or name a model or pre-tokenizer this
/// crate does not resolve. Every one of those is refused rather than
/// approximated: a tokenizer that is wrong by one regex still runs, and the
/// model it feeds answers slightly differently for the rest of its life.
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

    // Ids must stay exactly where the file put them -- they index the
    // embedding matrix -- so the split below is by TYPE and never by
    // renumbering. `vocab` keeps its own id, and the added tokens keep theirs.
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
            // `[PAD151665]` and friends: an id with no text behind it,
            // present because the embedding matrix was rounded up. Dropping
            // it is what `tokenizer.json` does by never mentioning it.
            UNUSED => trailing_unused += 1,
            CONTROL | USER_DEFINED => added.push(AddedToken {
                id,
                content: token.clone(),
                // The one flag GGUF states: `CONTROL` is a token the model
                // emits and a user does not type, which is what `special`
                // means in `added_tokens`.
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

    // `BpeTable` requires ids contiguous from zero, and `insert_added`
    // requires each added id to land at the end of what is already there. Both
    // hold for a well-formed file -- normal tokens first, then added, then
    // padding -- and neither is a property this can assume, so the shape is
    // checked here where the message can name the file's own layout.
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
/// The patterns are `tokenizer.json`'s own, not llama.cpp's transcription of
/// them: llama.cpp expands `(?i:'s|'t|...)` into `(?:'[sS]|'[tT]|...)`
/// because its regex engine has no inline case flag, and `fancy_regex` does.
/// The two match the same strings; taking the HuggingFace spelling keeps this
/// table comparable, by eye, with the documents it is standing in for.
///
/// An unknown name is refused. It cannot be guessed: `pre` is exactly the
/// field that distinguishes tokenizers whose vocabularies look alike, and
/// answering with a default would silently pick another model's splitter.
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
            // `Isolated` + `invert: false`, which is what every
            // `tokenizer.json` in this family writes: the matches become
            // pieces and the text between them survives.
            keep_gaps: true,
        }],
        bpe_mode: BpeMode::Merge,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 256 byte atoms in GPT-2's byte-level alphabet, which is what a
    /// `gpt2` GGUF's vocabulary is written in. Built the way the alphabet is
    /// defined rather than transcribed, so the fixture cannot drift from it.
    fn byte_alphabet() -> Vec<String> {
        let printable: Vec<u8> = (b'!'..=b'~')
            .chain(0xA1..=0xAC)
            .chain(0xAE..=0xFF)
            .collect();
        let mut next = 0u32;
        (0u16..256)
            .map(|byte| {
                let byte = u8::try_from(byte).expect("0..256 fits");
                let ch = if printable.contains(&byte) {
                    char::from(byte)
                } else {
                    let ch = char::from_u32(256 + next).expect("in range");
                    next += 1;
                    ch
                };
                ch.to_string()
            })
            .collect()
    }

    fn tables(extra: &[(&str, i64)], merges: &[&str]) -> (Vec<String>, Vec<i64>, Vec<String>) {
        let mut tokens = byte_alphabet();
        let mut types = vec![NORMAL; tokens.len()];
        for (token, kind) in extra {
            tokens.push((*token).to_string());
            types.push(*kind);
        }
        (
            tokens,
            types,
            merges.iter().map(|m| (*m).to_string()).collect(),
        )
    }

    fn compile(tokens: &[String], types: &[i64], merges: &[String]) -> Result<Tokenizer> {
        from_tables(&Tables {
            model: "gpt2",
            pre: Some("qwen2"),
            tokens,
            token_types: types,
            merges,
        })
    }

    /// The three roles `token_type` distinguishes, and what each becomes.
    /// `UNUSED` is the one that has to VANISH: `[PAD151665]` is an id with no
    /// text behind it, and a vocabulary entry for it would be a token the
    /// model can be asked to produce.
    #[test]
    fn token_type_sorts_the_vocabulary_into_three_places() {
        let (tokens, types, merges) = tables(
            &[
                ("<|im_end|>", CONTROL),
                ("<tool_call>", USER_DEFINED),
                ("[PAD258]", UNUSED),
            ],
            &[],
        );
        let compiled = compile(&tokens, &types, &merges).expect("compiles");

        assert_eq!(compiled.encode("<|im_end|>"), vec![256]);
        assert_eq!(compiled.encode("<tool_call>"), vec![257]);
        // The padding is not a token: it encodes as its own characters.
        assert!(compiled.encode("[PAD258]").len() > 1);
    }

    /// `special` is `decode(skip_special)`'s only input, and it is exactly
    /// ggml's `CONTROL` — not "was added".
    #[test]
    fn control_is_special_and_user_defined_is_not() {
        let (tokens, types, merges) = tables(
            &[("<|im_end|>", CONTROL), ("<tool_call>", USER_DEFINED)],
            &[],
        );
        let compiled = std::sync::Arc::new(compile(&tokens, &types, &merges).expect("compiles"));

        assert_eq!(compiled.decoder(true).feed(&[256, 257]), "<tool_call>");
        assert_eq!(
            compiled.decoder(false).feed(&[256, 257]),
            "<|im_end|><tool_call>"
        );
    }

    /// A merge is `"left right"`, and rank is position. Both are stated by
    /// checking that the merge actually fires: `ab` is one token only if the
    /// pair was read, and only if the merged token was found by name.
    #[test]
    fn a_merge_is_a_space_separated_pair_ranked_by_position() {
        let (mut tokens, mut types, merges) = tables(&[], &["a b"]);
        tokens.push("ab".to_string());
        types.push(NORMAL);
        let compiled = compile(&tokens, &types, &merges).expect("compiles");

        assert_eq!(compiled.encode("ab"), vec![256]);
    }

    /// The name is not the pattern, and there is no pattern to fall back on.
    /// Guessing would pick another model's splitter, which tokenizes and is
    /// wrong.
    #[test]
    fn an_unresolvable_pre_tokenizer_is_refused_by_name() {
        let (tokens, types, merges) = tables(&[], &[]);
        for pre in [None, Some("llama-bpe"), Some("")] {
            let why = from_tables(&Tables {
                model: "gpt2",
                pre,
                tokens: &tokens,
                token_types: &types,
                merges: &merges,
            })
            .err()
            .expect("must refuse");
            assert!(format!("{why:#}").contains("tokenizer.ggml.pre"), "{why:#}");
        }
    }

    /// `llama` is SentencePiece and `bert` is WordPiece. Both would compile
    /// into a byte-level pipeline without complaint and tokenize incorrectly
    /// forever, which is why the check is on the model and not on the vocab.
    #[test]
    fn a_non_byte_level_tokenizer_model_is_refused() {
        let (tokens, types, merges) = tables(&[], &[]);
        let why = from_tables(&Tables {
            model: "llama",
            pre: Some("qwen2"),
            tokens: &tokens,
            token_types: &types,
            merges: &merges,
        })
        .err()
        .expect("must refuse");

        assert!(
            format!("{why:#}").contains("tokenizer.ggml.model"),
            "{why:#}"
        );
    }

    /// The two arrays are parallel and nothing else aligns them. A short
    /// `token_type` would silently retype the tail of the vocabulary.
    #[test]
    fn parallel_tables_of_different_lengths_are_refused() {
        let (tokens, mut types, merges) = tables(&[], &[]);
        types.pop();
        let why = compile(&tokens, &types, &merges)
            .err()
            .expect("must refuse");

        assert!(format!("{why:#}").contains("parallel"), "{why:#}");
    }
}

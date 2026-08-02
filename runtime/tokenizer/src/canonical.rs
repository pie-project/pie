//! `pie.tokenizer/1` — the compiled tokenizer as bytes.
//!
//! Every external tokenizer format compiles into one of a small number of
//! pipelines (see the crate docs). That compilation is the expensive,
//! quirk-ridden part, and today it runs at every serve boot: an ~11 MB
//! `tokenizer.json` is parsed and its hash maps rebuilt, or a tiktoken rank
//! file is read and its merge table *synthesized* by trying every split point
//! of every token. This module is the serialization of what that compilation
//! produces, so it can run once at import instead.
//!
//! # What is stored, and why each piece
//!
//! | Object | Content |
//! |---|---|
//! | `tokenizer/vocab_bytes` | every token's bytes, concatenated in id order |
//! | `tokenizer/vocab_offsets` | `n + 1` little-endian `u32`, so token `i` is `[off[i], off[i+1])` |
//! | `tokenizer/merge_table` | little-endian `u32` quads `(left, right, rank, merged)` |
//! | `tokenizer/byte_fallback` | 256 little-endian `u32`, [`NO_TOKEN`] where absent |
//! | `tokenizer/descriptor` | the pipeline and the added tokens, as JSON |
//!
//! Three of those are less obvious than they look:
//!
//! - **The merge table is quads, not ranks.** [`BpeTable`](crate::bpe::BpeTable)
//!   keys merges by token-id *pair*, and `merged` is not always derivable: the
//!   HF path has `merged = token_to_id[left ++ right]`, but the tiktoken path
//!   synthesizes merges by trying split points and keeping the lowest rank,
//!   which can pick a different token than concatenation would name.
//! - **The byte-fallback table is stored, not recomputed.** It is populated
//!   only on the HF path; the tiktoken path leaves all 256 entries empty even
//!   though the vocabulary contains byte atoms. Recomputing it by scanning for
//!   `<0xNN>` literals would silently *change* how a tiktoken tokenizer
//!   decodes rather than reproduce it.
//! - **The encode map is not stored at all.** `bytes → id` is derived on load
//!   under "lowest id wins", and the writer refuses any table where that fails
//!   to reproduce the original ([`BpeTable::encode_map_is_derivable`]).
//!
//! # Versioning
//!
//! The descriptor carries [`VERSION`], and the reader refuses anything else.
//! This schema freezes internal enums into an on-disk contract, so it evolves
//! by version bump and regeneration from provenance — never by a reader
//! guessing a default for a field an older writer did not emit.
//!
//! # Deliberately not covered
//!
//! No general-purpose tokenizer interchange. This expresses exactly what pie's
//! pipelines support and rejects everything else at import; covering HF's full
//! normalizer/unigram/wordpiece space would re-invent `tokenizer.json`.
//!
//! bos/eos/`chat_template` are also absent, because they are not part of a
//! tokenizer here — they live as per-architecture Rust under
//! `model/src/instruct/`. Serializing them would create fields nothing
//! reads until that code is refactored, and a schema commitment bought for
//! nothing is the one kind this format cannot take back.

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::bpe::BpeTable;
use crate::{AddedToken, BpeMode, DummyPrefix, Pipeline, Splitter, Tokenizer};

/// The schema this module reads and writes.
pub const VERSION: &str = "pie.tokenizer/1";

/// The `byte_fallback` entry meaning "no token for this byte".
///
/// `u32::MAX` cannot collide with a real id: a vocabulary that large would
/// not fit the `u32` ids the tokenizer uses throughout.
pub const NO_TOKEN: u32 = u32::MAX;

/// Object names, relative to the artifact's metadata namespace.
pub const VOCAB_BYTES: &str = "tokenizer/vocab_bytes";
pub const VOCAB_OFFSETS: &str = "tokenizer/vocab_offsets";
pub const MERGE_TABLE: &str = "tokenizer/merge_table";
pub const BYTE_FALLBACK: &str = "tokenizer/byte_fallback";
pub const DESCRIPTOR: &str = "tokenizer/descriptor";

/// Every object of a serialized tokenizer, in the order they are written.
pub const OBJECTS: [&str; 5] = [
    BYTE_FALLBACK,
    DESCRIPTOR,
    MERGE_TABLE,
    VOCAB_BYTES,
    VOCAB_OFFSETS,
];

/// A compiled tokenizer, serialized.
///
/// Field order matches [`OBJECTS`], which is ascending by name — what the
/// artifact writer needs, since canonical `.zt` form requires objects in
/// ascending name order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalTokenizer {
    pub byte_fallback: Vec<u8>,
    pub descriptor: Vec<u8>,
    pub merge_table: Vec<u8>,
    pub vocab_bytes: Vec<u8>,
    pub vocab_offsets: Vec<u8>,
}

impl CanonicalTokenizer {
    /// The objects to write, paired with their names, in ascending name order.
    pub fn objects(&self) -> [(&'static str, &[u8]); 5] {
        [
            (BYTE_FALLBACK, &self.byte_fallback),
            (DESCRIPTOR, &self.descriptor),
            (MERGE_TABLE, &self.merge_table),
            (VOCAB_BYTES, &self.vocab_bytes),
            (VOCAB_OFFSETS, &self.vocab_offsets),
        ]
    }

    /// Collects the objects back from whatever holds them.
    ///
    /// `read` is given each name in [`OBJECTS`] and returns its bytes. A name
    /// it cannot supply is a hard error: a tokenizer missing one of its five
    /// objects is not a tokenizer, and guessing an empty default for a merge
    /// table would produce one that silently tokenizes character by character.
    pub fn from_objects(mut read: impl FnMut(&str) -> Option<Vec<u8>>) -> Result<Self> {
        let mut fetch =
            |name: &str| read(name).with_context(|| format!("the artifact has no '{name}' object"));
        Ok(Self {
            byte_fallback: fetch(BYTE_FALLBACK)?,
            descriptor: fetch(DESCRIPTOR)?,
            merge_table: fetch(MERGE_TABLE)?,
            vocab_bytes: fetch(VOCAB_BYTES)?,
            vocab_offsets: fetch(VOCAB_OFFSETS)?,
        })
    }

    /// Total size of the serialized form.
    pub fn byte_size(&self) -> usize {
        self.objects().iter().map(|(_, bytes)| bytes.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// Descriptor
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
struct Descriptor {
    version: String,
    pipeline: PipelineDescriptor,
    added_tokens: Vec<AddedTokenDescriptor>,
}

/// The pipeline, as data.
///
/// Tagged by `kind` rather than by position so a reader fails on an unknown
/// pipeline with a name in the message. Regexes travel as their pattern
/// strings; splitter *order* is semantic, since they apply as a sequence.
/// One splitter, as data.
///
/// Serializes as a BARE PATTERN STRING whenever the stage keeps the gaps
/// between matches — which is every splitter that existed before the
/// OLMo-2 `Removed+invert` encoding was supported, and still the common
/// case. So an artifact written before this field existed reads back as
/// `Isolated`, which is what those models are, rather than as "the field
/// is absent" — the distinction the format has to get right, because a
/// wrong default here silently drops text between matches.
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
#[serde(untagged)]
enum SplitterDescriptor {
    /// `Split { behavior: Isolated }` — the historical form.
    Isolated(String),
    /// `Split { behavior: Removed, invert: true }` — matches become
    /// pieces and the text between them is dropped.
    Explicit { pattern: String, keep_gaps: bool },
}

impl SplitterDescriptor {
    fn pattern(&self) -> &str {
        match self {
            Self::Isolated(p) => p,
            Self::Explicit { pattern, .. } => pattern,
        }
    }

    fn keep_gaps(&self) -> bool {
        match self {
            Self::Isolated(_) => true,
            Self::Explicit { keep_gaps, .. } => *keep_gaps,
        }
    }
}

/// How the sentencepiece dummy prefix is injected, as data.
///
/// Defaults to `None`, which is what every `ByteFallbackReplace` artifact
/// written before this field existed was: Gemma, the only member of that
/// profile at the time.
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Default, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum DummyPrefixDescriptor {
    #[default]
    None,
    EverySegment,
    FirstSegment,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum PipelineDescriptor {
    ByteLevelRegex {
        nfc: bool,
        splitters: Vec<SplitterDescriptor>,
        prefer_whole_token: bool,
    },
    ByteFallbackReplace {
        normalizer_from: String,
        normalizer_to: String,
        unk_token_id: Option<u32>,
        #[serde(default)]
        dummy_prefix: DummyPrefixDescriptor,
        #[serde(default)]
        strip_decoder_marker: bool,
    },
    RawChar,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
struct AddedTokenDescriptor {
    id: u32,
    content: String,
    special: bool,
    /// Boundary flags, absent from artifacts written before they were
    /// honoured. `false` is what every such token was: the encoder did
    /// not consume whitespace beside a match, so reading them back as
    /// clear reproduces exactly the behaviour that artifact was written
    /// with.
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

impl Tokenizer {
    /// Serializes this tokenizer as `pie.tokenizer/1`.
    ///
    /// Fails rather than writing something lossy: a vocabulary with a hole, or
    /// an encode map that cannot be derived from the decode table, has no
    /// representation here and is refused with the reason.
    pub fn to_canonical(&self) -> Result<CanonicalTokenizer> {
        ensure!(
            self.bpe.encode_map_is_derivable(),
            "this tokenizer's encode map cannot be recovered from its vocabulary, so \
             serializing it would change how it tokenizes; this happens when an added \
             token overwrote an id whose original bytes are still encodable"
        );

        let decode = self.bpe.decode_table();
        let mut vocab_bytes = Vec::new();
        let mut vocab_offsets = Vec::with_capacity((decode.len() + 1) * 4);
        vocab_offsets.extend_from_slice(&0u32.to_le_bytes());
        for (id, bytes) in decode.iter().enumerate() {
            let bytes = bytes.as_deref().with_context(|| {
                format!("token id {id} carries no bytes; a vocabulary with a hole in it has no canonical form")
            })?;
            vocab_bytes.extend_from_slice(bytes);
            let end = u32::try_from(vocab_bytes.len())
                .context("the concatenated vocabulary exceeds 4 GiB")?;
            vocab_offsets.extend_from_slice(&end.to_le_bytes());
        }

        let quads = self.bpe.merge_quads();
        let mut merge_table = Vec::with_capacity(quads.len() * 16);
        for (left, right, rank, merged) in quads {
            for value in [left, right, rank, merged] {
                merge_table.extend_from_slice(&value.to_le_bytes());
            }
        }

        let mut byte_fallback = Vec::with_capacity(256 * 4);
        for entry in self.bpe.byte_fallback_table() {
            byte_fallback.extend_from_slice(&entry.unwrap_or(NO_TOKEN).to_le_bytes());
        }

        let descriptor = Descriptor {
            version: VERSION.to_string(),
            pipeline: describe_pipeline(&self.pipeline),
            added_tokens: self.added_token_descriptors(),
        };
        let descriptor = serde_json::to_vec(&descriptor).context("encoding the descriptor")?;

        Ok(CanonicalTokenizer {
            byte_fallback,
            descriptor,
            merge_table,
            vocab_bytes,
            vocab_offsets,
        })
    }

    /// The added tokens, recovered in the order they were registered.
    ///
    /// `added_tokens` is in registration order (it indexes the matcher's
    /// patterns), so that order is what comes back; `content` comes from
    /// the vocabulary and `special` from the sorted set.
    fn added_token_descriptors(&self) -> Vec<AddedTokenDescriptor> {
        self.added_tokens
            .iter()
            .map(|token| AddedTokenDescriptor {
                id: token.id,
                content: self
                    .bpe
                    .id_to_bytes(token.id)
                    .map(String::from_utf8_lossy)
                    .unwrap_or_default()
                    .into_owned(),
                special: self.special_token_ids.binary_search(&token.id).is_ok(),
                lstrip: token.lstrip,
                rstrip: token.rstrip,
            })
            .collect()
    }
}

fn describe_pipeline(pipeline: &Pipeline) -> PipelineDescriptor {
    match pipeline {
        Pipeline::ByteLevelRegex {
            nfc,
            splitters,
            bpe_mode,
        } => PipelineDescriptor::ByteLevelRegex {
            nfc: *nfc,
            splitters: splitters
                .iter()
                .map(|s| {
                    if s.keep_gaps {
                        SplitterDescriptor::Isolated(s.regex.as_str().to_string())
                    } else {
                        SplitterDescriptor::Explicit {
                            pattern: s.regex.as_str().to_string(),
                            keep_gaps: false,
                        }
                    }
                })
                .collect(),
            prefer_whole_token: *bpe_mode == BpeMode::PreferWholeToken,
        },
        Pipeline::ByteFallbackReplace {
            normalizer_from,
            normalizer_to,
            unk_token_id,
            dummy_prefix,
            strip_decoder_marker,
        } => PipelineDescriptor::ByteFallbackReplace {
            normalizer_from: normalizer_from.clone(),
            normalizer_to: normalizer_to.clone(),
            unk_token_id: *unk_token_id,
            dummy_prefix: match dummy_prefix {
                DummyPrefix::None => DummyPrefixDescriptor::None,
                DummyPrefix::EverySegment => DummyPrefixDescriptor::EverySegment,
                DummyPrefix::FirstSegment => DummyPrefixDescriptor::FirstSegment,
            },
            strip_decoder_marker: *strip_decoder_marker,
        },
        Pipeline::RawChar => PipelineDescriptor::RawChar,
    }
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

impl Tokenizer {
    /// Rebuilds a tokenizer from its `pie.tokenizer/1` objects.
    ///
    /// This is the whole of the load path an artifact needs — no format
    /// sniffing, no sibling files, no merge synthesis.
    pub fn from_canonical(objects: &CanonicalTokenizer) -> Result<Self> {
        let descriptor: Descriptor =
            serde_json::from_slice(&objects.descriptor).context("decoding the descriptor")?;
        ensure!(
            descriptor.version == VERSION,
            "this artifact's tokenizer is {:?}, and this build reads {VERSION}; \
             regenerate it with `pie model import --force`",
            descriptor.version
        );

        let offsets = read_u32s(&objects.vocab_offsets, "vocab_offsets")?;
        ensure!(
            !offsets.is_empty(),
            "vocab_offsets is empty; it holds one more entry than there are tokens"
        );
        ensure!(
            offsets[0] == 0,
            "vocab_offsets starts at {} rather than 0",
            offsets[0]
        );
        let mut vocab = Vec::with_capacity(offsets.len() - 1);
        for (id, window) in offsets.windows(2).enumerate() {
            let (start, end) = (window[0] as usize, window[1] as usize);
            ensure!(
                start <= end && end <= objects.vocab_bytes.len(),
                "token id {id} spans [{start}, {end}) outside a {}-byte vocabulary",
                objects.vocab_bytes.len()
            );
            vocab.push(objects.vocab_bytes[start..end].to_vec());
        }
        ensure!(
            offsets[offsets.len() - 1] as usize == objects.vocab_bytes.len(),
            "vocab_offsets accounts for {} of {} vocabulary bytes",
            offsets[offsets.len() - 1],
            objects.vocab_bytes.len()
        );

        let flat = read_u32s(&objects.merge_table, "merge_table")?;
        ensure!(
            flat.len() % 4 == 0,
            "merge_table holds {} values, which is not a whole number of \
             (left, right, rank, merged) quads",
            flat.len()
        );
        let merges: Vec<(u32, u32, u32, u32)> = flat
            .chunks_exact(4)
            .map(|quad| (quad[0], quad[1], quad[2], quad[3]))
            .collect();

        let fallback = read_u32s(&objects.byte_fallback, "byte_fallback")?;
        ensure!(
            fallback.len() == 256,
            "byte_fallback holds {} entries rather than 256",
            fallback.len()
        );
        let mut byte_fallback_ids = [None; 256];
        for (byte, &id) in fallback.iter().enumerate() {
            byte_fallback_ids[byte] = (id != NO_TOKEN).then_some(id);
        }

        let bpe = BpeTable::from_canonical(vocab, &merges, byte_fallback_ids)?;
        let pipeline = rebuild_pipeline(descriptor.pipeline)?;
        let added_tokens = descriptor
            .added_tokens
            .into_iter()
            .map(|token| AddedToken {
                id: token.id,
                content: token.content,
                special: token.special,
                lstrip: token.lstrip,
                rstrip: token.rstrip,
            })
            .collect();
        Tokenizer::new(bpe, pipeline, added_tokens)
    }
}

fn rebuild_pipeline(descriptor: PipelineDescriptor) -> Result<Pipeline> {
    Ok(match descriptor {
        PipelineDescriptor::ByteLevelRegex {
            nfc,
            splitters,
            prefer_whole_token,
        } => {
            let splitters = splitters
                .iter()
                .map(|d| {
                    let pattern = d.pattern();
                    Ok(Splitter {
                        regex: fancy_regex::Regex::new(pattern)
                            .with_context(|| format!("recompiling splitter {pattern:?}"))?,
                        keep_gaps: d.keep_gaps(),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Pipeline::ByteLevelRegex {
                nfc,
                splitters,
                bpe_mode: if prefer_whole_token {
                    BpeMode::PreferWholeToken
                } else {
                    BpeMode::Merge
                },
            }
        }
        PipelineDescriptor::ByteFallbackReplace {
            normalizer_from,
            normalizer_to,
            unk_token_id,
            dummy_prefix,
            strip_decoder_marker,
        } => Pipeline::ByteFallbackReplace {
            normalizer_from,
            normalizer_to,
            unk_token_id,
            dummy_prefix: match dummy_prefix {
                DummyPrefixDescriptor::None => DummyPrefix::None,
                DummyPrefixDescriptor::EverySegment => DummyPrefix::EverySegment,
                DummyPrefixDescriptor::FirstSegment => DummyPrefix::FirstSegment,
            },
            strip_decoder_marker,
        },
        PipelineDescriptor::RawChar => Pipeline::RawChar,
    })
}

fn read_u32s(bytes: &[u8], what: &str) -> Result<Vec<u32>> {
    if !bytes.len().is_multiple_of(4) {
        bail!("{what} is {} bytes, not a whole number of u32", bytes.len());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Everything the format must reproduce, checked by behavior rather than
    /// by field equality: the round-tripped tokenizer encodes and decodes the
    /// same text to the same ids.
    fn assert_round_trips(original: &Tokenizer, samples: &[&str]) -> Tokenizer {
        let canonical = original.to_canonical().unwrap();
        let rebuilt = match Tokenizer::from_canonical(&canonical) {
            Ok(rebuilt) => rebuilt,
            Err(err) => panic!("round trip failed: {err}"),
        };

        assert_eq!(rebuilt.vocab_size(), original.vocab_size());
        assert_eq!(rebuilt.special_token_ids(), original.special_token_ids());
        for text in samples {
            let ids = original.encode(text);
            assert_eq!(rebuilt.encode(text), ids, "encoding {text:?}");
            assert_eq!(
                rebuilt.decode(&ids, false),
                original.decode(&ids, false),
                "decoding {text:?}"
            );
        }

        // Serialization is deterministic — the merge table is a hash map, and
        // an artifact that changed bytes between runs would defeat the
        // content-addressed identity the format is built on.
        assert_eq!(rebuilt.to_canonical().unwrap(), canonical);
        rebuilt
    }

    #[test]
    fn a_raw_char_tokenizer_round_trips() {
        let vocab: Vec<String> = ["a", "b", "c", "ab", "abc", " "]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let tokenizer = Tokenizer::from_vocab(&vocab);
        assert_round_trips(&tokenizer, &["abc", "a b c", "cab"]);
    }

    #[test]
    fn the_descriptor_names_its_version_and_pipeline() {
        let vocab: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let canonical = Tokenizer::from_vocab(&vocab).to_canonical().unwrap();
        let descriptor: Descriptor = serde_json::from_slice(&canonical.descriptor).unwrap();
        assert_eq!(descriptor.version, VERSION);
        assert_eq!(descriptor.pipeline, PipelineDescriptor::RawChar);
    }

    #[test]
    fn the_offsets_array_holds_one_more_entry_than_there_are_tokens() {
        let vocab: Vec<String> = ["a", "bb", "ccc"].iter().map(|s| s.to_string()).collect();
        let canonical = Tokenizer::from_vocab(&vocab).to_canonical().unwrap();
        let offsets = read_u32s(&canonical.vocab_offsets, "offsets").unwrap();
        assert_eq!(offsets, [0, 1, 3, 6]);
        assert_eq!(canonical.vocab_bytes, b"abbccc");
    }

    #[test]
    fn a_version_this_build_does_not_read_is_refused() {
        let vocab: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let mut canonical = Tokenizer::from_vocab(&vocab).to_canonical().unwrap();
        let text = String::from_utf8(canonical.descriptor.clone()).unwrap();
        canonical.descriptor = text.replace(VERSION, "pie.tokenizer/99").into_bytes();

        match Tokenizer::from_canonical(&canonical) {
            Ok(_) => panic!("a future schema version was accepted"),
            Err(err) => assert!(
                err.to_string().contains("pie.tokenizer/99"),
                "unexpected error: {err}"
            ),
        }
    }

    #[test]
    fn a_truncated_object_is_refused_rather_than_read_short() {
        let vocab: Vec<String> = ["a", "bb"].iter().map(|s| s.to_string()).collect();
        let good = Tokenizer::from_vocab(&vocab).to_canonical().unwrap();

        let mut short_fallback = good.clone();
        short_fallback.byte_fallback.truncate(4);
        assert!(Tokenizer::from_canonical(&short_fallback).is_err());

        let mut ragged_merges = good.clone();
        ragged_merges.merge_table.extend_from_slice(&[0, 0, 0, 0]);
        assert!(Tokenizer::from_canonical(&ragged_merges).is_err());

        let mut short_vocab = good.clone();
        short_vocab.vocab_bytes.pop();
        assert!(Tokenizer::from_canonical(&short_vocab).is_err());

        let mut misaligned = good;
        misaligned.vocab_offsets.push(0);
        assert!(Tokenizer::from_canonical(&misaligned).is_err());
    }

    #[test]
    fn from_objects_names_the_object_it_could_not_find() {
        let err = CanonicalTokenizer::from_objects(|name| (name != MERGE_TABLE).then(Vec::new))
            .unwrap_err();
        assert!(err.to_string().contains(MERGE_TABLE), "{err}");
    }
}

//! `pie.tokenizer/1` against the pipelines it actually has to carry.
//!
//! The unit tests in `canonical.rs` cover the byte layout and the refusals,
//! but they build tokenizers from `from_vocab`, which is the `RawChar` fixture
//! path: no merges, no added tokens, no regex splitters, no byte fallback.
//! Everything that makes the format non-trivial is therefore untested there.
//!
//! These run the round trip against the same fixtures `profiles.rs` checks
//! against HuggingFace's own tokenizer — a byte-level BPE profile (Qwen-style:
//! NFC, regex splitters, an explicit merge table, a special token) and a
//! byte-fallback profile (Gemma-style: a `▁` normalizer, `<0xNN>` atoms, an
//! unk token). The assertion is behavioral: the rebuilt tokenizer must encode
//! and decode every sample in the shared corpus identically, because that —
//! not field equality — is what "the artifact serves the same model" means.

mod common;

use common::{MergeFormat, TEXTS, byte_level_json, gemma_json};
use pie_tokenizer::Tokenizer;
use pie_tokenizer::canonical::{CanonicalTokenizer, VERSION};
use serde_json::json;

fn load(json: &serde_json::Value) -> Tokenizer {
    let bytes = serde_json::to_vec(json).unwrap();
    std::str::from_utf8(&bytes)
        .unwrap()
        .parse::<Tokenizer>()
        .unwrap()
}

/// Round-trips `original` and returns the serialized form.
fn assert_round_trips(original: &Tokenizer, what: &str) -> CanonicalTokenizer {
    let canonical = original
        .to_canonical()
        .unwrap_or_else(|err| panic!("{what}: serializing: {err}"));
    let rebuilt = Tokenizer::from_canonical(&canonical)
        .unwrap_or_else(|err| panic!("{what}: rebuilding: {err}"));

    assert_eq!(
        rebuilt.vocab_size(),
        original.vocab_size(),
        "{what}: vocab size"
    );
    assert_eq!(
        rebuilt.special_token_ids(),
        original.special_token_ids(),
        "{what}: special tokens"
    );

    for text in TEXTS {
        let expected = original.encode(text);
        assert_eq!(rebuilt.encode(text), expected, "{what}: encoding {text:?}");
        // Both `skip_special` settings: the special-id set is rebuilt from the
        // descriptor, and only the `true` path consults it.
        for skip in [false, true] {
            assert_eq!(
                rebuilt.decode(&expected, skip),
                original.decode(&expected, skip),
                "{what}: decoding {text:?} (skip_special={skip})"
            );
        }
    }

    // Byte-for-byte reproducible: the merge table is a hash map, so an
    // unsorted serialization would produce a different artifact — and a
    // different manifest digest — on every run.
    let again = rebuilt.to_canonical().unwrap();
    assert_eq!(
        again, canonical,
        "{what}: serialization is not deterministic"
    );
    canonical
}

/// Qwen-style: NFC, one regex splitter, an explicit merge table, a special
/// token above the base vocabulary.
#[test]
fn a_byte_level_bpe_profile_round_trips() {
    let tokenizer = load(&byte_level_json(
        json!({"type": "NFC"}),
        &[r"\p{N}|[^\p{N}]+"],
        false,
        MergeFormat::Tuple,
        false,
    ));
    let canonical = assert_round_trips(&tokenizer, "byte-level");

    // The merge table is not empty — otherwise this test would pass on a
    // serializer that dropped merges entirely, since a vocabulary of byte
    // atoms alone still decodes.
    assert!(
        !canonical.merge_table.is_empty(),
        "the fixture's merges did not survive"
    );
    assert_eq!(canonical.merge_table.len() % 16, 0);
}

/// Qwen-3.5-style: several splitters applied in sequence. Their *order* is
/// semantic, and a serialization that reordered them would still round-trip
/// the vocabulary while changing how text is split.
#[test]
fn splitter_order_survives() {
    let tokenizer = load(&byte_level_json(
        json!({"type": "NFC"}),
        &[r"\p{N}", r"[\p{L}\p{M}]+", r"[^\p{L}\p{M}\p{N}]+"],
        false,
        MergeFormat::String,
        false,
    ));
    let canonical = assert_round_trips(&tokenizer, "multi-splitter");

    let descriptor: serde_json::Value = serde_json::from_slice(&canonical.descriptor).unwrap();
    assert_eq!(descriptor["version"], VERSION);
    assert_eq!(
        descriptor["pipeline"]["splitters"],
        json!([r"\p{N}", r"[\p{L}\p{M}]+", r"[^\p{L}\p{M}\p{N}]+"])
    );
}

/// `ignore_merges` compiles to the `PreferWholeToken` BPE mode, which changes
/// which ids a known whole piece produces. It is one bool in the descriptor
/// and nothing else records it.
#[test]
fn the_prefer_whole_token_mode_survives() {
    let tokenizer = load(&byte_level_json(
        json!({"type": "NFC"}),
        &[r"\p{N}|[^\p{N}]+"],
        true,
        MergeFormat::Tuple,
        false,
    ));
    let canonical = assert_round_trips(&tokenizer, "ignore-merges");

    let descriptor: serde_json::Value = serde_json::from_slice(&canonical.descriptor).unwrap();
    assert_eq!(descriptor["pipeline"]["prefer_whole_token"], json!(true));
}

/// Gemma-style: a `Replace` normalizer, `<0xNN>` byte atoms, an unk token.
/// This is the profile whose byte-fallback table is populated.
#[test]
fn a_byte_fallback_profile_round_trips() {
    let tokenizer = load(&gemma_json());
    let canonical = assert_round_trips(&tokenizer, "byte-fallback");

    let descriptor: serde_json::Value = serde_json::from_slice(&canonical.descriptor).unwrap();
    assert_eq!(
        descriptor["pipeline"]["kind"],
        json!("byte_fallback_replace")
    );
    assert_eq!(descriptor["pipeline"]["normalizer_from"], json!(" "));
    assert_eq!(descriptor["pipeline"]["normalizer_to"], json!("▁"));

    // Every one of the 256 entries resolves to a real token here, which is
    // what distinguishes this profile from the byte-level one and what the
    // decoder depends on for invalid UTF-8.
    let entries: Vec<u32> = canonical
        .byte_fallback
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    assert_eq!(entries.len(), 256);
    assert!(
        entries.iter().all(|&id| id != u32::MAX),
        "the byte-fallback table did not survive"
    );
}

/// The byte-level profile leaves the byte-fallback table empty, and that
/// asymmetry is a fact about the tokenizer rather than an accident — a
/// serializer that "helpfully" recomputed the table by scanning for `<0xNN>`
/// literals would change how these tokenizers decode.
#[test]
fn an_absent_byte_fallback_table_stays_absent() {
    let tokenizer = load(&byte_level_json(
        json!({"type": "NFC"}),
        &[r"\p{N}|[^\p{N}]+"],
        false,
        MergeFormat::Tuple,
        false,
    ));
    let canonical = tokenizer.to_canonical().unwrap();
    let entries: Vec<u32> = canonical
        .byte_fallback
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    assert!(entries.iter().all(|&id| id == u32::MAX));
}

/// The objects survive a trip through a name-addressed store, which is how
/// they reach an artifact: written under `__meta__/…`, read back by name.
#[test]
fn the_objects_survive_a_name_addressed_round_trip() {
    let tokenizer = load(&gemma_json());
    let canonical = tokenizer.to_canonical().unwrap();

    let store: std::collections::HashMap<String, Vec<u8>> = canonical
        .objects()
        .iter()
        .map(|(name, bytes)| (name.to_string(), bytes.to_vec()))
        .collect();
    assert_eq!(store.len(), 5);

    let recovered = CanonicalTokenizer::from_objects(|name| store.get(name).cloned()).unwrap();
    assert_eq!(recovered, canonical);

    let rebuilt = Tokenizer::from_canonical(&recovered).unwrap();
    assert_eq!(rebuilt.encode("a b"), tokenizer.encode("a b"));
}

/// Objects are handed to the writer in ascending name order, because
/// canonical `.zt` form requires it and the writer trusts its caller.
#[test]
fn objects_are_offered_in_ascending_name_order() {
    let canonical = load(&gemma_json()).to_canonical().unwrap();
    let names: Vec<&str> = canonical.objects().iter().map(|(name, _)| *name).collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    assert_eq!(names, sorted);
}

/// The round trip against a real tokenizer, when one is pointed at.
///
/// The fixtures above are small by construction — a few hundred tokens, a
/// handful of merges. A production vocabulary is ~150k tokens and ~150k
/// merges, which is where the offset arithmetic, the `u32` bounds and the
/// derivability of the encode map are actually under load. Set
/// `PIE_TOKENIZER_FIXTURE` to a `tokenizer.json` or `tiktoken.model` to run
/// it; the test is a no-op otherwise, since the file cannot be vendored.
#[test]
fn a_real_tokenizer_round_trips() {
    let Ok(path) = std::env::var("PIE_TOKENIZER_FIXTURE") else {
        return;
    };
    let path = std::path::PathBuf::from(path);
    let tokenizer = Tokenizer::from_file(&path)
        .unwrap_or_else(|err| panic!("loading {}: {err:#}", path.display()));

    let canonical = assert_round_trips(&tokenizer, &path.display().to_string());
    eprintln!(
        "{}: {} tokens, {} merges, {} KiB serialized ({} KiB vocab, {} KiB merges)",
        path.display(),
        tokenizer.vocab_size(),
        canonical.merge_table.len() / 16,
        canonical.byte_size() / 1024,
        canonical.vocab_bytes.len() / 1024,
        canonical.merge_table.len() / 1024,
    );
}

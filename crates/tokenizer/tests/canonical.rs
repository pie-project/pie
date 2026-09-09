mod common;

use common::{MergeFormat, TEXTS, byte_level_json, gemma_json};
use serde_json::json;
use tokenizer::Tokenizer;
use tokenizer::canonical::{CanonicalTokenizer, VERSION};

fn load(json: &serde_json::Value) -> Tokenizer {
    let bytes = serde_json::to_vec(json).unwrap();
    std::str::from_utf8(&bytes)
        .unwrap()
        .parse::<Tokenizer>()
        .unwrap()
}

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
        for skip in [false, true] {
            assert_eq!(
                rebuilt.decode(&expected, skip),
                original.decode(&expected, skip),
                "{what}: decoding {text:?} (skip_special={skip})"
            );
        }
    }

    let again = rebuilt.to_canonical().unwrap();
    assert_eq!(
        again, canonical,
        "{what}: serialization is not deterministic"
    );
    canonical
}

#[test]
fn canonical_every_case() {
    a_byte_level_bpe_profile_round_trips();
    splitter_order_survives();
    the_prefer_whole_token_mode_survives();
    a_byte_fallback_profile_round_trips();
    an_absent_byte_fallback_table_stays_absent();
    the_objects_survive_a_name_addressed_round_trip();
    objects_are_offered_in_ascending_name_order();
}

fn a_byte_level_bpe_profile_round_trips() {
    let tokenizer = load(&byte_level_json(
        json!({"type": "NFC"}),
        &[r"\p{N}|[^\p{N}]+"],
        false,
        MergeFormat::Tuple,
        false,
    ));
    let canonical = assert_round_trips(&tokenizer, "byte-level");

    assert!(
        !canonical.merge_table.is_empty(),
        "the fixture's merges did not survive"
    );
    assert_eq!(canonical.merge_table.len() % 16, 0);
}

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

    let entries: Vec<u32> = canonical
        .byte_fallback
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    assert_eq!(entries.len(), 256);
    assert!(
        entries.iter().all(|&id| id != u32::MAX),
        "the byte-fallback table did not survive"
    );
}

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
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    assert!(entries.iter().all(|&id| id == u32::MAX));
}

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

fn objects_are_offered_in_ascending_name_order() {
    let canonical = load(&gemma_json()).to_canonical().unwrap();
    let names: Vec<&str> = canonical.objects().iter().map(|(name, _)| *name).collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    assert_eq!(names, sorted);
}

mod common;

use std::sync::Arc;

use common::{MergeFormat, byte_level_json, gemma_json, unigram_json};
use serde_json::json;
use tokenizer::Tokenizer;
use tokenizers::Tokenizer as HfTokenizer;

fn assert_exact(json: &serde_json::Value, texts: &[&str]) {
    let bytes = serde_json::to_vec(json).unwrap();
    let pie = Arc::new(
        std::str::from_utf8(&bytes)
            .unwrap()
            .parse::<Tokenizer>()
            .unwrap(),
    );
    let hf = HfTokenizer::from_bytes(&bytes).unwrap();

    assert_eq!(pie.vocab_size(), hf.get_vocab_size(true));
    for &text in texts {
        let ids = pie.encode(text);
        let hf_ids = hf.encode(text, false).unwrap().get_ids().to_vec();
        assert_eq!(ids, hf_ids, "encoding {text:?}");
        assert_eq!(
            pie.decode(&hf_ids, false),
            hf.decode(&hf_ids, false).unwrap(),
            "HF→Pie decoding {text:?}"
        );
        assert_eq!(
            pie.decode(&ids, false),
            hf.decode(&ids, false).unwrap(),
            "Pie→HF decoding {text:?}"
        );
        assert_eq!(
            pie.decode(&hf_ids, true),
            hf.decode(&hf_ids, true).unwrap(),
            "special-token filtering {text:?}"
        );

        let mut decoder = pie.decoder(false);
        let mut incremental = String::new();
        for token in &hf_ids {
            incremental.push_str(&decoder.feed(std::slice::from_ref(token)));
        }
        incremental.push_str(&decoder.finish());
        assert_eq!(
            incremental,
            pie.decode(&hf_ids, false),
            "incremental decoding {text:?}"
        );
    }
}

#[test]
fn profiles_every_case() {
    qwen3_profile_is_exact();
    qwen36_string_merges_are_exact();
    glm_and_nemotron_ignore_merges_are_exact();
    deepseek_multi_regex_profile_is_exact();
    gemma_byte_fallback_profile_is_exact();
    grammar_bytes_are_decoder_aware_and_exclude_specials();
    a_unigram_walks_and_wraps_exactly_as_hugging_face_does();
    a_unigram_survives_being_baked_and_read_back();
    a_bpe_tokenizer_writes_no_score_plane();
}

fn qwen3_profile_is_exact() {
    let tokenizer = byte_level_json(
        json!({"type": "NFC"}),
        &[r"\p{N}|[^\p{N}]+"],
        false,
        MergeFormat::Tuple,
        false,
    );
    assert_exact(&tokenizer, &["abc", "1234", "a\u{0301}", "<|special|>abc"]);
}

fn qwen36_string_merges_are_exact() {
    let tokenizer = byte_level_json(
        json!({"type": "NFC"}),
        &[r"\p{N}|[\p{L}\p{M}]+|[^\p{L}\p{M}\p{N}]+"],
        false,
        MergeFormat::String,
        false,
    );
    assert_exact(&tokenizer, &["abc", "1234", "a\u{0301}", "Hello!"]);
}

fn glm_and_nemotron_ignore_merges_are_exact() {
    let tokenizer = byte_level_json(
        serde_json::Value::Null,
        &[r"\p{N}{1,3}|[^\p{N}]+"],
        true,
        MergeFormat::Tuple,
        false,
    );
    assert_exact(&tokenizer, &["abc", "1234", "abc<|special|>"]);

    let pie: Tokenizer = tokenizer.to_string().parse().unwrap();
    assert_eq!(pie.encode("abc"), vec![259]);
    assert_eq!(pie.encode("1234"), vec![257, b'4' as u32]);
}

fn deepseek_multi_regex_profile_is_exact() {
    let tokenizer = byte_level_json(
        json!({"type": "Sequence", "normalizers": []}),
        &[
            r"\p{N}{1,3}",
            r"[一-龥぀-ゟ゠-ヿ]+",
            r"[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s+",
        ],
        false,
        MergeFormat::Tuple,
        true,
    );
    assert_exact(&tokenizer, &["abc", "1234", "你好1234", "<|special|>abc"]);
}

fn gemma_byte_fallback_profile_is_exact() {
    let tokenizer = gemma_json();
    assert_exact(&tokenizer, &["a b", "叫", "<special>a b"]);

    let pie: Tokenizer = tokenizer.to_string().parse().unwrap();
    assert_eq!(pie.decode(&[0xE5 + 6, 0x8F + 6], false), "��");
}

fn grammar_bytes_are_decoder_aware_and_exclude_specials() {
    let byte_level = byte_level_json(
        serde_json::Value::Null,
        &[r".+"],
        true,
        MergeFormat::Tuple,
        false,
    );
    let pie: Tokenizer = byte_level.to_string().parse().unwrap();
    assert_eq!(pie.decoded_token_bytes(0xC3), Some(&[0xC3][..]));
    assert_eq!(pie.decoded_token_bytes(0xA9), Some(&[0xA9][..]));
    assert_eq!(pie.decoded_token_bytes(260), None);
    assert!(!pie.sorted_token_ids().contains(&260));

    let gemma: Tokenizer = gemma_json().to_string().parse().unwrap();
    assert_eq!(gemma.decoded_token_bytes(4), Some(&b"a "[..]));
    assert_eq!(gemma.decoded_token_bytes(6 + 0xE5), Some(&[0xE5][..]));
}

fn a_unigram_walks_and_wraps_exactly_as_hugging_face_does() {
    let json = unigram_json();
    let bytes = serde_json::to_vec(&json).unwrap();
    let pie = std::str::from_utf8(&bytes)
        .unwrap()
        .parse::<Tokenizer>()
        .unwrap();
    let hf = HfTokenizer::from_bytes(&bytes).unwrap();
    assert_eq!(pie.vocab_size(), hf.get_vocab_size(true));

    for text in ["ab", "a red", "red", "a  b", "", "aQb", "abab", "d"] {
        let ids = pie.encode(text);
        let hf_ids = hf.encode(text, true).unwrap().get_ids().to_vec();
        assert_eq!(ids, hf_ids, "encoding {text:?}");
        assert_eq!(
            pie.decode(&ids, true),
            hf.decode(&ids, true).unwrap(),
            "decoding {text:?}"
        );
    }
}

fn a_unigram_survives_being_baked_and_read_back() {
    let bytes = serde_json::to_vec(&unigram_json()).unwrap();
    let pie = std::str::from_utf8(&bytes)
        .unwrap()
        .parse::<Tokenizer>()
        .unwrap();

    let baked = pie.to_canonical().expect("a Unigram bakes");
    assert!(
        baked.unigram_scores.is_some(),
        "a Unigram writes its score plane; without it the walk cannot be rebuilt"
    );
    assert_eq!(baked.unigram_scores.as_ref().unwrap().len(), 12 * 4);

    let names: Vec<&str> = baked.objects().iter().map(|(name, _)| *name).collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    assert_eq!(
        names, sorted,
        "canonical `.zt` form requires ascending names"
    );

    let back = Tokenizer::from_canonical(&baked).expect("and reads back");
    for text in ["ab", "a red", "red", "a  b", "", "aQb", "abab", "d"] {
        assert_eq!(back.encode(text), pie.encode(text), "encoding {text:?}");
        let ids = pie.encode(text);
        assert_eq!(
            back.decode(&ids, true),
            pie.decode(&ids, true),
            "decoding {text:?}"
        );
    }
}

fn a_bpe_tokenizer_writes_no_score_plane() {
    let bytes = serde_json::to_vec(&gemma_json()).unwrap();
    let pie = std::str::from_utf8(&bytes)
        .unwrap()
        .parse::<Tokenizer>()
        .unwrap();
    let baked = pie.to_canonical().expect("a BPE bakes");
    assert!(baked.unigram_scores.is_none());
    assert_eq!(baked.objects().len(), tokenizer::canonical::OBJECTS.len());
}

mod common;

use std::sync::Arc;

use common::{MergeFormat, byte_level_json, gemma_json, mistral_json, phi3_json};
use pie_tokenizer::Tokenizer;
use serde_json::json;
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
        let pie_ids = pie.encode(text);
        let hf_ids = hf.encode(text, false).unwrap().get_ids().to_vec();
        assert_eq!(pie_ids, hf_ids, "encoding {text:?}");
        assert_eq!(
            pie.decode(&hf_ids, false),
            hf.decode(&hf_ids, false).unwrap(),
            "HF→Pie decoding {text:?}"
        );
        assert_eq!(
            pie.decode(&pie_ids, false),
            hf.decode(&pie_ids, false).unwrap(),
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

#[test]
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

#[test]
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

#[test]
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

#[test]
fn gemma_byte_fallback_profile_is_exact() {
    let tokenizer = gemma_json();
    assert_exact(&tokenizer, &["a b", "叫", "<special>a b"]);

    let pie: Tokenizer = tokenizer.to_string().parse().unwrap();
    assert_eq!(pie.decode(&[0xE5 + 6, 0x8F + 6], false), "��");
}

#[test]
fn phi3_sentencepiece_profile_is_exact() {
    let tokenizer = phi3_json();
    assert_exact(
        &tokenizer,
        &[
            "",
            "a b",
            " a",
            "a ",
            "  a  b  ",
            "a\tb",
            "叫",
            "a叫b",
            "<s>a b",
            "<|end|>a",
            "a <|end|>  b",
            "a<|assistant|> a",
            "<s><|end|>",
        ],
    );

    let pie: Tokenizer = tokenizer.to_string().parse().unwrap();
    // Dummy prefix on encode, Strip undoing it on decode.
    assert_eq!(pie.encode("a b"), vec![5, 6]);
    assert_eq!(pie.decode(&[5, 6], false), "a b");
    // rstrip'd added tokens consume the following whitespace.
    assert_eq!(pie.encode("a <|end|>  b"), vec![5, 2, 263, 6]);
    // Byte fallback still round-trips.
    assert_eq!(pie.decode(&pie.encode("叫"), false), "叫");
}

#[test]
fn mistral_metaspace_profile_is_exact() {
    let tokenizer = mistral_json();
    assert_exact(
        &tokenizer,
        &[
            "",
            "a b",
            " a",
            "a ",
            "  a  b  ",
            "▁a",
            "a\tb",
            "叫",
            "a叫b",
            "<s>a b",
            "<s> a",
            "[INST]a",
            "a [INST] b",
            "[INST]a b[INST]",
            "<s>[INST]",
        ],
    );

    let pie: Tokenizer = tokenizer.to_string().parse().unwrap();
    // Dummy prefix at the input start, Strip undoing it on decode.
    assert_eq!(pie.encode("a b"), vec![5, 6]);
    assert_eq!(pie.decode(&[5, 6], false), "a b");
    // A leading space already produces the marker → no extra prefix.
    assert_eq!(pie.encode(" a"), vec![5]);
    // Segments after a special token receive no prefix.
    assert_eq!(pie.encode("[INST]a b"), vec![263, 3, 6]);
    // Byte fallback still round-trips.
    assert_eq!(pie.decode(&pie.encode("叫"), false), "叫");
}

#[test]
fn metaspace_shape_variants_are_rejected() {
    // Metaspace splitting has different segmentation semantics.
    let mut tokenizer = mistral_json();
    tokenizer["pre_tokenizer"]["split"] = json!(true);
    let error = tokenizer.to_string().parse::<Tokenizer>().err().unwrap();
    assert!(error.to_string().contains("Metaspace.split must be false"));

    // Only the "first" prepend scheme is this profile.
    for scheme in ["always", "never"] {
        let mut tokenizer = mistral_json();
        tokenizer["pre_tokenizer"]["prepend_scheme"] = json!(scheme);
        let error = tokenizer.to_string().parse::<Tokenizer>().err().unwrap();
        assert!(
            error
                .to_string()
                .contains(&format!("unsupported Metaspace prepend_scheme: {scheme}"))
        );
    }

    // A missing prepend_scheme defaults to "always" in HF → reject.
    let mut tokenizer = mistral_json();
    tokenizer["pre_tokenizer"]
        .as_object_mut()
        .unwrap()
        .remove("prepend_scheme");
    let error = tokenizer.to_string().parse::<Tokenizer>().err().unwrap();
    assert!(error.to_string().contains("requires prepend_scheme"));

    // Legacy serializations carrying add_prefix_space are ambiguous → reject.
    let mut tokenizer = mistral_json();
    tokenizer["pre_tokenizer"]["add_prefix_space"] = json!(true);
    let error = tokenizer.to_string().parse::<Tokenizer>().err().unwrap();
    assert!(
        error
            .to_string()
            .contains("add_prefix_space is unsupported")
    );

    // An active normalizer is not part of the Metaspace profile.
    let mut tokenizer = mistral_json();
    tokenizer["normalizer"] = json!({"type": "NFC"});
    let error = tokenizer.to_string().parse::<Tokenizer>().err().unwrap();
    assert!(
        error
            .to_string()
            .contains("unsupported metaspace normalizer: NFC")
    );

    // The decoder must end in the dummy-prefix Strip.
    let mut tokenizer = mistral_json();
    tokenizer["decoder"]["decoders"]
        .as_array_mut()
        .unwrap()
        .pop();
    let error = tokenizer.to_string().parse::<Tokenizer>().err().unwrap();
    assert!(
        error
            .to_string()
            .contains("must contain Replace, ByteFallback, Fuse, Strip")
    );

    // The decoder Replace must reverse the Metaspace replacement.
    let mut tokenizer = mistral_json();
    tokenizer["decoder"]["decoders"][0]["pattern"]["String"] = json!("_");
    let error = tokenizer.to_string().parse::<Tokenizer>().err().unwrap();
    assert!(
        error
            .to_string()
            .contains("decoder Replace must reverse the normalizer")
    );

    // Metaspace requires byte fallback.
    let mut tokenizer = mistral_json();
    tokenizer["model"]["byte_fallback"] = json!(false);
    let error = tokenizer.to_string().parse::<Tokenizer>().err().unwrap();
    assert!(error.to_string().contains("requires byte_fallback"));
}

#[test]
fn phi3_shape_variants_are_rejected() {
    // A sentencepiece decoder without the trailing Strip is not this profile.
    let mut tokenizer = phi3_json();
    tokenizer["decoder"]["decoders"]
        .as_array_mut()
        .unwrap()
        .pop();
    let error = tokenizer.to_string().parse::<Tokenizer>().err().unwrap();
    assert!(
        error
            .to_string()
            .contains("must contain Replace, ByteFallback, Fuse, Strip")
    );

    // The Prepend marker must match the Replace marker.
    let mut tokenizer = phi3_json();
    tokenizer["normalizer"]["normalizers"][0]["prepend"] = json!("_");
    let error = tokenizer.to_string().parse::<Tokenizer>().err().unwrap();
    assert!(
        error
            .to_string()
            .contains("Prepend must inject the Replace marker")
    );

    // Sentencepiece requires byte fallback.
    let mut tokenizer = phi3_json();
    tokenizer["model"]["byte_fallback"] = json!(false);
    let error = tokenizer.to_string().parse::<Tokenizer>().err().unwrap();
    assert!(error.to_string().contains("requires byte_fallback"));
}

#[test]
fn unsupported_legacy_shapes_are_rejected() {
    let mut tokenizer = byte_level_json(
        serde_json::Value::Null,
        &[r".+"],
        false,
        MergeFormat::Tuple,
        false,
    );
    tokenizer["pre_tokenizer"]["pretokenizers"][1]["use_regex"] = json!(true);
    let error = tokenizer.to_string().parse::<Tokenizer>().err().unwrap();
    assert!(
        error
            .to_string()
            .contains("ByteLevel.use_regex must be false")
    );

    let mut tokenizer = gemma_json();
    tokenizer["added_tokens"][0]["single_word"] = json!(true);
    let error = tokenizer.to_string().parse::<Tokenizer>().err().unwrap();
    assert!(
        error
            .to_string()
            .contains("unsupported added-token single_word flag")
    );

    let mut tokenizer = gemma_json();
    tokenizer["model"]["future_semantics"] = json!(true);
    let error = tokenizer.to_string().parse::<Tokenizer>().err().unwrap();
    assert!(format!("{error:#}").contains("unknown field"));

    let mut tokenizer = gemma_json();
    tokenizer["added_tokens"][0]["content"] = json!("");
    let error = tokenizer.to_string().parse::<Tokenizer>().err().unwrap();
    assert!(format!("{error:#}").contains("cannot be empty"));
}

#[test]
fn non_ascii_added_tokens_preserve_literal_bytes() {
    let mut tokenizer = byte_level_json(
        serde_json::Value::Null,
        &[r".+"],
        true,
        MergeFormat::Tuple,
        false,
    );
    let vocab = tokenizer["model"]["vocab"].as_object_mut().unwrap();
    vocab.remove("<|special|>");
    vocab.insert("<｜User｜>".into(), json!(260));
    tokenizer["added_tokens"][0]["content"] = json!("<｜User｜>");

    assert_exact(&tokenizer, &["<｜User｜>hello"]);
    let pie: Tokenizer = tokenizer.to_string().parse().unwrap();
    assert_eq!(pie.token_to_id("<｜User｜>"), Some(260));
    assert_eq!(pie.decode(&[260], false), "<｜User｜>");
}

#[test]
fn malformed_vocab_and_merges_are_rejected() {
    let mut malformed_merge = byte_level_json(
        serde_json::Value::Null,
        &[r".+"],
        false,
        MergeFormat::Tuple,
        false,
    );
    malformed_merge["model"]["merges"]
        .as_array_mut()
        .unwrap()
        .push(json!(["missing", "tokens"]));
    let error = malformed_merge
        .to_string()
        .parse::<Tokenizer>()
        .err()
        .unwrap();
    assert!(format!("{error:#}").contains("unknown left token"));

    let mut sparse_ids = gemma_json();
    sparse_ids["model"]["vocab"]["a"] = json!(300);
    let error = sparse_ids.to_string().parse::<Tokenizer>().err().unwrap();
    assert!(format!("{error:#}").contains("contiguous"));
}

#[test]
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

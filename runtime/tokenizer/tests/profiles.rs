mod common;

use std::sync::Arc;

use common::{MergeFormat, byte_level_json, gemma_json};
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
    tokenizer["added_tokens"][0]["lstrip"] = json!(true);
    let error = tokenizer.to_string().parse::<Tokenizer>().err().unwrap();
    assert!(
        error
            .to_string()
            .contains("unsupported added-token boundary flags")
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

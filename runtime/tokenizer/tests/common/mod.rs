#![allow(dead_code)]

use serde_json::{Value, json};
use std::sync::OnceLock;

pub const TEXTS: &[&str] = &[
    "",
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "  spaces   and\ttabs  ",
    "line1\nline2\n\nline4",
    "no\rnewline\r\nwindows\r\nstyle",
    "fn main() { let x = 42; println!(\"{x}\"); }",
    "día café naïve résumé über",
    "日本語テスト 中文测试 한국어",
    "مرحبا بالعالم",
    "שלום עולם",
    "Привет, мир!",
    "Hello 你好 こんにちは مرحبا Привет 🌍",
    "👨\u{200D}👩\u{200D}👧\u{200D}👦 family",
    "3.14159265358979323846264338327950288419716939937510",
    "1e-10 2.5e+03 -0.001 +42 0xDEADBEEF",
    "{\"name\": \"test\", \"values\": [1, 2, 3]}",
    "<|im_start|>system\nHello<|im_end|>",
    "<｜User｜>Hello<｜Assistant｜>",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "\u{200B}\u{200B}\u{200B}",
    "a\u{0300}a\u{0301}a\u{0302}a\u{0303}",
    "\u{00A0}non-breaking\u{00A0}spaces\u{00A0}",
    "a",
    "Ω",
    "🎵",
    "\n",
    "\t",
    " ",
];

#[derive(Clone, Copy)]
pub enum MergeFormat {
    String,
    Tuple,
}

fn mapped(bytes: &[u8]) -> String {
    let alphabet = bytes_to_unicode();
    bytes.iter().map(|byte| alphabet[*byte as usize]).collect()
}

fn bytes_to_unicode() -> &'static [char; 256] {
    static ALPHABET: OnceLock<[char; 256]> = OnceLock::new();
    ALPHABET.get_or_init(|| {
        let mut direct = Vec::new();
        direct.extend(b'!'..=b'~');
        direct.extend(0xA1..=0xAC);
        direct.extend(0xAE..=0xFF);

        let mut table = ['\0'; 256];
        let mut extra = 0u32;
        for byte in 0u16..=255 {
            table[byte as usize] = if direct.contains(&(byte as u8)) {
                char::from_u32(byte as u32).unwrap()
            } else {
                let character = char::from_u32(256 + extra).unwrap();
                extra += 1;
                character
            };
        }
        table
    })
}

pub fn byte_level_json(
    normalizer: Value,
    regexes: &[&str],
    ignore_merges: bool,
    merge_format: MergeFormat,
    added_normalized: bool,
) -> Value {
    let mut vocab = serde_json::Map::new();
    for byte in 0u16..=255 {
        vocab.insert(mapped(&[byte as u8]), json!(byte));
    }

    let merged = [
        (b"12".as_slice(), 256u32),
        (b"123".as_slice(), 257),
        (b"1234".as_slice(), 258),
        (b"abc".as_slice(), 259),
    ];
    for (bytes, id) in merged {
        vocab.insert(mapped(bytes), json!(id));
    }
    vocab.insert("<|special|>".into(), json!(260));

    let merge_pairs = [("1", "2"), ("12", "3"), ("123", "4")];
    let merges = merge_pairs
        .into_iter()
        .map(|(left, right)| match merge_format {
            MergeFormat::String => json!(format!(
                "{} {}",
                mapped(left.as_bytes()),
                mapped(right.as_bytes())
            )),
            MergeFormat::Tuple => json!([mapped(left.as_bytes()), mapped(right.as_bytes())]),
        })
        .collect::<Vec<_>>();

    let mut pretokenizers = regexes
        .iter()
        .map(|pattern| {
            json!({
                "type": "Split",
                "pattern": {"Regex": pattern},
                "behavior": "Isolated",
                "invert": false
            })
        })
        .collect::<Vec<_>>();
    pretokenizers.push(json!({
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": true,
        "use_regex": false
    }));

    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [{
            "id": 260,
            "content": "<|special|>",
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": added_normalized,
            "special": true
        }],
        "normalizer": normalizer,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": pretokenizers
        },
        "post_processor": null,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": true,
            "trim_offsets": true,
            "use_regex": true
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": ignore_merges,
            "vocab": vocab,
            "merges": merges
        }
    })
}

pub fn gemma_json() -> Value {
    let mut vocab = serde_json::Map::new();
    for (token, id) in [
        ("<unk>", 0u32),
        ("a", 1),
        ("▁", 2),
        ("b", 3),
        ("a▁", 4),
        ("a▁b", 5),
    ] {
        vocab.insert(token.into(), json!(id));
    }
    for byte in 0u16..=255 {
        vocab.insert(format!("<0x{byte:02X}>"), json!(6 + byte));
    }
    vocab.insert("<special>".into(), json!(262));

    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [{
            "id": 262,
            "content": "<special>",
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": true
        }],
        "normalizer": {
            "type": "Replace",
            "pattern": {"String": " "},
            "content": "▁"
        },
        "pre_tokenizer": {
            "type": "Split",
            "pattern": {"String": " "},
            "behavior": "MergedWithPrevious",
            "invert": false
        },
        "post_processor": null,
        "decoder": {
            "type": "Sequence",
            "decoders": [
                {
                    "type": "Replace",
                    "pattern": {"String": "▁"},
                    "content": " "
                },
                {"type": "ByteFallback"},
                {"type": "Fuse"}
            ]
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": "<unk>",
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": true,
            "byte_fallback": true,
            "ignore_merges": false,
            "vocab": vocab,
            "merges": [["a", "▁"], ["a▁", "b"]]
        }
    })
}

//! Exact compatibility against an official Kimi tiktoken snapshot.

use std::path::PathBuf;
use std::sync::Arc;

use pie_tokenizer::Tokenizer;

#[test]
#[ignore = "requires the official Kimi K2 tokenizer snapshot"]
fn official_kimi_k2_golden_vectors() {
    let dir = PathBuf::from(
        std::env::var_os("PIE_KIMI_TOKENIZER_DIR")
            .expect("set PIE_KIMI_TOKENIZER_DIR to a Kimi K2 tokenizer snapshot"),
    );
    let tokenizer = Arc::new(Tokenizer::from_file(&dir.join("tiktoken.model")).unwrap());

    assert_eq!(tokenizer.vocab_size(), 163_840);
    assert_eq!(tokenizer.token_to_id("<|im_user|>"), Some(163_587));
    assert_eq!(
        tokenizer.token_to_id("<|reserved_token_163600|>"),
        Some(163_600)
    );

    let cases: &[(&str, &[u32])] = &[
        ("", &[]),
        ("Hello, world!", &[19_180, 11, 2_695, 0]),
        (
            "  spaces   and\ttabs  ",
            &[220, 14_803, 256, 316, 5_604, 5_609, 256],
        ),
        (
            "12345678901234567890",
            &[6_694, 12_972, 16_242, 16_349, 18_439, 22_523, 2_788],
        ),
        (
            "Hello 你好 こんにちは 한국어 مرحبا 🌍",
            &[
                19_180, 220, 33_845, 220, 16_444, 29_194, 6_880, 43_566, 8_831, 78_560, 54_078,
                28_763, 61_561, 7_570, 4_265, 1_063, 17_137, 44_755,
            ],
        ),
        (
            "👨‍👩‍👧‍👦 family",
            &[
                64_390, 101, 67_963, 64_390, 102, 67_963, 64_390, 100, 67_963, 64_390, 99, 3_545,
            ],
        ),
        (
            "fn main() {\n    println!(\"Hello\");\n}",
            &[
                10_964, 2_777, 539, 440, 274, 47_647, 28_547, 19_180, 1_752, 92,
            ],
        ),
        ("<|im_user|>hello<|im_end|>", &[163_587, 22_931, 163_586]),
        (
            r#"<|tool_call_begin|>{"name":"f"}<|tool_call_end|>"#,
            &[163_597, 8_264, 1_152, 7_471, 69, 16_934, 163_599],
        ),
        ("<|reserved_token_163600|>", &[163_600]),
    ];

    for &(text, expected) in cases {
        assert_eq!(tokenizer.encode(text), expected, "encoding {text:?}");
        assert_eq!(tokenizer.decode(expected, false), text, "decoding {text:?}");
        let mut decoder = tokenizer.decoder(false);
        let mut incremental = String::new();
        for token in expected {
            incremental.push_str(&decoder.feed(std::slice::from_ref(token)));
        }
        incremental.push_str(&decoder.finish());
        assert_eq!(incremental, text, "incremental decoding {text:?}");
    }
}

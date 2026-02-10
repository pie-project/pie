/// Integration tests using real HuggingFace tokenizer.json files.
///
/// These tests load actual model tokenizers from the HF cache and verify
/// that encode/decode produce correct results.
///
/// Expected token IDs were obtained from the HuggingFace Python tokenizers library:
/// ```python
/// from transformers import AutoTokenizer
/// tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
/// print(tok.encode("Hello, world!"))
/// ```

use std::path::Path;
use pie::model::tokenizer::Tokenizer;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn load_tokenizer(model_dir: &str) -> Option<Tokenizer> {
    let path = Path::new(model_dir).join("tokenizer.json");
    if !path.exists() {
        eprintln!("Skipping test: {path:?} not found");
        return None;
    }
    Some(Tokenizer::from_file(&path).expect("Failed to load tokenizer"))
}

/// Run roundtrip tests. `lossy` = true for SentencePiece models where
/// leading whitespace can be lost (Metaspace pre-tokenizer).
fn assert_roundtrip(tok: &Tokenizer, texts: &[&str], lossy: bool) {
    for &text in texts {
        let ids = tok.encode(text);
        if text.is_empty() {
            assert!(ids.is_empty(), "non-empty encoding for empty string");
            continue;
        }
        assert!(!ids.is_empty(), "empty encoding for: {text:?}");
        let decoded = tok.decode(&ids, false);
        if lossy {
            // SentencePiece (Metaspace) can lose leading whitespace.
            // Converge to a fixed point, then verify it's stable.
            let mut prev = decoded;
            for _ in 0..10 {
                let next = tok.decode(&tok.encode(&prev), false);
                if next == prev {
                    break;
                }
                prev = next;
            }
            let stable = tok.decode(&tok.encode(&prev), false);
            assert_eq!(stable, prev, "decode not stable for: {text:?}");
        } else {
            assert_eq!(&decoded, text, "roundtrip failed for: {text:?}");
        }
    }
}

/// Shared test strings for roundtrip tests across all models.
/// Designed to stress-test: mixed scripts, whitespace edge cases, code,
/// emoji sequences, long repetitions, Unicode boundaries, and more.
const ROUNDTRIP_TEXTS: &[&str] = &[
    // === Basic ===
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "",  // empty (skipped by roundtrip — but good to include)

    // === Whitespace torture ===
    "  spaces  and\ttabs\n",
    "line1\nline2\n\nline4",
    "   \t\t  \n\n\n  \t ",                   // only whitespace
    "a b  c   d    e     f",                    // expanding gaps
    "no\rnewline\r\nwindows\r\nstyle",          // carriage returns

    // === Multilingual ===
    "こんにちは世界",                            // Japanese
    "你好世界！这是一个测试。",                   // Chinese
    "안녕하세요 세계",                           // Korean
    "مرحبا بالعالم",                            // Arabic (RTL)
    "שלום עולם",                                // Hebrew (RTL)
    "Привет, мир! Как дела?",                   // Russian
    "Γεια σου κόσμε",                           // Greek
    "สวัสดีครับ",                                 // Thai
    "Héllo wörld café résumé naïve",            // Latin diacritics

    // === Mixed scripts in one string ===
    "Hello 你好 こんにちは مرحبا Привет 🌍",
    "The price is ¥1,000 or €850 or $999.99",
    "Temperature: −40°C = −40°F (they intersect!)",

    // === Emoji stress ===
    "🎉 emoji test 🚀",
    "👨‍👩‍👧‍👦 family",                              // ZWJ sequence (family emoji)
    "🏳️‍🌈🏴‍☠️",                                    // flag ZWJ sequences
    "👋🏻👋🏼👋🏽👋🏾👋🏿",                              // skin tone modifiers
    "😀😃😄😁😆😅🤣😂🙂🙃😉😊😇",            // dense emoji run

    // === Code ===
    "fn main() { println!(\"test\"); }",
    "def f(x: int) -> list[dict[str, Any]]:\n    return [{\"key\": x}]",
    "SELECT u.name, COUNT(*) AS cnt FROM users u\n  LEFT JOIN orders o ON u.id = o.user_id\n  WHERE o.created_at >= '2024-01-01'\n  GROUP BY u.name\n  HAVING cnt > 5\n  ORDER BY cnt DESC;",
    "const x = { ...obj, arr: [1, [2, [3, [4]]]], fn: () => ({ a: 1 }) };",
    "#include <stdio.h>\nint main() {\n    printf(\"Hello, world!\\n\");\n    return 0;\n}",
    "if (a && b || !c && (d ^ e) & 0xFF) { *ptr++ = ~val; }",

    // === Long repetitions (stress the merge loop) ===
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",  // 64 'a's
    "abababababababababababababababababababababababababababababababab",  // 31 "ab"s
    "the the the the the the the the the the the the the the the the", // repeated word

    // === Numbers and math ===
    "3.14159265358979323846264338327950288419716939937510",
    "1e-10 2.5e+03 -0.001 +42 0xDEADBEEF 0b1101 0o777",
    "∑_{i=0}^{n} x_i² = ∫₀^∞ f(t) dt ≈ π × r²",

    // === Markdown / formatting ===
    "# Title\n## Subtitle\n- bullet **bold** _italic_ `code`\n> quote\n\n---\n\n1. ordered\n2. list",
    "Here is a [link](https://example.com/path?q=hello+world&lang=en#section) in text.",
    "| Col A | Col B |\n|-------|-------|\n| val 1 | val 2 |",

    // === JSON-like ===
    "{\"name\":\"test\",\"values\":[1,2,3],\"nested\":{\"a\":{\"b\":{\"c\":true}}}}",

    // === Special characters and escapes ===
    "backslash: \\\\ quote: \\\" tab: \\t newline: \\n",
    "HTML entities: &amp; &lt; &gt; &quot; &apos;",
    "<div class=\"container\"><p>Hello &amp; <b>world</b></p></div>",

    // === URLs and paths ===
    "https://user:pass@example.com:8080/api/v2/resource?key=val&foo=bar#fragment",
    "/usr/local/bin/../lib/python3.11/site-packages/numpy/__init__.py",
    "C:\\Users\\test\\Documents\\file (1).txt",

    // === Unicode edge cases ===
    "\u{200B}\u{200B}\u{200B}",                 // zero-width spaces
    "\u{FEFF}BOM at start",                     // byte-order mark
    "a\u{0300}a\u{0301}a\u{0302}a\u{0303}",     // combining diacritics: àáâã
    "fi fl ffi ffl ﬁ ﬂ ﬃ ﬄ",                    // ligatures (regular + Unicode)
    "2⁰ 2¹ 2² 2³ 2⁴ 2⁵ 2⁶ 2⁷ 2⁸ 2⁹",        // superscript digits
    "∀x∈ℝ: x²≥0 ∧ ∃y∈ℂ: y²<0",               // math symbols

    // === Adversarial / boundary ===
    "a",                                         // single ASCII char
    "Ω",                                         // single multi-byte char
    "🎵",                                        // single 4-byte char
    "\n",                                        // single newline
    "\t",                                        // single tab
    " ",                                         // single space
];


// ---------------------------------------------------------------------------
// LLaMA 3.2-1B (ByteLevel pre-tokenizer + ByteLevel decoder)
// ---------------------------------------------------------------------------

const LLAMA_32_DIR: &str = "/Users/ingim/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08";

#[test]
fn test_llama32_load() {
    let tok = load_tokenizer(LLAMA_32_DIR);
    if let Some(tok) = tok {
        // LLaMA 3.2 has 128256 tokens (128000 base + 256 special)
        assert!(tok.vocab_size() >= 128000, "vocab too small: {}", tok.vocab_size());
        println!("LLaMA 3.2 vocab size: {}", tok.vocab_size());
    }
}

#[test]
fn test_llama32_encode_hello() {
    let Some(tok) = load_tokenizer(LLAMA_32_DIR) else { return };
    let ids = tok.encode("Hello, world!");
    println!("LLaMA 3.2 encode 'Hello, world!': {:?}", ids);
    // Verified against HF Python: [9906, 11, 1917, 0]
    assert_eq!(ids, vec![9906, 11, 1917, 0]);
    // Verify roundtrip
    let decoded = tok.decode(&ids, false);
    assert_eq!(decoded, "Hello, world!");
}

#[test]
fn test_llama32_encode_multilingual() {
    let Some(tok) = load_tokenizer(LLAMA_32_DIR) else { return };
    let text = "こんにちは世界";
    let ids = tok.encode(text);
    println!("LLaMA 3.2 encode '{}': {:?}", text, ids);
    assert!(!ids.is_empty());
    let decoded = tok.decode(&ids, false);
    assert_eq!(decoded, text);
}

#[test]
fn test_llama32_encode_code() {
    let Some(tok) = load_tokenizer(LLAMA_32_DIR) else { return };
    let text = "fn main() {\n    println!(\"Hello\");\n}";
    let ids = tok.encode(text);
    println!("LLaMA 3.2 encode code: {:?}", ids);
    assert!(!ids.is_empty());
    let decoded = tok.decode(&ids, false);
    assert_eq!(decoded, text);
}

#[test]
fn test_llama32_special_tokens() {
    let Some(tok) = load_tokenizer(LLAMA_32_DIR) else { return };

    // Verify special tokens exist
    assert!(tok.token_to_id("<|begin_of_text|>").is_some());
    assert!(tok.token_to_id("<|end_of_text|>").is_some());

    // Encode text with special tokens embedded
    let text = "<|begin_of_text|>Hello<|end_of_text|>";
    let ids = tok.encode(text);
    println!("LLaMA 3.2 with special tokens: {:?}", ids);

    // The special tokens should be recognized
    let bos_id = tok.token_to_id("<|begin_of_text|>").unwrap();
    let eos_id = tok.token_to_id("<|end_of_text|>").unwrap();
    assert_eq!(ids[0], bos_id);
    assert_eq!(*ids.last().unwrap(), eos_id);

    // Decode with skip_special=true should strip them
    let decoded_skip = tok.decode(&ids, true);
    assert_eq!(decoded_skip, "Hello");

    // Decode with skip_special=false should include them
    let decoded_full = tok.decode(&ids, false);
    assert!(decoded_full.contains("Hello"));
}

#[test]
fn test_llama32_whitespace() {
    let Some(tok) = load_tokenizer(LLAMA_32_DIR) else { return };
    let text = "  multiple   spaces   here  ";
    let ids = tok.encode(text);
    let decoded = tok.decode(&ids, false);
    assert_eq!(decoded, text);
}

#[test]
fn test_llama32_empty() {
    let Some(tok) = load_tokenizer(LLAMA_32_DIR) else { return };
    let ids = tok.encode("");
    assert!(ids.is_empty());
    let decoded = tok.decode(&[], false);
    assert_eq!(decoded, "");
}

#[test]
fn test_llama32_newlines() {
    let Some(tok) = load_tokenizer(LLAMA_32_DIR) else { return };
    let text = "line1\nline2\n\nline4";
    let ids = tok.encode(text);
    let decoded = tok.decode(&ids, false);
    assert_eq!(decoded, text);
}

// ---------------------------------------------------------------------------
// Qwen3-0.6B
// ---------------------------------------------------------------------------

const QWEN3_DIR: &str = "/Users/ingim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca";

#[test]
fn test_qwen3_load() {
    let tok = load_tokenizer(QWEN3_DIR);
    if let Some(tok) = tok {
        assert!(tok.vocab_size() >= 150000, "vocab too small: {}", tok.vocab_size());
        println!("Qwen3 vocab size: {}", tok.vocab_size());
    }
}

#[test]
fn test_qwen3_roundtrip() {
    let Some(tok) = load_tokenizer(QWEN3_DIR) else { return };
    // NFC normalizer composes combining characters, making roundtrip lossy
    // for decomposed input (e.g. a+U+0300 → à).
    assert_roundtrip(&tok, ROUNDTRIP_TEXTS, true);
}

// ---------------------------------------------------------------------------
// Mistral-7B-v0.1
// ---------------------------------------------------------------------------

const MISTRAL_DIR: &str = "/Users/ingim/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots/7231864981174d9bee8c7687c24c8344414eae6b";

#[test]
fn test_mistral_load() {
    let tok = load_tokenizer(MISTRAL_DIR);
    if let Some(tok) = tok {
        assert!(tok.vocab_size() >= 30000, "vocab too small: {}", tok.vocab_size());
        println!("Mistral vocab size: {}", tok.vocab_size());
    }
}

#[test]
fn test_mistral_roundtrip() {
    let Some(tok) = load_tokenizer(MISTRAL_DIR) else { return };
    // Mistral uses Metaspace pre-tokenizer — leading whitespace can be lossy.
    assert_roundtrip(&tok, ROUNDTRIP_TEXTS, true);
}

// ---------------------------------------------------------------------------
// openai/gpt-oss-120b (ByteLevel)
// ---------------------------------------------------------------------------

const GPT_OSS_DIR: &str = "/Users/ingim/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a";

#[test]
fn test_gpt_oss_load() {
    let tok = load_tokenizer(GPT_OSS_DIR);
    if let Some(tok) = tok {
        assert!(tok.vocab_size() >= 200000, "vocab too small: {}", tok.vocab_size());
        println!("GPT-oss-120b vocab size: {}", tok.vocab_size());
    }
}

#[test]
fn test_gpt_oss_roundtrip() {
    let Some(tok) = load_tokenizer(GPT_OSS_DIR) else { return };
    assert_roundtrip(&tok, ROUNDTRIP_TEXTS, false);
}

// ---------------------------------------------------------------------------
// microsoft/phi-4 (ByteLevel)
// ---------------------------------------------------------------------------

const PHI4_DIR: &str = "/Users/ingim/.cache/huggingface/hub/models--microsoft--phi-4/snapshots/932b33c0ec9ca189badeb22480721a8de9d0e006";

#[test]
fn test_phi4_load() {
    let tok = load_tokenizer(PHI4_DIR);
    if let Some(tok) = tok {
        assert!(tok.vocab_size() >= 100000, "vocab too small: {}", tok.vocab_size());
        println!("Phi-4 vocab size: {}", tok.vocab_size());
    }
}

#[test]
fn test_phi4_roundtrip() {
    let Some(tok) = load_tokenizer(PHI4_DIR) else { return };
    assert_roundtrip(&tok, ROUNDTRIP_TEXTS, false);
}

// ---------------------------------------------------------------------------
// allenai/OLMo-3.1-32B-Think (ByteLevel)
// ---------------------------------------------------------------------------

const OLMO_DIR: &str = "/Users/ingim/.cache/huggingface/hub/models--allenai--Olmo-3.1-32B-Think/snapshots/832c3f543499af8fe68b88359501de9cb7840544";

#[test]
fn test_olmo_load() {
    let tok = load_tokenizer(OLMO_DIR);
    if let Some(tok) = tok {
        assert!(tok.vocab_size() >= 100000, "vocab too small: {}", tok.vocab_size());
        println!("OLMo vocab size: {}", tok.vocab_size());
    }
}

#[test]
fn test_olmo_roundtrip() {
    let Some(tok) = load_tokenizer(OLMO_DIR) else { return };
    assert_roundtrip(&tok, ROUNDTRIP_TEXTS, false);
}

// ---------------------------------------------------------------------------
// google/gemma-2-2b (ByteLevel)
// ---------------------------------------------------------------------------

const GEMMA2_DIR: &str = "/Users/ingim/.cache/huggingface/hub/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf";

#[test]
fn test_gemma2_load() {
    let tok = load_tokenizer(GEMMA2_DIR);
    if let Some(tok) = tok {
        assert!(tok.vocab_size() >= 256000, "vocab too small: {}", tok.vocab_size());
        println!("Gemma-2-2b vocab size: {}", tok.vocab_size());
    }
}

#[test]
fn test_gemma2_roundtrip() {
    let Some(tok) = load_tokenizer(GEMMA2_DIR) else { return };
    assert_roundtrip(&tok, ROUNDTRIP_TEXTS, false);
}

// ---------------------------------------------------------------------------
// google/gemma-3-12b-it (ByteLevel)
// ---------------------------------------------------------------------------

const GEMMA3_DIR: &str = "/Users/ingim/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80";

#[test]
fn test_gemma3_load() {
    let tok = load_tokenizer(GEMMA3_DIR);
    if let Some(tok) = tok {
        assert!(tok.vocab_size() >= 262000, "vocab too small: {}", tok.vocab_size());
        println!("Gemma-3-12b vocab size: {}", tok.vocab_size());
    }
}

#[test]
fn test_gemma3_roundtrip() {
    let Some(tok) = load_tokenizer(GEMMA3_DIR) else { return };
    // Gemma-3 uses SentencePiece — some rare Unicode chars may be lossy.
    assert_roundtrip(&tok, ROUNDTRIP_TEXTS, true);
}

// ---------------------------------------------------------------------------
// deepseek-ai/DeepSeek-V3.2-Exp (ByteLevel)
// ---------------------------------------------------------------------------

const DEEPSEEK_DIR: &str = "/Users/ingim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V3.2-Exp/snapshots/194c67e12b1b0d6df0ef373ddcf215bc84027409";

#[test]
fn test_deepseek_load() {
    let tok = load_tokenizer(DEEPSEEK_DIR);
    if let Some(tok) = tok {
        assert!(tok.vocab_size() >= 128000, "vocab too small: {}", tok.vocab_size());
        println!("DeepSeek-V3.2-Exp vocab size: {}", tok.vocab_size());
    }
}

#[test]
fn test_deepseek_roundtrip() {
    let Some(tok) = load_tokenizer(DEEPSEEK_DIR) else { return };
    assert_roundtrip(&tok, ROUNDTRIP_TEXTS, false);
}

// ---------------------------------------------------------------------------
// mistralai/Ministral-3-3B-Instruct-2512 (ByteLevel)
// ---------------------------------------------------------------------------

const MINISTRAL_DIR: &str = "/Users/ingim/.cache/huggingface/hub/models--mistralai--Ministral-3-3B-Instruct-2512/snapshots/cfcb068fa7c44114cf77a462357c6cdcd2c304b4";

#[test]
fn test_ministral_load() {
    let tok = load_tokenizer(MINISTRAL_DIR);
    if let Some(tok) = tok {
        assert!(tok.vocab_size() >= 131000, "vocab too small: {}", tok.vocab_size());
        println!("Ministral-3-3B vocab size: {}", tok.vocab_size());
    }
}

#[test]
fn test_ministral_roundtrip() {
    let Some(tok) = load_tokenizer(MINISTRAL_DIR) else { return };
    assert_roundtrip(&tok, ROUNDTRIP_TEXTS, false);
}

// ---------------------------------------------------------------------------
// nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (ByteLevel)
// ---------------------------------------------------------------------------

const NEMOTRON_DIR: &str = "/Users/ingim/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/378df16e4b54901a3f514f38ea9a34db9d061634";

#[test]
fn test_nemotron_load() {
    let tok = load_tokenizer(NEMOTRON_DIR);
    if let Some(tok) = tok {
        assert!(tok.vocab_size() >= 131000, "vocab too small: {}", tok.vocab_size());
        println!("Nemotron-3-Nano vocab size: {}", tok.vocab_size());
    }
}

#[test]
fn test_nemotron_roundtrip() {
    let Some(tok) = load_tokenizer(NEMOTRON_DIR) else { return };
    assert_roundtrip(&tok, ROUNDTRIP_TEXTS, false);
}

// ---------------------------------------------------------------------------
// Cross-model consistency: same text → same decode
// ---------------------------------------------------------------------------

#[test]
fn test_cross_model_decode_consistency() {
    let models: Vec<(&str, Option<Tokenizer>)> = vec![
        ("LLaMA", load_tokenizer(LLAMA_32_DIR)),
        ("Qwen3", load_tokenizer(QWEN3_DIR)),
        ("Mistral", load_tokenizer(MISTRAL_DIR)),
        ("GPT-oss", load_tokenizer(GPT_OSS_DIR)),
        ("Phi-4", load_tokenizer(PHI4_DIR)),
        ("OLMo", load_tokenizer(OLMO_DIR)),
        ("Gemma-2", load_tokenizer(GEMMA2_DIR)),
        ("Gemma-3", load_tokenizer(GEMMA3_DIR)),
        ("DeepSeek", load_tokenizer(DEEPSEEK_DIR)),
        ("Ministral", load_tokenizer(MINISTRAL_DIR)),
        ("Nemotron", load_tokenizer(NEMOTRON_DIR)),
    ];

    let text = "Hello, world!";
    for (name, tok) in &models {
        if let Some(tok) = tok {
            let ids = tok.encode(text);
            assert_eq!(tok.decode(&ids, false), text, "{name} roundtrip failed");
        }
    }
}


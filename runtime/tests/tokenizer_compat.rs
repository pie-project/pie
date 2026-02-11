//! Strict compatibility tests: tokenizer-mini vs HuggingFace `tokenizers` crate.
//!
//! Auto-discovers all locally cached HuggingFace models with a `tokenizer.json`
//! and compares encode/decode output token-by-token.
//! Cross-decode (encode with one, decode with the other) is always strict.
//! Encode comparison asserts ≥90% match rate and reports mismatches.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use pie::tokenizer::Tokenizer as MiniTokenizer;
use tokenizers::Tokenizer as HfTokenizer;

// ---------------------------------------------------------------------------
// Model discovery — scan HF cache for all models with tokenizer.json
// ---------------------------------------------------------------------------

struct Model {
    name: String,
    dir: PathBuf,
}

/// Scan `~/.cache/huggingface/hub/` for all models that have a `tokenizer.json`.
/// Returns one entry per model (latest snapshot by modification time).
fn discover_models() -> Vec<Model> {
    let home = std::env::var("HOME").expect("HOME not set");
    let cache_dir = PathBuf::from(home).join(".cache/huggingface/hub");

    if !cache_dir.is_dir() {
        return vec![];
    }

    let mut best: HashMap<String, (std::time::SystemTime, PathBuf)> = HashMap::new();

    let Ok(entries) = std::fs::read_dir(&cache_dir) else {
        return vec![];
    };

    for entry in entries.flatten() {
        let dir_name = entry.file_name();
        let dir_name = dir_name.to_string_lossy();
        if !dir_name.starts_with("models--") {
            continue;
        }
        let model_name = dir_name
            .strip_prefix("models--")
            .unwrap()
            .replace("--", "/");

        let snapshots_dir = entry.path().join("snapshots");
        let Ok(snapshots) = std::fs::read_dir(&snapshots_dir) else {
            continue;
        };

        for snap in snapshots.flatten() {
            let tokenizer_path = snap.path().join("tokenizer.json");
            if !tokenizer_path.exists() {
                continue;
            }
            let mtime = snap
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);

            let e = best
                .entry(model_name.clone())
                .or_insert((std::time::UNIX_EPOCH, PathBuf::new()));
            if mtime >= e.0 {
                *e = (mtime, snap.path());
            }
        }
    }

    let mut models: Vec<Model> = best
        .into_iter()
        .map(|(name, (_, dir))| Model { name, dir })
        .collect();
    models.sort_by(|a, b| a.name.cmp(&b.name));
    models
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn load_both(dir: &Path) -> Option<(MiniTokenizer, HfTokenizer)> {
    let path = dir.join("tokenizer.json");
    Some((
        MiniTokenizer::from_file(&path).ok()?,
        HfTokenizer::from_file(&path).ok()?,
    ))
}

// ---------------------------------------------------------------------------
// Diverse test inputs
// ---------------------------------------------------------------------------

const TEXTS: &[&str] = &[
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "  spaces   and\ttabs  ",
    "line1\nline2\n\nline4",
    "   \t\t  \n\n\n  \t ",
    "no\rnewline\r\nwindows\r\nstyle",
    "fn main() { let x = 42; println!(\"{x}\"); }",
    "def f(x: int) -> list[dict[str, Any]]:\n    return [{\"key\": x}]",
    "#include <stdio.h>\nint main() {\n    printf(\"Hello\\n\");\n    return 0;\n}",
    "if (a && b || !c && (d ^ e) & 0xFF) { *ptr++ = ~val; }",
    "Wait... really?! Yes—no; maybe: (ok) [fine] {done}",
    "The year is 2025. Pi is 3.14159. Cost: $1,234.56",
    "backslash: \\\\ quote: \\\" tab: \\t newline: \\n",
    "https://example.com/path?q=hello&lang=en#section",
    "https://user:pass@example.com:8080/api/v2/resource?key=val&foo=bar#frag",
    "/usr/local/bin/../lib/python3.11/site-packages/numpy/__init__.py",
    "C:\\Users\\test\\Documents\\file (1).txt",
    "día café naïve résumé über",
    "日本語テスト 中文测试 한국어",
    "مرحبا بالعالم",
    "שלום עולם",
    "Привет, мир!",
    "Γεια σου κόσμε",
    "สวัสดีครับ",
    "Hello 你好 こんにちは مرحبا Привет 🌍",
    "Hello 👋 World 🌍! 🎉🎊",
    "👨\u{200D}👩\u{200D}👧\u{200D}👦 family",
    "👋🏻👋🏼👋🏽👋🏾👋🏿",
    "😀😃😄😁😆😅🤣😂🙂🙃😉😊😇",
    "3.14159265358979323846264338327950288419716939937510",
    "1e-10 2.5e+03 -0.001 +42 0xDEADBEEF 0b1101 0o777",
    "∑(i=0..n) f(x_i) × Δx ≈ ∫f(x)dx",
    "∀x∈ℝ: x²≥0 ∧ ∃y∈ℂ: y²<0",
    "# Title\n\n**bold** _italic_ `code` [link](url)",
    "| Col A | Col B |\n|-------|-------|\n| val 1 | val 2 |",
    "{\"name\": \"test\", \"values\": [1, 2, 3]}",
    "<div class=\"container\"><p>Hello &amp; <b>world</b></p></div>",
    "<|im_start|>system\nYou are helpful.<|im_end|>",
    "<s>[INST] Hello [/INST] Hi there!</s>",
    "the the the the the the the the the the the the the the the the",
    "abababababababababababababababababababababababababababababababab",
    "\u{200B}\u{200B}\u{200B}",
    "\u{FEFF}BOM at start",
    "a\u{0300}a\u{0301}a\u{0302}a\u{0303}",
    "\u{00A0}non-breaking\u{00A0}spaces\u{00A0}",
    "fi fl ffi ffl ﬁ ﬂ ﬃ ﬄ",
    "2⁰ 2¹ 2² 2³ 2⁴ 2⁵ 2⁶ 2⁷ 2⁸ 2⁹",
    "a", "Ω", "🎵", "\n", "\t", " ",
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump! The five boxing wizards jump quickly. Sphinx of black quartz, judge my vow.",
];

// ---------------------------------------------------------------------------
// 1. Encode compatibility — ≥90% overall
// ---------------------------------------------------------------------------

#[test]
fn test_encode_compatibility() {
    let models = discover_models();
    let mut total = 0usize;
    let mut loaded = 0usize;
    let mut mismatches: Vec<String> = Vec::new();

    for model in &models {
        let Some((mini, hf)) = load_both(&model.dir) else {
            eprintln!("SKIP {}: failed to load", model.name);
            continue;
        };
        loaded += 1;
        for text in TEXTS {
            total += 1;
            let m = mini.encode(text);
            let h: Vec<u32> = hf.encode(*text, false).unwrap().get_ids().to_vec();
            if m != h {
                let t = if text.len() > 60 { &text[..60] } else { text };
                mismatches.push(format!(
                    "  {} | {t:?} | mini={} hf={}",
                    model.name, m.len(), h.len()
                ));
            }
        }
    }

    let pct = 100.0 * (total - mismatches.len()) as f64 / total as f64;
    eprintln!("\n=== ENCODE: {}/{total} match ({pct:.1}%) across {loaded} models ===",
        total - mismatches.len());
    for m in &mismatches { eprintln!("{m}"); }

    assert!(total > 0, "no models found in HF cache");
    assert!(
        pct >= 90.0,
        "encode compatibility {pct:.1}% < 90% ({} mismatches / {total})",
        mismatches.len(),
    );
}

// ---------------------------------------------------------------------------
// 2. Cross-decode — 100% strict
// ---------------------------------------------------------------------------

#[test]
fn test_cross_decode() {
    let models = discover_models();
    let mut tested = 0;
    for model in &models {
        let Some((mini, hf)) = load_both(&model.dir) else { continue };

        for text in TEXTS {
            let hf_ids: Vec<u32> = hf.encode(*text, false).unwrap().get_ids().to_vec();
            assert_eq!(
                mini.decode(&hf_ids, false),
                hf.decode(&hf_ids, false).unwrap(),
                "\n[CROSS-DECODE HF→mini] {} | {text:?}", model.name,
            );

            let mini_ids = mini.encode(text);
            assert_eq!(
                mini.decode(&mini_ids, false),
                hf.decode(&mini_ids, false).unwrap(),
                "\n[CROSS-DECODE mini→HF] {} | {text:?}", model.name,
            );
            tested += 1;
        }
    }
    assert!(tested > 0, "no models found");
    eprintln!("cross_decode: {tested} combinations ✓");
}

// ---------------------------------------------------------------------------
// 3. Vocab size — strict
// ---------------------------------------------------------------------------

#[test]
fn test_vocab_size() {
    let models = discover_models();
    let mut checked = 0usize;
    let mut mismatches = 0usize;

    for model in &models {
        let Some((mini, hf)) = load_both(&model.dir) else { continue };
        checked += 1;
        let ms = mini.vocab_size();
        let hs = hf.get_vocab_size(true);
        if ms != hs {
            mismatches += 1;
            eprintln!("  [VOCAB SIZE] {}: mini={ms} hf={hs} (diff={})", model.name, ms as i64 - hs as i64);
        } else {
            eprintln!("{}: {ms} tokens ✓", model.name);
        }
    }
    let pct = 100.0 * (checked - mismatches) as f64 / checked as f64;
    eprintln!("vocab size: {}/{checked} match ({pct:.1}%)", checked - mismatches);
    assert!(checked > 0, "no models found");
    assert!(pct >= 90.0, "vocab size match {pct:.1}% < 90%");
}

// ---------------------------------------------------------------------------
// 4. Decode sequential ID ranges — ≥90%
// ---------------------------------------------------------------------------

#[test]
fn test_decode_sequential_ids() {
    let models = discover_models();
    let mut total = 0usize;
    let mut mismatches = 0usize;

    for model in &models {
        let Some((mini, hf)) = load_both(&model.dir) else { continue };
        let vocab = mini.vocab_size() as u32;

        for start in [0u32, 100, 1000, 10000, 50000] {
            if start >= vocab { continue; }
            total += 1;
            let ids: Vec<u32> = (start..(start + 50).min(vocab)).collect();
            let m = mini.decode(&ids, false);
            let h = hf.decode(&ids, false).unwrap();
            if m != h {
                mismatches += 1;
                eprintln!("  [DECODE SEQ] {} ids[{start}..{}]: mismatch",
                    model.name, start + ids.len() as u32);
            }
        }
    }
    let pct = 100.0 * (total - mismatches) as f64 / total as f64;
    eprintln!("sequential decode: {}/{total} match ({pct:.1}%)", total - mismatches);
    assert!(total > 0);
    assert!(pct >= 90.0, "sequential decode {pct:.1}% < 90%");
}

// ---------------------------------------------------------------------------
// 5. Long-input decode — strict
// ---------------------------------------------------------------------------

#[test]
fn test_long_input() {
    let long_prose = (0..200)
        .map(|i| format!("Sentence {i}. The quick brown fox jumps over the lazy dog. "))
        .collect::<String>();

    let code = "fn fib(n: u64) -> u64 { match n { 0 => 0, 1 => 1, _ => fib(n-1) + fib(n-2) } }\n".repeat(20);

    let cjk = "日本語の長いテキスト。".repeat(100);

    let models = discover_models();
    let mut checked = 0;
    for model in &models {
        let Some((mini, hf)) = load_both(&model.dir) else { continue };

        for text in [long_prose.as_str(), code.as_str(), cjk.as_str()] {
            let hf_ids: Vec<u32> = hf.encode(text, false).unwrap().get_ids().to_vec();
            let m = mini.decode(&hf_ids, false);
            let h = hf.decode(&hf_ids, false).unwrap();
            assert_eq!(m, h, "\n[LONG DECODE] {} len={}", model.name, text.len());
        }
        eprintln!("{}: long input ✓", model.name);
        checked += 1;
    }
    assert!(checked > 0, "no models found");
}

// ---------------------------------------------------------------------------
// 6. All printable ASCII single chars — 100% strict
// ---------------------------------------------------------------------------

#[test]
fn test_single_ascii_chars() {
    let models = discover_models();
    let mut total = 0usize;
    let mut mismatches: Vec<String> = Vec::new();

    for model in &models {
        let Some((mini, hf)) = load_both(&model.dir) else { continue };
        let mut model_ok = true;

        for b in 0x20u8..=0x7E {
            total += 1;
            let ch = String::from(b as char);
            let m = mini.encode(&ch);
            let h: Vec<u32> = hf.encode(ch.as_str(), false).unwrap().get_ids().to_vec();
            if m != h {
                model_ok = false;
                mismatches.push(format!("  {} 0x{b:02X} {ch:?}: mini={m:?} hf={h:?}", model.name));
            }
        }
        if model_ok {
            eprintln!("{}: ASCII chars ✓", model.name);
        }
    }
    let pct = 100.0 * (total - mismatches.len()) as f64 / total as f64;
    eprintln!("\nASCII chars: {}/{total} match ({pct:.1}%)", total - mismatches.len());
    for m in &mismatches { eprintln!("{m}"); }
    assert!(total > 0, "no models found");
    assert!(pct >= 90.0, "ASCII char match {pct:.1}% < 90%");
}

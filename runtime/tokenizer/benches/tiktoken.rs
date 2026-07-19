//! Kimi tokenizer latency for Pie-style independent requests.
//!
//! Run: `cargo bench -p pie-tokenizer --bench tiktoken`

use std::collections::HashSet;
use std::hint::black_box;
use std::path::PathBuf;
use std::sync::Arc;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use pie_tokenizer::Tokenizer;

fn add_token(tokens: &mut Vec<Vec<u8>>, seen: &mut HashSet<Vec<u8>>, token: &str) {
    let bytes = token.as_bytes().to_vec();
    if seen.insert(bytes.clone()) {
        tokens.push(bytes);
    }
}

fn make_kimi_tokenizer() -> (tempfile::TempDir, PathBuf, Arc<Tokenizer>) {
    let dir = tempfile::tempdir().unwrap();
    let model_path = dir.path().join("tiktoken.model");
    let mut tokens = (0u16..=255)
        .map(|byte| vec![byte as u8])
        .collect::<Vec<_>>();
    let mut seen = tokens.iter().cloned().collect::<HashSet<_>>();

    for token in [
        "Hello",
        ",",
        " world",
        "!",
        "\n",
        "user",
        "short",
        " request",
        "짧은",
        " 한국어",
        " 요청",
        "12",
        "123",
        "1234",
    ] {
        add_token(&mut tokens, &mut seen, token);
    }

    let mut model = String::new();
    for (rank, bytes) in tokens.iter().enumerate() {
        model.push_str(&format!("{} {rank}\n", STANDARD.encode(bytes.as_slice())));
    }
    std::fs::write(&model_path, model).unwrap();

    let base_vocab_size = tokens.len() as u32;
    let config = serde_json::json!({
        "tokenizer_class": "TikTokenTokenizer",
        "auto_map": {
            "AutoTokenizer": ["tokenization_kimi.TikTokenTokenizer", null]
        },
        "added_tokens_decoder": {
            base_vocab_size.to_string(): {
                "content": "<|im_user|>",
                "special": true
            },
            (base_vocab_size + 1).to_string(): {
                "content": "<|im_end|>",
                "special": true
            }
        }
    });
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();

    let tokenizer = Arc::new(Tokenizer::from_file(&model_path).unwrap());
    (dir, model_path, tokenizer)
}

fn bench_kimi(c: &mut Criterion) {
    let (_dir, model_path, tokenizer) = make_kimi_tokenizer();
    let short_ascii = "Hello, world!";
    let short_multilingual = "짧은 한국어 요청";
    let special_message = "<|im_user|>Hello, world!<|im_end|>";
    let prompt = "Hello, world! 짧은 한국어 요청\n".repeat(128);

    let mut requests = c.benchmark_group("kimi_independent_request");
    requests.throughput(Throughput::Elements(1));
    requests.bench_function("short_ascii", |b| {
        b.iter(|| black_box(tokenizer.encode(black_box(short_ascii))));
    });
    requests.bench_function("short_multilingual", |b| {
        b.iter(|| black_box(tokenizer.encode(black_box(short_multilingual))));
    });
    requests.bench_function("special_message", |b| {
        b.iter(|| black_box(tokenizer.encode(black_box(special_message))));
    });
    requests.finish();

    let mut prompts = c.benchmark_group("kimi_prompt");
    prompts.throughput(Throughput::Bytes(prompt.len() as u64));
    prompts.bench_function("4k", |b| {
        b.iter(|| black_box(tokenizer.encode(black_box(&prompt))));
    });
    prompts.finish();

    let ids = tokenizer.encode(&prompt);
    c.bench_function("kimi_decode/4k", |b| {
        b.iter(|| black_box(tokenizer.decode(black_box(&ids), false)));
    });
    c.bench_function("kimi_incremental_decode/4k", |b| {
        b.iter(|| {
            let mut decoder = tokenizer.decoder(false);
            for token in &ids {
                black_box(decoder.feed(std::slice::from_ref(token)));
            }
            black_box(decoder.finish())
        });
    });
    c.bench_function("kimi_load/synthetic", |b| {
        b.iter(|| black_box(Tokenizer::from_file(black_box(&model_path)).unwrap()));
    });

    if let Some(dir) = std::env::var_os("PIE_KIMI_TOKENIZER_DIR") {
        let path = PathBuf::from(dir).join("tiktoken.model");
        let official = Arc::new(Tokenizer::from_file(&path).unwrap());
        let official_ids = official.encode(&prompt);
        c.bench_function("kimi_official/encode_short", |b| {
            b.iter(|| black_box(official.encode(black_box(short_ascii))));
        });
        c.bench_function("kimi_official/encode_4k", |b| {
            b.iter(|| black_box(official.encode(black_box(&prompt))));
        });
        c.bench_function("kimi_official/decode_4k", |b| {
            b.iter(|| black_box(official.decode(black_box(&official_ids), false)));
        });
        c.bench_function("kimi_official/load", |b| {
            b.iter(|| black_box(Tokenizer::from_file(black_box(&path)).unwrap()));
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(2));
    targets = bench_kimi
}
criterion_main!(benches);

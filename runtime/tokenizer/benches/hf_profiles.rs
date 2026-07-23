//! Modern Hugging Face profile latency for Pie-style requests.
//!
//! Run: `cargo bench -p pie-tokenizer --bench hf_profiles`

#[path = "../tests/common/mod.rs"]
mod common;

use std::hint::black_box;
use std::path::Path;
use std::sync::Arc;

use common::{MergeFormat, byte_level_json, gemma_json};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use pie_tokenizer::Tokenizer;
use serde_json::json;

fn load(value: serde_json::Value) -> (String, Arc<Tokenizer>) {
    let source = value.to_string();
    let tokenizer = Arc::new(source.parse().unwrap());
    (source, tokenizer)
}

fn bench_hf_profiles(c: &mut Criterion) {
    let (qwen_source, qwen) = load(byte_level_json(
        json!({"type": "NFC"}),
        &[r"\p{N}|[\p{L}\p{M}]+|[^\p{L}\p{M}\p{N}]+"],
        false,
        MergeFormat::String,
        false,
    ));
    let (deepseek_source, deepseek) = load(byte_level_json(
        json!({"type": "Sequence", "normalizers": []}),
        &[
            r"\p{N}{1,3}",
            r"[一-龥぀-ゟ゠-ヿ]+",
            r"[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s+",
        ],
        false,
        MergeFormat::Tuple,
        true,
    ));
    let (glm_source, glm_nemotron) = load(byte_level_json(
        serde_json::Value::Null,
        &[r"\p{N}{1,3}|[^\p{N}]+"],
        true,
        MergeFormat::Tuple,
        false,
    ));
    let (gemma_source, gemma) = load(gemma_json());

    let short = "Hello 你好 1234!";
    let mut requests = c.benchmark_group("hf_profile_independent_request");
    requests.throughput(Throughput::Elements(1));
    for (name, tokenizer) in [
        ("qwen", qwen.as_ref()),
        ("deepseek", deepseek.as_ref()),
        ("glm_nemotron", glm_nemotron.as_ref()),
        ("gemma", gemma.as_ref()),
    ] {
        requests.bench_function(name, |b| {
            b.iter(|| black_box(tokenizer.encode(black_box(short))));
        });
    }
    requests.finish();

    let prompt = "Hello 你好 1234! Structured output and tool calls.\n".repeat(96);
    let mut prompts = c.benchmark_group("hf_profile_4k_prompt");
    prompts.throughput(Throughput::Bytes(prompt.len() as u64));
    for (name, tokenizer) in [
        ("qwen", qwen.as_ref()),
        ("deepseek", deepseek.as_ref()),
        ("glm_nemotron", glm_nemotron.as_ref()),
        ("gemma", gemma.as_ref()),
    ] {
        prompts.bench_function(name, |b| {
            b.iter(|| black_box(tokenizer.encode(black_box(&prompt))));
        });
    }
    prompts.finish();

    let encoded = [
        ("qwen", &qwen, qwen.encode(&prompt)),
        ("deepseek", &deepseek, deepseek.encode(&prompt)),
        ("glm_nemotron", &glm_nemotron, glm_nemotron.encode(&prompt)),
        ("gemma", &gemma, gemma.encode(&prompt)),
    ];
    let mut decode = c.benchmark_group("hf_profile_decode_4k");
    for (name, tokenizer, ids) in &encoded {
        decode.bench_function(*name, |b| {
            b.iter(|| black_box(tokenizer.decode(black_box(ids), false)));
        });
    }
    decode.finish();

    let mut incremental = c.benchmark_group("hf_profile_incremental_decode_4k");
    for (name, tokenizer, ids) in &encoded {
        incremental.bench_function(*name, |b| {
            b.iter(|| {
                let mut decoder = tokenizer.decoder(false);
                for token in ids {
                    black_box(decoder.feed(std::slice::from_ref(token)));
                }
                black_box(decoder.finish())
            });
        });
    }
    incremental.finish();

    let mut loading = c.benchmark_group("hf_profile_load");
    for (name, source) in [
        ("qwen", &qwen_source),
        ("deepseek", &deepseek_source),
        ("glm_nemotron", &glm_source),
        ("gemma", &gemma_source),
    ] {
        loading.bench_function(name, |b| {
            b.iter(|| black_box(source.parse::<Tokenizer>().unwrap()));
        });
    }
    loading.finish();

    bench_official_profiles(c, short, &prompt);
}

fn bench_official_profiles(c: &mut Criterion, short: &str, prompt: &str) {
    let Some(root) = std::env::var_os("PIE_TOKENIZER_BENCH_FIXTURES_DIR") else {
        return;
    };
    for fixture in ["qwen36", "deepseek-v4", "gemma4", "glm52", "nemotron3"] {
        let path = Path::new(&root).join(fixture).join("tokenizer.json");
        let tokenizer = Arc::new(Tokenizer::from_file(&path).unwrap());
        let ids = tokenizer.encode(prompt);

        c.bench_function(&format!("official_encode_short/{fixture}"), |b| {
            b.iter(|| black_box(tokenizer.encode(black_box(short))));
        });
        c.bench_function(&format!("official_encode_4k/{fixture}"), |b| {
            b.iter(|| black_box(tokenizer.encode(black_box(prompt))));
        });
        c.bench_function(&format!("official_decode_4k/{fixture}"), |b| {
            b.iter(|| black_box(tokenizer.decode(black_box(&ids), false)));
        });
        c.bench_function(&format!("official_load/{fixture}"), |b| {
            b.iter(|| black_box(Tokenizer::from_file(black_box(&path)).unwrap()));
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(2));
    targets = bench_hf_profiles
}
criterion_main!(benches);

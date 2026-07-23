//! Single-request grammar benchmarks.
//!
//! Run:
//!   cargo bench -p pie-grammar --bench grammar
//!   PIE_GRAMMAR_BENCH_VOCAB=151936 cargo bench -p pie-grammar --bench grammar
//!
//! These benchmarks cover the native grammar crate. WIT list transfer and
//! sampler binding costs belong in an engine-level benchmark.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use pie_grammar::bitmask;
use pie_grammar::compiled_grammar::CompiledGrammar;
use pie_grammar::compiler::GrammarCompiler;
use pie_grammar::grammar::Grammar;
use pie_grammar::json_schema::{JsonSchemaOptions, json_schema_to_grammar};
use pie_grammar::matcher::GrammarMatcher;
use pie_grammar::regex::regex_to_grammar;
use pie_tokenizer::Tokenizer;

const EBNF: &str = r#"
root ::= "[" person ("," person)* "]"
person ::= "{" "\"name\"" ":" string "," "\"age\"" ":" number "}"
string ::= "\"" [^"\\]* "\""
number ::= [0-9]+
"#;

const JSON_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "integer", "minimum": 0, "maximum": 130}
  },
  "required": ["name", "age"],
  "additionalProperties": false
}"#;

const OPTIONAL_JSON_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "a": {"type": "integer"},
    "b": {"type": "string"},
    "c": {"type": "boolean"},
    "d": {"type": "number"},
    "e": {"type": "null"},
    "f": {"type": "integer"}
  },
  "minProperties": 2,
  "maxProperties": 5,
  "additionalProperties": false
}"#;

const REGEX: &str = r#"[a-zA-Z_][a-zA-Z0-9_]{0,31}"#;
const UNICODE_REGEX: &str = "ww我😁";
const JSON_SAMPLE: &str = r#"{"name":"alice","age":42}"#;

fn benchmark_vocab_size() -> usize {
    std::env::var("PIE_GRAMMAR_BENCH_VOCAB")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&size| size >= 128)
        .unwrap_or(8_192)
}

fn synthetic_tokenizer(vocab_size: usize) -> Arc<Tokenizer> {
    let mut vocab: Vec<String> = (0u8..=127).map(|byte| String::from(byte as char)).collect();
    vocab.extend((vocab.len()..vocab_size).map(|id| format!("token_{id:06x}")));
    Arc::new(Tokenizer::from_vocab(&vocab))
}

fn token_ids(tokenizer: &Tokenizer, input: &str) -> Vec<u32> {
    input
        .bytes()
        .map(|byte| {
            tokenizer
                .token_to_id(&String::from(byte as char))
                .expect("benchmark vocabulary contains every ASCII byte")
        })
        .collect()
}

fn bench_frontends(c: &mut Criterion) {
    let mut group = c.benchmark_group("grammar/frontend");

    group.bench_function("ebnf", |b| {
        b.iter(|| Grammar::from_ebnf(black_box(EBNF), "root").unwrap())
    });
    group.bench_function("regex_ascii", |b| {
        b.iter(|| regex_to_grammar(black_box(REGEX)).unwrap())
    });
    group.bench_function("regex_unicode", |b| {
        b.iter(|| regex_to_grammar(black_box(UNICODE_REGEX)).unwrap())
    });
    group.bench_function("json_schema", |b| {
        b.iter(|| {
            json_schema_to_grammar(black_box(JSON_SCHEMA), &JsonSchemaOptions::default()).unwrap()
        })
    });
    group.bench_function("json_schema_optional", |b| {
        b.iter(|| {
            json_schema_to_grammar(
                black_box(OPTIONAL_JSON_SCHEMA),
                &JsonSchemaOptions::default(),
            )
            .unwrap()
        })
    });

    group.finish();
}

fn bench_compile(c: &mut Criterion) {
    let vocab_size = benchmark_vocab_size();
    let tokenizer = synthetic_tokenizer(vocab_size);
    let ebnf = Grammar::from_ebnf(EBNF, "root").unwrap();
    let schema = json_schema_to_grammar(JSON_SCHEMA, &JsonSchemaOptions::default()).unwrap();
    let optional_schema =
        json_schema_to_grammar(OPTIONAL_JSON_SCHEMA, &JsonSchemaOptions::default()).unwrap();

    let mut group = c.benchmark_group("grammar/compile");
    group.sample_size(10);

    group.bench_with_input(BenchmarkId::new("ebnf", vocab_size), &vocab_size, |b, _| {
        b.iter(|| CompiledGrammar::new(black_box(&ebnf), black_box(&tokenizer)))
    });
    group.bench_with_input(
        BenchmarkId::new("json_schema", vocab_size),
        &vocab_size,
        |b, _| b.iter(|| CompiledGrammar::new(black_box(&schema), black_box(&tokenizer))),
    );
    group.bench_with_input(
        BenchmarkId::new("json_schema_optional", vocab_size),
        &vocab_size,
        |b, _| b.iter(|| CompiledGrammar::new(black_box(&optional_schema), black_box(&tokenizer))),
    );

    group.finish();
}

fn bench_matcher_creation(c: &mut Criterion) {
    let vocab_size = benchmark_vocab_size();
    let tokenizer = synthetic_tokenizer(vocab_size);
    let compiler = GrammarCompiler::new(tokenizer.clone());
    let options = JsonSchemaOptions::default();
    let compiled = compiler.compile_json_schema(JSON_SCHEMA, &options).unwrap();

    let mut group = c.benchmark_group("grammar/request_setup");
    group.bench_with_input(
        BenchmarkId::new("compiled_cache_hit", vocab_size),
        &vocab_size,
        |b, _| b.iter(|| compiler.compile_json_schema(black_box(JSON_SCHEMA), black_box(&options))),
    );
    group.bench_with_input(
        BenchmarkId::new("matcher_from_compiled", vocab_size),
        &vocab_size,
        |b, _| b.iter(|| GrammarMatcher::with_compiled(black_box(compiled.clone()), Vec::new(), 0)),
    );
    group.bench_with_input(
        BenchmarkId::new("warm_json_schema", vocab_size),
        &vocab_size,
        |b, _| {
            b.iter(|| {
                let compiled = compiler
                    .compile_json_schema(black_box(JSON_SCHEMA), black_box(&options))
                    .unwrap();
                GrammarMatcher::with_compiled(compiled, Vec::new(), 0)
            })
        },
    );
    group.finish();
}

fn bench_mask_generation(c: &mut Criterion) {
    let vocab_size = benchmark_vocab_size();
    let tokenizer = synthetic_tokenizer(vocab_size);
    let grammar = json_schema_to_grammar(JSON_SCHEMA, &JsonSchemaOptions::default()).unwrap();
    let compiled = Arc::new(CompiledGrammar::new(&grammar, &tokenizer));
    let mut matcher = GrammarMatcher::with_compiled(compiled, Vec::new(), 0);
    let mut bitmask = vec![0u32; bitmask::bitmask_size(vocab_size)];

    matcher.fill_next_token_bitmask(&mut bitmask);

    let mut group = c.benchmark_group("grammar/mask_cached");
    group.throughput(Throughput::Bytes((bitmask.len() * size_of::<u32>()) as u64));
    group.bench_function("borrowed", |b| {
        b.iter(|| matcher.fill_next_token_bitmask(black_box(&mut bitmask)))
    });
    group.bench_function("owned_wit_shape", |b| {
        b.iter(|| black_box(matcher.fill_next_token_mask()))
    });
    group.finish();
}

fn bench_sequential_request(c: &mut Criterion) {
    let vocab_size = benchmark_vocab_size();
    let tokenizer = synthetic_tokenizer(vocab_size);
    let grammar = json_schema_to_grammar(JSON_SCHEMA, &JsonSchemaOptions::default()).unwrap();
    let compiled = Arc::new(CompiledGrammar::new(&grammar, &tokenizer));
    let input_ids = token_ids(&tokenizer, JSON_SAMPLE);
    let mask_words = bitmask::bitmask_size(vocab_size);

    let mut group = c.benchmark_group("grammar/sequential_request");
    group.throughput(Throughput::Elements(input_ids.len() as u64));
    group.bench_function(
        BenchmarkId::new("mask_and_accept", format!("{vocab_size}_vocab")),
        |b| {
            b.iter_batched(
                || {
                    (
                        GrammarMatcher::with_compiled(compiled.clone(), Vec::new(), 0),
                        vec![0u32; mask_words],
                    )
                },
                |(mut matcher, mut mask)| {
                    for &token_id in &input_ids {
                        matcher.fill_next_token_bitmask(&mut mask);
                        black_box(&mask);
                        assert!(matcher.accept_token(token_id));
                    }
                    assert!(matcher.can_terminate());
                },
                BatchSize::SmallInput,
            )
        },
    );
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    targets =
        bench_frontends,
        bench_compile,
        bench_matcher_creation,
        bench_mask_generation,
        bench_sequential_request
}
criterion_main!(benches);

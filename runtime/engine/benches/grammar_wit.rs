//! Actual Wasmtime/WIT grammar-call benchmark.
//!
//! Usage:
//!   cargo bench -p pie-engine --bench grammar_wit
//!   PIE_GRAMMAR_WIT_VOCAB=151936 cargo bench -p pie-engine --bench grammar_wit

#[path = "../tests/common/env.rs"]
mod env;
#[allow(dead_code)]
#[path = "../tests/common/inferlets.rs"]
mod inferlets;
#[allow(dead_code)]
#[path = "../tests/common/mock_device.rs"]
mod mock_device;

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use env::{MockEnv, create_mock_env};
use mock_device::EchoBehavior;
use pie_engine::inferlet::process;
use pie_engine::inferlet::program::{self, ProgramName};
use tempfile::TempDir;

const INFERLET: &str = "grammar-wit-bench";

struct BenchState {
    #[allow(dead_code)]
    env: MockEnv,
    #[allow(dead_code)]
    tokenizer_dir: TempDir,
    rt: tokio::runtime::Runtime,
    program: ProgramName,
}

fn synthetic_tokenizer(vocab_size: usize) -> (TempDir, PathBuf) {
    assert!(vocab_size >= 256);
    let dir = TempDir::new().expect("create tokenizer tempdir");
    let path = dir.path().join("tokenizer.json");
    let mut fixture: serde_json::Value =
        serde_json::from_str(include_str!("../tests/common/fixtures/test_tokenizer.json")).unwrap();
    fixture["pre_tokenizer"] = serde_json::json!({
        "type": "Sequence",
        "pretokenizers": [
            {
                "type": "Split",
                "pattern": {"Regex": ".+"},
                "behavior": "Isolated",
                "invert": false
            },
            {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": false
            }
        ]
    });

    fixture["added_tokens"] = serde_json::Value::Array(Vec::new());
    let mut vocab = serde_json::Map::new();
    for byte in 0u16..=255 {
        vocab.insert(gpt2_byte_char(byte as u8).to_string(), byte.into());
    }
    for id in 256..vocab_size {
        vocab.insert(format!("token_{id:06x}"), id.into());
    }

    fn gpt2_byte_char(byte: u8) -> char {
        if matches!(byte, 0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff) {
            return char::from_u32(byte as u32).unwrap();
        }
        let offset = (0u16..=byte as u16)
            .filter(|value| !matches!(*value as u8, 0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff))
            .count() as u32
            - 1;
        char::from_u32(0x100 + offset).unwrap()
    }
    fixture["model"]["vocab"] = vocab.into();
    fixture["model"]["merges"] = serde_json::Value::Array(Vec::new());
    std::fs::write(&path, serde_json::to_vec(&fixture).unwrap()).unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::to_vec(&serde_json::json!({
            "vocab_size": vocab_size,
            "num_hidden_layers": 1
        }))
        .unwrap(),
    )
    .unwrap();
    (dir, path)
}

fn build_release_inferlet() -> Vec<u8> {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/inferlets");
    let status = Command::new("cargo")
        .args([
            "build",
            "--release",
            "--target",
            "wasm32-wasip2",
            "-p",
            INFERLET,
        ])
        .current_dir(&workspace)
        .status()
        .expect("build benchmark inferlet");
    assert!(status.success(), "benchmark inferlet build failed");
    std::fs::read(
        workspace
            .join("target/wasm32-wasip2/release")
            .join("grammar_wit_bench.wasm"),
    )
    .unwrap()
}

fn state(vocab_size: usize) -> BenchState {
    let (tokenizer_dir, tokenizer_path) = synthetic_tokenizer(vocab_size);
    let rt = tokio::runtime::Runtime::new().unwrap();
    let env = create_mock_env("grammar-wit-bench-model", 1, 16, Arc::new(EchoBehavior(42)));
    let mut config = env.config();
    config.model.tokenizer_path = tokenizer_path;
    let wasm = build_release_inferlet();
    let manifest = inferlets::read_inferlet_manifest(INFERLET);
    let program = ProgramName::parse("grammar-wit-bench@0.1.0").unwrap();

    rt.block_on(async {
        pie_engine::bootstrap::bootstrap(config).await.unwrap();
        program::add(wasm, manifest, true).await.unwrap();
        program::install(&program).await.unwrap();
    });
    BenchState {
        env,
        tokenizer_dir,
        rt,
        program,
    }
}

fn run(state: &BenchState, iterations: usize, rounds: usize) -> String {
    let input = serde_json::json!({
        "iterations": iterations,
        "rounds": rounds,
    })
    .to_string();
    let receiver = state.rt.block_on(async {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        process::spawn(
            "grammar-bench".into(),
            state.program.clone(),
            input,
            None,
            false,
            Some(sender),
        )
        .unwrap();
        receiver
    });
    state.rt.block_on(async {
        tokio::time::timeout(Duration::from_secs(120), receiver)
            .await
            .expect("grammar WIT benchmark timed out")
            .expect("result channel dropped")
            .expect("benchmark inferlet failed")
    })
}

fn main() {
    let vocab_size = std::env::var("PIE_GRAMMAR_WIT_VOCAB")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(151_936);
    let iterations = std::env::var("PIE_GRAMMAR_WIT_ITERATIONS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1_000);
    let rounds = std::env::var("PIE_GRAMMAR_WIT_ROUNDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(7);

    let state = state(vocab_size);
    let result = run(&state, iterations, rounds);
    println!("grammar_wit vocab={vocab_size} {result}");
}

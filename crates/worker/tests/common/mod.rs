#![allow(dead_code)]

use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use ::runtime::inferlet::program::ProgramName;
use worker::WorkerHandle;

fn snapshot_in_hub(repo: &str) -> String {
    let home = std::env::var("HF_HOME").unwrap_or_else(|_| {
        format!("{}/.cache/huggingface", std::env::var("HOME").unwrap_or_default())
    });
    let hub = format!("{home}/hub/models--{}", repo.replace('/', "--"));
    let snapshots = std::path::Path::new(&hub).join("snapshots");
    std::fs::read_dir(&snapshots)
        .ok()
        .and_then(|mut dir| dir.next().and_then(Result::ok))
        .map(|entry| entry.path().to_string_lossy().into_owned())
        .unwrap_or_else(|| snapshots.to_string_lossy().into_owned())
}

pub const DEFAULT_REPO: &str = "Qwen/Qwen3-0.6B";

pub const DEFAULT_GDN_REPO: &str = "Qwen/Qwen3.5-0.8B";

pub fn snapshot() -> String {
    std::env::var("PIE_CUDA_TEST_SNAPSHOT").unwrap_or_else(|_| snapshot_in_hub(DEFAULT_REPO))
}

pub fn gdn_snapshot() -> String {
    std::env::var("PIE_CUDA_TEST_GDN_SNAPSHOT").unwrap_or_else(|_| snapshot_in_hub(DEFAULT_GDN_REPO))
}

pub fn cuda_toml_for(snapshot_path: &str) -> String {
    let scratch = std::env::temp_dir().join("pie-cuda-test-scratch");
    let _ = std::fs::create_dir_all(&scratch);
    let kv = std::env::var("PIE_CUDA_TEST_KV_PAGES")
        .map(|v| format!("max_total_pages = {v}\n"))
        .unwrap_or_default();
    format!(
        "[server]\n\
         host = \"127.0.0.1\"\n\
         port = 0\n\
         \n\
         [sandbox]\n\
         allow_fs = true\n\
         fs_scratch_dir = \"{scratch}\"\n\
         \n\
         \n\
         [model]\n\
         name = \"default\"\n\
         model = \"{snapshot_path}\"\n\
         \n\
         [engine]\n\
         type = \"cuda_native\"\n\
         device = [\"cuda:0\"]\n\
         gpu_mem_utilization = 0.90\n\
         {kv}",
        scratch = scratch.display(),
    )
}

pub fn cuda_toml() -> String {
    cuda_toml_for(&snapshot())
}

fn wire_tracing() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("error"));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(std::io::stderr)
            .try_init();
    });
}

pub async fn boot_cuda_model(snapshot_path: &str) -> WorkerHandle {
    wire_tracing();
    let cfg =
        worker::Config::parse(&cuda_toml_for(snapshot_path)).expect("parse cuda worker config");
    worker::run(cfg).await.expect("boot embedded cuda engine")
}

pub async fn boot_cuda() -> WorkerHandle {
    boot_cuda_model(&snapshot()).await
}

pub fn load_curated_inferlet(name: &str) -> (Vec<u8>, ProgramName) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples")
        .join(name);
    assert!(
        dir.is_dir(),
        "no inferlet fixture at {} -- this is the path, not `cargo`",
        dir.display()
    );
    let status = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "--release"])
        .current_dir(&dir)
        .status()
        .unwrap_or_else(|e| panic!("spawn cargo build for {name}: {e}"));
    assert!(status.success(), "build {name} failed");

    let wasm_path = dir
        .join("target/wasm32-wasip2/release")
        .join(format!("{}.wasm", name.replace('-', "_")));
    let wasm_path = if wasm_path.exists() {
        wasm_path
    } else {
        dir.join("../target/wasm32-wasip2/release")
            .join(format!("{}.wasm", name.replace('-', "_")))
    };
    let wasm =
        std::fs::read(&wasm_path).unwrap_or_else(|e| panic!("read {}: {e}", wasm_path.display()));
    let program_name = ::runtime::inferlet::program::identify(
        &format!("{name}.wasm"),
        None,
        &wasm,
    )
    .expect("the example names itself")
    .name;
    (wasm, program_name)
}

pub async fn install_inferlet(name: &str) -> ProgramName {
    let (wasm, program_name) = load_curated_inferlet(name);
    let installed = ::runtime::inferlet::program::add(wasm, &format!("{name}.wasm"), None, true)
        .await
        .expect("add program");
    assert_eq!(installed, program_name);
    ::runtime::inferlet::program::install(&program_name)
        .await
        .expect("install program");
    program_name
}

pub async fn spawn_text(
    program: &ProgramName,
    prompt: &str,
    max_tokens: usize,
) -> Result<String, String> {
    let input = format!(r#"{{"prompt":{prompt:?},"max_tokens":{max_tokens}}}"#);
    spawn_input(program, &input).await
}

pub async fn spawn_input(program: &ProgramName, input_json: &str) -> Result<String, String> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    ::runtime::inferlet::process::spawn(
        "cuda-test".into(),
        program.clone(),
        input_json.to_string(),
        None,
        false,
        Some(tx),
    )
    .expect("spawn process");
    match tokio::time::timeout(Duration::from_secs(180), rx).await {
        Err(_) => Err("no answer within 180s".to_string()),
        Ok(result) => result.expect("process result channel dropped"),
    }
}

pub async fn spawn_inferlet(name: &str, input_json: &str) -> Result<String, String> {
    let program = install_inferlet(name).await;
    spawn_input(&program, input_json).await
}

pub fn assert_coherent(text: &str, min_words: usize) {
    let words = text
        .split(|c: char| !c.is_alphabetic())
        .filter(|w| w.chars().count() >= 3)
        .count();
    let dense: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    let alnum = dense.iter().filter(|c| c.is_alphanumeric()).count();
    assert!(
        words >= min_words,
        "completion has {words} word(s) of 3+ letters, wanted {min_words}: {text:?}"
    );
    assert!(
        !dense.is_empty() && alnum * 5 >= dense.len() * 2,
        "completion is {alnum}/{} alphanumeric, wanted two fifths: {text:?}",
        dense.len()
    );
}

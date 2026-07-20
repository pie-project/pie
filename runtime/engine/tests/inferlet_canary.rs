//! Phase-6 canary: representative curated inferlet fixtures RUN on the mock driver.
//!
//! Complements `e2e.rs` (the small test inferlets in `tests/inferlets/`): this
//! builds a few realistic inferlets from the repository test fixtures and spawns them through the
//! full stack — SDK `Context` facade → `kv-working-set` → forward descriptors →
//! mock `EchoBehavior` driver — capturing the **process result** so we assert the
//! inferlet actually *succeeded* (not merely "completed", which is also true on
//! error). The mock echoes a constant in-vocab token, so this validates the
//! pipeline runs cleanly end to end (no trap / hang / host error), not model
//! output correctness.

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

mod common;
use common::{MockEnv, create_mock_env, mock_device::EchoBehavior};

use pie_engine::inferlet::process;
use pie_engine::inferlet::program::{Manifest, ProgramName};
use tokio::sync::oneshot;

/// Per-process bootstrap (LazyLock-backed runtime globals → bootstrap exactly once).
struct Harness {
    rt: tokio::runtime::Runtime,
    #[allow(dead_code)]
    env: MockEnv, // keep the mock backend (RPC servers) alive
    #[allow(dead_code)]
    fs_tmp: tempfile::TempDir, // backs the wasm fs preopen (snapshot blobs)
}

static HARNESS: OnceLock<Harness> = OnceLock::new();

fn harness() -> &'static Harness {
    HARNESS.get_or_init(|| {
        // EchoBehavior token must be in-vocab for the 20-token test tokenizer and
        // not `<eos>` (id 0) / `<bos>` (id 1), so decode succeeds and generation
        // runs to `max_tokens` rather than erroring or stopping immediately.
        let env = create_mock_env("test-model", 1, 256, Arc::new(EchoBehavior(7)));
        // Grant the wasm instance a filesystem preopen so the replay-snapshot
        // facade can write/read its `/scratch` blobs.
        let fs_tmp = tempfile::TempDir::new().unwrap();
        let mut config = env.config();
        config.runtime.allow_fs = true;
        config.runtime.fs_scratch_dir = fs_tmp.path().to_path_buf();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            pie_engine::bootstrap::bootstrap(config).await.unwrap();
        });
        Harness { rt, env, fs_tmp }
    })
}

/// Build a curated inferlet fixture → wasm + manifest + id.
fn load_curated_inferlet(name: &str) -> (Vec<u8>, Manifest, ProgramName) {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/inferlets");
    let dir = workspace.join(name);
    let status = Command::new("cargo")
        .args([
            "build",
            "--target",
            "wasm32-wasip2",
            "--release",
            "-p",
            name,
        ])
        .current_dir(&workspace)
        .status()
        .unwrap_or_else(|e| panic!("spawn cargo build for {name}: {e}"));
    assert!(status.success(), "build {name} failed");

    let wasm_path = workspace
        .join("target/wasm32-wasip2/release")
        .join(format!("{}.wasm", name.replace('-', "_")));
    let wasm =
        std::fs::read(&wasm_path).unwrap_or_else(|e| panic!("read {}: {e}", wasm_path.display()));
    let manifest =
        Manifest::parse(&std::fs::read_to_string(dir.join("Pie.toml")).unwrap()).unwrap();
    let program_name = ProgramName::parse(&format!("{name}@{}", manifest.package.version)).unwrap();
    (wasm, manifest, program_name)
}

/// Build + install + spawn a curated inferlet fixture on the mock driver and return its
/// **result** (`Ok(output)` on success, `Err(msg)` on a host/inferlet error).
/// Panics only if it doesn't finish within `timeout`.
fn run_curated_inferlet(name: &str, input: &str, timeout: Duration) -> Result<String, String> {
    let h = harness();
    let (wasm, manifest, program_name) = load_curated_inferlet(name);
    h.rt.block_on(async {
        pie_engine::inferlet::program::add(wasm, manifest, true)
            .await
            .unwrap();
        pie_engine::inferlet::program::install(&program_name)
            .await
            .unwrap();
        let (tx, rx) = oneshot::channel();
        let _pid = process::spawn(
            "test-user".into(),
            program_name,
            input.into(),
            None,
            false,
            Some(tx),
        )
        .expect("spawn");
        tokio::time::timeout(timeout, rx)
            .await
            .expect("inferlet should finish within timeout")
            .expect("process result channel dropped")
    })
}

/// Assert an inferlet ran to a successful result, surfacing the error otherwise.
fn assert_ok(name: &str, result: Result<String, String>) {
    if let Err(e) = &result {
        panic!("{name} errored on the mock driver: {e}");
    }
}

/// Masked-decode canary — attention-sink (echo, SDK-minimization ①): the
/// raw-WIT / keep-core rewrite (no `Context`/`Sampler` facade) drives a decode
/// loop that attaches a position-deterministic sink+window `attention_mask` in
/// the `carrier::submit_pass_with` bind seam. A small `window_size` makes the
/// mask path fire within a few steps, so this exercises the mask attach +
/// forward on the mock end to end (not just the no-mask fallback).
#[test]
#[ignore = "author-managed device-geometry KV (flat CSR pages, rolling-window \
            overwrite in a fixed pool) with device-carried tokens has no engine \
            execution class yet: the Host class cannot advance past a device \
            sample (pending RV-3 committed-value echo) and the DecodeEnvelope \
            class requires [lanes,pages] matrix pages"]
fn attention_sink_masked_decode_runs() {
    assert_ok(
        "attention-sink",
        run_curated_inferlet(
            "attention-sink",
            r#"{"prompt":"hi","max_tokens":6,"sink_size":1,"window_size":2}"#,
            Duration::from_secs(60),
        ),
    );
}

/// Masked-decode canary — sliding-window-attention: same
/// keep-core rewrite, sliding-window mask via the `submit_pass_with` bind seam.
/// A small `window_size` fires the mask path within a few steps.
#[test]
#[ignore = "same author-managed device-geometry KV gap as attention-sink \
            (pending RV-3 committed-value echo)"]
fn sliding_window_attention_masked_decode_runs() {
    assert_ok(
        "sliding-window-attention",
        run_curated_inferlet(
            "sliding-window-attention",
            r#"{"prompt":"hi","max_tokens":6,"window_size":2}"#,
            Duration::from_secs(60),
        ),
    );
}

/// Benchmark-inferlet canary — text-completion-bench (the V5 reference
/// workload): chat-templated prefill, in-graph greedy sampling, device
/// loop-carried decode submitted `DEFAULT_RUNAHEAD_DEPTH` ahead, exact token
/// accounting in the Return envelope. `ignore_eos` keeps the run at exactly
/// `max_tokens` (the mock echoes a constant token, which would otherwise
/// depend on the chat template's stop set).
#[test]
fn text_completion_bench_runs() {
    let result = run_curated_inferlet(
        "text-completion-bench",
        r#"{"prompt":"hi","max_tokens":6,"temperature":0.0,"ignore_eos":true}"#,
        Duration::from_secs(60),
    );
    assert_ok("text-completion-bench", result.clone());
    let out = result.unwrap();
    assert!(
        out.contains("\"num_output_tokens\":6"),
        "expected exactly max_tokens emissions, got: {out}"
    );
}

/// In-band `-1` validity canary — MTP speculative decoding: a fixed `k+1`
/// verify-and-extend window rides a loop-carried envelope where shape decides
/// slots and `-1` decides existence (RV-12). The committed-token output is
/// `-1`-padded per window and acceptance is device-decided (eq/cumprod in
/// stage), so this exercises the envelope classification, the mock's envelope
/// execution, and TOKEN_PAD unpadding end to end. The mock's MTP drafts echo a
/// constant token, so acceptance actually varies across rounds.
#[test]
fn mtp_speculative_decoding_pads_and_verifies() {
    assert_ok(
        "mtp-speculative-decoding",
        run_curated_inferlet(
            "mtp-speculative-decoding",
            r#"{"prompt":"hi","max_tokens":8}"#,
            Duration::from_secs(120),
        ),
    );
}

/// Chat completion canary: runs prompt prefill, top-p sampling, and decode on
/// the mock driver.
#[test]
#[ignore = "same author-managed device-geometry KV gap as attention-sink \
            (pending RV-3 committed-value echo)"]
fn chat_completion_runs() {
    assert_ok(
        "chat-completion",
        run_curated_inferlet(
            "chat-completion",
            r#"{"prompt":"hi","max_tokens":4}"#,
            Duration::from_secs(60),
        ),
    );
}

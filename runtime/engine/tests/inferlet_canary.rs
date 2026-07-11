//! Phase-6 canary: representative **production** inferlets RUN on the mock driver.
//!
//! Complements `e2e.rs` (the small test inferlets in `tests/inferlets/`): this
//! builds a few real inferlets from `../inferlets/` and spawns them through the
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

/// Build a production inferlet (`../inferlets/<name>`) → wasm + manifest + id.
fn load_prod_inferlet(name: &str) -> (Vec<u8>, Manifest, ProgramName) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../inferlets")
        .join(name);
    let status = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "--release"])
        .current_dir(&dir)
        .status()
        .unwrap_or_else(|e| panic!("spawn cargo build for {name}: {e}"));
    assert!(status.success(), "build {name} failed");

    let wasm_path = dir
        .join("target/wasm32-wasip2/release")
        .join(format!("{}.wasm", name.replace('-', "_")));
    let wasm =
        std::fs::read(&wasm_path).unwrap_or_else(|e| panic!("read {}: {e}", wasm_path.display()));
    let manifest =
        Manifest::parse(&std::fs::read_to_string(dir.join("Pie.toml")).unwrap()).unwrap();
    let program_name = ProgramName::parse(&format!("{name}@{}", manifest.package.version)).unwrap();
    (wasm, manifest, program_name)
}

/// Build + install + spawn a production inferlet on the mock driver and return its
/// **result** (`Ok(output)` on success, `Err(msg)` on a host/inferlet error).
/// Panics only if it doesn't finish within `timeout`.
fn run_prod_inferlet(name: &str, input: &str, timeout: Duration) -> Result<String, String> {
    let h = harness();
    let (wasm, manifest, program_name) = load_prod_inferlet(name);
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

/// Fork-heavy KV canary: best-of-n forks the base context per candidate
/// (`Context::fork()` → `kv-working-set.fork`), generates each, ranks.
#[test]
fn best_of_n_fork_heavy_runs() {
    assert_ok(
        "best-of-n",
        run_prod_inferlet(
            "best-of-n",
            r#"{"question":"hi","num_candidates":3,"max_tokens":4}"#,
            Duration::from_secs(60),
        ),
    );
}

/// Speculative-decoding canary: text-completion-spec sets `Speculation::Default`
/// (every forward requests `output_speculative_tokens(true)`). Exercises the
/// prefill path: the forward resolves only the `valid_tokens` prefix of the
/// kv-context (trailing reserved slots are written this pass as kv-output, never
/// read), so the turn-1 prefill no longer errors on a reserved slot.
#[test]
fn text_completion_spec_runs() {
    assert_ok(
        "text-completion-spec",
        run_prod_inferlet(
            "text-completion-spec",
            r#"{"prompt":"hi","max_tokens":4}"#,
            Duration::from_secs(60),
        ),
    );
}

/// Snapshot canary: demo-persistent-kv `smart` mode saves the turn-1 context to a
/// `/scratch` blob (`snapshot::save`) then reopens it (`snapshot::open`) and
/// REPLAYS the token log via a keep-core carrier prefill, then continues — the
/// raw-WIT / keep-core rewrite (no `Context` facade; the thin `snapshot::` data +
/// wasi:filesystem I/O primitive, `ptir-snapshot-keepcore-spec`). Exercises the
/// prefill + snapshot replay round-trip end-to-end on the mock driver.
#[test]
fn snapshot_save_open_runs() {
    assert_ok(
        "demo-persistent-kv",
        run_prod_inferlet(
            "demo-persistent-kv",
            r#"{"mode":"smart","turn1":"hi","turn2":"yo","max_tokens":3,"snapshot":"canary_snap"}"#,
            Duration::from_secs(60),
        ),
    );
}

/// Masked-decode canary — attention-sink (echo, SDK-minimization ①): the
/// raw-WIT / keep-core rewrite (no `Context`/`Sampler` facade) drives a decode
/// loop that attaches a position-deterministic sink+window `attention_mask` in
/// the `carrier::submit_pass_with` bind seam. A small `window_size` makes the
/// mask path fire within a few steps, so this exercises the mask attach +
/// forward on the mock end to end (not just the no-mask fallback).
#[test]
fn attention_sink_masked_decode_runs() {
    assert_ok(
        "attention-sink",
        run_prod_inferlet(
            "attention-sink",
            r#"{"prompt":"hi","max_tokens":6,"sink_size":1,"window_size":2}"#,
            Duration::from_secs(60),
        ),
    );
}

/// Masked-decode canary — windowed-attention (echo, SDK-minimization ①): same
/// keep-core rewrite, sliding-window mask via the `submit_pass_with` bind seam.
/// A small `window_size` fires the mask path within a few steps.
#[test]
fn windowed_attention_masked_decode_runs() {
    assert_ok(
        "windowed-attention",
        run_prod_inferlet(
            "windowed-attention",
            r#"{"prompt":"hi","max_tokens":6,"window_size":2}"#,
            Duration::from_secs(60),
        ),
    );
}

/// Chat-EOS pipelined canary — text-completion (echo, SDK-minimization ①): the
/// raw-WIT / keep-core rewrite (no `Context`/`Generator`/`Sampler` facade) drives
/// the depth-1 EOS-rollback pipeline — each step speculates the next forward via
/// `carrier::submit_pass` and rolls an over-shot pass back with
/// `carrier::discard_pass` on a stop — with a parametric top-p `sampler_program`
/// and kept `chat` templating + streaming `chat::Decoder` detok. This exercises
/// the full lowering + pipelined-carrier + stop/decoder path on the mock end to
/// end (rollback token-identity vs the sequential stream is a GPU MATCH gate, per
/// `ptir-pipelined-eos-rollback-spec §7`).
#[test]
fn text_completion_pipelined_eos_runs() {
    assert_ok(
        "text-completion",
        run_prod_inferlet(
            "text-completion",
            r#"{"prompt":"hi","max_tokens":4}"#,
            Duration::from_secs(60),
        ),
    );
}

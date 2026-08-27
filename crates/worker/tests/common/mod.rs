//! Shared real-hardware (`cuda_native`) test harness.
//!
//! Boots the worker's prod embedded path in-proc — `worker::run` in
//! SingleNode mode loads the model onto the GPU via the embedded cuda driver and
//! co-resides `::engine::bootstrap::bootstrap` — then drives inferlets through the
//! same in-proc `program::add` → `process::spawn` flow the mock canary uses,
//! bypassing the gateway/client edge (no msgpack/JSON codec, no identity header,
//! no `pie-server-py`).
//!
//! Reused by the cuda validation tests (`cuda_forward` = dense forward; the
//! Lane-C CAS-dedup + Lane-D fold-parity tests compose on these helpers). Every
//! cuda test is `#[ignore]`d (real GPU + `--features driver-cuda-13`) and boots
//! ONCE per process (global engine state forbids a second boot).
//!
//! The model snapshot is overridable via `PIE_CUDA_TEST_SNAPSHOT` (a local HF
//! snapshot dir — R3: the worker never downloads); the default is the Qwen3-0.6B
//! dense model on the reference box. Use a GDN model (e.g. Qwen3.5-0.8B) for RS
//! fold validation.

#![allow(dead_code)]

use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use ::engine::inferlet::program::{Manifest, ProgramName};
use worker::WorkerHandle;

/// Default local HF snapshot (Qwen3-0.6B dense) on the reference box. Override
/// with `PIE_CUDA_TEST_SNAPSHOT=/path/to/snapshot` for another model/host.
pub const DEFAULT_SNAPSHOT: &str = "/home/ingim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca";

/// Local HF snapshot for the Qwen3.5-0.8B GDN (hybrid linear-attention) model —
/// the RS-fold validation model. Override with `PIE_CUDA_TEST_GDN_SNAPSHOT`.
pub const DEFAULT_GDN_SNAPSHOT: &str = "/home/ingim/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17";

/// The dense model snapshot path (env-overridable).
pub fn snapshot() -> String {
    std::env::var("PIE_CUDA_TEST_SNAPSHOT").unwrap_or_else(|_| DEFAULT_SNAPSHOT.to_string())
}

/// The GDN/hybrid-RS model snapshot path (env-overridable) — for fold validation.
pub fn gdn_snapshot() -> String {
    std::env::var("PIE_CUDA_TEST_GDN_SNAPSHOT").unwrap_or_else(|_| DEFAULT_GDN_SNAPSHOT.to_string())
}

/// Single-model worker config for `snapshot_path`: `cuda_native`, no cluster
/// (→ SingleNode), client edge on an ephemeral loopback port (unused — tests
/// drive `process::spawn` directly in-proc).
pub fn cuda_toml_for(snapshot_path: &str) -> String {
    // A writable scratch dir so snapshot-using canaries (demo-persistent-kv:
    // Context::save/open over `/scratch`) work; harmless for fs-free inferlets.
    let scratch = std::env::temp_dir().join("pie-cuda-test-scratch");
    let _ = std::fs::create_dir_all(&scratch);
    // Weight streaming is off unless asked for. A run with it on must produce
    // the same tokens as a run with it off, which is the whole point of
    // comparing across two processes -- one boot per process is the harness's
    // standing constraint.
    let streaming = if std::env::var("PIE_CUDA_TEST_STREAM_EXPERTS").as_deref() == Ok("1") {
        // The two knobs are `ByteSize` and carry a UNIT -- `expert_cache` and
        // `expert_host_cache`, not `expert_cache_gb` / `expert_host_cache_gb`,
        // which is what this wrote until a run answered
        //
        //     invalid [model.driver.options] for driver type CudaNative:
        //     unknown field `expert_cache_gb`
        //
        // The ENV knobs keep their `_GB` names, because a fraction of a GiB is
        // how an operator thinks about a slab meant to be too small (0.0004
        // GiB is under half a mebibyte, which forces an eviction per layer and
        // is the only way the eviction path runs at all). Rendered in bytes so
        // no fraction is lost on the way.
        let gib = |name: &str| -> Option<u64> {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .filter(|v| *v > 0.0)
                .map(|v| (v * 1024.0 * 1024.0 * 1024.0) as u64)
        };
        // Omitted rather than zero when unset: the field is `Option<ByteSize>`
        // and its doc says an absent key means "derive one at bootstrap",
        // while a zero would be a slab with no slots in it.
        let slab = gib("PIE_CUDA_TEST_EXPERT_CACHE_GB")
            .map(|b| format!("expert_cache = \"{b}B\"\n"))
            .unwrap_or_default();
        let host = gib("PIE_CUDA_TEST_EXPERT_HOST_CACHE_GB")
            .map(|b| format!("expert_host_cache = \"{b}B\"\n"))
            .unwrap_or_default();
        format!("stream_routed_experts = true\n{slab}{host}")
    } else {
        String::new()
    };
    // The KV cache is otherwise sized from whatever VRAM is left, so changing
    // the expert slab silently changes the page count -- and with it the
    // attention plan and its reduction order. Two runs meant to differ only in
    // residency would then differ in numerics too, which is not a comparison.
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
         [driver]\n\
         type = \"cuda_native\"\n\
         device = [\"cuda:0\"]\n\
         gpu_mem_utilization = 0.90\n\
         memory_profile = \"latency\"\n\
         {kv}{streaming}",
        scratch = scratch.display(),
    )
}

/// Worker config for the default dense model.
pub fn cuda_toml() -> String {
    cuda_toml_for(&snapshot())
}

/// Route `tracing` to stderr, once per process, at whatever `RUST_LOG` says
/// (`error` if it says nothing).
///
/// The reason this exists rather than being left to whoever wants it: the
/// interesting CUDA failures are not the ones this harness asserts on. A
/// device instantiation that will not compile or load is reported by
/// `kernels-cuda::jit::ctx::said` through `tracing::error!` and NOWHERE else
/// -- the `KernelError::Device { call, code }` it returns holds a `&'static str` and so
/// cannot carry the driver's sentence -- so the engine's message stops at
/// "the compile, the load or the launch refused; see the log". Without a
/// subscriber there is no log, and a `CUDA_ERROR_ILLEGAL_ADDRESS` in one
/// kernel reads as an unrelated module failing to load in the next, because
/// that error is sticky.
fn wire_tracing() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("error"));
        // `try_init` and not `init`: a second harness in the same process, or
        // an engine that wired its own, is not this function's failure.
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(std::io::stderr)
            .try_init();
    });
}

/// Boot the embedded cuda engine in-proc with an explicit model snapshot (loads
/// it onto the GPU + bootstraps the runtime). Use a GDN snapshot for RS fold
/// validation, the dense default otherwise. Caller holds the handle and
/// `shutdown()`s it.
pub async fn boot_cuda_model(snapshot_path: &str) -> WorkerHandle {
    wire_tracing();
    let cfg =
        worker::Config::parse(&cuda_toml_for(snapshot_path)).expect("parse cuda worker config");
    worker::run(cfg).await.expect("boot embedded cuda engine")
}

/// Boot the embedded cuda engine with the default dense model (Qwen3-0.6B).
pub async fn boot_cuda() -> WorkerHandle {
    boot_cuda_model(&snapshot()).await
}

/// Build a curated inferlet fixture → wasm + manifest + program id.
///
/// The fixtures are at the REPOSITORY's `tests/inferlets`, which is two levels
/// above this crate's manifest and not one: this crate lives at
/// `crates/worker`. It read `../tests/inferlets` until every caller of this
/// helper was `#[ignore]`d and nothing noticed, and the way it failed is worth
/// the extra line below. `Command::current_dir` on a directory that is not
/// there does not report the directory -- it reports `spawn cargo build for
/// text-completion: No such file or directory`, which reads as a missing
/// `cargo` and sent the first person to look at `PATH`.
pub fn load_curated_inferlet(name: &str) -> (Vec<u8>, Manifest, ProgramName) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/inferlets")
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
    // `tests/inferlets` is a cargo workspace, so a member's artifact lands in
    // the shared target dir one level up, not beside its manifest. Non-members
    // build in place. Accept either rather than caring which this one is.
    let wasm_path = if wasm_path.exists() {
        wasm_path
    } else {
        dir.join("../target/wasm32-wasip2/release")
            .join(format!("{}.wasm", name.replace('-', "_")))
    };
    let wasm =
        std::fs::read(&wasm_path).unwrap_or_else(|e| panic!("read {}: {e}", wasm_path.display()));
    let manifest =
        Manifest::parse(&std::fs::read_to_string(dir.join("Pie.toml")).unwrap()).unwrap();
    let program_name = ProgramName::parse(&format!("{name}@{}", manifest.package.version)).unwrap();
    (wasm, manifest, program_name)
}

/// Build + add + install an inferlet once; returns its program id for repeated
/// spawns (one install per process; spawn many).
pub async fn install_inferlet(name: &str) -> ProgramName {
    let (wasm, manifest, program_name) = load_curated_inferlet(name);
    ::engine::inferlet::program::add(wasm, manifest, true)
        .await
        .expect("add program");
    ::engine::inferlet::program::install(&program_name)
        .await
        .expect("install program");
    program_name
}

/// Spawn one inferlet run and capture its result (`Ok(text)` / `Err(msg)`) — the
/// result-captured pattern that surfaces host/forward errors (e.g. the lost-KV
/// -commit bug) instead of a silent "completed". Panics only on timeout.
pub async fn spawn_text(
    program: &ProgramName,
    prompt: &str,
    max_tokens: usize,
) -> Result<String, String> {
    let input = format!(r#"{{"prompt":{prompt:?},"max_tokens":{max_tokens}}}"#);
    spawn_input(program, &input).await
}

/// Spawn an already-installed inferlet with a raw JSON input string, capturing
/// its result.
pub async fn spawn_input(program: &ProgramName, input_json: &str) -> Result<String, String> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    ::engine::inferlet::process::spawn(
        "cuda-test".into(),
        program.clone(),
        input_json.to_string(),
        None,
        false,
        Some(tx),
    )
    .expect("spawn process");
    // A timeout is an `Err` rather than a panic, because "did not answer" is
    // a RESULT about the inferlet -- four of the curated fixtures give exactly
    // that on CUDA today (see `cuda_canaries`' census) -- and a caller that
    // wants it fatal says `.expect()`, which is what every one of them does.
    // A panic here instead reports the harness's deadline as though it were
    // the assertion the caller wrote.
    match tokio::time::timeout(Duration::from_secs(180), rx).await {
        Err(_) => Err("no answer within 180s".to_string()),
        Ok(result) => result.expect("process result channel dropped"),
    }
}

/// Build + add + install + spawn an arbitrary curated inferlet fixture with a raw
/// JSON input (for canaries — fork / spec / snapshot — that take
/// inferlet-specific inputs). One-shot: installs then spawns.
pub async fn spawn_inferlet(name: &str, input_json: &str) -> Result<String, String> {
    let program = install_inferlet(name).await;
    spawn_input(&program, input_json).await
}

/// Refuse a completion that is not made of WORDS.
///
/// "Non-empty" is the assertion these forward gates used to carry, and it is
/// worth almost nothing: a model whose weights are being read correctly and a
/// model whose every RMSNorm reduces over the wrong axis both emit sixteen
/// tokens. The qwen3.5 hybrid answered `"The capital of France is"` with
///
///     "\n\n\nqu.c.\n\n. / } )\n0\n -"
///
/// for as long as nothing looked, and a non-empty check passed on it every
/// time. What separates that from a real completion is not length and not
/// perplexity — it is that a working model emits WORDS, and a broken one emits
/// punctuation and fragments.
///
/// So: at least `min_words` runs of three-or-more letters, and at least two
/// fifths of the non-space characters alphanumeric. Both thresholds are far
/// below anything a coherent English completion produces and far above what
/// degenerate sampling from a scrambled residual stream produces, which is the
/// only band where a gate like this is worth having. It says nothing about
/// WHICH words — that claim belongs to the A/B fixtures against transformers,
/// which compare logits.
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

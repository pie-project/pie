//! Shared boot helper for the `bin/pie` cuda_native integration tests
//! (Phase-2 WS7 — the programmable-sampler 4090 real-driver pass).
//!
//! `boot_4090()` owns ALL cuda_native boot details (TOML + Mode + addresses)
//! so the capability test bodies (golf's Client-submit + hotel's
//! `sampler_assert`) stay pure submit+assert and can't drift from the boot.
//! Imported via `mod common;` in each integration-test file.
//!
//! ONE boot per process (the runtime owns process-global singletons — `auth`
//! panics on a 2nd boot; the driver grabs a fixed POSIX shmem), so every test
//! that calls `boot_4090()` must live in its own `#[ignore]` test process.

// Not every integration-test file uses every helper (each `mod common;` is a
// separate compilation), so silence unused-helper warnings per test binary.
#![allow(dead_code)]

use anyhow::Result;
use pie::derive::derive_standalone;
use pie::run_standalone;

/// Install a `tracing` subscriber driven by `RUST_LOG` so the inproc
/// forward-path debug probes (`engine::driver::inproc`) and any other `tracing`
/// events surface on the diagnostic runs. Idempotent + non-panicking: a 2nd
/// call (or a boot that already set a global) is a silent no-op.
pub fn init_trace() {
    use tracing_subscriber::{EnvFilter, fmt};
    let _ = fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .try_init();
}

/// Default model for the 4090 pass — HF-cached, resolved to a local snapshot
/// path (the cuda_native worker never downloads, per the R3 policy).
pub const QWEN3_0_6B_REPO: &str = "Qwen/Qwen3-0.6B";

/// Resolve `Qwen/Qwen3-0.6B` to its **local HF cache snapshot dir** (the dir
/// holding `config.json` + `model.safetensors` + `tokenizer.json`). The
/// cuda_native worker enforces R3 (never downloads), so `hf_repo` must be a
/// local path, and the snapshot hash is machine-specific — resolve it at
/// runtime from `$HF_HOME`/`~/.cache/huggingface/hub`.
pub fn resolve_qwen3_snapshot() -> Result<String> {
    let hub = std::env::var("HF_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_default();
            std::path::PathBuf::from(home).join(".cache/huggingface")
        })
        .join("hub/models--Qwen--Qwen3-0.6B/snapshots");
    let snap = std::fs::read_dir(&hub)
        .with_context(|| {
            format!(
                "qwen-3-0.6b not in HF cache at {} — run `huggingface-cli download Qwen/Qwen3-0.6B`",
                hub.display()
            )
        })?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("config.json").exists() && p.join("model.safetensors").exists())
        .with_context(|| format!("no complete qwen-3-0.6b snapshot under {}", hub.display()))?;
    Ok(snap.to_string_lossy().into_owned())
}

/// The cuda_native standalone TOML (`[controller]/[gateway]/[worker]`).
///
/// `binary_path` is omitted (accepted-but-ignored for cuda_native — the
/// standalone embeds the driver as a static lib). The CUDA driver loads
/// `config.json` + `model.safetensors` + `tokenizer.json` from the snapshot
/// dir, so `hf_repo` is a **local snapshot path** (R3: the worker never
/// downloads). `device` is an array (`["cuda:0"]`); auth off; gateway on an
/// ephemeral loopback port.
pub fn cuda_standalone_toml(hf_repo: &str) -> String {
    cuda_standalone_toml_util(hf_repo, 0.85)
}

/// [`cuda_standalone_toml`] with a caller-set `gpu_mem_utilization` — the knob
/// that sizes the KV pool. A LOW value shrinks the pool so a modest fleet
/// over-fills it (the Task-B contention e2e, `cuda_contention`). Tune per box.
pub fn cuda_standalone_toml_util(hf_repo: &str, gpu_mem_utilization: f64) -> String {
    cuda_standalone_toml_capped(hf_repo, gpu_mem_utilization, 0)
}

/// Like [`cuda_standalone_toml_util`] but with an explicit KV-page cap
/// (`[batching].total_pages`; 0 = derive from util). Forces a tiny deterministic
/// KV pool for the contention/preempt e2e independent of the forward-layout floor.
pub fn cuda_standalone_toml_capped(
    hf_repo: &str,
    gpu_mem_utilization: f64,
    total_pages: u32,
) -> String {
    // cpu_pages (the runtime KV stash pool for suspend/restore) is derived from
    // the cuda driver's `swap_pool_size` (translate.rs:117). MANDATORY > 0 for
    // suspend/restore (with swap_pool_size=0 the runtime cpu_pages=0 → every suspend is
    // all-cold → the fix makes suspends inert (freed_now=0 → decline), and pre-fix
    // it silently dropped written KV → "slot 0 has no written page"). Default 512
    // (host RAM is cheap; must hold the fleet's stashed overage). Override via
    // PIE_KV_CPU_PAGES.
    let swap_pool_size: u32 = std::env::var("PIE_KV_CPU_PAGES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    format!(
        "         [server]\n\
         port = 0\n\
         \n\
         [model]\n\
         name = \"qwen3\"\n\
         model = \"{hf_repo}\"\n\
         \n\
         [driver]\n\
         type = \"cuda_native\"\n\
         device = [\"cuda:0\"]\n\
         \n\
         gpu_mem_utilization = {gpu_mem_utilization}\n\
         {cap}\
         swap_pool_size = {swap_pool_size}\n",
        // Omitted rather than zeroed: `0 = derive` was retired when the
        // sentinels went, and `max_total_pages` is an Option now -- absence IS
        // the request to derive from gpu_mem_utilization.
        cap = if total_pages > 0 {
            format!("         max_total_pages = {total_pages}\n")
        } else {
            String::new()
        }
    )
}

/// Boot the embedded standalone (controller + gateway + worker) with the real
/// CUDA driver + qwen-3-0.6b on the 4090. The client edge is at
/// `handle.listen_addr` (`ws://{listen_addr}` for the `pie-client`).
pub async fn boot_4090() -> Result<pie::StandaloneHandle> {
    let snapshot = resolve_qwen3_snapshot()?;
    let (controller, gateway, worker) = derive_standalone(&cuda_standalone_toml(&snapshot))?;
    run_standalone(controller, gateway, worker).await
}

/// [`boot_4090`] at an explicit `[runtime] frame_dispatch_depth` — the
/// engine's enqueue horizon in frames. `cuda_deep_coverify` needs the engine's
/// depth to MATCH the chain depth its carrier submits, and config is the only
/// way to say so: the depth used to be an env var the engine silently clamped,
/// so a test asking for 4 quietly ran the engine at 3.
pub async fn boot_4090_dispatch_depth(depth: u32) -> Result<pie::StandaloneHandle> {
    let snapshot = resolve_qwen3_snapshot()?;
    let toml = format!(
        "{}\n[runtime]\nframe_dispatch_depth = {depth}\n",
        cuda_standalone_toml(&snapshot)
    );
    let (controller, gateway, worker) = derive_standalone(&toml)?;
    run_standalone(controller, gateway, worker).await
}

/// [`boot_4090`] with a SMALL KV pool (low `gpu_mem_utilization`) so a modest
/// fleet over-fills it — the Task-B preempt/restore over-capacity e2e
/// (`cuda_contention`). Contention is now forced by the explicit KV-page cap
/// (`PIE_CONTENTION_TOTAL_PAGES`, charlie — `[batching].total_pages`, mirrors
/// metal), NOT by util: util only needs to clear the ~0.3 forward-layout floor
/// so the driver boots (util < ~0.3 → fatal "no viable forward/KV layout"). The
/// cap then shrinks the KV pool to exactly N pages (`min(kv_pages, cap)`), so a
/// modest fleet genuinely over-fills it deterministically (CI-friendly).
pub const SMALL_POOL_GPU_MEM_UTIL: f64 = 0.3;

pub async fn boot_4090_kv_cap(total_pages: u32) -> Result<pie::StandaloneHandle> {
    let util = std::env::var("PIE_CONTENTION_UTIL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(SMALL_POOL_GPU_MEM_UTIL);
    eprintln!("[contention] boot_4090_kv_cap util={util} total_pages={total_pages}");
    let snapshot = resolve_qwen3_snapshot()?;
    let (controller, gateway, worker) =
        derive_standalone(&cuda_standalone_toml_capped(&snapshot, util, total_pages))?;
    run_standalone(controller, gateway, worker).await
}

pub async fn boot_4090_small_kv() -> Result<pie::StandaloneHandle> {
    // charlie: explicit KV-page cap (deterministic tiny pool, independent of the
    // forward-layout budget floor). Default 8 forces genuine contention out-of-the
    // -box; `PIE_CONTENTION_TOTAL_PAGES=0` restores the derive-from-util path.
    let total_pages: u32 = std::env::var("PIE_CONTENTION_TOTAL_PAGES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    boot_4090_kv_cap(total_pages).await
}

/// Default MTP model for the native-drafter de-risk (Qwen3.5-0.8B GDN backbone +
/// a 1-layer MTP head). HF-cached, resolved to a local snapshot path (R3: never
/// downloads). The MTP head auto-activates in the driver on MTP-weight presence
/// (entry.cpp `wire_system_drafter`); `[model.driver.options].mtp_num_drafts`
/// sets the draft count K (0 disables → the non-spec baseline).
pub const QWEN35_0_8B_REPO: &str = "Qwen/Qwen3.5-0.8B";

/// Resolve `Qwen/Qwen3.5-0.8B` to its local HF cache snapshot dir (mirrors
/// [`resolve_qwen3_snapshot`]; the snapshot hash is machine-specific).
pub fn resolve_qwen35_snapshot() -> Result<String> {
    let hub = std::env::var("HF_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_default();
            std::path::PathBuf::from(home).join(".cache/huggingface")
        })
        .join("hub/models--Qwen--Qwen3.5-0.8B/snapshots");
    let snap = std::fs::read_dir(&hub)
        .with_context(|| {
            format!(
                "qwen3.5-0.8b not in HF cache at {} — run `huggingface-cli download Qwen/Qwen3.5-0.8B`",
                hub.display()
            )
        })?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.join("config.json").exists()
                && (p.join("model.safetensors").exists()
                    || p.join("model.safetensors.index.json").exists())
        })
        .with_context(|| format!("no complete qwen3.5-0.8b snapshot under {}", hub.display()))?;
    Ok(snap.to_string_lossy().into_owned())
}

/// The cuda_native standalone TOML for an MTP model. Same shape as
/// [`cuda_standalone_toml`] but `name = "default"` so the driver auto-detects
/// the architecture (GDN + MTP head) from the snapshot's `config.json` rather
/// than being pinned to the dense `qwen3` path.
pub fn cuda_mtp_standalone_toml(hf_repo: &str, mtp_num_drafts: u32) -> String {
    format!(
        "         [server]\n\
         port = 0\n\
         \n\
         [model]\n\
         name = \"default\"\n\
         model = \"{hf_repo}\"\n\
         \n\
         [driver]\n\
         type = \"cuda_native\"\n\
         device = [\"cuda:0\"]\n\
         \n\
         gpu_mem_utilization = 0.85\n\
         mtp_num_drafts = {mtp_num_drafts}\n"
    )
}

/// Boot the embedded standalone with the real CUDA driver + Qwen3.5-0.8B (the
/// MTP model) on the 4090. K (native draft tokens) is `mtp_num_drafts`, passed
/// through `[model.driver.options]` like any other driver setting -- 0 disables
/// speculation and gives the non-spec baseline. Client edge at
/// `handle.listen_addr`.
pub async fn boot_4090_mtp(mtp_num_drafts: u32) -> Result<pie::StandaloneHandle> {
    let snapshot = resolve_qwen35_snapshot()?;
    let (controller, gateway, worker) =
        derive_standalone(&cuda_mtp_standalone_toml(&snapshot, mtp_num_drafts))?;
    run_standalone(controller, gateway, worker).await
}

/// K for the MTP suites, from `PIE_MTP_DRAFT_TOKENS`. A harness parameter, not
/// engine config: it selects which arm of a manual A/B to boot, and is handed
/// to the driver as `mtp_num_drafts`.
pub fn mtp_draft_tokens(default_k: u32) -> u32 {
    std::env::var("PIE_MTP_DRAFT_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .map(|k: u32| k.min(32))
        .unwrap_or(default_k)
}

// ── The dummy driver, and the three gates that stood on it ─────────────
//
// `dummy_standalone_toml` and `boot_dummy` STOOD HERE. They fabricated
// everything a portable driver reads from weights -- no GPU, no 20 GB load,
// near-instant boot -- so a gate could exercise the driver-AGNOSTIC client
// edge (connect -> add_program -> launch -> forward round-trip) on a machine
// with no CUDA and no artifact.
//
// The driver they named is deleted. `DriverKind` accepts `cuda_native`,
// `metal`, `vulkan` and `wgpu` and nothing else, so `type = "dummy"` no longer
// parses -- `worker::config` refuses it before a boot is attempted, and there
// is no driverless boot left anywhere in this tree. That is a decision made
// upstream and recorded in this crate's `Cargo.toml`: *"there is no fallback:
// the dummy driver these no-GPU diagnostics used to run against is deleted, so
// a build with neither feature reaches no device."* These helpers were what
// that sentence had not finished removing.
//
// THREE gates rested on them, and they went three different ways, which is
// worth writing down because "the dummy driver is gone" is not by itself an
// answer to what a gate was measuring.
//
// * `dummy_add_program` -- the chunked-`add_program` session-bridge deadlock
//   repro -- took `boot_vulkan()` and is `vulkan_add_program` now. What it
//   measures is the gateway/worker turn model under a ~12 MB upload, which is
//   driver-agnostic; it only ever wanted the CHEAPEST boot, and on a machine
//   with no CUDA the Vulkan driver is that. It costs an artifact env var it
//   did not use to need, which is the whole price.
//
// * tests/boot_smoke.rs and tests/boot_artifact.rs, in the root `pie` package,
//   are DELETED -- named without backticks because they are not there to be
//   opened, which is the rule `model-loader`'s citation gate enforces. Both
//   booted the standalone against a synthetic snapshot -- a 256-token
//   byte-level BPE and a four-byte `embed_tokens` tensor --
//   which only the dummy load planner ever accepted. A real driver cannot
//   serve four bytes, so there was no config to repoint them at: their subject
//   was the fabricating driver itself.
//
//   They are not replaced so much as already covered from both sides. The
//   end-to-end half -- convert a checkpoint, boot from the artifact alone,
//   round-trip a turn through the real client edge -- is what
//   `vulkan_chat_completion_e2e` and `vulkan_two_conversations` in this
//   directory do, against a real model on a real device. The artifact half --
//   that the objects written under `__meta__/` come back out of a `.zt` by
//   name and rebuild the same tokenizer -- is `tests/artifact_tokenizer.rs`,
//   which asserts it directly and needs no boot at all.
//
//   What is genuinely lost is that those two ran in a plain `cargo test` with
//   no GPU and no env var, and the gates that took over do not: they are
//   `#[ignore]`d and want `PIE_VULKAN_ARTIFACT`. That is a real reduction in
//   what CI notices on a machine with no device, and it is stated here rather
//   than discovered later. It is the cost of the dummy driver's deletion, not
//   of this edit.

// ── Client submit (golf) ────────────────────────────────────────────────────
//
// The capability-test half: build a capability inferlet to wasm, submit it to
// the engine `boot_4090()` brought up, and return its structured-JSON result
// for hotel's `sampler_assert`. Pure client-side (no GPU), so it compiles +
// type-checks Rust-only (without `--features driver-cuda-13`).

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Context;
use client::client::Client;

/// Build the capability inferlets to `wasm32-wasip2` and return
/// `(wasm, manifest)` for `name` ∈ {"generate", "mirostat", "grammar"}. Paths resolve from
/// the `bin/pie` crate dir to the runtime test-inferlets workspace. Builds both
/// (one cargo invocation) so a multi-capability harness pays the build once.
pub fn build_inferlet(name: &str) -> (PathBuf, PathBuf) {
    let workspace =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../crates/engine/tests/inferlets");
    let ok = Command::new("cargo")
        .args([
            "build",
            "--target",
            "wasm32-wasip2",
            "-p",
            "generate",
            "-p",
            "mirostat",
            "-p",
            "grammar",
        ])
        .current_dir(&workspace)
        .status()
        .expect("spawn cargo build for capability inferlets")
        .success();
    assert!(ok, "capability inferlet wasm build failed");
    let wasm = workspace.join(format!("target/wasm32-wasip2/debug/{name}.wasm"));
    let manifest = workspace.join(format!("{name}/Pie.toml"));
    assert!(wasm.exists(), "missing inferlet wasm: {}", wasm.display());
    assert!(
        manifest.exists(),
        "missing manifest: {}",
        manifest.display()
    );
    (wasm, manifest)
}

/// Submit a capability inferlet to the running engine at `listen_addr` and
/// return its structured-JSON result. Builds `name` (`mirostat`/`grammar`),
/// then runs the canonical submit flow (connect → authenticate → add_program →
/// launch_process → `wait_for_return`). `program_name` is `{name}@{version}`
/// (e.g. `mirostat@0.1.0`); `input` is the inferlet's JSON run-params (e.g.
/// `{"max_tokens":48}`, or `{}` for defaults).
pub async fn run_inferlet(
    listen_addr: &std::net::SocketAddr,
    name: &str,
    program_name: &str,
    input: &str,
) -> Result<String> {
    let (wasm, manifest) = build_inferlet(name);

    // The gateway serves the multi-turn client WebSocket at `/v1/ws`
    // (`gateway/src/ingress/mod.rs`), gated on the `x-pie-identity` trust-edge
    // header (else 401). Standalone has no edge proxy, so inject it here.
    let client = Client::connect_with_identity(&format!("ws://{listen_addr}/v1/ws"), "test-user")
        .await
        .with_context(|| format!("connect to engine at ws://{listen_addr}/v1/ws"))?;
    // The bench/test engine disables public-key auth, so this returns early.
    client
        .authenticate("test-user", &None)
        .await
        .context("authenticate")?;
    client
        .add_program(&wasm, &manifest, true)
        .await
        .with_context(|| format!("add_program {program_name}"))?;

    let mut proc = client
        .launch_process(program_name.to_string(), input.to_string(), true)
        .await
        .with_context(|| format!("launch_process {program_name}"))?;

    proc.wait_for_return().await
}

/// The standalone TOML for a Vulkan deployment.
///
/// `kv_pages` sizes the pool. `kernels` STOOD beside it, naming the SPIR-V
/// directory the seam read at `create`; the modules are in the binary now, so
/// this block states one knob. There is no `gpu_mem_utilization` here:
/// this driver does not derive a pool from a fraction of the card, it is told
/// how many pages to hold. `device` is stated because the config requires it
/// and ignored because `Device::open` takes the first Vulkan device the
/// loader reports -- a selector here would be a setting nothing acts on.
pub fn vulkan_standalone_toml(artifact: &str) -> String {
    vulkan_standalone_toml_with_pages(artifact, 256)
}

/// The same, with the pool sized by the caller.
///
/// A test that wants the pool to RUN OUT needs this: 256 pages is four
/// thousand tokens, which no gate here comes close to, so the driver's
/// `Exhausted` answer and the engine's re-post are never taken at the default.
#[must_use]
pub fn vulkan_standalone_toml_with_pages(artifact: &str, kv_pages: u32) -> String {
    vulkan_standalone_toml_named(artifact, kv_pages, "qwen3")
}

/// The same, with `[model] name` stated by the caller.
///
/// The name is a LABEL, not a selector: the driver reads the architecture out
/// of the artifact's embedded `model/config`, so a Qwen2 artifact serves a
/// Qwen2 whatever this says. It is parameterised anyway because a gate that
/// boots a second architecture under the first one's name reads as though the
/// name were doing work, and the next person to change the boot path would
/// have to re-derive that it is not.
#[must_use]
pub fn vulkan_standalone_toml_named(artifact: &str, kv_pages: u32, name: &str) -> String {
    format!(
        "         [server]\n\
         port = 0\n\
         \n\
         [model]\n\
         name = \"{name}\"\n\
         model = \"{artifact}\"\n\
         \n\
         [driver]\n\
         type = \"vulkan\"\n\
         device = [\"vulkan:0\"]\n\
         kv_pages = {kv_pages}\n"
    )
}

/// The standalone TOML for the WebGPU driver.
///
/// No artifact, which is the whole difference from the Vulkan block now that
/// both backends carry their shaders in the binary: the weights come from the
/// `$PIE_HOME` model cache that `pie serve` reads, so a gate names a model the
/// cache already holds rather than a file it was handed.
///
/// `device` says `gpu:0` and no driver reads it as a selector: `wgpu` asks the
/// platform for an adapter itself. It is written because `device` is required
/// of every driver and `cuda:0` would be a lie about the hardware.
///
/// `name` is the ARCHITECTURE label. It was hard-coded to `qwen3` here while
/// the Vulkan twin took it as a parameter, which meant no wgpu gate could
/// serve a second architecture even though the driver can: the one file whose
/// whole subject is "a different model" would have been the one place still
/// saying `qwen3`. See `wgpu_second_model`.
#[must_use]
pub fn wgpu_standalone_toml_named(model: &str, name: &str, kv_pages: u32) -> String {
    format!(
        "         [server]\n\
         port = 0\n\
         \n\
         [model]\n\
         name = \"{name}\"\n\
         model = \"{model}\"\n\
         \n\
         [driver]\n\
         type = \"wgpu\"\n\
         device = [\"gpu:0\"]\n\
         activation_dtype = \"bfloat16\"\n\
         kv_pages = {kv_pages}\n"
    )
}

/// [`wgpu_standalone_toml_named`] under the architecture every other gate here
/// serves.
#[must_use]
pub fn wgpu_standalone_toml(model: &str, kv_pages: u32) -> String {
    wgpu_standalone_toml_named(model, "qwen3", kv_pages)
}

/// Boot the embedded standalone with the real WebGPU driver.
///
/// `PIE_WGPU_MODEL` names the model in `$PIE_HOME`'s cache, defaulting to the
/// one every other gate here uses. There is no artifact variable because this
/// driver quantizes through the load plan and reads what the cache holds.
pub async fn boot_wgpu() -> Result<pie::StandaloneHandle> {
    boot_wgpu_with_pages(256).await
}

/// The same, with the pool sized by the caller -- which is how a gate asks for
/// pool PRESSURE rather than for room.
pub async fn boot_wgpu_with_pages(kv_pages: u32) -> Result<pie::StandaloneHandle> {
    let model = std::env::var("PIE_WGPU_MODEL")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "Qwen--Qwen3-0.6B-optimized".to_string());
    boot_wgpu_named(&model, "qwen3", kv_pages).await
}

/// Boot a NAMED model under a NAMED architecture.
///
/// One deployment serves one model, so every other boot here takes the
/// default. A gate that means to prove a SECOND architecture has to name both
/// halves: the cache entry, which is a different `.zt`, and the label, which
/// the engine reports and hashes. The label is not a selector -- the driver
/// reads the architecture out of the artifact -- but the one gate whose whole
/// subject is a different model should not be the place that says `qwen3`.
pub async fn boot_wgpu_named(
    model: &str,
    name: &str,
    kv_pages: u32,
) -> Result<pie::StandaloneHandle> {
    let (controller, gateway, worker) =
        derive_standalone(&wgpu_standalone_toml_named(model, name, kv_pages))?;
    run_standalone(controller, gateway, worker).await
}

/// Boot the embedded standalone with the real Vulkan driver.
///
/// `PIE_VULKAN_ARTIFACT` names a `.zt` that `pie model build --backend
/// vulkan` authored -- an artifact rather than a snapshot because this driver
/// reads its declared quantization out of the embedded `model/config`, and
/// one carrying its tokenizer because the runtime parses one at boot
/// unconditionally. The compiled modules need no variable: they are in the
/// binary under `kernels-vulkan/native`.
/// A colon-separated list is accepted and the first entry used: a deployment
/// serves one model, and the engine's own tests take the list.
pub async fn boot_vulkan() -> Result<pie::StandaloneHandle> {
    boot_vulkan_with_pages(256).await
}

/// The same, with the pool sized by the caller. See
/// [`vulkan_standalone_toml_with_pages`].
pub async fn boot_vulkan_with_pages(kv_pages: u32) -> Result<pie::StandaloneHandle> {
    boot_vulkan_nth(0, "qwen3", kv_pages).await
}

/// Boot the `nth` artifact of `PIE_VULKAN_ARTIFACT` under `[model] name`.
///
/// One deployment serves one model, so every other boot here takes entry 0.
/// A gate that means to prove a SECOND architecture cannot say that with an
/// env var it shares with the first, and inventing a second variable would
/// leave two places to keep in step. The list is already colon-separated
/// because the engine's own tests take it whole, so the second entry is where
/// a second model already lives.
pub async fn boot_vulkan_nth(
    nth: usize,
    name: &str,
    kv_pages: u32,
) -> Result<pie::StandaloneHandle> {
    let artifacts = std::env::var("PIE_VULKAN_ARTIFACT")
        .ok()
        .filter(|v| !v.is_empty())
        .context("PIE_VULKAN_ARTIFACT names the built artifact")?;
    let artifact = artifacts
        .split(':')
        .nth(nth)
        .filter(|v| !v.is_empty())
        .with_context(|| {
            format!(
                "PIE_VULKAN_ARTIFACT has no entry {nth}: {artifacts:?}. A second entry is a \
                 second model's `.zt`, built by `pie model build --backend vulkan`."
            )
        })?
        .to_string();
    let (controller, gateway, worker) =
        derive_standalone(&vulkan_standalone_toml_named(&artifact, kv_pages, name))?;
    run_standalone(controller, gateway, worker).await
}

//! The shared boots every gate in this directory comes up on, and the one
//! checkpoint they all come up on.
//!
//! There are two families here and the split is the wave they were written in.
//! `boot_cuda*` is the LEGACY family: the standalone with the engine's own
//! `gpu_mem_utilization` deriving every ceiling, which is what the pre-palo
//! harnesses were written against. `boot_serving*` is the palo B3 family: the
//! same standalone with the ceilings STATED, because the arena reserves
//! `max_tokens` rows of a vocabulary-wide logit column and an 8192-row default
//! is 8 GiB of arena for a gate that fires eight tokens.
//!
//! Both resolve the SAME checkpoint. There is exactly one dense SKU the
//! catalog, `runtime::model::ROWS` and the reference device all agree about —
//! `qwen35-d0.8b-bf16-kv-bf16` — and a gate that booted anything else in this
//! tree would be measuring a load refusal.
//!
//! ONE boot per process (the runtime owns process-global singletons — `auth`
//! panics on a 2nd boot; the engine grabs a fixed POSIX shmem), so every test
//! that boots must live in its own `#[ignore]` test process.

// Not every integration-test file uses every helper (each `mod common;` is a
// separate compilation), so silence unused-helper warnings per test binary.
#![allow(dead_code)]

use anyhow::{Context, Result};
use pie::derive::derive_standalone;
use pie::run_standalone;

/// Install a `tracing` subscriber driven by `RUST_LOG` so the inproc
/// forward-path debug probes (`runtime::engine::inproc`) and any other `tracing`
/// events surface on the diagnostic runs. Idempotent + non-panicking: a 2nd
/// call (or a boot that already set a global) is a silent no-op.
pub fn init_trace() {
    use tracing_subscriber::{EnvFilter, fmt};
    let _ = fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .try_init();
}

// ── The qwen-3-0.6b default STOOD HERE ──────────────────────────────────
//
// `QWEN3_0_6B_REPO` and `resolve_qwen3_snapshot` named `Qwen/Qwen3-0.6B` and
// every `boot_4090*` below resolved through them. This build ships no SKU that
// checkpoint can be: `models::qwen_3::IMPORTS` claims an artifact by NAME, and
// all five of its rows ask for `model.language_model.layers.*` at a qwen3.5
// geometry, which a Qwen3-0.6B file does not hold. `runtime::model::ROWS` has no
// id for it either. So `runtime::engine::load` cannot identify it and the boot
// dies before a fire, for a reason that has nothing to do with what any gate
// here measures.
//
// The boots now resolve the one dense SKU the catalog, the serving table and
// the L40S all agree about -- `qwen35-d0.8b-bf16-kv-bf16`, out of the stock
// `Qwen/Qwen3.5-0.8B` snapshot -- which is the checkpoint every palo-era gate
// in this directory already pins its continuation against.

/// The cuda_native standalone TOML (`[controller]/[gateway]/[worker]`).
///
/// `binary_path` is omitted (accepted-but-ignored for cuda_native — the
/// standalone embeds the engine as a static lib). The CUDA engine loads
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
    // `swap_pool_size` STOOD HERE (with a PIE_KV_CPU_PAGES override): it fed
    // the runtime's cpu_pages stash pool for suspend/restore while the boot
    // document carried it to the C++ engine. The key retired with that
    // document -- `crates/worker/src/translate.rs` states `cpu_pages: 0` for
    // every engine, so there is no host stash to size and a config stating the
    // key refuses by name.
    format!(
        "         [server]\n\
         port = 0\n\
         \n\
         [model]\n\
         name = \"qwen35\"\n\
         model = \"{hf_repo}\"\n\
         \n\
         [engine]\n\
         type = \"cuda_native\"\n\
         device = [\"cuda:0\"]\n\
         \n\
         gpu_mem_utilization = {gpu_mem_utilization}\n\
         {cap}",
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
/// CUDA engine against the shipping dense SKU. The client edge is at
/// `handle.listen_addr` (`ws://{listen_addr}` for the `pie-client`).
///
/// The LEGACY boot: every ceiling is derived from `gpu_mem_utilization`, which
/// is what the pre-palo harnesses in this directory were written against. A
/// gate that wants the ceilings stated wants [`boot_serving`].
pub async fn boot_cuda() -> Result<pie::StandaloneHandle> {
    let snapshot = resolve_qwen35_snapshot()?;
    let (controller, gateway, worker) = derive_standalone(&cuda_standalone_toml(&snapshot))?;
    run_standalone(controller, gateway, worker).await
}

/// [`boot_cuda`] with a SMALL KV pool (low `gpu_mem_utilization`) so a modest
/// fleet over-fills it — the preempt/restore over-capacity e2e
/// (`cuda_contention`). Contention is forced by the explicit KV-page cap
/// (`PIE_CONTENTION_TOTAL_PAGES` → `[batching].total_pages`), NOT by util: util
/// only needs to clear the ~0.3 forward-layout floor so the engine boots
/// (util < ~0.3 → fatal "no viable forward/KV layout"). The cap then shrinks
/// the KV pool to exactly N pages (`min(kv_pages, cap)`), so a modest fleet
/// genuinely over-fills it deterministically.
pub const SMALL_POOL_GPU_MEM_UTIL: f64 = 0.3;

pub async fn boot_cuda_kv_cap(total_pages: u32) -> Result<pie::StandaloneHandle> {
    let util = std::env::var("PIE_CONTENTION_UTIL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(SMALL_POOL_GPU_MEM_UTIL);
    eprintln!("[contention] boot_cuda_kv_cap util={util} total_pages={total_pages}");
    let snapshot = resolve_qwen35_snapshot()?;
    let (controller, gateway, worker) =
        derive_standalone(&cuda_standalone_toml_capped(&snapshot, util, total_pages))?;
    run_standalone(controller, gateway, worker).await
}

/// The checkpoint the MTP harnesses were written against (a Qwen3.5-0.8B GDN
/// backbone with a 1-layer MTP head bolted on).
///
/// **THIS BUILD DECLARES NO DRAFT ARM ON IT.** `models::qwen_3::CATALOG` puts
/// the `mtp` export on exactly one row — `qwen36-27b-bf16-kv-bf16`, the only
/// checkpoint in the catalog that publishes fifteen `mtp.*` planes — so a
/// Qwen3.5-0.8B snapshot loads here as the plain `qwen35-d0.8b` dense row and
/// `Fault::Draftless` is what a drafting lane gets. Every harness that boots
/// through here is `#[ignore]`d on that fact and says so in its reason.
pub const QWEN35_0_8B_REPO: &str = "Qwen/Qwen3.5-0.8B";

/// Resolve `Qwen/Qwen3.5-0.8B` to its local HF cache snapshot dir (R3: the
/// worker never downloads, and the snapshot hash is machine-specific, so the
/// path is resolved at runtime out of `$HF_HOME`/`~/.cache/huggingface/hub`).
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
/// [`cuda_standalone_toml`] but `name = "default"` so the engine auto-detects
/// the architecture (GDN + MTP head) from the snapshot's `config.json` rather
/// than being pinned to the dense `qwen3` path.
/// `mtp_num_drafts` is gone from the TOML: the key retired with the boot
/// document (no engine read it), so K has no config spelling and the engine's
/// own draft-window default rules. The parameter is kept so the A/B harnesses
/// keep their call shape and their logs keep saying which arm was asked for.
pub fn cuda_mtp_standalone_toml(hf_repo: &str, _mtp_num_drafts: u32) -> String {
    format!(
        "         [server]\n\
         port = 0\n\
         \n\
         [model]\n\
         name = \"default\"\n\
         model = \"{hf_repo}\"\n\
         \n\
         [engine]\n\
         type = \"cuda_native\"\n\
         device = [\"cuda:0\"]\n\
         \n\
         gpu_mem_utilization = 0.85\n"
    )
}

/// Boot the embedded standalone with the real CUDA engine against
/// [`QWEN35_0_8B_REPO`]. K (native draft tokens) no longer crosses --
/// `mtp_num_drafts` retired with the boot document, so the argument only
/// names which arm the caller intended (see `cuda_mtp_standalone_toml`).
/// Client edge at `handle.listen_addr`.
pub async fn boot_cuda_mtp(mtp_num_drafts: u32) -> Result<pie::StandaloneHandle> {
    let snapshot = resolve_qwen35_snapshot()?;
    let (controller, gateway, worker) =
        derive_standalone(&cuda_mtp_standalone_toml(&snapshot, mtp_num_drafts))?;
    run_standalone(controller, gateway, worker).await
}

/// **THE B3 SERVING BOOT.** The standalone over the CUDA shell, serving the
/// one dense SKU the catalog and the L40S agree about
/// (`qwen35-d0.8b-bf16-kv-bf16`), with the ceilings stated rather than
/// planner-derived: the arena reserves `max_tokens` rows of a vocabulary-wide
/// logit column, so an 8192-row default is 8 GiB of arena for a test that
/// fires eight tokens.
///
/// `[model] name` is the deployment's name, not the checkpoint's — the SKU is
/// identified from the checkpoint's own tensors by `runtime::engine::load`.
pub fn serving_standalone_toml(checkpoint: &str) -> String {
    format!(
        "[server]\n\
         port = 0\n\
         \n\
         [model]\n\
         name = \"qwen35\"\n\
         model = \"{checkpoint}\"\n\
         \n\
         [engine]\n\
         type = \"cuda_native\"\n\
         device = [\"cuda:0\"]\n\
         gpu_mem_utilization = 0.85\n\
         kv_page_size = 16\n\
         max_forward_tokens = 512\n\
         max_forward_requests = 8\n\
         max_total_pages = 2048\n"
    )
}

/// The prompt every serving gate asks, and why it is this one: the answer is a
/// single well-known token, so a continuation that is merely fluent still
/// fails.
pub const SERVING_PROMPT: &str = "The capital of France is";

/// What greedy decoding of [`SERVING_PROMPT`] answers with on
/// `qwen35-d0.8b-bf16`, for the first sixteen tokens.
///
/// **SIXTEEN, BECAUSE THE PROMPT IS FIVE AND A KV PAGE IS SIXTEEN.** The
/// continuation crosses a page boundary, which is the first thing a frozen
/// page CSR gets wrong — and the sixteenth token is where it goes wrong.
///
/// It lives here rather than in one test because it is what the `palo B3`
/// arms are diffed against: `text-completion` produces it with the token
/// travelling through the host, `token-healing` produces it with the token
/// carried on the device, and the run-ahead A/B produces it at each of two
/// `frame_size`s. Those arms are spread over several processes because a
/// BOOT is what they differ by (one boot per process — the engine grabs the
/// device and `auth` panics on a second), and a constant is how several
/// processes agree about a fact. Launches within one boot are not spread:
/// `cuda_launch_isolation` is the gate that says so.
pub const SERVING_GREEDY_16: &str =
    " Paris.\nThe capital of France is Paris.\nThe capital of France is";

/// Boot the standalone the way `pie serve` does, against the HF-cache
/// snapshot of Qwen3.5-0.8B.
/// Boot a standalone from a TOML this caller wrote.
///
/// The helpers above each build one config and boot it, which is right for a
/// gate that varies one knob of a known shape. A gate that has to state a
/// checkpoint AND a budget the others do not offer would otherwise grow a
/// third parameter on two of them; this is the door for that.
pub async fn boot_from_toml(toml: &str) -> Result<pie::StandaloneHandle> {
    let (controller, gateway, worker) = derive_standalone(toml)?;
    run_standalone(controller, gateway, worker).await
}

pub async fn boot_serving() -> Result<pie::StandaloneHandle> {
    boot_serving_frame(None).await
}

/// [`boot_serving`] at a stated run-ahead width.
///
/// **k IS THE RUN-AHEAD DEPTH A GUEST SEES**, and it is stated here rather
/// than left to the default because the `palo B3` A/B is exactly the two
/// values: at `frame_size = 1` `submit_frame` short-circuits before
/// `validate_frame` and every fire settles before the next is built, which is
/// one host round trip per token; at `2` a frame carries two ordered slots
/// and slot 1 consumes the channel slot 0 published, which is the chained
/// decode. `None` keeps the runtime's own default so
/// [`boot_serving`]'s callers are unchanged.
///
/// It rides `[runtime]`, which is where the frame knobs live in the file and
/// in `worker::config::RuntimeConfig` alike.
pub async fn boot_serving_frame(frame_size: Option<u32>) -> Result<pie::StandaloneHandle> {
    let checkpoint = resolve_qwen35_snapshot()?;
    let mut toml = serving_standalone_toml(&checkpoint);
    if let Some(k) = frame_size {
        toml.push_str(&format!("\n[runtime]\nframe_size = {k}\n"));
    }
    let (controller, gateway, worker) = derive_standalone(&toml)?;
    run_standalone(controller, gateway, worker).await
}

/// K for the MTP suites, from `PIE_MTP_DRAFT_TOKENS`. A harness parameter, not
/// runtime config: it selects which arm of a manual A/B to boot. It used to be
/// handed to the engine as `mtp_num_drafts`; that key retired with the boot
/// document, so it now only labels the run.
pub fn mtp_draft_tokens(default_k: u32) -> u32 {
    std::env::var("PIE_MTP_DRAFT_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .map(|k: u32| k.min(32))
        .unwrap_or(default_k)
}

// ── The dummy engine, and the three gates that stood on it ─────────────
//
// `dummy_standalone_toml` and `boot_dummy` STOOD HERE. They fabricated
// everything a portable engine reads from weights -- no GPU, no 20 GB load,
// near-instant boot -- so a gate could exercise the engine-AGNOSTIC client
// edge (connect -> add_program -> launch -> forward round-trip) on a machine
// with no CUDA and no artifact.
//
// The engine they named is deleted. `EngineKind` accepts `cuda_native`,
// `metal`, `vulkan` and `wgpu` and nothing else, so `type = "dummy"` no longer
// parses -- `worker::config` refuses it before a boot is attempted, and there
// is no engineless boot left anywhere in this tree. That is a decision made
// upstream and recorded in this crate's `Cargo.toml`: *"there is no fallback:
// the dummy engine these no-GPU diagnostics used to run against is deleted, so
// a build with neither feature reaches no device."* These helpers were what
// that sentence had not finished removing.
//
// What is genuinely lost is that the two root-package boot gates ran in a
// plain `cargo test` with no GPU and no env var, and the gates that took over
// do not: they are `#[ignore]`d and want a device. That is a real reduction in
// what CI notices on a machine with no device, and it is stated here rather
// than discovered later. It is the cost of the dummy engine's deletion, not of
// this edit. The end-to-end half -- boot from a snapshot and round-trip a turn
// through the real client edge -- is `cuda_serve_round_trip` in this
// directory, against a real model on a real device.

// ── `build_inferlet` and `run_inferlet` STOOD HERE ──────────────────────
//
// They built `-p generate -p mirostat -p grammar` out of
// `crates/runtime/tests/inferlets` and submitted the result over the client
// websocket. BOTH halves of that are gone: the guest workspace moved to
// `tests/inferlets`, and none of those three packages is in it -- the sampler
// capability suite they served (`programmable_sampler_4090`, `cuda_mirostat19`,
// `cuda_grammar_op`, `cuda_grammar_late`) went with the engine-baked sampler
// plane it was written for, and what asks that question now is
// `engine-cuda/tests/program_parity` (the emitted guest kernels diffed against
// the host interpreter, ring for ring). It was two: runtime/tests/cuda_program_
// epilogue asked the same question through the serving stack, and was deleted
// as misplaced -- it drove `Engine::submit` directly with no runtime above it,
// so it lived in a crate it did not test.
//
// The gates that still submit a guest build it themselves against
// `tests/inferlets`, one package each, which is what lets a harness name the
// fixture it actually drives instead of paying for three.

// ── The shader-plane boots STOOD HERE ───────────────────────────────────
//
// `vulkan_standalone_toml*`, `wgpu_standalone_toml*`, `boot_vulkan*` and
// `boot_wgpu*`, together with the fifteen gates that called them, went with
// R3: the vulkan and wgpu engines are out of the workspace until their
// baker executors land (P5), so this crate has no feature that reaches one.
// They come back with the engines.

//! Standalone composition-root boot smoke — the Phase-2 regression guard (build/packaging).
//!
//! The `bin/pie` analogue of the gateway M3 smoke: boots the **embedded controller + gateway +
//! worker** in one process over loopback via the composition seam `pie_bin::run_standalone`
//! (= `controller::embed` → `EmbeddedControl` → `gateway::bind(worker_listen :0)` →
//! `worker::run_with(.., EmbeddedControl, [gw.worker_addr])` → `gw.serve()`, binding ephemeral),
//! then proves the planes co-reside + the real client path round-trips one turn.
//!
//! **One boot per test process.** The runtime owns process-global singletons — `auth` panics
//! "Service already spawned" on a 2nd boot (`runtime/src/auth.rs:31`) and the dummy driver grabs a
//! fixed POSIX shmem (`/pie_shmem_g0`) — so the gate is a *single* boot-once test that runs both
//! the Tier-1 plane/addr checks and the ping-through-ingress check sequentially. (The same
//! constraint applies to the `run` follow-on: one standalone per process.)
//!
//! Compiles against delta's `run_standalone` stub today (it `bail!`s); goes green the instant golf's
//! P5a `compose.rs` overlay lands in the consolidation assembly — that green is the Phase-2 gate.
//!
//! Fixture: `tests/fixtures/smoke-model/tokenizer.json` — a real **256-token byte-level-BPE** tokenizer
//! (charlie's pure-stdlib generator replicating `model/tokenizer/bpe.rs build_byte_to_unicode` exactly;
//! `model.type=BPE`, `pre_tokenizer.type=ByteLevel`, empty `merges` → each byte = 1 token). **Boot-validated**
//! (booted `bin/worker` → exit 0). The runtime parses the tokenizer at boot unconditionally
//! (`model::register` → `Tokenizer::from_file`), so it must be valid — this is. `Ping` never tokenizes
//! (broker `server.rs:721`), so it covers boot + ping; a real *generation* turn (the `#[ignore]`d
//! fast-follow) additionally needs an inferlet program + `driver.vocab_size = 256`.

use anyhow::Result;
use pie_bin::derive::derive_standalone;
use pie_bin::{Mode, run_standalone};

/// The one standalone TOML (`[controller]`/`[gateway]`/`[worker]` sections); `derive_standalone`
/// splits + hands each section to its role lib's `Config::parse`. `Mode::Local` pins the client edge
/// to loopback but keeps the *configured port* (so `pie local` has a predictable address), so the test
/// must itself request an ephemeral one — `[gateway] listen = 127.0.0.1:0` — else both checks collide on
/// the `0.0.0.0:8080` default ("Address already in use"). `worker_listen` is already forced ephemeral by
/// compose. The worker runs the always-linked **dummy** driver against a local snapshot (no GPU, no
/// download — R3); auth off. The dummy driver's `[..options]` are explicit (`vocab_size = 256` matches
/// the 256-token fixture; `arch_name` required — the fixture dir has only `tokenizer.json`, no
/// `config.json` for the standalone to auto-discover them from).
fn standalone_toml(snapshot: &str) -> String {
    format!(
        "[controller]\n\
         \n\
         [gateway]\n\
         listen = \"127.0.0.1:0\"\n\
         \n\
         [worker]\n\
         [worker.auth]\n\
         enabled = false\n\
         \n\
         [[worker.model]]\n\
         name = \"smoke\"\n\
         hf_repo = \"{snapshot}\"\n\
         \n\
         [worker.model.driver]\n\
         type = \"dummy\"\n\
         device = [\"cpu\"]\n\
         \n\
         [worker.model.driver.options]\n\
         vocab_size = 256\n\
         arch_name = \"qwen3\"\n"
    )
}

/// Absolute path to the committed fixture snapshot dir (contains `tokenizer.json`).
fn fixture_snapshot() -> String {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures/smoke-model");
    p.to_string_lossy().into_owned()
}

async fn boot() -> Result<pie_bin::StandaloneHandle> {
    let (controller, gateway, worker) = derive_standalone(&standalone_toml(&fixture_snapshot()))?;
    run_standalone(controller, gateway, worker, Mode::Local).await
}

/// THE GATE — boots ONCE (process-global singletons forbid a 2nd boot) and runs both checks:
/// (1) Tier-1: the composition root assembles all three planes in-proc over loopback, the worker
/// dials in, and `StandaloneHandle` surfaces both resolved ephemeral addrs. (2) Ping: a `Ping`
/// round-trips the real client path (REST → ingress → session → dispatch → worker → push_tokens →
/// SSE) without tokenization, proving the whole path is wired end-to-end, not just boot. Then drains.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn standalone_boots_three_planes_pings_and_drains() -> Result<()> {
    let pie = boot().await?;

    // (1) Tier-1: three planes co-reside, worker dialed in, both ephemeral loopback addrs resolved.
    assert_ne!(
        pie.listen_addr.port(),
        0,
        "client-facing edge must bind a real ephemeral port"
    );
    assert_ne!(
        pie.worker_addr.port(),
        0,
        "worker dial-in must bind a real ephemeral port (worker co-resides + dialed in)"
    );
    assert!(
        pie.listen_addr.ip().is_loopback() && pie.worker_addr.ip().is_loopback(),
        "standalone is loopback-only"
    );

    // (2) Ping through ingress — exercises the full client path with no tokenization. Serialize the
    // body with serde_json (a bin/pie dep) — reqwest here lacks the `json` feature.
    let payload = serde_json::to_vec(&serde_json::json!({ "type": "ping", "corr_id": 1 }))?;
    let resp = reqwest::Client::new()
        .post(format!("http://{}/v1/generate", pie.listen_addr))
        .header("x-pie-identity", "smoke/test") // REQUIRED trust-edge gate (else 401)
        .header("content-type", "application/json")
        .body(payload)
        .send()
        .await?;
    assert_eq!(resp.status(), 200, "ingress one-shot must accept the turn");

    let body = resp.text().await?; // tiny for a single Ping turn
    assert!(
        body.contains("[DONE]"),
        "the turn must stream back and reach the clean [DONE] sentinel; got: {body}"
    );

    pie.shutdown().await; // drain all three planes cleanly
    Ok(())
}

/// FAST-FOLLOW (non-gating, `#[ignore]`d so it never co-boots with the gate). A real *generation*
/// turn that actually tokenizes + decodes. The tokenizer fixture is ready (boot-validated 256-token
/// BPE); the remaining blocker is the turn itself: a generation goes through an **inferlet program**
/// (`AddProgram` + `LaunchProcess`) — there is no self-contained "generate text" `ClientMessage` —
/// and the dummy driver's `vocab_size = 256` (already set) keeps its random ids decoding through the
/// 256 tokens. Un-ignore once an inferlet fixture lands (and run it in its own process — see the
/// one-boot-per-process constraint above).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "fast-follow: a generation turn needs an inferlet program (AddProgram+LaunchProcess); tokenizer fixture + vocab_size=256 are ready; one boot per process"]
async fn standalone_generates_a_real_turn() -> Result<()> {
    let pie = boot().await?;
    // TODO: AddProgram(inferlet) + LaunchProcess → drain SSE token chunks then [DONE].
    pie.shutdown().await;
    Ok(())
}
